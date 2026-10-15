"""Background topology-optimisation runs: a sweep of mass targets, each warm-started from the last.

Results (densities per step plus the grid placement) are saved under <project>/topo/ so they survive restarts,
with a key of the intent they were run on so the UI can say when they are out of date.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from .machining import MachiningSpec
from .prismatic import Prismatic25D, setup_faces
from .setup import Setup, build_setup
from .simp import Optimizer

SWEEP_FACTORS = [1.5, 1.0, 0.65]  # of the mass target: a step either side, warm-started high to low
FIRST_ITERS, NEXT_ITERS = 100, 50
MACHINING_ITERS = 150  # warm-started from the raw result; the clevis (3 depths) took 14
MIN_FRACTION, MAX_FRACTION = 0.02, 0.8


def intent_key(project) -> str:
    d = project.data
    blob = d.intent.model_dump_json() + json.dumps({k: v.role for k, v in d.parts.items()}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class TopoRunError(ValueError):
    pass


def run_fractions(setup: Setup, mass: float | None, sweep: bool) -> tuple[list[float], int]:
    """Volume fractions to optimise for (high to low) and which one is the mass target.

    One run at the mass target by default; a sweep adds a heavier and a lighter step around it.
    """
    if not mass or mass <= 0:
        raise TopoRunError("Set a mass target (Design rules, or the field in this panel) to optimise for")
    t = mass / setup.full_mass()
    if not MIN_FRACTION <= t <= MAX_FRACTION:
        raise TopoRunError(f"A {mass / 0.45359237:.3f} lb target is {100 * t:.1f}% of the design space; it needs to be "
                           f"between {100 * MIN_FRACTION:.0f}% and {100 * MAX_FRACTION:.0f}%")
    if not sweep:
        return [round(t, 4)], 0
    fr = [round(min(MAX_FRACTION, max(MIN_FRACTION, t * k)), 4) for k in SWEEP_FACTORS]
    return fr, SWEEP_FACTORS.index(1.0)


def cut_to_count(part, n: float, active: np.ndarray, iters: int = 30) -> np.ndarray:
    """Bisect a cut level in (0, 1) so the 0/1 part `part(cut)` has as close to n active elements as it can."""
    lo, hi = 0.0, 1.0
    best = part(0.5)
    for _ in range(iters):
        mid = (lo + hi) / 2
        cand = part(mid)
        k = int((cand & active).sum())
        if abs(k - n) < abs(int((best & active).sum()) - n):
            best = cand
        if k > n:
            lo = mid
        else:
            hi = mid
    return best


@dataclass
class StepInfo:
    index: int
    fraction: float
    mass: float  # kg, from the (grey) densities
    compliance: list[float]  # N·mm per load case
    iterations: int
    converged: bool
    seconds: float
    label: str = "raw"  # "raw" or the machining set-up it was constrained to
    # As a 0/1 part: the raw result cut at 0.5, or the sharp machined part. The like-for-like comparison.
    solid_mass: float = 0.0  # kg
    solid_compliance: list[float] = field(default_factory=list)  # N·mm per load case


@dataclass
class Result:
    """A finished (or partly finished) sweep: what the viewer needs to draw any step."""

    key: str
    budget: int
    grid: list[int]
    h: float
    frame_origin: list[float]
    origin: list[float]
    axes: list[list[float]]
    density: float
    void: np.ndarray
    steps: list[StepInfo] = field(default_factory=list)
    x: list[np.ndarray] = field(default_factory=list)
    target_step: int | None = None
    setup: dict = field(default_factory=dict)
    machining: dict | None = None  # the MachiningSpec used, as a dict

    def meta(self) -> dict:
        return {
            "key": self.key, "budget": self.budget, "grid": self.grid, "h": self.h, "target_step": self.target_step,
            "steps": [s.__dict__ for s in self.steps], "setup": self.setup, "machining": self.machining,
        }

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        meta = self.meta() | {"frame_origin": self.frame_origin, "origin": self.origin, "axes": self.axes, "density": self.density}
        arrays = {"void": self.void} | {f"x{i}": (x * 255).round().astype(np.uint8) for i, x in enumerate(self.x)}
        tmp = folder / "result.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(folder / "result.npz")
        (folder / "result.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, folder: Path) -> Result | None:
        try:
            meta = json.loads((folder / "result.json").read_text())
            arr = np.load(folder / "result.npz")
        except (OSError, ValueError):
            return None
        steps = [StepInfo(**s) for s in meta["steps"]]
        return cls(meta["key"], meta["budget"], meta["grid"], meta["h"], meta["frame_origin"], meta["origin"], meta["axes"],
                   meta["density"], arr["void"], steps, [arr[f"x{i}"] / 255.0 for i in range(len(steps))],
                   meta.get("target_step"), meta.get("setup", {}), meta.get("machining"))

    def to_world(self, grid_pts: np.ndarray) -> np.ndarray:
        return np.asarray(self.frame_origin) + (grid_pts + np.asarray(self.origin)) @ np.asarray(self.axes)

    def mesh(self, x: np.ndarray, cutoff: float) -> dict:
        """Iso-surface of the density field at `cutoff`, in world mm, plus the mass of everything above it."""
        cutoff = float(min(0.95, max(0.05, cutoff)))
        mass = float((x >= cutoff).sum() * self.h**3 * self.density)
        padded = np.pad(x, 1)  # closes the surface at the edges of the box
        if padded.max() < cutoff:
            return {"positions": [], "indices": [], "mass": 0.0, "cutoff": cutoff}
        verts, faces, _n, _v = marching_cubes(padded, level=cutoff, spacing=(self.h,) * 3)
        world = self.to_world(verts - 0.5 * self.h)  # padded index p is element p-1, centred at (p-0.5)·h
        return {"positions": world.round(3).ravel().tolist(), "indices": faces.ravel().tolist(), "mass": mass, "cutoff": cutoff}


@dataclass
class TopoJob:
    id: str
    budget: int
    status: str = "running"  # running | done | failed | cancelled
    phase: str = "setup"  # setup | optimising | done
    fractions: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    step: int = 0
    iteration: int = 0
    history: list[dict] = field(default_factory=list)  # {step, it, c, vol, s}
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    setup: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    live: np.ndarray | None = None

    def to_dict(self) -> dict:
        skip = {"cancel", "live"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}


class Runner:
    def __init__(self) -> None:
        self.job: TopoJob | None = None
        self.result: Result | None = None  # last saved sweep
        self.current: Result | None = None  # the sweep being built (grid placement for live views)
        self._thread: threading.Thread | None = None
        self._folder: Path | None = None

    def attach(self, folder: Path) -> None:
        """Load the last saved result for a project (once)."""
        if self._folder != folder:
            self._folder = folder
            self.result = Result.load(folder)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, project, budget: int, mass: float | None = None, sweep: bool = False,
              machining: MachiningSpec | None = None, wait: bool = False) -> TopoJob:
        """Optimise for the mass target. With `machining`, the raw optimum is found first and then constrained to
        the machining set-up, warm-started from it (from scratch the clevis stalled at 1.7x the raw compliance),
        so every machinable result comes with the raw one at the same mass for comparison."""
        if self.running():
            raise RuntimeError("A topology optimisation is already running")
        roles = {k: v.role for k, v in project.data.parts.items()}
        intent = project.data.intent.model_copy(deep=True)
        setup = build_setup(intent, project.shape, roles, budget)  # raises TopoSetupError before a job exists
        if machining is not None and sweep:
            raise TopoRunError("A machinable run already compares against the raw result; sweep raw runs only")
        fractions, target_step = run_fractions(setup, mass if mass is not None else intent.rules.mass_target, sweep)
        labels = ["raw"] * len(fractions)
        if machining is not None:
            setup.problem.filter_radius = machining.filter_radius(setup.grid.h)  # same for both, a fair comparison
            fractions, labels, target_step = fractions * 2, ["raw", machining.label(setup.grid.shape)], 1
        job = TopoJob(id=uuid.uuid4().hex[:8], budget=setup.budget, setup=setup.summary())
        g = setup.grid
        res = Result(intent_key(project), setup.budget, [g.nx, g.ny, g.nz], g.h, list(setup.frame.origin), setup.origin.tolist(),
                     setup.axes.tolist(), setup.density, setup.problem.void, setup=job.setup)
        self.job, self.current = job, res
        folder = project.root / "topo"
        self._folder = folder
        job.fractions, job.labels, res.target_step = fractions, labels, target_step
        res.machining = machining.__dict__ if machining is not None else None
        args = (job, setup, res, folder, machining)
        self._thread = threading.Thread(target=self._run, args=args, daemon=True)
        self._thread.start()
        if wait:
            self._thread.join()
        return job

    def cancel(self) -> None:
        if self.job is not None:
            self.job.cancel.set()

    def _run(self, job: TopoJob, setup: Setup, res: Result, folder: Path, machining: MachiningSpec | None) -> None:
        try:
            fractions = job.fractions
            opt = Optimizer(setup.problem)
            g = setup.grid
            job.phase = "optimising"
            raw = None
            for i, frac in enumerate(fractions):
                job.step, t0 = i, time.time()

                def on_iter(st, xp, i=i):
                    job.iteration = st.it
                    job.history.append({"step": i, "it": st.it, "c": st.compliance, "vol": st.volume, "s": st.seconds})
                    job.live = xp.astype(np.float32)
                    return not job.cancel.is_set()

                if job.labels[i] == "raw":
                    r = opt.run(frac, x0=raw, max_iter=FIRST_ITERS if raw is None else NEXT_ITERS, on_iter=on_iter)
                else:
                    shape = setup.grid.shape
                    design = Prismatic25D(setup.problem, setup_faces(machining.axis_index(shape), machining.open_end()),
                                          machining.depths, setup.problem.filter_radius)
                    r = Optimizer(setup.problem, design).run(frac, x0=design.from_physical(res.x[-1]),
                                                             max_iter=MACHINING_ITERS, on_iter=on_iter)
                if job.cancel.is_set():
                    break
                raw = r.raw
                mass = float(r.x[~setup.problem.void].sum() * g.h**3 * setup.density)
                # Like-for-like: both become 0/1 parts at exactly the target mass (raw by a density cutoff, the
                # machined part by its map cut level), and are solved as such. The machined part is shown sharp.
                n_target = frac * opt.available
                if job.labels[i] == "raw":
                    solid = cut_to_count(lambda c: r.x >= c, n_target, ~setup.problem.void)
                    shown = r.x
                else:
                    solid = cut_to_count(lambda c: design.sharp(r.raw, c) >= 0.5, n_target, ~setup.problem.void)
                    shown = solid.astype(float)
                solid_c = opt.solid_compliance(solid)
                solid_m = float(solid[~setup.problem.void].sum() * g.h**3 * setup.density)
                res.steps.append(StepInfo(i, frac, mass, r.compliance, len(r.history), r.converged, time.time() - t0,
                                          job.labels[i], solid_m, solid_c))
                res.x.append(shown.astype(np.float32))
                res.save(folder)
                self.result = res
            job.status = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:  # report, don't kill the server
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.phase = "done"
            job.finished = time.time()
            job.live = None
