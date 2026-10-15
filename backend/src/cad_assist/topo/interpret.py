"""Read a machinable (2.5D) topology result as a machined plate and write it as parametric features.

The plate axis is the design frame's Z. Seen down Z the result is a profile (the outline), through cut-outs (holes in
the outline other than the pins) and stepped thicknesses: the top and bottom faces sit at a few levels, which become
pockets milled from +Z and -Z. Every choice errs towards keeping material, so the model is never weaker than the
voxels it came from: a merged level takes its shallowest depth, pocket and cut-out outlines are opened by the cutter
radius (a cutter can't reach narrower), and a thicker island inside a pocket is kept.

The output is an ordinary feature list (profile_plate, profile_cut, pin_bore) that the user and the AI can edit.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

from build123d import IN, Shape

from ..features.context import FeatureError
from ..features.library import _profile_face
from ..features.model import FeatureInstance
from ..intent.models import Intent
from .job import Result
from .setup import Setup, clearance_shapes, inside_mask
from .simp import Optimizer

LB = 0.45359237  # kg
PLATE_COLUMNS = 0.95  # share of columns that must be one solid run through Z for the plate reading to hold
SIMPLIFY = 0.6  # outline simplification tolerance, in elements
MAX_LEVELS = 3  # floor depths per side, as the 2.5D run had


class InterpretError(ValueError):
    pass


@dataclass
class Pocket:
    side: str  # "+z" | "-z"
    depth: float  # mm
    outline: list[list[float]]
    islands: list[list[list[float]]]


@dataclass
class Reading:
    z_min: float
    z_max: float
    outline: list[list[float]]
    cutouts: list[list[list[float]]]
    pockets: list[Pocket]
    bores: list[str]
    corner_radius: float
    notes: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)  # design rules applied on top of the result
    pins: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # bore id -> (x, y, diameter), design frame
    web: dict[str, float] = field(default_factory=dict)  # side -> depth of the floor covering most of that face, mm
    spots: list[dict] = field(default_factory=list)  # {pin, side, depth, radius}: cuts a fork's tines need round a pin

    def features(self) -> list[FeatureInstance]:
        r = round(self.corner_radius, 4)
        out = [FeatureInstance(id="F1", kind="profile_plate", params={
            "outline": self.outline, "z_min": round(self.z_min, 4), "z_max": round(self.z_max, 4), "corner_radius": r})]
        cuts = [{"outline": c, "through": "yes"} for c in self.cutouts]
        cuts += [{"outline": p.outline, "islands": p.islands, "side": p.side, "depth": round(p.depth, 4)} for p in self.pockets]
        for params in cuts:
            out.append(FeatureInstance(id=f"F{len(out) + 1}", kind="profile_cut", params=params | {"corner_radius": r}, refs={"base": "F1"}))
        for b in self.bores:
            out.append(FeatureInstance(id=f"F{len(out) + 1}", kind="pin_bore", refs={"pin": b, "base": "F1"}))
        return out


def _disk(r_cells: float) -> np.ndarray:
    n = int(np.ceil(r_cells))
    ax = np.arange(-n, n + 1)
    return (ax[:, None] ** 2 + ax[None, :] ** 2) <= r_cells**2


def _trace(mask: np.ndarray, origin: np.ndarray, h: float) -> list[list[float]]:
    """Outer boundary of one connected region, simplified, as design-frame [x, y] mm."""
    contours = find_contours(np.pad(mask, 1).astype(float), 0.5)
    c = max(contours, key=len)
    c = approximate_polygon(c, tolerance=SIMPLIFY)[:-1]  # closed: last point repeats the first
    xy = origin + (c - 0.5) * h  # padded index p is element p-1, centred at (p - 0.5)·h
    return [[round(float(x), 4), round(float(y), 4)] for x, y in xy]


def _components(mask: np.ndarray, min_cells: float) -> list[np.ndarray]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return [lab == i for i in range(1, n + 1) if (lab == i).sum() >= min_cells]


def _levels(depth: np.ndarray, region: np.ndarray) -> list[int]:
    """Pocket floor depths (in elements, > 0) on one side: at most MAX_LEVELS floors, chosen so that rounding every
    column up to the floor above it adds the least material (exact, by dynamic programming over the distinct depths).
    Columns at depth 0 stay at the stock face."""
    vals, area = np.unique(depth[region], return_counts=True)
    vals, area = vals.tolist(), area.tolist()
    if vals and vals[0] == 0:
        vals, area = vals[1:], area[1:]
    m = len(vals)
    if m == 0:
        return []

    def cost(i: int, j: int) -> float:  # columns i..j share floor vals[i]
        return sum(area[k] * (vals[k] - vals[i]) for k in range(i, j + 1))

    k_max = min(MAX_LEVELS, m)
    best = {(0, j): (cost(0, j), [0]) for j in range(m)}  # (groups used - 1, last index) -> (cost, group starts)
    for g in range(1, k_max):
        for j in range(g, m):
            best[(g, j)] = min(((best[(g - 1, i - 1)][0] + cost(i, j), best[(g - 1, i - 1)][1] + [i]) for i in range(g, j + 1)),
                               key=lambda t: t[0])
    _c, starts = min((best[(g, m - 1)] for g in range(k_max)), key=lambda t: t[0])
    return [vals[i] for i in starts]


def _clearances(clear: dict, result: Result, cols: np.ndarray, mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column (design-frame mm): the highest the part may reach and the lowest, clear of every shape in `clear`
    by its gap, found by casting a ray along the design Z through the column centre."""
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector

    hi = np.full(cols.shape, np.inf)
    lo = np.full(cols.shape, -np.inf)
    o, ax, h = np.asarray(result.frame_origin, float), np.asarray(result.axes, float), result.h
    g0 = np.asarray(result.origin, float)
    idx = np.argwhere(cols)
    for shape, gap in clear.values():
        inter = IntCurvesFace_ShapeIntersector()
        inter.Load(shape.wrapped, 1e-6)
        for i, j in idx:
            # Off-grid nudge keeps rays off edges shared by faces, where hits double up (as in setup.inside_mask).
            p = o + np.array([g0[0] + (i + 0.5) * h + 1.3e-4, g0[1] + (j + 0.5) * h + 0.7e-4, 0.0]) @ ax
            inter.Perform(gp_Lin(gp_Pnt(*p), gp_Dir(*ax[2])), -1e9, 1e9)
            ws = sorted(inter.WParameter(n) for n in range(1, inter.NbPnt() + 1))
            for a, b in zip(ws[0::2], ws[1::2]):  # solid spans along the ray, in design Z
                if a >= mid[i, j]:
                    hi[i, j] = min(hi[i, j], a - gap)
                elif b <= mid[i, j]:
                    lo[i, j] = max(lo[i, j], b + gap)
    return hi, lo


def read_plate(result: Result, step: int, intent: Intent, clear: dict | None = None) -> Reading:
    x = result.x[step] >= 0.5
    h = result.h
    origin = np.asarray(result.origin, float)
    if not x.any():
        raise InterpretError("The result is empty")
    filled = x.any(axis=2)
    runs = np.abs(np.diff(np.pad(x, ((0, 0), (0, 0), (1, 1))).astype(np.int8), axis=2)).sum(axis=2) // 2
    single = (runs[filled] == 1).mean()
    if single < PLATE_COLUMNS:
        raise InterpretError(f"Only {single:.0%} of the columns are one solid run through Z, so this isn't a plate along Z")
    nx, ny, nz = x.shape
    tool_d = (result.machining or {}).get("tool_diameter", 6.35)
    r_cut = tool_d / 2
    notes: list[str] = []

    # Pins along Z: their holes are drilled, not traced.
    centres = (np.indices((nx, ny)).transpose(1, 2, 0) + 0.5) * h + origin[:2]
    bores, pin_disks, lugs = [], np.zeros((nx, ny), bool), np.zeros((nx, ny), bool)
    rules: list[str] = []
    pin_xy: dict[str, tuple[float, float, float]] = {}
    e_full = intent.rules.lug_edge_full_bearing
    axes = np.asarray(result.axes)
    for b in (o for o in intent.objects if o.kind == "bore"):
        a = axes @ np.asarray(b.frame.z, float)
        if abs(a[2]) < 0.999 * np.linalg.norm(a):
            notes.append(f"{b.id}: pin not along Z; drill it by hand")
            continue
        c = axes @ (np.asarray(b.frame.origin, float) - np.asarray(result.frame_origin, float))
        pin_xy[b.id] = (float(c[0]), float(c[1]), b.diameter)
        dist = np.linalg.norm(centres - c[:2], axis=2)
        disk = dist <= b.diameter / 2 + h
        if (filled & ndimage.binary_dilation(disk, iterations=2)).any():
            pin_disks |= disk
            bores.append(b.id)
            # Lug edge distance: the result keeps only the ring it was forced to; grow the lug to full bearing.
            e_now = (dist[~(filled | disk)].min() - h / 2) / b.diameter
            if e_now < e_full:
                lugs |= dist <= e_full * b.diameter + h / 2
                rules.append(f"{b.id}: lug edge distance e/D {e_now:.2f} in the result; grown to {e_full:g} (full bearing)")

    profile = filled | pin_disks | lugs
    parts = _components(profile, 1)
    profile = max(parts, key=lambda m: m.sum())
    if len(parts) > 1:
        notes.append(f"{len(parts) - 1} loose piece(s) dropped")
    x = x & profile[:, :, None]
    has = x.any(axis=2)
    zlo = np.where(has, np.argmax(x, axis=2), nz)
    zhi = np.where(has, nz - np.argmax(x[:, :, ::-1], axis=2), 0)
    top, bot = int(zhi[has].max()), int(zlo[has].min())
    # Added lug material takes the thickness of the nearest lug column (which already fits between the fork's tines).
    grown = profile & ~has  # includes the ring round each pin, outside the drilled hole
    if grown.any():
        _d, (ii, jj) = ndimage.distance_transform_edt(~has, return_indices=True)
        zlo, zhi = np.where(grown, zlo[ii, jj], zlo), np.where(grown, zhi[ii, jj], zhi)
        has = has | grown
    outline = _trace(ndimage.binary_fill_holes(profile), origin[:2], h)

    # Through cut-outs: holes in the profile that aren't pins.
    min_area = np.pi * (r_cut / h) ** 2
    holes = ndimage.binary_fill_holes(profile) & ~profile
    cutouts = [_trace(m, origin[:2], h) for m in _components(holes, min_area)]

    # Pockets: on each side, the columns whose face sits at least a level in from the stock face. Keeping material is a
    # preference; clearing the mating parts is not. Their faces are found exactly, by a ray down each column through
    # the real geometry (a voxel only says whether its centre is clear), and the part keeps the assembly gap from them.
    solid = has
    opener = _disk(r_cut / h)
    z0 = origin[2]
    top_mm, bot_mm = z0 + zhi * h, z0 + zlo * h  # per column, where the result's material ends
    allow_hi, allow_lo = _clearances(clear or {}, result, solid, (top_mm + bot_mm) / 2)
    col_top, col_bot = np.minimum(top_mm, allow_hi), np.maximum(bot_mm, allow_lo)
    stock_top, stock_bot = float(col_top[solid].max()), float(col_bot[solid].min())
    if stock_top < z0 + top * h - 1e-6 or stock_bot > z0 + bot * h + 1e-6:
        rules.append(f"Stock trimmed to {_inch(stock_top - stock_bot)} so the thickest part fits between the fork's tines")
    outside = ~ndimage.binary_fill_holes(profile) | holes  # no material: cutting here is free
    lo_xy, hi_xy = origin[:2] - 3 * h, origin[:2] + (np.array([nx, ny]) + 3) * h
    frame = [[float(lo_xy[0]), float(lo_xy[1])], [float(hi_xy[0]), float(lo_xy[1])], [float(hi_xy[0]), float(hi_xy[1])], [float(lo_xy[0]), float(hi_xy[1])]]
    pockets: list[Pocket] = []
    web: dict[str, float] = {}
    spots: list[dict] = []
    sides = (("+z", stock_top - top_mm, stock_top - allow_hi), ("-z", bot_mm - stock_bot, allow_lo - stock_bot))
    for side, want, need in sides:
        want = np.where(solid, np.round(want / h), 0).astype(int)  # whole elements, as the result had them
        need = np.where(solid, np.maximum(need, 0), 0.0)  # mm, exact
        levels = [d * h for d in _levels(want, solid)]
        # The floor covering most of this face: the web, as a machinist would pocket it in one go.
        if levels:
            snapped = np.array([max([L for L in levels if L <= w * h + 1e-6], default=0.0) for w in want[solid]])
            areas = {L: int((snapped == L).sum()) for L in levels}
            web[side] = max(areas, key=areas.get)
        # What each fork's tines need cut round its pin, exactly.
        for bid in bores:
            px, py, dia = pin_xy[bid]
            dist = np.linalg.norm(centres - np.array([px, py]), axis=2)
            near = solid & (need > 1e-3) & (dist <= 2.5 * e_full * dia)
            if near.any():
                spots.append({"pin": bid, "side": side, "depth": float(np.ceil(need[near].max() * 1000) / 1000),
                              "radius": float(dist[near].max() + h / 2 + r_cut),
                              "columns": (centres[near] - np.array([px, py])).tolist()})  # offsets from the pin, mm
        # Each depth a mating part needs gets a floor of its own unless a level already reaches it within half an element.
        for q in sorted({float(np.ceil(v * 1000) / 1000) for v in need[need > 1e-3]}):  # rounded up, never shallower
            if not any(q - 1e-6 <= L <= q + h / 2 for L in levels):
                # A floor just shallower would leave a step of a few thou next to this one: take it down to q.
                levels = [L for L in levels if not q - h / 2 <= L < q] + [q]
        levels.sort()
        # Open pockets are nested (a deeper one leaves standing all a shallower one did, and more). Each reuses the
        # shallower one's island outlines exactly and traces only the newly standing material, overlapping it by an
        # element, so a wall shared by two floors is one wall, not two a few thou apart.
        prev_standing, prev_islands = None, []
        for i, L in enumerate(levels):
            region = ndimage.binary_opening(solid & (want * h >= L - 1e-6), structure=opener)
            # Columns whose clearance this is the shallowest level to reach, cutter-wide.
            prev = levels[i - 1] if i else 0.0
            must = solid & (need > prev + 1e-6) & (need <= L + 1e-6)
            if must.any():
                region |= ndimage.binary_dilation(must, structure=opener) & solid
                rules.append(f"{int(must.sum())} column(s) cut {_inch(L)} from {side.upper()} to clear the mating parts")
            # A pocket that runs off the part is milled as "face down to this depth, except what stands": its outline is
            # a frame round the whole part, so it can't leave knife-edge rims along the profile; the islands are the
            # only walls traced. A pocket inside the part is traced as it is.
            comps = _components(region, min_area)
            edge = [m for m in comps if (ndimage.binary_dilation(m, iterations=2) & outside).any()]
            groups = ([np.any(edge, axis=0)] if edge else []) + [m for m in comps if not any(m is e for e in edge)]
            for gi, m in enumerate(groups):
                open_pocket = bool(edge) and gi == 0
                if open_pocket:
                    loop = frame
                    standing = solid & ~m
                    if prev_standing is not None:
                        new = ndimage.binary_dilation(standing & ~prev_standing) & standing
                        islands = prev_islands + [isl for c in _components(new, min_area) if len(isl := _trace(c, origin[:2], h)) >= 3]
                    else:
                        islands = [isl for c in _components(standing, min_area) if len(isl := _trace(c, origin[:2], h)) >= 3]
                    prev_standing, prev_islands = standing, islands
                else:
                    loop = _trace(m, origin[:2], h)
                    standing = ndimage.binary_fill_holes(m) & ~m & has & ~outside  # thicker material inside the pocket
                    # Islands smaller than the cutter can leave standing are cut with the pocket.
                    islands = [isl for c in _components(standing, min_area) if len(isl := _trace(c, origin[:2], h)) >= 3]
                try:
                    _profile_face(loop, r_cut, round_concave=False)
                except FeatureError:
                    notes.append(f"A {_inch(L)} deep pocket from {side.upper()} is narrower than the cutter, so it stays solid")
                    continue
                pockets.append(Pocket(side, L, loop, islands))
    return Reading(stock_bot, stock_top, outline, cutouts, pockets, bores, r_cut, notes, rules,
                   {b: pin_xy[b] for b in bores}, web, spots)


@dataclass
class ModelCheck:
    """The parametric part against the topology result it came from, both as 0/1 voxels on the same grid."""

    mass: float  # kg, voxels
    compliance: list[float]  # N·mm per load case
    ref_mass: float
    ref_compliance: list[float]
    solid: np.ndarray | None = None  # the model's voxels on the topology grid

    @property
    def stiffness_ratio(self) -> float:
        """Mean stiffness of the model over the reference (1.0 = as stiff; > 1 stiffer)."""
        return float(np.mean([r / c for r, c in zip(self.ref_compliance, self.compliance)]))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "solid"} | {"stiffness_ratio": self.stiffness_ratio}


def check_model(setup: Setup, result: Result, step: int, world: Shape) -> ModelCheck:
    if [setup.grid.nx, setup.grid.ny, setup.grid.nz] != result.grid:
        raise InterpretError("The set-up has changed since this result; rerun the topology optimisation")
    solid = inside_mask(world, setup.grid, setup.origin, setup.frame, setup.axes) & ~setup.problem.void
    comp = Optimizer(setup.problem).solid_compliance(solid.astype(float))
    ref = result.steps[step]
    return ModelCheck(float(solid.sum() * setup.grid.h**3 * setup.density), comp, ref.solid_mass, ref.solid_compliance, solid)


# ---------------------------------------------------------------- the job behind "Make parametric model"


@dataclass
class InterpretJob:
    id: str
    step: int
    status: str = "running"  # running | done | failed
    log: list[dict] = field(default_factory=list)  # {kind: read|rule|sketch|ai|build|skip|check|question|done|error, text}
    revision: int = 0  # bumps every time the project's design changes
    overlay: dict | None = None  # traced outlines in world mm, for the viewport
    preview: dict | None = None  # mesh of the part so far, while it is being built
    mode: str = "rules"  # rules | ai
    stress: bool = False  # AI drafts: run a coarse stress analysis on every build
    instruction: str = ""
    tokens: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    check: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def say(self, kind: str, text: str) -> None:
        self.log.append({"kind": kind, "text": text, "t": round(time.time() - self.started, 1)})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("preview", "cancel")}


def _inch(mm: float) -> str:
    return f"{mm / IN:.3f} in"


def _describe(fi: FeatureInstance) -> str:
    p = fi.params
    if fi.kind == "profile_plate":
        return (f"{fi.id} profile plate: {_inch(p['z_max'] - p['z_min'])} stock, {len(p['outline'])}-point outline, "
                f"inside corners R{_inch(p['corner_radius'])}")
    if fi.kind == "profile_cut" and p.get("through") == "yes":
        return f"{fi.id} through cut-out: {len(p['outline'])}-point outline"
    if fi.kind == "profile_cut":
        isl = f", {len(p['islands'])} island(s) left standing" if p.get("islands") else ""
        return f"{fi.id} pocket from {p['side'].upper()}: {_inch(p['depth'])} deep, {len(p['outline'])}-point outline{isl}"
    if fi.kind == "pin_bore":
        return f"{fi.id} pin bore for {fi.refs['pin']}"
    if fi.kind == "pad":
        return f"{fi.id} pad of {fi.refs['sketch']}: {_inch(p['z_max'] - p['z_min'])} thick"
    if fi.kind == "sketch_pocket":
        how = "through" if p.get("through") == "yes" else f"{_inch(p['depth'])} deep from {p['side'].upper()}"
        return f"{fi.id} pocket of {fi.refs['sketch']}: {how}"
    return f"{fi.id} {fi.kind}"


def _overlay(rd: Reading, result: Result) -> dict:
    o, ax = np.asarray(result.frame_origin, float), np.asarray(result.axes, float)

    def world(pts, z):
        a = np.array([[x, y, z] for x, y in pts], float)
        return (o + a @ ax).round(3).ravel().tolist()

    return {
        "outline": world(rd.outline, rd.z_max),
        "cutouts": [world(c, rd.z_max) for c in rd.cutouts],
        "pockets": [{"side": pk.side, "depth": pk.depth,
                     "points": world(pk.outline, rd.z_max if pk.side == "+z" else rd.z_min)} for pk in rd.pockets],
    }


def _ai_landed(job, project, result):
    """Show each AI candidate in the viewport as it is built."""
    from ..features.context import frame_plane
    from ..geometry.tessellate import mesh_shape

    def landed(design, build) -> None:
        if build.local is not None:
            job.preview = mesh_shape(frame_plane(design_frame_of(project)).location * build.local, "part").to_dict()
        job.overlay = sketch_overlay(design.sketches, result)
        job.revision += 1
    return landed


def design_frame_of(project):
    from ..features.design import design_frame

    return design_frame(project.data.intent)


def sketch_overlay(sketches, result: Result) -> dict:
    """Every sketch's curves as world polylines (drawn in the sketch plane at the part's mid-thickness for now)."""
    from ..sketch.geometry import edges

    o, ax = np.asarray(result.frame_origin, float), np.asarray(result.axes, float)
    out = []
    for sk in sketches:
        lines = []
        for e in edges(sk):
            pts = np.array([[(e @ (i / 24)).X, (e @ (i / 24)).Y, 0.0] for i in range(25)])
            lines.append((o + pts @ ax).round(3).ravel().tolist())
        out.append({"id": sk.id, "name": sk.name, "polylines": lines})
    return {"sketches": out}


class _InterpreterAI:
    def _run_ai(self, job, project, result, step, rd, setup_of, client, max_builds, model) -> None:
        from ..ai.sketch_draft import run_draft

        job.say("ai", f"Asking Claude to draft the part ({max_builds} builds at most)…")
        best = run_draft(job, project, result, step, rd, setup_of, job.instruction, client, max_builds, model,
                         on_build=_ai_landed(job, project, result), stress=job.stress)
        if best is None:
            job.error, job.status = "Claude did not produce a part that builds", "failed"
            job.say("error", job.error)
            return
        project.data.design = best["design"]
        project.save()
        project.adopt_build(best["build"])
        job.overlay = sketch_overlay(best["design"].sketches, result)
        mc = best["check"]
        job.check = mc.to_dict() | {"brep_mass": best["build"].mass}
        job.revision += 1
        job.say("check", f"Kept build {best['n']}: {best['build'].mass / LB:.3f} lb, stiffness {mc.stiffness_ratio - 1:+.1%} vs the result")
        job.status = "done"


class Interpreter(_InterpreterAI):
    def __init__(self) -> None:
        self.job: InterpretJob | None = None
        self._thread: threading.Thread | None = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, project, result: Result, step: int, setup_of: Callable[[], Setup], mode: str = "rules",
              instruction: str = "", client=None, max_builds: int = 5, model: str = "", stress: bool = False) -> InterpretJob:
        if self.running():
            raise RuntimeError("Already making a parametric model")
        self.job = InterpretJob(id=uuid.uuid4().hex[:8], step=step, mode=mode, instruction=instruction, stress=stress)
        args = (self.job, project, result, step, setup_of, client, max_builds, model)
        self._thread = threading.Thread(target=self._run, args=args, daemon=True)
        self._thread.start()
        return self.job

    def cancel(self) -> None:
        if self.running() and self.job is not None:
            self.job.cancel.set()

    def _run(self, job: InterpretJob, project, result: Result, step: int, setup_of, client=None, max_builds: int = 5, model: str = "") -> None:
        from ..features.context import frame_plane
        from ..features.design import build_design, intent_density
        from ..geometry.tessellate import mesh_shape
        from ..sketch.solve import solve
        from .sketch_seed import seed_design

        try:
            label = result.steps[step].label
            job.say("read", f"Reading the {label} result as a plate machined from +Z and −Z")
            roles = {k: v.role for k, v in project.data.parts.items()}
            rd = read_plate(result, step, project.data.intent, clearance_shapes(project.data.intent, project.shape, roles))
            job.say("read", f"Stock {_inch(rd.z_max - rd.z_min)} thick (Z {_inch(rd.z_min)} to {_inch(rd.z_max)}); "
                            f"{len(rd.pins)} pins; {len(rd.cutouts)} cut-out(s) in the result")
            job.say("read", "Floors covering most of each face: " + ", ".join(f"{_inch(d)} from {sd.upper()}" for sd, d in rd.web.items())
                    + f"; every inside corner ≥ R{_inch(rd.corner_radius)} (the cutter)")
            for n in rd.rules:
                job.say("rule", n)
            if job.mode == "ai":
                self._run_ai(job, project, result, step, rd, setup_of, client, max_builds, model)
                return
            design, notes = seed_design(rd, result.h)
            for n in notes:
                job.say("sketch", n)
            solved = []
            for sk in design.sketches:
                sk = solve(sk)
                r = sk.report
                ents = sum(1 for e in sk.entities if e.kind != "point")
                state = "fully constrained" if r.dof == 0 else f"{r.dof} degree(s) of freedom left to drag"
                trouble = f"; conflicting {', '.join(r.conflicting)}" if r.conflicting else ""
                trouble += f"; redundant {', '.join(r.redundant)}" if r.redundant else ""
                job.say("sketch" if r.status == "solved" and not trouble else "skip",
                        f"{sk.id} {sk.name}: {ents} curves, {len(sk.constraints)} constraints, {state}{trouble}")
                solved.append(sk)
            design.sketches = solved
            job.overlay = sketch_overlay(design.sketches, result)
            feats = design.features
            by_id = {fi.id: fi for fi in feats}

            def landed(fr, part, frame) -> None:
                fi = by_id[fr.id]
                if part is not None:
                    world = frame_plane(frame).location * part
                    job.preview = mesh_shape(world, "part").to_dict()
                    job.revision += 1
                mass = part.volume * 1e-9 * intent_density(project.data.intent) if part is not None else 0.0
                if fr.error:
                    job.say("skip", f"{fi.id} {fi.kind} failed: {fr.error}")
                else:
                    job.say("build", f"{_describe(fi)} · part now {mass / LB:.3f} lb")

            build = build_design(project.data.intent, design, on_feature=landed)
            project.data.design = design
            project.save()
            project.adopt_build(build)
            job.revision += 1
            job.say("check", "Voxelising the part onto the topology mesh to compare stiffness at the same supports and load")
            mc = check_model(setup_of(), result, step, build.world)
            job.check = mc.to_dict() | {"brep_mass": build.mass}
            ratio = mc.stiffness_ratio
            job.say("check", f"{mc.mass / LB:.3f} lb vs {mc.ref_mass / LB:.3f} lb ({mc.mass / mc.ref_mass - 1:+.0%}); "
                             f"stiffness {ratio - 1:+.1%} vs the {label} result")
            fails = [c for c in build.checks if c.status != "ok"]
            job.say("done", f"{len(feats)} features. Rule checks: {sum(c.status == 'fail' for c in fails)} failing, "
                            f"{sum(c.status == 'warn' for c in fails)} warnings. Edit any feature to change it.")
            job.status = "done"
        except (InterpretError, FeatureError) as exc:
            job.error, job.status = str(exc), "failed"
            job.say("error", str(exc))
        except Exception as exc:  # noqa: BLE001 - report kernel failures in the log rather than dying silently
            job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"
            job.say("error", job.error)
        finally:
            job.finished = time.time()
