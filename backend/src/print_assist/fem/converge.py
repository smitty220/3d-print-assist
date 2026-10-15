"""Converged peak stresses: refine the mesh locally at each hot spot and solve again until its peak stops moving.

A hot spot is a local von Mises peak within BAND of the governing one. Each round puts a sphere of influence on every
spot still converging, its element size the spot's measured local size over RATIO, and re-meshes the whole part (the
rest of the mesh is unchanged, so the model grows little). A spot's peak is the largest von Mises within its sphere.

  converged  the peak moved less than `tol` since the previous mesh
  singular   the peak keeps rising and the change is not shrinking: a sharp re-entrant corner or a point load,
             where refining raises the peak without limit (it needs a fillet, not a finer mesh)
  limit      out of rounds (or degrees of freedom) before either

For a smooth peak and quadratic elements each change is about RATIO^-2 (0.44) of the one before; at a sharp 90-degree
inside corner it stays about the same (stress ~ h^-0.46), so SINGULAR_RATIO sits between the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controls import MeshSetup, Sphere
from .mesh import Mesh

BAND = 0.10  # spots within 10% of the governing von Mises
MAX_SPOTS = 4  # per load case
RATIO = 1.5  # each round's local element size: the last mesh's measured size over this
RADIUS = 3.0  # sphere radius, in local element sizes on the first mesh
TOL = 0.05
MIN_REFINE = 1.3  # a spot is judged against the last mesh at least this much coarser there (Gmsh can fall short of a size)
MAX_ROUNDS = 4
SINGULAR_RATIO = 0.7
MAX_DOF = 1_500_000  # about 10 GB to solve


@dataclass
class Spot:
    id: str
    case: str
    region: str
    center: list[float]
    radius: float
    history: list[dict] = field(default_factory=list)  # per mesh: {"dof", "h" (mm), "vm" (MPa)}
    status: str = "converging"  # converging | converged | singular | limit
    factor: float = 1.0  # contact / cosine at this spot on the first mesh (when the rounds solve the cosine model)

    @property
    def vm(self) -> float:
        return self.history[-1]["vm"] * self.factor

    def change(self, k: int = -1) -> float | None:
        """Relative change of the peak into mesh k from the one before."""
        h = self.history
        n = len(h) + k if k < 0 else k
        return h[n]["vm"] / h[n - 1]["vm"] - 1 if n >= 1 else None

    def reference(self) -> dict | None:
        """The latest earlier mesh at least MIN_REFINE coarser at this spot than the last one: what convergence is
        judged against."""
        h = self.history[-1]["h"]
        return next((x for x in reversed(self.history[:-1]) if x["h"] >= MIN_REFINE * h), None)

    def extrapolated(self) -> float | None:
        """Richardson-style estimate of the mesh-independent peak from the last three meshes (changes shrinking
        geometrically), or None."""
        if len(self.history) < 3:
            return None
        d1 = self.history[-2]["vm"] - self.history[-3]["vm"]
        d2 = self.history[-1]["vm"] - self.history[-2]["vm"]
        if abs(d1) < 1e-12 or not 0 < d2 / d1 < SINGULAR_RATIO:
            return None
        q = d2 / d1
        return (self.history[-1]["vm"] + d2 * q / (1 - q)) * self.factor

    def to_dict(self) -> dict:
        return {"id": self.id, "case": self.case, "region": self.region, "center": self.center, "radius": round(self.radius, 3),
                "history": self.history, "status": self.status, "factor": self.factor, "vm": self.vm,
                "change": self.change() if (r := self.reference()) is None else self.history[-1]["vm"] / r["vm"] - 1,
                "extrapolated": self.extrapolated()}


def local_size(mesh: Mesh, center, n: int = 30) -> float:
    """Mean corner-edge length of the `n` tetrahedra nearest `center` (mm)."""
    c = mesh.nodes[mesh.tets[:, :4]]
    d = np.linalg.norm(c.mean(axis=1) - np.asarray(center, float), axis=1)
    near = c[np.argsort(d)[:n]]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return float(np.mean([np.linalg.norm(near[:, i] - near[:, j], axis=1) for i, j in pairs]))


def spot_peak(mesh: Mesh, vm: np.ndarray, center, radius: float) -> float:
    d = np.linalg.norm(mesh.nodes - np.asarray(center, float), axis=1)
    inside = d <= radius
    return float(vm[inside].max()) if inside.any() else float(vm[np.argmin(d)])


def hot_spots(mesh: Mesh, vm_by_case: dict[str, np.ndarray], region_of=None, band: float = BAND, max_spots: int = MAX_SPOTS) -> list[Spot]:
    """Local peaks within `band` of each case's governing von Mises, at least 1.5 sphere radii apart, highest first."""
    out = []
    for case, vm in vm_by_case.items():
        top = vm.max()
        cand = np.flatnonzero(vm >= (1 - band) * top)
        cand = cand[np.argsort(-vm[cand])]
        picked: list[Spot] = []
        for i in cand:
            x = mesh.nodes[i]
            if any(np.linalg.norm(x - s.center) < 1.5 * s.radius for s in picked):
                continue
            h = local_size(mesh, x)
            s = Spot(f"{case}/{len(picked) + 1}", case, region_of(x) if region_of else "", x.round(3).tolist(), RADIUS * h)
            s.history.append({"dof": int(mesh.n_dof), "h": round(h, 4), "vm": float(vm[i])})
            picked.append(s)
            if len(picked) >= max_spots:
                break
        out += picked
    return out


def refined_setup(setup: MeshSetup, spots: list[Spot], sizes: dict[str, float]) -> MeshSetup:
    """`setup` plus a sphere of influence per spot at its target size. Edge divisions near a spot are raised to match
    (a divided edge ignores size fields), and the minimum size lowered to let the spheres through."""
    s = setup.model_copy(deep=True)
    for sp in spots:
        h = sizes[sp.id]
        s.spheres.append(Sphere(center=sp.center, radius=round(sp.radius, 4), size=round(h, 4), label=f"peak {sp.id}"))
        for e in s.edges:
            if np.linalg.norm(np.asarray(e.ref) - sp.center) < sp.radius + e.length / 2:
                e.n = max(e.n, int(np.ceil(e.length / h)))
        s.min_size = min(s.min_size, round(0.5 * h, 4))
    s.notes = s.notes + [f"Peak convergence: {len(spots)} refinement sphere(s)"]
    return s


def converge(setup: MeshSetup, mesh: Mesh, vm_by_case: dict[str, np.ndarray], run, region_of=None, base_fields=None,
             tol: float = TOL, max_rounds: int = MAX_ROUNDS, log=None, cancel=None, on_round=None) -> list[Spot]:
    """Refine at the hot spots of the solved `mesh` until each peak converges (or is found singular).

    `run(setup, round)` meshes the part with `setup`, solves it and returns (mesh, {case: von Mises per node}).
    `base_fields`: when the rounds solve a different (faster) model than `vm_by_case` came from, that model's von Mises
    on `mesh`. Spots are still found on `vm_by_case`; their histories are in the round model, and each spot's factor
    (the two models' peaks in its sphere, e.g. contact / cosine) carries the converged value back.
    `on_round(round, spots)` is told after each round (progress)."""
    say = log or (lambda *_: None)
    spots = hot_spots(mesh, vm_by_case, region_of)
    for sp in spots:
        peak = spot_peak(mesh, vm_by_case[sp.case], sp.center, sp.radius)
        sp.history[0]["vm"] = peak
        if base_fields is not None:
            sp.history[0]["vm"] = spot_peak(mesh, base_fields[sp.case], sp.center, sp.radius)
            sp.factor = peak / sp.history[0]["vm"]
    say(f"{len(spots)} hot spot(s) within {BAND:.0%} of the peak: "
        + "; ".join(f"{sp.id} {sp.region} {sp.history[0]['vm'] * sp.factor:.0f} MPa (element {sp.history[0]['h']:.2f} mm"
                    + (f", contact / cosine {sp.factor:.2f}" if base_fields is not None else "") + ")" for sp in spots))
    sizes = {sp.id: sp.history[0]["h"] for sp in spots}
    push = {sp.id: 1.0 for sp in spots}  # how far Gmsh fell short of each spot's last target (next target times this)
    last = mesh
    for rnd in range(1, max_rounds + 1):
        live = [sp for sp in spots if sp.status == "converging"]
        if not live or (cancel is not None and cancel.is_set()):
            break
        for sp in live:
            sizes[sp.id] = local_size(last, sp.center) / RATIO * push[sp.id]
        s = refined_setup(setup, spots, sizes)
        say(f"Round {rnd}: refining {len(live)} spot(s) to " + ", ".join(f"{sp.id} {sizes[sp.id]:.2f} mm" for sp in live))
        last, fields = run(s, rnd)
        for sp in spots:
            if sp.status != "converging":
                continue
            h = local_size(last, sp.center)
            push[sp.id] = float(np.clip(push[sp.id] * sizes[sp.id] / h, 0.3, 1.0))
            sp.history.append({"dof": int(last.n_dof), "h": round(h, 4), "vm": spot_peak(last, fields[sp.case], sp.center, sp.radius)})
            d, prev = sp.change(), sp.change(-2)
            ref = sp.reference()
            if ref is not None and abs(sp.history[-1]["vm"] / ref["vm"] - 1) < tol:
                sp.status = "converged"
            elif prev is not None and d > 0 and prev > 0 and d >= SINGULAR_RATIO * prev:
                sp.status = "singular"
            note = {"converged": f"converged (against the {ref['h']:.2f} mm mesh)" if ref else "", "singular": "not settling: a singularity",
                    "converging": "still moving" if ref else f"mesh only {sp.history[-2]['h'] / h:.2f}x finer here; refining further"}[sp.status]
            say(f"  {sp.id} {sp.region}: {sp.history[-2]['vm']:.0f} -> {sp.history[-1]['vm']:.0f} MPa ({d:+.1%}) at "
                f"{h:.2f} mm, {last.n_dof:,} DOF: {note}")
        if on_round:
            on_round(rnd, spots)
        if last.n_dof > MAX_DOF:
            say(f"Stopping: {last.n_dof:,} DOF is the most this machine should solve")
            break
    for sp in spots:
        if sp.status == "converging":
            sp.status = "limit"
    return spots
