"""Stress analysis of the designed part: mesh it, put every pin's force on its bore as cosine bearing pressure (the
supports' reactions from statics), solve, and report stresses by region with margins of safety.

Supports and loads act through pins along a common axis (a bell crank, a lever, a fitting on pins). The reactions
come from equilibrium in the plane square to the pins (a pin reacts any force in that plane, a link only along its
rod), so the load set on the part is self-balanced and the part is held only against rigid-body drift: three points,
whose reactions are reported and should be zero. Margins use von Mises stress against the material allowables:
limit load against Fty, ultimate load (limit x factor) against Ftu.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..intent.models import Intent, resolved_support
from .element import principal, surface_points, traction_load, von_mises
from .controls import PRESETS, MeshSetup, estimate_gb, propose
from .mesh import Mesh, mesh_shape
from .solve import solve

LBF = 4.4482216  # N


class AnalysisError(ValueError):
    pass


@dataclass
class PinForce:
    id: str
    role: str  # load | pin | link
    force: np.ndarray  # (3,) N on the part, world


@dataclass
class Region:
    name: str
    vm: float  # MPa, peak von Mises
    p1: float  # MPa, peak max principal
    at: list[float]  # world mm, where the von Mises peak is


@dataclass
class CaseResult:
    id: str
    label: str
    category: str
    factor: float
    pins: list[PinForce]
    regions: list[Region]
    vm_max: float
    p1_max: float
    at: list[float]
    u_max: float  # mm
    ms_yield: float
    ms_ultimate: float
    residual: float  # N, largest support reaction (a balanced load set gives ~0)
    bearing: str = "cosine"  # cosine (assumed bearing pressure) | contact (pins and forks, frictionless contact)
    contact: dict = field(default_factory=dict)  # per pin: {"p_max" MPa, "pin_vm" MPa, "statics" N}; "passes"
    calibration: dict = field(default_factory=dict)  # region -> contact peak / cosine peak on the same mesh
    cosine_vm: np.ndarray = field(repr=False, default=None)  # contact runs: the cosine model's von Mises per node, same mesh
    vm_nodes: np.ndarray = field(repr=False, default=None)
    p1_nodes: np.ndarray = field(repr=False, default=None)
    u_nodes: np.ndarray = field(repr=False, default=None)
    disp: np.ndarray = field(repr=False, default=None)  # (N, 3) mm


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def pin_forces(intent: Intent, case, shape_of) -> list[PinForce]:
    """Forces on the part at every pin for one load case: the applied load plus the reactions that balance it."""
    from ..topo.setup import link_direction

    bores = {o.id: o for o in intent.objects if o.kind == "bore"}
    tgt = intent.get(case.target)
    if tgt is None or tgt.kind != "bore":
        raise AnalysisError(f"{case.id}: stress analysis handles loads applied at pin bores; {case.target} is not one")
    axis = _unit(tgt.frame.z)
    for b in bores.values():
        if abs(abs(_unit(b.frame.z) @ axis) - 1) > 1e-3:
            raise AnalysisError("Stress analysis needs every pin on a common axis direction (a plate part)")
    F = np.asarray(case.force, float)
    M = np.asarray(case.moment, float)
    if abs(F @ axis) > 1e-6 * max(1.0, np.linalg.norm(F)) or np.linalg.norm(M - axis * (M @ axis)) > 1e-6:
        raise AnalysisError(f"{case.id}: out-of-plane load components are not handled yet")
    # Plane coordinates: e1, e2 square to the pins.
    e1 = _unit(np.cross(axis, [1, 0, 0]) if abs(axis[0]) < 0.9 else np.cross(axis, [0, 1, 0]))
    e2 = np.cross(axis, e1)
    unknowns, cols = [], []  # (bore id, direction) per unknown force component
    for b in bores.values():
        role = resolved_support(intent, b)
        if role in ("pin", "fixed"):
            unknowns += [(b.id, e1), (b.id, e2)]
        elif role == "link":
            d = link_direction(b, shape_of(b.face.part))
            unknowns.append((b.id, d - axis * (d @ axis)))
    if len(unknowns) != 3:
        raise AnalysisError(f"The supports give {len(unknowns)} reaction components in the plane; statics needs exactly 3 "
                            "(a pin and a link, or a pin and ... ) for the reactions to be determinate")
    o = np.asarray(tgt.frame.origin, float)
    A = np.zeros((3, 3))
    for k, (bid, d) in enumerate(unknowns):
        r = np.asarray(bores[bid].frame.origin, float) - o
        A[0, k], A[1, k] = d @ e1, d @ e2
        A[2, k] = np.cross(r, d) @ axis
    rhs = -np.array([F @ e1, F @ e2, M @ axis])
    try:
        x = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        raise AnalysisError("The supports cannot balance the load (a mechanism): check the link direction") from None
    out: dict[str, np.ndarray] = {tgt.id: F.copy()}
    for (bid, d), v in zip(unknowns, x):
        out[bid] = out.get(bid, np.zeros(3)) + d * v
    roles = {b.id: ("load" if b.id == tgt.id else resolved_support(intent, b)) for b in bores.values()}
    return [PinForce(k, roles[k], v) for k, v in out.items() if k in roles]


def bore_tris(mesh: Mesh, origin, axis, radius: float, tol: float = 0.02) -> np.ndarray:
    """Boundary triangles on the cylinder of `radius` about the pin axis."""
    a = _unit(axis)
    v = mesh.nodes[mesh.tris] - np.asarray(origin, float)
    rho = np.linalg.norm(v - (v @ a)[..., None] * a, axis=-1)
    return mesh.tris[(np.abs(rho - radius) < tol * radius).all(axis=1)]


def bearing_load(mesh: Mesh, tris: np.ndarray, origin, axis, force: np.ndarray) -> np.ndarray:
    """Cosine bearing pressure on a bore, scaled so the resultant is exactly `force` (on the part): the pin presses on
    the half of the bore surface the force points to, p = p0 cos(theta) from the force direction."""
    a = _unit(axis)
    o = np.asarray(origin, float)
    Fm = np.linalg.norm(force)
    if Fm < 1e-9:
        return np.zeros(mesh.n_dof)
    fh = force / Fm

    def radial(x):
        v = x - o
        r = v - (v @ a)[:, None] * a
        return r / np.linalg.norm(r, axis=1)[:, None]

    def shape_(x):
        e = radial(x)
        return np.maximum(0.0, e @ fh)[:, None] * e  # cos(theta) on the loaded half, pushing outward

    x, _n, w = surface_points(mesh.nodes, tris)
    resultant = (shape_(x) * w[:, None]).sum(axis=0)
    p0 = Fm / (resultant @ fh)
    f = traction_load(mesh.nodes, tris, lambda q: p0 * shape_(q), mesh.nodes.shape[0])
    # Facets leave the resultant a hair off (~0.05%): spread the difference over the loaded nodes in proportion to
    # their load, so the resultant is exactly the force.
    fv = f.reshape(-1, 3)
    mag = np.linalg.norm(fv, axis=1)
    fv -= np.outer(mag / mag.sum(), fv.sum(axis=0) - force)
    return fv.ravel()


def _hold_rigid(mesh: Mesh) -> np.ndarray:
    """Three nodes, six DOFs: no rigid-body motion and no restraint beyond it (a 3-2-1 set, far apart)."""
    n = mesh.nodes[mesh.corner_nodes()]
    ids = mesh.corner_nodes()
    c = n.mean(axis=0)
    a = ids[np.argmin(np.linalg.norm(n - c, axis=1))]
    pa = mesh.nodes[a]
    b = ids[np.argmax(np.linalg.norm(n - pa, axis=1))]
    ab = _unit(mesh.nodes[b] - pa)
    d = n - pa
    lateral = np.linalg.norm(d - (d @ ab)[:, None] * ab, axis=1)
    cc = ids[np.argmax(lateral)]
    # a: all three; b: the two square to ab (approximately); c: one more.
    k = int(np.argmax(np.abs(ab)))
    others = [i for i in range(3) if i != k]
    nrm = np.cross(ab, mesh.nodes[cc] - pa)
    kc = int(np.argmax(np.abs(nrm)))
    return np.array([3 * a, 3 * a + 1, 3 * a + 2, 3 * b + others[0], 3 * b + others[1], 3 * cc + kc])


def mesh_part(part, setup: MeshSetup, log=None) -> Mesh:
    """Mesh the part with the given controls."""
    try:
        return mesh_shape(part, size=setup.size, curvature=setup.curvature, min_size=setup.min_size, log=log,
                          edges=[e.model_dump() for e in setup.edges], faces=[f.model_dump() for f in setup.faces],
                          spheres=[s.model_dump() for s in setup.spheres])
    except RuntimeError as exc:
        raise AnalysisError(str(exc)) from None


def mesh_metrics(mesh: Mesh, intent: Intent) -> dict:
    """What a stress engineer checks before solving: size, element quality, and the elements around and through each
    pin bore (where the peak stresses are)."""
    q = mesh.quality if mesh.quality is not None else np.ones(len(mesh.tets))
    hist, _ = np.histogram(np.clip(q, 0, 1), bins=10, range=(0, 1))
    out = {"tets": int(len(mesh.tets)), "nodes": int(len(mesh.nodes)), "dof": int(mesh.n_dof), "gb": round(estimate_gb(mesh.n_dof), 1),
           "quality": {"min": float(q.min()), "mean": float(q.mean()), "below_0_2": int((q < 0.2).sum()), "histogram": hist.tolist()},
           "bores": [], "notes": list(mesh.notes)}
    corners = np.zeros(len(mesh.nodes), bool)
    corners[np.unique(mesh.tets[:, :4])] = True
    for b in (o for o in intent.objects if o.kind == "bore"):
        tris = bore_tris(mesh, b.frame.origin, b.frame.z, b.diameter / 2)
        if not len(tris):
            continue
        a = _unit(b.frame.z)
        ids = np.unique(tris[:, :3])
        ids = ids[corners[ids]]
        v = mesh.nodes[ids] - np.asarray(b.frame.origin, float)
        h = v @ a
        e1 = _unit(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]))
        ang = np.arctan2(v @ np.cross(a, e1), v @ e1)
        tol = 1e-3 * max(1.0, b.diameter)
        rim = ids[np.abs(h - h.max()) < tol]  # corner nodes on the top rim: elements around
        around = int(len(rim))
        # Through the wall: distinct heights of corner nodes in a narrow wedge, averaged over a few wedges.
        w = 2 * np.pi / max(around, 8) / 2
        through = []
        for a0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            d = np.abs((ang - a0 + np.pi) % (2 * np.pi) - np.pi) < w
            if d.sum() >= 2:
                levels = np.unique(np.round(h[d] / tol))
                through.append(len(levels) - 1)
        out["bores"].append({"id": b.id, "around": around, "through": round(float(np.median(through)), 1) if through else None,
                             "thickness": round(float(h.max() - h.min()), 3)})
    return out


def contact_joints(intent: Intent, pins: list[PinForce], shape_of, part_tris: dict, mesh: Mesh) -> list:
    """The pins to model for one load case: every supported bore (pivot or link) and the loaded one, each in its fork
    (the mating part's tine gap; a single eye gets a gap just wider than the lug)."""
    from ..topo.setup import link_direction
    from .contact import Joint

    bores = {o.id: o for o in intent.objects if o.kind == "bore"}
    out = []
    for p in pins:
        b = bores[p.id]
        a = _unit(b.frame.z)
        o = np.asarray(b.frame.origin, float)
        h = (mesh.nodes[np.unique(part_tris[p.id])] - o) @ a
        gap = b.gap if b.gap is not None else float(np.ptp(h)) + 0.5
        role = {"load": "load", "link": "link"}.get(p.role, "pivot")
        link = None
        if role == "link":
            d = link_direction(b, shape_of(b.face.part))
            link = _unit(d - a * (d @ a))
        out.append(Joint(p.id, role, o, a, b.diameter, gap, link_dir=link, force=p.force if role == "load" else None, guess=p.force))
    return out


def analyse(intent: Intent, part, shape_of, preset: str = "medium", log=None, setup: MeshSetup | None = None, mesh: Mesh | None = None,
            monitor=None, bearing: str = "cosine") -> tuple[Mesh, list[CaseResult], dict]:
    """Analyse the part (world coordinates) under every load case. Returns the mesh, per-case results and stats.
    The mesh is `mesh` if given, else made with `setup` (else the best-practice rules at `preset`).

    `bearing`: "cosine" puts each pin's statics force on its bore as cosine pressure (fast, linear); "contact" models
    the pins and forks with frictionless contact (fem/contact.py), and also solves the cosine model on the same mesh so
    each region's ratio calibrates the fast model."""
    say = log or (lambda *_: None)
    mon = monitor or (lambda *_a, **_k: None)
    bores = [o for o in intent.objects if o.kind == "bore"]
    cases = [o for o in intent.objects if o.kind == "load_case" and o.target and (any(o.force) or any(o.moment))]
    if not cases:
        raise AnalysisError("Add a load case first")
    mat = intent.rules.material_props()
    t0 = time.time()
    if mesh is None:
        setup = setup or propose(intent, part, preset)
        mon("mesh")
        mesh = mesh_part(part, setup)
        say("mesh", f"{len(mesh.tets):,} quadratic tetrahedra, {mesh.n_dof:,} degrees of freedom ({setup.size:.1f} mm global, "
                    f"{len(setup.spheres)} sphere(s), {len(setup.edges)} edge division(s), {len(setup.faces)} mapped face(s)) "
                    f"in {time.time() - t0:.1f} s")
        for n in mesh.notes:
            say("mesh", n)
    tris = {b.id: bore_tris(mesh, b.frame.origin, b.frame.z, b.diameter / 2) for b in bores}
    for b in bores:
        if not len(tris[b.id]):
            raise AnalysisError(f"No bore surface found for {b.id}: is the pin bore drilled in the part?")
    fixed = _hold_rigid(mesh)
    results = []
    stats = {"tets": len(mesh.tets), "dof": mesh.n_dof, "preset": preset, "material": mat.model_dump(), "seconds": {}}
    mon("loads", dof=mesh.n_dof)
    for lc in cases:
        pins = pin_forces(intent, lc, shape_of)
        say("statics", f"{lc.id}: " + ", ".join(f"{p.id} ({p.role}) {np.linalg.norm(p.force) / LBF:,.0f} lbf" for p in pins))
        f = np.zeros(mesh.n_dof)
        for p in pins:
            b = next(x for x in bores if x.id == p.id)
            f += bearing_load(mesh, tris[p.id], b.frame.origin, b.frame.z, p.force)
        imbalance = np.linalg.norm(f.reshape(-1, 3).sum(axis=0))
        t1 = time.time()
        sol = solve(mesh, mat.E, mat.nu, f, fixed, monitor=lambda stage, **kw: mon(stage, case=lc.id, **kw))
        residual = float(np.abs(sol.reactions[fixed]).max())
        stress, disp = sol.stress, sol.u.reshape(-1, 3)
        contact, calibration, cosine_vm = {}, {}, None
        if bearing == "contact":
            from .contact import solve_contact

            cosine_vm = von_mises(sol.stress)
            cosine_regions = {g.name: g.vm for g in _regions(mesh, np.arange(len(mesh.nodes)), cosine_vm, principal(sol.stress)[:, 0], bores)}
            joints = contact_joints(intent, pins, shape_of, tris, mesh)
            say("contact", f"{lc.id}: pins in contact with their bores (clearance {joints[0].clearance / 25.4:.4f} in, steel pins, "
                           "frictionless); first guess from statics")
            cr = solve_contact(mesh, joints, {j.id: tris[j.id] for j in joints}, mat.E, mat.nu,
                               monitor=lambda stage, **kw: mon(stage, case=lc.id, **kw), log=lambda t: say("contact", t))
            stress, disp, residual = cr.stress, cr.u, cr.hold
            statics = {p.id: p.force for p in pins}
            contact = {"passes": len(cr.passes), "penetration": cr.penetration,
                       "pins": {k: {"p_max": float(cr.pressure[k][1].max()), "pin_vm": cr.pin_vm[k],
                                    "force": cr.pin_forces[k].round(3).tolist(), "statics": statics[k].round(3).tolist()} for k in cr.pin_forces}}
            pins = [PinForce(p.id, p.role, cr.pin_forces[p.id]) for p in pins]
            worst = max(np.linalg.norm(cr.pin_forces[k] - statics[k]) / max(np.linalg.norm(statics[k]), 1e-9) for k in statics)
            say("contact", f"{lc.id}: contact settled in {len(cr.passes)} passes; pin forces from contact within {worst:.2%} of statics")
            calibration = cosine_regions
        stats["seconds"][lc.id] = time.time() - t1
        vm = von_mises(stress)
        p1 = principal(stress)[:, 0]
        U = np.linalg.norm(disp, axis=1)
        corner = np.arange(len(mesh.nodes))  # every node: mid-edge nodes carry their own (averaged) stress estimates
        regions = _regions(mesh, corner, vm, p1, bores)
        if calibration:
            calibration = {g.name: g.vm / calibration[g.name] for g in regions if calibration.get(g.name)}
        k = corner[np.argmax(vm[corner])]
        ult = lc.factor if lc.category == "limit" else 1.0
        lim = 1.0 if lc.category == "limit" else 1 / lc.factor
        vmax = float(vm[corner].max())
        results.append(CaseResult(
            lc.id, lc.label or lc.id, lc.category, lc.factor, pins, regions, vmax, float(p1[corner].max()),
            mesh.nodes[k].round(3).tolist(), float(U.max()),
            mat.fty / (vmax * lim) - 1, mat.ftu / (vmax * ult) - 1, residual, bearing, contact, calibration, cosine_vm, vm, p1, U, disp))
        say("solve", f"{lc.id}: peak von Mises {vmax:.0f} MPa ({vmax / 6.894757:.1f} ksi) at {regions[0].name if regions else '?'}; "
                     f"max displacement {U.max():.4f} mm; load set balanced to {imbalance:.2e} N, hold reactions {residual:.2e} N")
    return mesh, results, stats


def region_labels(x: np.ndarray, bores) -> np.ndarray:
    """The region of each point: a pin's lug (within 1.6 pin diameters of its axis), else the web / rim."""
    x = np.atleast_2d(x)
    label = np.full(len(x), "web / rim", dtype=object)
    for b in bores:
        a = _unit(b.frame.z)
        v = x - np.asarray(b.frame.origin, float)
        rho = np.linalg.norm(v - (v @ a)[:, None] * a, axis=1)
        label[rho < 1.6 * b.diameter] = f"{b.id} lug"
    return label


def _regions(mesh: Mesh, corner: np.ndarray, vm: np.ndarray, p1: np.ndarray, bores) -> list[Region]:
    """Peak stresses near each pin (within 1.6 pin diameters of its axis: the lug) and elsewhere (web, rim, fillets)."""
    label = region_labels(mesh.nodes[corner], bores)
    out = []
    for name in dict.fromkeys(label):
        m = label == name
        i = np.argmax(vm[corner][m])
        node = corner[m][i]
        out.append(Region(str(name), float(vm[node]), float(p1[corner][m].max()), mesh.nodes[node].round(3).tolist()))
    return sorted(out, key=lambda r: -r.vm)
