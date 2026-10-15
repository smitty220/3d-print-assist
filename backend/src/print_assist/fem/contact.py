"""Pin contact: the part and an elastic steel pin in each bore, frictionless contact between them, the forks holding the
pins. Replaces the assumed cosine bearing pressure: where each pin bears, over what arc and at what height through the
lug come out of the solve, as do the clearance's effect and the out-of-plane couple of bearing at different heights.

Each fork grips its pin's two stubs (outside the lug, in the tines) as a rigid body that holds the pin square and stops it
spinning, so every stub node moves with the fork: a pivot fork is fixed; a link fork is held along the link and free
across it in the plane; the loading fork is free in the plane and carries the applied force. All are held along the pin
axis. The part is held only along the pins (frictionless pins cannot stop it sliding along them); that reaction should
be ~0.

Contact is enforced at the Gauss points of the bore surface against the pin surface (small sliding: each point keeps the
pin facet it faced at the start) with a penalty, and an active set: points that penetrate are added, points that open
are dropped, until the set stops changing. The first set is the loaded half of each bore, from statics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .element import TRI_N, TRI_WEIGHTS, _tri_shape, elasticity, gauss_stresses, nodal_stresses, stiffness, surface_points
from .mesh import Mesh, mesh_shape
from .solve import rigid_modes

PIN_STEEL = (200_000.0, 0.29)  # MPa, -: alloy steel pin (generic handbook values)
CLEARANCE = 0.0005 * 25.4  # mm, diametral: a snug fit
STUB = 0.5  # x D: the length of pin gripped in each tine
PENALTY = 10.0  # x E_part / h: contact stiffness per area (penetration ~1% of the clearance at bearing stress)
MAX_PASSES = 30


class ContactError(RuntimeError):
    pass


@dataclass
class Joint:
    id: str
    role: str  # pivot | link | load
    origin: np.ndarray  # mid-gap, world mm
    axis: np.ndarray  # unit
    bore_d: float  # mm
    gap: float  # mm, between the tines (the lug sits in it)
    clearance: float = CLEARANCE
    link_dir: np.ndarray | None = None  # link: unit, in the plane
    force: np.ndarray | None = None  # load: N on the part, in the plane
    guess: np.ndarray | None = None  # direction of the pin's force on the part (statics), for the first active set
    stub: float | None = None  # mm gripped in each tine (default STUB x D)
    spheres: list = field(default_factory=list)  # pin mesh refinement [{center, radius, size}]


@dataclass
class ContactResult:
    u: np.ndarray  # (N_part, 3) mm, part displacement
    stress: np.ndarray  # (N_part, 6) MPa, part nodal stress
    pin_forces: dict  # joint id -> (3,) N, the pin's force on the part (from contact)
    fork_forces: dict  # joint id -> (3,) N, the fork's force on its pin
    pressure: dict  # joint id -> (points (P, 3), pressure (P,) MPa, averaged over each bore facet: point values are noisy)
    pin_vm: dict  # joint id -> peak von Mises in the pin, MPa
    hold: float  # N, the part's axial hold reaction
    passes: list = field(default_factory=list)  # [{"pass", "active", "changed", "iterations"}]
    penetration: float = 0.0  # mm, largest remaining (penalty)
    seconds: float = 0.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _plane(axis):
    e1 = _unit(np.cross(axis, [1, 0, 0]) if abs(axis[0]) < 0.9 else np.cross(axis, [0, 1, 0]))
    return e1, np.cross(axis, e1)


def pin_mesh(j: Joint, size: float, curvature: int = 32) -> tuple[Mesh, float]:
    """A solid pin of diameter bore - clearance across the gap and into both tines, in world coordinates."""
    from build123d import Cylinder, Location, Plane

    r = (j.bore_d - j.clearance) / 2
    length = j.gap + 2 * (j.stub if j.stub is not None else STUB * j.bore_d)
    cyl = Cylinder(r, length)  # along z, centred
    z = Plane(origin=tuple(j.origin), z_dir=tuple(j.axis))
    return mesh_shape(Location(z) * cyl, size=size, curvature=curvature, spheres=j.spheres or None), r


@dataclass
class _Points:
    """Contact points: bore Gauss points with their weights, radial directions and the rows mapping displacements to
    the gap."""

    joint: int
    x: np.ndarray  # (P, 3)
    w: np.ndarray  # (P,) mm^2
    n: np.ndarray  # (P, 3) outward radial
    g0: np.ndarray  # (P,) mm, gap with the pin concentric


def _gauss_rows(nodes, tris, offset: int, sign: float, n: np.ndarray, rows: list, cols: list, vals: list, row0: int):
    """Rows of the gap map from a surface's own nodes: sign * n . u(x_p), u interpolated with the 6-node triangle."""
    F = len(tris)
    for g in range(len(TRI_WEIGHTS)):
        for k in range(3):
            for nn in range(6):
                p = g * F + np.arange(F)
                rows.append(row0 + p)
                cols.append(3 * (offset + tris[:, nn]) + k)
                vals.append(sign * n[p, k] * TRI_N[g, nn])


def _pin_rows(pin_nodes, pin_tris, offset, j: Joint, x, n, rows, cols, vals, row0):
    """Rows of the gap map from the pin: -n . u_pin at each point's facing position, interpolated with the 6-node shape
    functions of the pin facet it faces (as the bore side is; found once, on the cylinder unrolled about the pin axis)."""
    a, o = j.axis, j.origin
    e1, e2 = _plane(a)

    def unroll(p):
        v = p - o
        return np.arctan2(v @ e2, v @ e1), v @ a

    cn = pin_tris[:, :3]
    th_c, h_c = unroll(pin_nodes[cn.ravel()])
    th_c, h_c = th_c.reshape(-1, 3), h_c.reshape(-1, 3)
    th_p, h_p = unroll(x)
    r = np.linalg.norm(pin_nodes[cn[0, 0]] - o - ((pin_nodes[cn[0, 0]] - o) @ a) * a)
    for i in range(len(x)):
        dth = (th_c - th_p[i] + np.pi) % (2 * np.pi) - np.pi  # facet corners relative to the point, unwrapped
        A = np.stack([r * dth, h_c - h_p[i]], axis=-1)  # (F, 3, 2)
        v0, v1, v2 = A[:, 0], A[:, 1], A[:, 2]
        d = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        ok = np.abs(d) > 1e-12
        l1 = np.where(ok, ((0 - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (0 - v0[:, 1])) / np.where(ok, d, 1), -1)
        l2 = np.where(ok, ((v1[:, 0] - v0[:, 0]) * (0 - v0[:, 1]) - (0 - v0[:, 0]) * (v1[:, 1] - v0[:, 1])) / np.where(ok, d, 1), -1)
        l0 = 1 - l1 - l2
        worst = np.minimum(np.minimum(l0, l1), l2)
        f = int(np.argmax(worst))  # the facet containing the point (or nearest to it)
        if worst[f] < -0.05:
            raise ContactError(f"{j.id}: a bore point faces no pin surface (is the lug inside the fork gap?)")
        N = _tri_shape((min(max(l1[f], 0.0), 1.0), min(max(l2[f], 0.0), 1.0)))[0]
        for k in range(3):
            rows.append(np.full(6, row0 + i))
            cols.append(3 * (offset + pin_tris[f]) + k)
            vals.append(-n[i, k] * N)


def solve_contact(part: Mesh, joints: list[Joint], bore_tris: dict, E: float, nu: float, pin_size: float | None = None,
                  rtol: float = 1e-8, monitor=None, log=None, fixed: np.ndarray | None = None) -> ContactResult:
    """Static solve of the part on elastic pins with frictionless contact (see the module notes). `fixed`: part DOFs
    held at zero as well (a clamped edge, for benchmarks and bolted parts); the axial hold is then not added."""
    mon = monitor or (lambda *_a, **_k: None)
    say = log or (lambda *_: None)
    t0 = time.time()
    # ---- bodies
    pins, radii = [], []
    for j in joints:
        tris = bore_tris[j.id]
        edge = np.linalg.norm(part.nodes[tris[:, 0]] - part.nodes[tris[:, 1]], axis=1).mean()
        m, r = pin_mesh(j, size=pin_size or max(0.25 * j.bore_d, 1.5 * edge))
        pins.append(m)
        radii.append(r)
    Np = len(part.nodes)
    offsets = np.cumsum([Np] + [len(m.nodes) for m in pins])[:-1]
    n_nodes = Np + sum(len(m.nodes) for m in pins)
    n = 3 * n_nodes
    mon("assemble", dof=n)
    Kp = stiffness(part.nodes, part.tets, elasticity(E, nu))
    blocks = [Kp] + [stiffness(m.nodes, m.tets, elasticity(*PIN_STEEL)) for m in pins]
    K = sp.block_diag(blocks, format="csr")
    nodes = np.vstack([part.nodes] + [m.nodes for m in pins])

    # ---- forks: stub nodes move with the fork; the fork's free motion per role
    tied = np.full(n_nodes, -1)  # node -> joint index
    fork_basis = []  # per joint: (3, k) directions the fork may move along
    f_fork = []  # per joint: (k,) generalised applied force
    for ji, (j, m, off) in enumerate(zip(joints, pins, offsets)):
        h = (m.nodes - j.origin) @ j.axis
        tied[off + np.flatnonzero(np.abs(h) >= j.gap / 2 - 1e-9)] = ji
        e1, e2 = _plane(j.axis)
        if j.role == "pivot":
            B = np.zeros((3, 0))
        elif j.role == "link":
            B = _unit(np.cross(j.axis, j.link_dir))[:, None]
        else:
            B = np.stack([e1, e2], axis=1)
        fork_basis.append(B)
        # The loading fork pushes its pin, which pushes the part: the force on the fork's DOFs is the force on the part.
        f_fork.append(B.T @ j.force if j.role == "load" else np.zeros(B.shape[1]))
    # Part held along the pins at one node near the middle.
    axis = joints[0].axis
    c = part.nodes.mean(axis=0)
    hold_node = int(np.argmin(np.linalg.norm(part.nodes - c, axis=1)))
    hold_dof = 3 * hold_node + int(np.argmax(np.abs(axis)))
    held = np.asarray(fixed, int) if fixed is not None and len(fixed) else np.array([hold_dof])

    # ---- reduction u = Q x
    free_full = np.ones(n, bool)
    for node in np.flatnonzero(tied >= 0):
        free_full[3 * node:3 * node + 3] = False
    free_full[held] = False
    free_idx = np.flatnonzero(free_full)
    nf = len(free_idx)
    q0 = np.cumsum([nf] + [B.shape[1] for B in fork_basis])
    nx = q0[-1]
    rows, cols, vals = [free_idx], [np.arange(nf)], [np.ones(nf)]
    for ji, B in enumerate(fork_basis):
        if not B.shape[1]:
            continue
        nds = np.flatnonzero(tied == ji)
        for k in range(3):
            for q in range(B.shape[1]):
                if abs(B[k, q]) > 0:
                    rows.append(3 * nds + k)
                    cols.append(np.full(len(nds), q0[ji] + q))
                    vals.append(np.full(len(nds), B[k, q]))
    Q = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nx))
    Kr = (Q.T @ K @ Q).tocsr()
    fr = np.zeros(nx)
    for ji in range(len(joints)):
        fr[q0[ji]:q0[ji + 1]] = f_fork[ji]

    # ---- contact points and the gap map g = g0 + A u
    pts = []
    rows, cols, vals = [], [], []
    row0 = 0
    for ji, (j, m, off, r_pin) in enumerate(zip(joints, pins, offsets, radii)):
        tris = bore_tris[j.id]
        x, _nn, w = surface_points(part.nodes, tris)
        v = x - j.origin
        rad = v - np.outer(v @ j.axis, j.axis)
        rho = np.linalg.norm(rad, axis=1)
        nrm = rad / rho[:, None]
        _gauss_rows(part.nodes, tris, 0, +1.0, nrm, rows, cols, vals, row0)
        on = np.abs(np.linalg.norm((m.nodes[m.tris[:, :3]] - j.origin) - np.einsum("fci,i->fc", m.nodes[m.tris[:, :3]] - j.origin, j.axis)[..., None] * j.axis, axis=-1) - r_pin) < 1e-3 * r_pin
        pin_tris = m.tris[on.all(axis=1)]
        _pin_rows(m.nodes, pin_tris, off, j, x, nrm, rows, cols, vals, row0)
        pts.append(_Points(ji, x, w, nrm, rho - r_pin))
        row0 += len(x)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(row0, n))
    Ar = (A @ Q).tocsr()
    x_all = np.vstack([p.x for p in pts])
    w_all = np.concatenate([p.w for p in pts])
    n_all = np.vstack([p.n for p in pts])
    g0 = np.concatenate([p.g0 for p in pts])
    jid = np.concatenate([np.full(len(p.x), p.joint) for p in pts])
    h_el = np.sqrt(w_all.sum() / max(len(w_all), 1) * 6)  # a typical contact facet edge
    eps = PENALTY * E / max(h_el, 1e-6)

    # ---- active set loop
    # First set: the arc a Hertz estimate gives (plane strain, capped at +-65 deg for snug fits) about the direction
    # the pin presses the part in, from statics.
    active = np.zeros(row0, bool)
    E1, n1 = PIN_STEEL
    Es = 1 / ((1 - n1 ** 2) / E1 + (1 - nu ** 2) / E)
    for ji, (j, r_pin) in enumerate(zip(joints, radii)):
        d = j.guess if j.guess is not None else j.force
        sel = jid == ji
        if d is None or np.linalg.norm(d) < 1e-9:
            active[sel] = g0[sel] <= np.min(g0[sel]) + 1e-12
            continue
        rb = j.bore_d / 2
        P = np.linalg.norm(d) / max(np.ptp(x_all[sel] @ j.axis), 1e-6)  # N per mm of bore
        Rs = r_pin * rb / max(rb - r_pin, 1e-9)
        half = min(np.sqrt(4 * P * Rs / (np.pi * Es)) / rb, np.radians(65))
        active[sel] = (n_all[sel] @ _unit(d)) > np.cos(half)
    reg = 1e-10 * float(np.median(Kr.diagonal()))
    Bm = np.zeros((nx, 6))
    Bfull = rigid_modes(nodes)
    Bm[:nf] = Bfull[free_idx]
    xsol = np.zeros(nx)
    passes = []
    mon("precondition", seconds=time.time() - t0)
    ml, first_its, tight = None, None, False
    for it in range(1, MAX_PASSES + 1):
        Wa = sp.diags(eps * w_all * active)
        M = (Kr + Ar.T @ Wa @ Ar + reg * sp.eye(nx)).tocsr()
        b = fr - Ar.T @ (eps * w_all * active * g0)
        # The contact term changes only at the bore surface: keep the AMG hierarchy until CG slows down.
        if ml is None or (first_its and passes and passes[-1]["iterations"] > 2 * first_its):
            ml = pyamg.smoothed_aggregation_solver(M, B=Bm, symmetry="symmetric", max_coarse=600, coarse_solver="splu")
            first_its = None
        tol = rtol if tight else max(rtol, 1e-6)  # loose while the set is moving, tight to confirm it
        its = [0]
        mon("iterate", iteration=0, residual=1.0, rtol=tol, contact_pass=it)
        bn = float(np.linalg.norm(b)) or 1.0

        def count(xk, M=M, b=b, bn=bn):
            its[0] += 1
            if monitor is not None and its[0] % 5 == 0:
                mon("iterate", iteration=its[0], residual=float(np.linalg.norm(b - M @ xk)) / bn, rtol=rtol, contact_pass=it)

        xsol, info = spla.cg(M, b, x0=xsol, rtol=tol, maxiter=3000, M=ml.aspreconditioner(cycle="V"), callback=count)
        if info != 0:
            raise ContactError(f"Contact pass {it}: the linear solve did not converge ({its[0]} iterations)")
        g = g0 + Ar @ xsol
        new = g < 0
        changed = int((new != active).sum())
        first_its = first_its or its[0]
        passes.append({"pass": it, "active": int(new.sum()), "changed": changed, "iterations": its[0], "rtol": tol})
        mon("contact", **passes[-1])
        say(f"contact pass {it}: {int(new.sum())} of {row0} points in contact, {changed} changed, {its[0]} CG iterations")
        if changed == 0:
            if tight:
                break
            tight, ml = True, None  # settled at the loose tolerance: solve tight (fresh AMG) and check again
            continue
        tight = False
        active = new
    else:
        raise ContactError(f"The contact set did not settle in {MAX_PASSES} passes")

    # ---- results
    u = Q @ xsol
    U = u.reshape(-1, 3)
    p = np.where(active, -eps * g, 0.0)  # MPa at each point
    fpt = (p * w_all)[:, None] * n_all  # force on the part at each point (the pin pushes the bore outward)
    pin_forces, pressure, fork_forces, pin_vm = {}, {}, {}, {}
    for ji, (j, m, off) in enumerate(zip(joints, pins, offsets)):
        sel = jid == ji
        pin_forces[j.id] = fpt[sel].sum(axis=0)
        fork_forces[j.id] = pin_forces[j.id].copy()  # the pin is in equilibrium: the fork balances its contact force
        F = len(bore_tris[j.id])
        pj, wj = p[sel].reshape(len(TRI_WEIGHTS), F), w_all[sel].reshape(len(TRI_WEIGHTS), F)  # g-major points
        facet = (pj * wj).sum(axis=0) / wj.sum(axis=0)
        pressure[j.id] = (x_all[sel], np.tile(facet, len(TRI_WEIGHTS)))
        sg = gauss_stresses(m.nodes, m.tets, elasticity(*PIN_STEEL), U[off:off + len(m.nodes)].ravel())
        s = nodal_stresses(m.tets, len(m.nodes), sg)
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) + 3 * (s[:, 3:] ** 2).sum(axis=1))
        pin_vm[j.id] = float(vm.max())
    Dp = elasticity(E, nu)
    up = U[:Np].ravel()
    sg = gauss_stresses(part.nodes, part.tets, Dp, up)
    stress = nodal_stresses(part.tets, Np, sg)
    # The hold's reaction: the part's axial internal force balance at the held DOF.
    contact_on_part = A[:, :3 * Np].T @ (p * w_all)  # generalised contact forces on the part's DOFs
    hold = float(np.abs((Kp @ up - contact_on_part)[held]).max()) if fixed is None else float(np.abs((Kp @ up - contact_on_part)[held]).sum())
    return ContactResult(U[:Np], stress, pin_forces, fork_forces, pressure, pin_vm, hold, passes,
                         float(max(0.0, -g[active].min())) if active.any() else 0.0, time.time() - t0)
