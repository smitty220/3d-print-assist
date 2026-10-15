"""Turn the design intent into a topology-optimisation problem on a voxel grid.

The grid fills the design-space box in its own frame (the primary mount's) with cubic elements. Each element
is design, forced empty (inside a mating part, a keep-out plus its clearance, a fastener shank or head-access
column, the pin path) or forced solid (a seat ring under each fastener, the lug ring(s) round the pin). The
element size is chosen so the elements left in the analysis (everything not forced empty) just fit a budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from build123d import IN, CenterOf, Shape
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_IN
from scipy import ndimage

from ..catalog import FASTENERS
from ..intent.models import MATERIALS, Bore, BoltPattern, DesignSpace, Frame, Intent, PlanarMount, resolved_support
from .fe import Grid, Springs
from .simp import Problem

DEFAULT_BUDGET = 60_000
MIN_BUDGET, MAX_BUDGET = 5_000, 400_000
MATE_GAP = 0.25  # mm of assembly clearance kept round mating parts


class TopoSetupError(ValueError):
    pass


@dataclass
class LoadInfo:
    id: str
    label: str
    target: str
    nodes: np.ndarray  # node indices the load is spread over
    forces: np.ndarray  # (n, 3) N, world axes


@dataclass
class SupportInfo:
    id: str
    kind: str  # "fixed" | "normal" | "pin" | "link"
    nodes: np.ndarray
    along: str = ""  # the held direction of a normal or link support: "X", "Y" or "Z" of the design frame, or "skew"


@dataclass
class Setup:
    problem: Problem
    frame: Frame
    origin: np.ndarray  # grid corner, design-frame coordinates (mm)
    axes: np.ndarray  # (3, 3) rows: design-frame x, y, z in world
    budget: int
    density: float  # kg/mm³
    supports: list[SupportInfo]
    loads: list[LoadInfo]
    notes: list[str] = field(default_factory=list)

    @property
    def grid(self) -> Grid:
        return self.problem.grid

    def to_world(self, grid_pts: np.ndarray) -> np.ndarray:
        """Grid coordinates (mm from the grid corner) to world."""
        return np.asarray(self.frame.origin) + (grid_pts + self.origin) @ self.axes

    @property
    def active_count(self) -> int:
        return int((~self.problem.void).sum())

    def full_mass(self) -> float:
        """kg if every non-void element were solid."""
        return self.active_count * self.grid.h**3 * self.density

    def summary(self) -> dict:
        p, g = self.problem, self.grid
        return {
            "grid": [g.nx, g.ny, g.nz],
            "h": g.h,
            "budget": self.budget,
            "elements": self.active_count,
            "design": int(p.design.sum()),
            "solid": int(p.solid.sum()),
            "void": int(p.void.sum()),
            "full_mass": self.full_mass(),
            "supports": [{"id": s.id, "kind": s.kind, "along": s.along, "nodes": len(s.nodes)} for s in self.supports],
            "loads": [{"id": lc.id, "label": lc.label, "target": lc.target, "nodes": len(lc.nodes)} for lc in self.loads],
            "notes": self.notes,
        }

    def preview(self) -> dict:
        """Geometry for the viewer: forced-solid element centres, supported nodes, load arrows (world mm)."""
        g = self.grid
        idx = np.argwhere(self.problem.solid)
        nodes = g.node_coords()
        return {
            "h": g.h,
            "axes": self.axes.round(6).tolist(),
            "solid": self.to_world((idx + 0.5) * g.h).round(3).ravel().tolist(),
            "supports": [{"id": s.id, "kind": s.kind, "points": self.to_world(nodes[s.nodes]).round(3).ravel().tolist()} for s in self.supports],
            "loads": [{"id": lc.id, "points": self.to_world(nodes[lc.nodes]).round(3).ravel().tolist(),
                       "forces": lc.forces.round(4).ravel().tolist()} for lc in self.loads],
        }


# ---------------------------------------------------------------- geometry helpers


def _axes(frame: Frame) -> np.ndarray:
    z = np.asarray(frame.z, float)
    z /= np.linalg.norm(z)
    x = np.asarray(frame.x, float)
    x -= z * (x @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def _cyl_coords(pts: np.ndarray, origin, axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t along the axis, radius from it, unit radial vectors) for world points."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    v = pts - np.asarray(origin, float)
    t = v @ a
    radial = v - t[:, None] * a
    rho = np.linalg.norm(radial, axis=1)
    return t, rho, radial / np.maximum(rho, 1e-12)[:, None]


def _in_polygon(xy: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    inside = np.zeros(len(xy), bool)
    px, py = xy[:, 0], xy[:, 1]
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        crosses = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < xc)
    return inside


def inside_mask(shape: Shape, grid: Grid, origin: np.ndarray, frame: Frame, axes: np.ndarray) -> np.ndarray:
    """Element centres inside a solid, by casting one ray per grid column along the frame Z (parity of hits)."""
    out = np.zeros(grid.shape, bool)
    bb = shape.bounding_box()
    corners = np.array([(x, y, z) for x in (bb.min.X, bb.max.X) for y in (bb.min.Y, bb.max.Y) for z in (bb.min.Z, bb.max.Z)])
    local = (corners - np.asarray(frame.origin)) @ axes.T - origin  # grid coordinates
    lo = np.floor(local.min(axis=0) / grid.h - 0.5).astype(int)
    hi = np.ceil(local.max(axis=0) / grid.h - 0.5).astype(int)
    n = np.array(grid.shape)
    lo, hi = np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1)
    if np.any(local.max(axis=0) < 0) or np.any(local.min(axis=0) > n * grid.h):
        return out
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shape.wrapped, 1e-6)
    zc = (np.arange(grid.nz) + 0.5) * grid.h
    fo = np.asarray(frame.origin)
    zdir = gp_Dir(*axes[2])
    classifier = None
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            # A tiny off-grid nudge keeps rays off edges shared by faces, where hits double up.
            gx, gy = (i + 0.5) * grid.h + 1.3e-4, (j + 0.5) * grid.h + 0.7e-4
            p = fo + (np.array([gx, gy, 0.0]) + origin) @ axes  # at grid z = 0
            inter.Perform(gp_Lin(gp_Pnt(*p), zdir), -1e9, 1e9)
            ws = sorted(inter.WParameter(k) for k in range(1, inter.NbPnt() + 1))
            ws = [w for k, w in enumerate(ws) if k == 0 or w - ws[k - 1] > 1e-7]
            if not ws:
                continue
            if len(ws) % 2 == 0:
                for a, b in zip(ws[0::2], ws[1::2]):
                    out[i, j] |= (zc > a) & (zc < b)
            else:  # tangent or seam hit: classify this column point by point
                if classifier is None:
                    classifier = BRepClass3d_SolidClassifier(shape.wrapped)
                for k in range(lo[2], hi[2] + 1):
                    q = fo + (np.array([gx, gy, zc[k]]) + origin) @ axes
                    classifier.Perform(gp_Pnt(*q), 1e-6)
                    out[i, j, k] = classifier.State() == TopAbs_IN
    return out


def link_direction(b: Bore, mating: Shape) -> np.ndarray:
    """Unit vector (world) a link support reacts along: pin centre to the link's centre of mass, square to the pin."""
    a = np.asarray(b.frame.z, float)
    a /= np.linalg.norm(a)
    c = mating.center(CenterOf.MASS)
    v = np.array([c.X, c.Y, c.Z]) - np.asarray(b.frame.origin, float)
    v -= a * (v @ a)
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise TopoSetupError(f"{b.id}: the link's centre of mass is on the pin axis, so its direction is unknown")
    return v / n


def _grow(mask: np.ndarray, h: float, dist: float) -> np.ndarray:
    """Elements whose centre is within `dist` of the region's surface (centre distance minus half an element)."""
    if not mask.any() or dist <= 0:
        return mask
    d = ndimage.distance_transform_edt(~mask, sampling=h)
    return d <= dist + h / 2


# ---------------------------------------------------------------- the build


def _rules(intent: Intent):
    return intent.rules


def _fastener(bp: BoltPattern):
    return FASTENERS.get(bp.fastener or "")


def _bolt_dims(bp: BoltPattern, h: float, rules) -> tuple[float, float, float, float]:
    """(shank-hole radius, head/washer radius, seat-ring radius, seat thickness) in mm."""
    f = _fastener(bp)
    hole_d = min(hh.diameter for hh in bp.holes)
    shank_d = f.shank_d * IN if f else hole_d
    r_hole = (f.hole_close * IN if f else hole_d) / 2
    r_head = (max(f.head_size, f.washer_od) * IN if f else 2 * hole_d) / 2
    r_ring = max(r_head, rules.bolt_edge_preferred * shank_d, r_hole + 2 * h)
    t_seat = max(2 * h, shank_d)
    return r_hole, r_head, r_ring, t_seat


def _bore_dims(b: Bore, h: float, rules) -> tuple[float, float, float]:
    """(pin radius, lug-ring radius, lug thickness) in mm."""
    r = b.diameter / 2
    return r, max(rules.lug_edge_min * b.diameter, r + 2 * h), max(2 * h, rules.lug_td_min * b.diameter)


def _hole_points(bp: BoltPattern, axes_bp: np.ndarray) -> np.ndarray:
    o = np.asarray(bp.frame.origin, float)
    return np.array([o + hh.center[0] * axes_bp[0] + hh.center[1] * axes_bp[1] for hh in bp.holes])


@dataclass
class _Grid:
    grid: Grid
    origin: np.ndarray
    centres: np.ndarray  # (ne, 3) world, C order
    void: np.ndarray
    solid: np.ndarray
    rings: dict[str, np.ndarray]  # interface id -> elements of its forced-solid ring(s)
    notes: list[str]


def _classify(intent: Intent, ds: DesignSpace, axes: np.ndarray, h: float, shapes: dict[str, tuple[Shape, float]]) -> _Grid:
    lo, hi = np.asarray(ds.min, float), np.asarray(ds.max, float)
    size = hi - lo
    n = np.maximum(2, np.floor(size / h + 1e-9).astype(int))
    origin = lo + np.array([(size[0] - n[0] * h) / 2, (size[1] - n[1] * h) / 2, 0.0])  # never past the box
    grid = Grid(int(n[0]), int(n[1]), int(n[2]), h)
    i, j, k = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij")
    gpts = (np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) + 0.5) * h
    centres = np.asarray(ds.frame.origin) + (gpts + origin) @ axes
    void = np.zeros(grid.shape, bool)
    solid = np.zeros(grid.shape, bool)
    rings: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for name, (shape, gap) in shapes.items():
        inside = inside_mask(shape, grid, origin, ds.frame, axes)
        void |= _grow(inside, h, gap)
    rules = _rules(intent)
    for o in intent.objects:
        if o.kind == "bolt_pattern":
            ax_bp = _axes(o.frame)
            r_hole, r_head, r_ring, t_seat = _bolt_dims(o, h, rules)
            ring = np.zeros(len(centres), bool)
            for p in _hole_points(o, ax_bp):
                t, rho, _ = _cyl_coords(centres, p, ax_bp[2])
                void.ravel()[rho <= r_hole] = True
                void.ravel()[(rho <= r_head) & (t > t_seat)] = True  # head, washer and tool access
                ring |= (rho > r_hole) & (rho <= r_ring) & (t >= 0) & (t <= t_seat)
            rings[o.id] = ring.reshape(grid.shape)
        elif o.kind == "bore":
            r, r_lug, t_lug = _bore_dims(o, h, rules)
            t, rho, _ = _cyl_coords(centres, o.frame.origin, o.frame.z)
            void.ravel()[rho <= r] = True  # pin path, both ways out of the box
            band = (rho > r) & (rho <= r_lug)
            if o.gap is None:  # male end: the part carries the pin on both sides of it
                half = o.length / 2 + MATE_GAP
                ring = band & (np.abs(t) >= half) & (np.abs(t) <= half + t_lug)
            else:  # forked end: one lug in the gap
                ring = band & (np.abs(t) <= o.gap / 2 - MATE_GAP)
            rings[o.id] = ring.reshape(grid.shape)
    for rid, ring in rings.items():
        clash = int((ring & void).sum())
        if clash:
            notes.append(f"{rid}: {clash} interface elements fall inside a keep-out, mating part or access path and stay empty")
        solid |= ring & ~void
        rings[rid] = ring & ~void
    return _Grid(grid, origin, centres, void, solid, rings, notes)


def _distribute(F: np.ndarray, M: np.ndarray, about: np.ndarray, pts: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Node forces summing to F with moment M about `about`: `base` (n, 3) carries F in some pattern, then a
    rigid 'spider' share fixes the moment so the resultant is exactly (F, M)."""
    c = pts.mean(axis=0)
    r = pts - c
    want = M + np.cross(about - c, F)
    have = np.cross(r, base).sum(axis=0)
    J = (r * r).sum() * np.eye(3) - r.T @ r
    theta = np.linalg.pinv(J) @ (want - have)
    return base + np.cross(theta, r)


def clearance_shapes(intent: Intent, shape_of: Callable[[str], Shape], roles: dict[str, str]) -> dict[str, tuple[Shape, float]]:
    """What the part must keep clear of, with the gap to keep: the mating parts the intent attaches to (other parts in
    the project may be alternatives sitting in the same place) and the keep-outs."""
    shapes: dict[str, tuple[Shape, float]] = {}
    used = {o.face.part for o in intent.objects if getattr(o, "face", None) is not None}
    for name in sorted(used):
        if roles.get(name, "mating") == "mating":
            shapes[name] = (shape_of(name), MATE_GAP)
    for ko in (o for o in intent.objects if o.kind == "keep_out"):
        shapes[ko.part] = (shape_of(ko.part), ko.clearance)
    return shapes


def build_setup(intent: Intent, shape_of: Callable[[str], Shape], roles: dict[str, str], budget: int = DEFAULT_BUDGET) -> Setup:
    budget = int(min(MAX_BUDGET, max(MIN_BUDGET, budget)))
    ds = next((o for o in intent.objects if o.kind == "design_space"), None)
    if ds is None:
        raise TopoSetupError("Add a design space first; the optimiser fills it with elements")
    cases = [o for o in intent.objects if o.kind == "load_case"]
    cases = [lc for lc in cases if lc.target and intent.get(lc.target) is not None and (any(lc.force) or any(lc.moment))]
    if not cases:
        raise TopoSetupError("Add at least one load case with a target interface and a non-zero force or moment")
    supported = [o for o in intent.objects if o.kind in ("planar_mount", "bolt_pattern", "bore") and resolved_support(intent, o) != "free"]
    if not supported:
        raise TopoSetupError("Nothing holds the part: set a bolt pattern or bore to Fixed (or the mount face to Normal)")

    axes = _axes(ds.frame)
    shapes = clearance_shapes(intent, shape_of, roles)

    size = np.asarray(ds.max) - np.asarray(ds.min)
    h = float(np.prod(size) / budget) ** (1 / 3)
    g = _classify(intent, ds, axes, h, shapes)
    for _ in range(6):
        active = int((~g.void).sum())
        if 0.97 * budget <= active <= budget:
            break
        h *= (active / budget) ** (1 / 3) * (1.003 if active > budget else 0.997)
        g = _classify(intent, ds, axes, h, shapes)
    while int((~g.void).sum()) > budget:
        h *= 1.01
        g = _classify(intent, ds, axes, h, shapes)

    grid = g.grid
    nodes_w = np.asarray(ds.frame.origin) + (grid.node_coords() + g.origin) @ axes
    active_nodes = np.zeros(grid.n_nodes, bool)
    active_nodes[grid.element_nodes()[np.flatnonzero(~g.void.ravel())].ravel()] = True

    def ring_nodes(rid: str) -> np.ndarray:
        el = np.flatnonzero(g.rings[rid].ravel())
        return np.unique(grid.element_nodes()[el].ravel()) if len(el) else np.array([], int)

    rules = _rules(intent)
    supports: list[SupportInfo] = []
    fixed: list[np.ndarray] = [np.zeros(0, np.int64)]
    spr: list[tuple] = []  # (a, b, world directions, k) spring groups
    extra: list[np.ndarray] = []  # pin-centre nodes of link supports, grid coordinates
    notes = list(g.notes)
    regions: dict[str, tuple[np.ndarray, str]] = {}  # interface -> (nodes, how loads spread over them)
    for o in intent.objects:
        if o.kind == "bolt_pattern":
            ax_bp = _axes(o.frame)
            r_hole, _r_head, r_ring, _t = _bolt_dims(o, grid.h, rules)
            cand = ring_nodes(o.id)
            keep = np.zeros(len(cand), bool)
            for p in _hole_points(o, ax_bp):
                t, rho, _ = _cyl_coords(nodes_w[cand], p, ax_bp[2])
                keep |= (np.abs(t) <= 0.5 * grid.h) & (rho <= r_ring + 0.5 * grid.h)  # clamped face under the fastener
            regions[o.id] = (cand[keep], "uniform")
        elif o.kind == "bore":
            r, _rl, _tl = _bore_dims(o, grid.h, rules)
            cand = ring_nodes(o.id)
            _t, rho, _ = _cyl_coords(nodes_w[cand], o.frame.origin, o.frame.z)
            regions[o.id] = (cand[rho <= r + 0.75 * grid.h], "bearing")  # the bore surface
        elif o.kind == "planar_mount":
            m_ax = _axes(o.frame)
            v = nodes_w - np.asarray(o.frame.origin)
            on_face = (np.abs(v @ m_ax[2]) <= 0.5 * grid.h) & active_nodes
            xy = np.stack([v @ m_ax[0], v @ m_ax[1]], axis=1)
            sel = np.flatnonzero(on_face)
            regions[o.id] = (sel[_in_polygon(xy[sel], o.footprint)], "uniform")

    for o in supported:
        nodes, _how = regions[o.id]
        nodes = nodes[active_nodes[nodes]]
        if not len(nodes):
            notes.append(f"{o.id}: no elements at this interface, so it holds nothing")
            continue
        kind = resolved_support(intent, o)
        along = ""
        if kind == "normal":
            nrm = _axes(o.frame)[2]
            comp = int(np.argmax(np.abs(axes @ nrm)))
            if abs(axes[comp] @ nrm) < 0.999:
                raise TopoSetupError(f"{o.id}: a Normal support needs the mount face square to the design-space box")
            fixed.append(3 * nodes + comp)  # grid DOFs are along the design-frame axes
            along = "XYZ"[comp]
        elif kind in ("pin", "link"):
            # Bearing springs, radial and along the pin, from each bore node to the pin: free to turn about it. A pin
            # support grounds them; a link joins them to a pin-centre node held only along the link (and the pin).
            a = np.asarray(o.frame.z, float)
            a /= np.linalg.norm(a)
            _t, _rho, e = _cyl_coords(nodes_w[nodes], o.frame.origin, a)
            k = grid.h  # about one solid element's stiffness (E = 1) per node
            if kind == "pin":
                other = np.full(len(nodes), -1)
            else:
                other = np.full(len(nodes), grid.n_nodes + len(extra))
                extra.append((np.asarray(o.frame.origin, float) - np.asarray(ds.frame.origin)) @ axes.T - g.origin)
                d = link_direction(o, shape_of(o.face.part))
                held = [d, a]
                spr.append((np.full(2, grid.n_nodes + len(extra) - 1), np.full(2, -1), np.array(held), np.full(2, k * len(nodes))))
                c = axes @ d
                along = "XYZ"[int(np.argmax(np.abs(c)))] if np.abs(c).max() > 0.999 else "skew"
            spr.append((np.concatenate([nodes, nodes]), np.concatenate([other, other]),
                        np.vstack([e, np.tile(a, (len(nodes), 1))]), np.full(2 * len(nodes), k)))
        else:
            fixed.append((3 * nodes[:, None] + np.arange(3)).ravel())
        supports.append(SupportInfo(o.id, kind, nodes, along))
    if not supports:
        raise TopoSetupError("None of the supports touch the elements; check the design space reaches them")

    loads: list[LoadInfo] = []
    vectors: list[np.ndarray] = []
    for lc in cases:
        tgt = intent.get(lc.target)
        nodes, how = regions.get(tgt.id, (np.array([], int), "uniform"))
        nodes = nodes[active_nodes[nodes]]
        if not len(nodes):
            raise TopoSetupError(f"{lc.id}: no elements at {tgt.id} to take the load")
        F, M = np.asarray(lc.force, float), np.asarray(lc.moment, float)
        pts = nodes_w[nodes]
        base = np.tile(F / len(nodes), (len(nodes), 1))
        if how == "bearing":
            a = np.asarray(tgt.frame.z, float)
            a /= np.linalg.norm(a)
            f_ax = a * (F @ a)
            f_perp = F - f_ax
            _t, _rho, e = _cyl_coords(pts, tgt.frame.origin, a)
            if np.linalg.norm(f_perp) > 1e-9:
                w = np.maximum(0.0, e @ (f_perp / np.linalg.norm(f_perp)))  # the pin bears on the side it pushes
                base = np.outer(w / w.sum(), f_perp) + f_ax / len(nodes)
        forces = _distribute(F, M, np.asarray(tgt.frame.origin, float), pts, base)
        vec = np.zeros(3 * grid.n_nodes)
        np.add.at(vec, (3 * nodes[:, None] + np.arange(3)).ravel(), (forces @ axes.T).ravel())  # world -> grid axes
        vectors.append(vec)
        loads.append(LoadInfo(lc.id, lc.label or lc.id, tgt.id, nodes, forces))
    if any(s.id in {lc.target for lc in loads} for s in supports):
        notes.append("A load case targets a supported interface; that load goes straight into the support")

    density = MATERIALS[rules.material] / 1e9
    springs = None
    if spr:
        springs = Springs(np.array(extra).reshape(-1, 3), np.concatenate([q[0] for q in spr]), np.concatenate([q[1] for q in spr]),
                          np.vstack([q[2] for q in spr]) @ axes.T, np.concatenate([q[3] for q in spr]))  # directions in grid axes
    problem = Problem(grid, g.void, g.solid, np.unique(np.concatenate(fixed)), vectors, springs=springs)
    return Setup(problem, ds.frame, g.origin, axes, budget, density, supports, loads, notes)
