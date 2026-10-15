"""Planar faces (exact lines and arcs from 2D booleans) → a constrained sketch: shared endpoints, tangency where edges
meet smoothly, arcs about a pin sharing the pin's (held) centre point, a radius dimension on every arc and circle,
horizontal/vertical lines marked as such."""

from __future__ import annotations

import math

from build123d import Edge, Face, GeomType

from .model import Arc, Circle, Constraint, Line, Point, Sketch, Spline

TOL = 1e-4  # mm


class _Builder:
    def __init__(self, sid: str, name: str, pins: dict[str, tuple[float, float]]):
        self.s = Sketch(id=sid, name=name)
        self.n = 0
        for ref, (x, y) in pins.items():
            self.s.entities.append(Point(id=f"pin_{ref}", x=x, y=y, ref=ref))

    def _id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def point(self, x: float, y: float) -> str:
        for e in self.s.entities:
            if isinstance(e, Point) and abs(e.x - x) < TOL and abs(e.y - y) < TOL:
                return e.id
        pid = self._id("p")
        self.s.entities.append(Point(id=pid, x=x, y=y))
        return pid

    def constrain(self, kind: str, refs: list[str], value: float | None = None) -> None:
        self.s.constraints.append(Constraint(id=self._id("k"), kind=kind, refs=refs, value=value))


def _xy(v) -> tuple[float, float]:
    return float(v.X), float(v.Y)


FIT_TOL = 1e-3  # mm: a sampled curve this close to a circle is drawn as an arc
LINE_TOL = 1e-2  # mm: this close to straight is a line (an arc that flat is an approximation's wobble, not a design)


def _recognise(e):
    """("line",) / ("arc", centre, radius) / None for a freeform edge (or a run of them) that is really straight or
    circular (offset curves come back from OpenCascade as B-spline approximations even where the source was a line or
    an arc). Judged over the whole run, so a curve is not chopped into straight-looking pieces."""
    edges = e.edges if isinstance(e, _Joined) else [e]
    pts = [x @ (i / 16) for x in edges for i in range(17)]
    a, z = pts[0], pts[-1]
    chord = z - a
    if chord.length > 1e-6:
        n = chord.normalized()
        if max(abs((p - a).X * n.Y - (p - a).Y * n.X) for p in pts) < LINE_TOL:
            return ("line",)
    m = pts[len(pts) // 2]
    # Circle through start, middle and end.
    ax, ay, bx, by, cx, cy = a.X, a.Y, m.X, m.Y, z.X, z.Y
    dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(dd) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / dd
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / dd
    r = math.hypot(ax - ux, ay - uy)
    if r < 20 * max(chord.length, 1e-9) and max(abs(math.hypot(p.X - ux, p.Y - uy) - r) for p in pts) < FIT_TOL:
        return ("arc", (ux, uy), r)
    return None


def _snap(b: _Builder, x: float, y: float, tol: float = 5e-3) -> str:
    """A point id for (x, y), reusing a nearby one (a fitted centre on a pin shares the pin's point)."""
    for e in b.s.entities:
        if isinstance(e, Point) and abs(e.x - x) < tol and abs(e.y - y) < tol:
            return e.id
    return b.point(x, y)


def _add_edge(b: _Builder, e: Edge) -> tuple[str, str] | None:
    """Add one edge; returns (entity id, kind) for tangency checks."""
    gt = e.geom_type
    if gt not in (GeomType.LINE, GeomType.CIRCLE):
        rec = _recognise(e)
        if rec and rec[0] == "line":
            p1, p2 = b.point(*_xy(e @ 0)), b.point(*_xy(e @ 1))
            lid = b._id("l")
            b.s.entities.append(Line(id=lid, p1=p1, p2=p2))
            return lid, "line"
        if rec and rec[0] == "arc":
            (ux, uy), r = rec[1], rec[2]
            a, z, m = e @ 0, e @ 1, e @ 0.5
            if (a.X - ux) * (m.Y - uy) - (a.Y - uy) * (m.X - ux) < 0:
                a, z = z, a
            aid = b._id("a")
            b.s.entities.append(Arc(id=aid, c=_snap(b, ux, uy), start=b.point(*_xy(a)), end=b.point(*_xy(z))))
            b.constrain("radius", [aid], r)
            return aid, "arc"
    if gt == GeomType.LINE:
        p1, p2 = b.point(*_xy(e @ 0)), b.point(*_xy(e @ 1))
        lid = b._id("l")
        b.s.entities.append(Line(id=lid, p1=p1, p2=p2))
        return lid, "line"
    if gt == GeomType.CIRCLE:
        c = e.arc_center
        r = e.radius
        cid = b.point(*_xy(c))
        if e.is_closed:
            kid = b._id("c")
            b.s.entities.append(Circle(id=kid, c=cid, r=r))
            b.constrain("radius", [kid], r)
            return kid, "circle"
        a, z, m = e @ 0, e @ 1, e @ 0.5
        ccw = (a.X - c.X) * (m.Y - c.Y) - (a.Y - c.Y) * (m.X - c.X) > 0
        if not ccw:
            a, z = z, a
        aid = b._id("a")
        b.s.entities.append(Arc(id=aid, c=cid, start=b.point(*_xy(a)), end=b.point(*_xy(z))))
        b.constrain("radius", [aid], r)
        return aid, "arc"
    # Anything else (offset curves of splines): a spline through points sampled along it.
    pts = [b.point(*_xy(e @ (i / 8))) for i in range(9)]
    pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    sid = b._id("s")
    b.s.entities.append(Spline(id=sid, points=pts))
    return sid, "spline"


def _add_wire(b: _Builder, es: list[Edge]):
    """Add a wire's edges in order; a run of freeform edges becomes one spline through all their points."""
    out, run = [], []

    def flush():
        if not run:
            return
        whole = run[0] if len(run) == 1 else _Joined(list(run))
        if _recognise(whole) is not None:  # the run is really a line or an arc
            out.append((whole, _add_edge(b, whole)))
            run.clear()
            return
        # A few points, evenly spaced along the run (about one per 8 mm, 3 to 9): few enough to drag by hand.
        dense = [e @ (i / 32) for e in run for i in range(33)]
        cum = [0.0]
        for p0, p1 in zip(dense, dense[1:]):
            cum.append(cum[-1] + (p1 - p0).length)
        n = int(min(9, max(3, round(cum[-1] / 8.0) + 1)))
        picks, j = [], 0
        for k in range(n):
            target = cum[-1] * k / (n - 1)
            while j < len(cum) - 1 and cum[j] < target:
                j += 1
            picks.append(dense[j])
        pts = []
        for q in picks:
            pid = b.point(*_xy(q))
            if not pts or pts[-1] != pid:
                pts.append(pid)
        sid = b._id("s")
        b.s.entities.append(Spline(id=sid, points=pts))
        merged = run[0] if len(run) == 1 else _Joined(list(run))
        out.append((merged, (sid, "spline")))
        run.clear()

    for e in es:
        if e.geom_type not in (GeomType.LINE, GeomType.CIRCLE):
            run.append(e)
        else:
            flush()
            out.append((e, _add_edge(b, e)))
    flush()
    return out


class _Joined:
    """Several edges standing in for one (a merged freeform run), for the end and tangent queries the joints need."""

    geom_type = None

    def __init__(self, edges):
        self.edges = edges

    def __matmul__(self, t):
        if t <= 0:
            return self.edges[0] @ 0
        if t >= 1:
            return self.edges[-1] @ 1
        k = min(len(self.edges) - 1, int(t * len(self.edges)))
        return self.edges[k] @ (t * len(self.edges) - k)

    def tangent_at(self, t):
        return self.edges[0].tangent_at(0) if t == 0 else self.edges[-1].tangent_at(1)


def faces_to_sketch(faces: list[Face], sid: str, name: str, pins: dict[str, tuple[float, float]] | None = None) -> Sketch:
    b = _Builder(sid, name, pins or {})
    for f in faces:
        for w in [f.outer_wire(), *f.inner_wires()]:
            es = list(w.order_edges()) if hasattr(w, "order_edges") else list(w.edges())
            added = _add_wire(b, es)
            before = len(b.s.constraints)
            # Tangency where consecutive edges meet smoothly (a line into an arc, an arc into an arc).
            for (e1, k1), (e2, k2) in zip(added, added[1:] + added[:1]):
                if k1 is None or k2 is None or len(added) < 2:
                    continue
                joint = _shared(e1, e2)
                if joint is None:
                    continue
                t1, t2 = e1.tangent_at(_end(e1, joint)), e2.tangent_at(_end(e2, joint))
                if k1[1] == "line" and k2[1] == "line":
                    if abs(t1.dot(t2)) < 1e-6:
                        b.constrain("perpendicular", [k1[0], k2[0]])
                    continue
                if abs(t1.X * t2.Y - t1.Y * t2.X) < 1e-6:
                    if "spline" in (k1[1], k2[1]):
                        _smooth_spline_end(b.s, k1 if k1[1] == "spline" else k2, joint)  # splines follow, not constrained
                        if k1[1] == k2[1] == "spline":
                            _smooth_spline_end(b.s, k2, joint)
                    else:
                        b.constrain("tangent", [k1[0], k2[0]])
            # A closed loop of lines square at every corner: the last right angle follows from the others.
            mine = b.s.constraints[before:]
            if len(added) > 2 and len(mine) == len(added) and all(c.kind == "perpendicular" for c in mine):
                b.s.constraints.remove(mine[-1])
    # Horizontal / vertical only on lines nothing else already orients (else the sketch is over-constrained).
    tangent = {r for c in b.s.constraints if c.kind in ("tangent", "perpendicular") for r in c.refs}
    for e in b.s.entities:
        if isinstance(e, Line) and e.id not in tangent:
            p, q = b.s.point(e.p1), b.s.point(e.p2)
            if abs(p.y - q.y) < TOL:
                b.constrain("horizontal", [e.id])
            elif abs(p.x - q.x) < TOL:
                b.constrain("vertical", [e.id])
    return b.s


def _smooth_spline_end(s: Sketch, k: tuple[str, str], joint) -> None:
    sp = next(e for e in s.entities if e.id == k[0])
    first = s.point(sp.points[0])
    if abs(first.x - joint.X) < TOL and abs(first.y - joint.Y) < TOL:
        sp.tangent_start = True
    else:
        sp.tangent_end = True


def _end(e: Edge, p) -> float:
    """0 or 1: which end of the edge is at p (param_at_point gives raw parameters on some curves)."""
    return 1.0 if ((e @ 1) - p).length < TOL else 0.0


def _shared(e1: Edge, e2: Edge):
    for p in (e1 @ 0, e1 @ 1):
        for q in (e2 @ 0, e2 @ 1):
            if (p - q).length < TOL:
                return p
    return None


def pin_arcs_concentric(s: Sketch) -> Sketch:
    """Arcs and circles centred on a pin already share its point; nothing to add. Kept for clarity of intent."""
    return s


def radius_of(s: Sketch, arc: Arc) -> float:
    c, a = s.point(arc.c), s.point(arc.start)
    return math.hypot(a.x - c.x, a.y - c.y)
