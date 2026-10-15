"""Sketch → exact planar faces. Closed profiles are found from the non-construction curves; nested profiles alternate
material and hole (as in CATIA's Pad), so one sketch can hold an outline, its cut-outs and islands inside those."""

from __future__ import annotations

import math

from build123d import Edge, Face, Plane, Vector, Wire

from .model import Arc, Circle, Line, Sketch, Spline

TOL = 1e-4  # mm, endpoint matching


class SketchGeometryError(ValueError):
    pass


def _v(s: Sketch, pid: str) -> Vector:
    p = s.point(pid)
    return Vector(p.x, p.y, 0)


def edges(s: Sketch) -> list[Edge]:
    out = []
    for e in s.entities:
        if getattr(e, "construction", False):
            continue
        if isinstance(e, Line):
            out.append(Edge.make_line(_v(s, e.p1), _v(s, e.p2)))
        elif isinstance(e, Circle):
            out.append(Edge.make_circle(e.r, Plane.XY.move(_loc(_v(s, e.c)))))
        elif isinstance(e, Arc):
            c, a, b = _v(s, e.c), _v(s, e.start), _v(s, e.end)
            r = (a - c).length
            t0 = math.atan2(a.Y - c.Y, a.X - c.X)
            t1 = math.atan2(b.Y - c.Y, b.X - c.X)
            if t1 <= t0:
                t1 += 2 * math.pi
            tm = (t0 + t1) / 2
            m = c + Vector(math.cos(tm), math.sin(tm), 0) * r
            out.append(Edge.make_three_point_arc(a, m, b))  # exact ends: the wire joins at the sketch's own points
        elif isinstance(e, Spline):
            n = len(e.points)
            t0 = join_tangent(s, e.id, e.points[0], e.points[1]) if e.tangent_start and n > 1 else None
            t1 = join_tangent(s, e.id, e.points[-1], e.points[-2]) if e.tangent_end and n > 1 else None
            out.extend(spline_edges([_v(s, p) for p in e.points], t0, -t1 if t1 is not None else None))
    return out


def spline_edges(pts: list[Vector], t0: Vector | None = None, t1: Vector | None = None) -> list[Edge]:
    """Catmull–Rom through the points as cubic Bezier segments: the curve the sketcher draws (frontend/src/sketch/draw.ts).
    An end with a unit tangent leaves along it, a third of the first chord long."""
    if len(pts) == 2 and t0 is None and t1 is None:
        return [Edge.make_line(*pts)]
    ext = [pts[0], *pts, pts[-1]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        chord = (p2 - p1).length / 3
        c1, c2 = p1 + (p2 - p0) * (1 / 6), p2 - (p3 - p1) * (1 / 6)
        if i == 1 and t0 is not None:
            c1 = p1 + t0 * chord
        if i == len(ext) - 3 and t1 is not None:
            c2 = p2 - t1 * chord
        out.append(Edge.make_bezier(p1, c1, c2, p2))
    return out


def join_tangent(s: Sketch, spline_id: str, end: str, nxt: str) -> Vector | None:
    """Unit direction a spline leaves `end` along: the tangent of the other curve on that point, toward `nxt`."""
    p, q = _v(s, end), _v(s, nxt)
    for e in s.entities:
        if e.id == spline_id or getattr(e, "construction", False):
            continue
        d = None
        if isinstance(e, Line) and end in (e.p1, e.p2):
            d = _v(s, e.p2) - _v(s, e.p1)
        elif isinstance(e, Arc) and end in (e.start, e.end):
            c = _v(s, e.c)
            d = Vector(-(p.Y - c.Y), p.X - c.X, 0)
        if d is not None:
            d = d.normalized()
            return d if d.dot(q - p) >= 0 else -d
    return None


def _loc(v: Vector):
    from build123d import Location

    return Location((v.X, v.Y, 0))


def regions(s: Sketch) -> list[Face]:
    """Material faces of the sketch: closed profiles, nested ones alternating material and hole."""
    es = edges(s)
    if not es:
        raise SketchGeometryError(f"{s.id} has no profile geometry")
    wires = Wire.combine(es, tol=TOL)
    open_ends = [w for w in wires if not w.is_closed]
    if open_ends:
        p = open_ends[0] @ 0
        raise SketchGeometryError(f"{s.id}: a profile is not closed (open end near {p.X / 25.4:.3f}, {p.Y / 25.4:.3f} in)")
    loops = sorted((Face(w) for w in wires), key=lambda f: -f.area)
    probe = [w @ 0 for w in (f.outer_wire() for f in loops)]
    depth = []
    parent: list[int | None] = []
    for i, f in enumerate(loops):
        hosts = [j for j in range(i) if loops[j].area > f.area and loops[j].is_inside(probe[i])]
        depth.append(len(hosts))
        parent.append(max(hosts, key=lambda j: depth[j]) if hosts else None)
    out = []
    for i, f in enumerate(loops):
        if depth[i] % 2:
            continue
        face = f
        for j, f2 in enumerate(loops):
            if parent[j] == i:
                face = face - f2
        out.extend(face.faces() if hasattr(face, "faces") else [face])
    return out
