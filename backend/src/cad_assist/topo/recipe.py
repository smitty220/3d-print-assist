"""A plate recipe: the design decisions for a pin-mounted machined plate (a bell crank, a lever), compiled into the
constrained sketches and pad / pocket features the engineer edits in the sketcher.

The AI drafts recipes (ai/sketch_draft.py); it never writes sketch entities. Lengths are inches here, at the AI's
interface; the compiled sketches are millimetres like everything else.

    outline   one edge per pair of neighbouring lugs, going counter-clockwise round the part:
              "tangent" (straight, tangent to both lugs), "arc" with a radius bulging "out" (convex) or "in"
              (concave), or "spline" through points (leaving and arriving tangent to the lugs)
    holes     lightening holes: corner points, corners rounded to at least the cutter radius
    web       pockets a rim inside the outline, clear of a boss at each lug, to a depth from each face
    spots     spot faces: a step square to the lug's arm, so the lug fits between its fork's tines
"""

from __future__ import annotations

import math

import numpy as np
from build123d import Edge, Face, Location, Plane, Vector, Wire
from pydantic import BaseModel, Field

from ..features.library import _offset
from ..features.model import Design, FeatureInstance
from ..sketch.from_faces import faces_to_sketch
from ..sketch.geometry import spline_edges

IN = 25.4


class RecipeError(ValueError):
    pass


class OutlineEdge(BaseModel):
    frm: str = Field(alias="from")
    to: str
    edge: str = "tangent"  # tangent | arc | spline
    radius: float | None = None  # in, for an arc
    bulge: str = "out"  # out (convex) | in (concave), for an arc
    through: list[tuple[float, float]] = Field(default_factory=list)  # in, for a spline

    model_config = {"populate_by_name": True}


class Hole(BaseModel):
    points: list[tuple[float, float]]  # in, the corners
    corner_radius: float = 0.125  # in


class Web(BaseModel):
    rim: float = 0.15  # in
    boss_gap: float = 0.0  # in
    depth_top: float = 0.0  # in, pocket depth from +Z
    depth_bottom: float = 0.0  # in, from -Z


class Spot(BaseModel):
    pin: str
    side: str  # +z | -z
    depth: float  # in
    step: float  # in, from the pin to the step, measured toward the part's middle


class Recipe(BaseModel):
    z_min: float  # in
    z_max: float  # in
    lugs: dict[str, float]  # pin id -> lug radius, in
    outline: list[OutlineEdge]
    holes: list[Hole] = Field(default_factory=list)
    web: Web | None = None
    spots: list[Spot] = Field(default_factory=list)
    corner_radius: float = 0.125  # in, the cutter


# ---------------------------------------------------------------- outline geometry (mm)


def _v(p) -> Vector:
    return Vector(float(p[0]), float(p[1]), 0)


def _unit(v: Vector) -> Vector:
    return v.normalized()


def _right(e: Vector) -> Vector:
    return Vector(e.Y, -e.X, 0)


def _circle_meet(c1: Vector, d1: float, c2: Vector, d2: float, side: Vector) -> Vector | None:
    """The point at distance d1 from c1 and d2 from c2 on the side `side` points to (two-circle intersection)."""
    L = (c2 - c1).length
    if L < 1e-9 or L > d1 + d2 or L < abs(d1 - d2):
        return None
    a = (d1 * d1 - d2 * d2 + L * L) / (2 * L)
    h = math.sqrt(max(0.0, d1 * d1 - a * a))
    e = (c2 - c1) * (1 / L)
    base = c1 + e * a
    n = _right(e)
    if n.dot(side) < 0:
        n = -n
    return base + n * h


def _arc(c: Vector, a: Vector, b: Vector, ccw: bool) -> Edge:
    """Arc about c from a to b, counter-clockwise or clockwise."""
    t0 = math.atan2(a.Y - c.Y, a.X - c.X)
    t1 = math.atan2(b.Y - c.Y, b.X - c.X)
    if ccw:
        while t1 <= t0:
            t1 += 2 * math.pi
    else:
        while t1 >= t0:
            t1 -= 2 * math.pi
    tm = (t0 + t1) / 2
    r = (a - c).length
    return Edge.make_three_point_arc(a, c + Vector(math.cos(tm), math.sin(tm), 0) * r, b)


def outline_face(rec: Recipe, pins: dict[str, tuple[float, float]]) -> Face:
    """The outline as an exact face: lug arcs joined by the recipe's edges, counter-clockwise."""
    order = [e.frm for e in rec.outline]
    if sorted(order) != sorted(pins) or [e.to for e in rec.outline] != order[1:] + order[:1]:
        raise RecipeError(f"The outline must go once round every pin, each edge from one pin to the next: {sorted(pins)}")
    C = {k: _v((pins[k][0], pins[k][1])) for k in pins}
    R = {}
    for k in pins:
        if k not in rec.lugs or rec.lugs[k] <= 0:
            raise RecipeError(f"Give a positive lug radius for {k}")
        R[k] = rec.lugs[k] * IN
    centroid = sum((C[k] for k in order), Vector(0, 0, 0)) * (1 / len(order))
    joins = []  # per edge: (start point on lug `frm`, end point on lug `to`, connecting edges)
    for e in rec.outline:
        c1, c2, r1, r2 = C[e.frm], C[e.to], R[e.frm], R[e.to]
        d = c2 - c1
        L = d.length
        out = _right(_unit(d))  # outward, for a counter-clockwise loop
        if out.dot((c1 + c2) * 0.5 - centroid) < 0 and len(order) > 2:
            raise RecipeError("List the outline edges counter-clockwise round the part")
        if e.edge in ("tangent", "spline"):
            cosf = (r1 - r2) / L
            if abs(cosf) >= 1:
                raise RecipeError(f"Lugs {e.frm} and {e.to} overlap; no tangent line between them")
            n = _unit(d) * cosf + out * math.sqrt(1 - cosf * cosf)
            t1, t2 = c1 + n * r1, c2 + n * r2
            if e.edge == "tangent":
                joins.append((t1, t2, [Edge.make_line(t1, t2)]))
            else:
                pts = [t1, *[_v((x * IN, y * IN)) for x, y in e.through], t2]
                ta = Vector(-(t1 - c1).Y, (t1 - c1).X, 0).normalized()  # counter-clockwise along lug 1
                tb = Vector(-(t2 - c2).Y, (t2 - c2).X, 0).normalized()
                ta = ta if ta.dot(pts[1] - t1) >= 0 else -ta
                tb = tb if tb.dot(t2 - pts[-2]) >= 0 else -tb
                joins.append((t1, t2, spline_edges(pts, ta, tb)))
        elif e.edge == "arc":
            if not e.radius or e.radius <= 0:
                raise RecipeError(f"Arc {e.frm}→{e.to} needs a radius")
            rb = e.radius * IN
            if e.bulge == "in":  # concave: a blend circle outside, touching both lugs from outside
                p = _circle_meet(c1, rb + r1, c2, rb + r2, out)
                if p is None:
                    raise RecipeError(f"Arc {e.frm}→{e.to}: radius too small to reach both lugs; at least {(L - r1 - r2) / 2 / IN:.3f} in")
                t1, t2 = c1 + _unit(p - c1) * r1, c2 + _unit(p - c2) * r2
                joins.append((t1, t2, [_arc(p, t1, t2, ccw=False)]))
            else:  # convex: a large circle inside, holding both lugs
                p = _circle_meet(c1, rb - r1, c2, rb - r2, -out)
                if p is None or rb <= max(r1, r2):
                    raise RecipeError(f"Arc {e.frm}→{e.to}: radius too small to wrap both lugs; at least {(L + r1 + r2) / 2 / IN:.3f} in")
                t1, t2 = c1 + _unit(c1 - p) * r1, c2 + _unit(c2 - p) * r2
                joins.append((t1, t2, [_arc(p, t1, t2, ccw=True)]))
        else:
            raise RecipeError(f"Unknown outline edge {e.edge!r}: use tangent, arc or spline")
    edges: list[Edge] = []
    n = len(rec.outline)
    for i, e in enumerate(rec.outline):
        _a, b_end, conn = joins[i]
        edges += conn
        nxt = rec.outline[(i + 1) % n]
        start_next = joins[(i + 1) % n][0]
        edges.append(_arc(C[nxt.frm], b_end, start_next, ccw=True))  # round the next lug, outside
    try:
        return Face(Wire(edges))
    except Exception as exc:  # noqa: BLE001 - self-intersecting loops
        raise RecipeError(f"The outline does not close into one clean loop ({exc})") from exc


# ---------------------------------------------------------------- the rest


def _disk(x: float, y: float, r: float) -> Face:
    return Face(Wire([Edge.make_circle(r, Plane.XY.move(Location((x, y, 0))))]))


def _minus(a: list[Face], b: list[Face]) -> list[Face]:
    out = []
    for f in a:
        g = f
        for k in b:
            g = g - k
        out.extend(g.faces())
    return [f for f in out if f.area > 1e-3]


def _round(faces: list[Face], r: float) -> list[Face]:
    return _offset(_offset(faces, -r), r) if faces else faces


def spot_rectangles(part: list[Face], pins: dict[str, tuple[float, float]], spots: list[tuple[str, float]], margin: float):
    """Spot-face regions: beyond a step square to each lug's arm (`step` mm from the pin, toward the middle of the pins),
    wide and long enough that the step is the only wall inside the part. Returns faces and the step lines."""
    centroid = np.mean(np.array(list(pins.values())), axis=0)
    ring = np.array([[(w @ (i / 400)).X, (w @ (i / 400)).Y] for f in part for w in [f.outer_wire()] for i in range(401)])
    rects, steps = [], []
    for pin, step in spots:
        pc = np.array(pins[pin])
        u = pc - centroid
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        t0 = -step
        rel = ring - pc
        t, w = rel @ u, rel @ v
        beyond = t >= t0
        W = (np.abs(w[beyond]).max() if beyond.any() else step) + margin
        T = (t[beyond].max() if beyond.any() else step) + margin
        corners = [pc + a * u + b * v for a, b in ((t0, -W), (T, -W), (T, W), (t0, W))]
        rects.append(Face(Wire.make_polygon([Vector(float(x), float(y), 0) for x, y in corners], close=True)))
        steps.append((pin, step, corners[0], corners[3]))
    return rects, steps


def compile_recipe(rec: Recipe, pins: dict[str, tuple[float, float]]) -> Design:
    """Recipe → sketches (constraints inferred, pins held) and features: pad, web pockets, spot faces, pin bores."""
    from .sketch_seed import _dimension_steps, offset_pocket

    if rec.z_max - rec.z_min < 0.05:
        raise RecipeError("z_max must be above z_min")
    r_cut = rec.corner_radius * IN
    outline = outline_face(rec, pins)
    holes = []
    for h in rec.holes:
        if len(h.points) < 3:
            raise RecipeError("A hole needs at least three corners")
        poly = Face(Wire.make_polygon([_v((x * IN, y * IN)) for x, y in h.points], close=True))
        rounded = _round([poly], max(h.corner_radius * IN, r_cut))
        if not rounded:
            raise RecipeError("A hole is narrower than the cutter")
        holes += rounded
    profile = _minus([outline], holes) if holes else [outline]
    if not profile:
        raise RecipeError("The holes remove the whole part")

    sketches, feats = [], []

    def sketch(faces, name):
        sid = f"S{len(sketches) + 1}"
        sketches.append(faces_to_sketch(faces, sid, name, pins))
        return sid

    def feat(kind, params, refs):
        feats.append(FeatureInstance(id=f"F{len(feats) + 1}", kind=kind, params=params, refs=refs))
        return feats[-1].id

    s_profile = sketch(profile, "Profile")
    pad = feat("pad", {"z_min": round(rec.z_min * IN, 4), "z_max": round(rec.z_max * IN, 4)}, {"sketch": s_profile})
    if rec.web and (rec.web.depth_top > 0 or rec.web.depth_bottom > 0):
        web = offset_pocket(sketches[0], f"S{len(sketches) + 1}", "Web pockets", rec.web.rim * IN, r_cut, rec.web.boss_gap * IN)
        sketches.append(web)
        for side, d in (("+z", rec.web.depth_top), ("-z", rec.web.depth_bottom)):
            if d > 0:
                feat("sketch_pocket", {"side": side, "depth": round(d * IN, 4)}, {"base": pad, "sketch": web.id})
    groups: dict[tuple[str, float], list[Spot]] = {}
    for s in rec.spots:
        if s.side not in ("+z", "-z") or s.depth <= 0 or s.step <= 0:
            raise RecipeError("Each spot face needs a side (+z or -z), a positive depth and a positive step")
        groups.setdefault((s.side, round(s.depth, 4)), []).append(s)
    for (side, depth), group in groups.items():
        rects, steps = spot_rectangles(profile, pins, [(s.pin, s.step * IN) for s in group], 0.25 * IN)
        u = rects[0]
        for r in rects[1:]:
            u = u + r
        sid = sketch(list(u.faces()), f"Spot faces {side.upper()} {depth:.3f} in")
        _dimension_steps(sketches[-1], steps)
        feat("sketch_pocket", {"side": side, "depth": round(depth * IN, 4)}, {"base": pad, "sketch": sid})
    for b in pins:
        feat("pin_bore", {}, {"pin": b, "base": pad})
    return Design(features=feats, sketches=sketches)
