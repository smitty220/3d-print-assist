"""A first sketch-based design for a pin-mounted plate (a bell crank, a lever), drafted from a topology reading.

Deterministic, as a starting point the engineer (and later the AI) edits: the outline is the hull of a lug circle at
every pin (lines tangent to the lugs), the result's cut-outs become filleted lightening holes, the faces are pocketed
to the web floors leaving a rim and a boss round each pin, and spot faces thin each lug to fit its fork.
"""

from __future__ import annotations

import numpy as np
from build123d import Edge, Face, Location, Plane, Vector, Wire
from skimage.measure import approximate_polygon

from ..features.library import _offset
from ..features.model import Design, FeatureInstance
from ..sketch.from_faces import faces_to_sketch
from .interpret import Reading, _inch

SIMPLIFY_HOLE = 4.0  # elements: cut-out outlines become a few straight sides before filleting


def _poly_face(pts) -> Face:
    return Face(Wire.make_polygon([Vector(float(x), float(y), 0) for x, y in pts], close=True))


def _disk(x: float, y: float, r: float) -> Face:
    return Face(Wire([Edge.make_circle(r, Plane.XY.move(Location((x, y, 0))))]))


def _union(faces: list[Face]) -> list[Face]:
    out = faces[0]
    for f in faces[1:]:
        out = out + f
    return list(out.faces())


def _minus(a: list[Face], b: list[Face]) -> list[Face]:
    out = []
    for f in a:
        g = f
        for k in b:
            g = g - k
        out.extend(g.faces())
    return [f for f in out if f.area > 1e-3]


def _round(faces: list[Face], r: float) -> list[Face]:
    """Round a cut's outside corners to the cutter radius (and drop anything narrower than the cutter)."""
    return _offset(_offset(faces, -r), r) if faces else faces


def seed_design(rd: Reading, h: float, rim: float | None = None) -> tuple[Design, list[str]]:
    notes: list[str] = []
    r_cut = rd.corner_radius
    pins = {k: (x, y) for k, (x, y, _d) in rd.pins.items()}
    if len(pins) < 3:
        raise ValueError("The first sketch needs three or more pins (a bell crank or lever); place features by hand")
    # Lug radius: full-bearing edge distance, rounded up to 0.005 in.
    R = float(np.ceil(max(1.5 * d for _x, _y, d in rd.pins.values()) / (0.005 * 25.4)) * 0.005 * 25.4)
    rim = rim if rim is not None else max(r_cut, 0.15 * 25.4)

    # Outline: the hull of the lug circles = the pin polygon grown by R (lines tangent to the lugs).
    centres = np.array(list(pins.values()))
    order = np.argsort(np.arctan2(*(centres - centres.mean(axis=0)).T[::-1]))
    hull = _offset([_poly_face(centres[order])], R)
    notes.append(f"Outline: lug arcs R{_inch(R)} (e/D 1.5) about the {len(pins)} pins, joined by tangent lines")

    # Lightening holes: the result's cut-outs, simplified to a few sides, kept a rim inside the outline, filleted.
    inner = _offset(hull, -rim)
    holes: list[Face] = []
    for c in rd.cutouts:
        pts = approximate_polygon(np.asarray(c), tolerance=SIMPLIFY_HOLE * h)
        if len(pts) < 4:
            continue
        f = _minus([_poly_face(pts[:-1])], _minus(hull, inner))  # clip to leave the rim
        f = [g for g in f if g.area > np.pi * r_cut**2]
        holes += _round(f, r_cut)
    if holes:
        notes.append(f"{len(holes)} lightening hole(s) from the result's cut-outs, corners R{_inch(r_cut)} (the cutter)")
    profile = _minus(hull, holes) if holes else hull

    # Web pockets: inside the rim, round a boss at each pin.
    bosses = [_disk(x, y, R) for x, y in pins.values()]
    web = _round(_minus(inner, bosses), r_cut)

    # Spot faces round the lugs, grouped by side and depth; if every pin needs one from a side, lower the stock instead.
    z_min, z_max = rd.z_min, rd.z_max
    spots = list(rd.spots)
    for side in ("+z", "-z"):
        mine = [s for s in spots if s["side"] == side]
        if len({s["pin"] for s in mine}) == len(pins):
            skim = min(s["depth"] for s in mine)
            if side == "+z":
                z_max -= skim
            else:
                z_min += skim
            for s in mine:
                s["depth"] -= skim
            notes.append(f"Stock {_inch(skim)} thinner from {side.upper()} so the hub fits its fork")
    spots = [s for s in spots if s["depth"] > 1e-3]
    web_depth = {side: d - ((rd.z_max - z_max) if side == "+z" else (z_min - rd.z_min)) for side, d in rd.web.items()}

    sketches, feats = [], []

    def sketch(faces, name):
        sid = f"S{len(sketches) + 1}"
        sketches.append(faces_to_sketch(faces, sid, name, pins))
        return sid

    def feat(kind, params, refs):
        feats.append(FeatureInstance(id=f"F{len(feats) + 1}", kind=kind, params=params, refs=refs))
        return feats[-1].id

    s_profile = sketch(profile, "Profile")
    pad = feat("pad", {"z_min": round(z_min, 4), "z_max": round(z_max, 4)}, {"sketch": s_profile})
    if web:
        s_web = sketch(web, "Web pockets")
        for side, d in web_depth.items():
            if d > 1e-3:
                feat("sketch_pocket", {"side": side, "depth": round(d, 4)}, {"base": pad, "sketch": s_web})
        notes.append(f"Web pockets inside a {_inch(rim)} rim, bosses R{_inch(R)} at the pins: "
                     + ", ".join(f"{_inch(d)} from {s.upper()}" for s, d in web_depth.items()))
    # A spot face thins a lug end with a straight step across its arm, square to it (a circle concentric with the lug
    # would cross the tangent sides at a shallow angle and leave a knife edge). The step sits a cutter radius inside the
    # last column the tines need cleared.
    centroid = centres.mean(axis=0)
    groups: dict[tuple[str, float], list[dict]] = {}
    for s in spots:
        groups.setdefault((s["side"], round(s["depth"], 3)), []).append(s)
    for (side, depth), group in groups.items():
        rects, steps = [], []
        for s in group:
            px, py = pins[s["pin"]]
            u = np.array([px, py]) - centroid
            u /= np.linalg.norm(u)
            v = np.array([-u[1], u[0]])
            t0 = float(min(np.asarray(s["columns"]) @ u)) - h / 2 - r_cut
            corners = [np.array([px, py]) + t * u + w * v for t, w in ((t0, -3 * R), (t0 + 4 * R, -3 * R), (t0 + 4 * R, 3 * R), (t0, 3 * R))]
            rects.append(_poly_face(corners))
            steps.append((s["pin"], -t0, corners[0], corners[3]))
        sid = sketch(_union(rects), f"Spot faces {side.upper()} {_inch(depth)}")
        _dimension_steps(sketches[-1], steps)
        feat("sketch_pocket", {"side": side, "depth": depth}, {"base": pad, "sketch": sid})
        notes.append(f"Spot face{'s' if len(group) > 1 else ''} at {', '.join(s['pin'] for s in group)}: a step square to the arm "
                     f"{', '.join(_inch(t) for _p, t, _a, _b in steps)} from the pin, {_inch(depth)} deep from {side.upper()}, "
                     "so the lug fits between the tines")
    for b in pins:
        feat("pin_bore", {}, {"pin": b, "base": pad})
    return Design(features=feats, sketches=sketches), notes


def _dimension_steps(sk, steps) -> None:
    """Dimension each step line by its distance from its pin (the number an engineer would change)."""
    from ..sketch.model import Constraint, Line

    for pin, dist, a, b in steps:
        for e in sk.entities:
            if not isinstance(e, Line):
                continue
            p, q = sk.point(e.p1), sk.point(e.p2)
            ends = {(round(p.x, 3), round(p.y, 3)), (round(q.x, 3), round(q.y, 3))}
            if ends == {(round(float(a[0]), 3), round(float(a[1]), 3)), (round(float(b[0]), 3), round(float(b[1]), 3))}:
                sk.constraints.append(Constraint(id=f"k_step_{pin}", kind="distance", refs=[f"pin_{pin}", e.id], value=round(dist, 4)))


def offset_pocket(profile, sid: str, name: str, rim: float, r_cut: float, boss_gap: float = 0.0):
    """A pocket sketch a standard offset inside a profile sketch: the profile's outer edge offset inward by `rim`, kept
    clear of a boss at each pin the size of that pin's lug arc in the profile (plus `boss_gap`), outside corners at the
    cutter radius. The lightening holes are cut through anyway, so the pocket may run over them."""
    from ..sketch.geometry import regions
    from ..sketch.model import Arc, Point

    faces = regions(profile)
    outer = [Face(f.outer_wire()) for f in faces]
    inner = _offset(outer, -rim)
    pins, bosses = {}, []
    for e in profile.entities:
        if isinstance(e, Point) and e.ref:
            pins[e.ref] = (e.x, e.y)
            radii = [np.hypot(profile.point(a.start).x - e.x, profile.point(a.start).y - e.y)
                     for a in profile.entities if isinstance(a, Arc) and a.c == e.id]
            if radii:
                bosses.append(_disk(e.x, e.y, max(radii) + boss_gap))
    web = _round(_minus(inner, bosses), r_cut)
    if not web:
        raise ValueError("Nothing left to pocket inside that rim and those bosses")
    return faces_to_sketch(web, sid, name, pins)
