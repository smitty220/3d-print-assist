import math

import pytest
from build123d import Face, IN, Vector, Wire

from cad_assist.features.design import build_design
from cad_assist.features.library import _offset
from cad_assist.features.model import Design
from cad_assist.features.model import FeatureInstance as FI
from cad_assist.intent.models import Intent
from cad_assist.sketch.from_faces import faces_to_sketch
from cad_assist.sketch.geometry import SketchGeometryError, regions
from cad_assist.sketch.model import Sketch
from cad_assist.sketch.solve import solve


def _pt(i, x, y, **kw):
    return {"id": i, "kind": "point", "x": x, "y": y, **kw}


def test_solver_makes_a_line_tangent_to_two_equal_lugs():
    s = Sketch(id="S1", entities=[
        _pt("c1", 0, 0, fixed=True), _pt("c2", 100, 0, fixed=True),
        {"id": "k1", "kind": "circle", "c": "c1", "r": 14}, {"id": "k2", "kind": "circle", "c": "c2", "r": 10},
        _pt("a", 2, 15), _pt("b", 98, 12), {"id": "l", "kind": "line", "p1": "a", "p2": "b"},
    ], constraints=[
        {"id": "t1", "kind": "tangent", "refs": ["l", "k1"]}, {"id": "t2", "kind": "tangent", "refs": ["k2", "l"]},
        {"id": "r", "kind": "radius", "refs": ["k1"], "value": 14.2}, {"id": "e", "kind": "equal", "refs": ["k1", "k2"]},
    ])
    out = solve(s)
    assert out.report.status == "solved" and not out.report.conflicting and not out.report.redundant
    # Equal radii, so the common tangent is parallel to the centre line, one radius away: y = 14.2 at both ends.
    assert out.point("a").y == pytest.approx(14.2, abs=1e-6) and out.point("b").y == pytest.approx(14.2, abs=1e-6)
    assert out.get("k2").r == pytest.approx(14.2, abs=1e-9)
    assert out.report.dof == 2  # the line's two ends may still slide along it


def test_regions_nest_material_and_holes():
    sq = lambda pre, x0, y0, x1, y1: (  # noqa: E731
        [_pt(f"{pre}{i}", x, y) for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]
        + [{"id": f"{pre}l{i}", "kind": "line", "p1": f"{pre}{i}", "p2": f"{pre}{(i + 1) % 4}"} for i in range(4)])
    s = Sketch(id="S1", entities=sq("o", 0, 0, 40, 40) + sq("i", 10, 10, 30, 30) + [_pt("m", 20, 20), {"id": "k", "kind": "circle", "c": "m", "r": 5}])
    # Outer square material, inner square a hole, the circle inside it an island again.
    assert sum(f.area for f in regions(s)) == pytest.approx(1600 - 400 + math.pi * 25, rel=1e-9)
    s.entities = [e for e in s.entities if e.id != "il3"]
    with pytest.raises(SketchGeometryError, match="not closed"):
        regions(s)


def test_faces_to_sketch_infers_a_fully_constrained_lug_hull():
    R = 14.0
    tri = Face(Wire.make_polygon([Vector(0, 0, 0), Vector(100, 0, 0), Vector(0, 60, 0)], close=True))
    hull = _offset([tri], R)
    s = faces_to_sketch(hull, "S1", "Profile", {"IF-00": (0, 0), "IF-01": (100, 0), "IF-02": (0, 60)})
    kinds = [c.kind for c in s.constraints]
    assert kinds.count("tangent") == 6 and kinds.count("radius") == 3
    out = solve(s)
    assert out.report.dof == 0 and not out.report.redundant and not out.report.conflicting
    # Hull of three equal disks: triangle area + perimeter × R + πR².
    perimeter = 100 + 60 + math.hypot(100, 60)
    assert sum(f.area for f in regions(out)) == pytest.approx(3000 + perimeter * R + math.pi * R * R, rel=1e-9)


def test_pad_and_pocket_from_sketches():
    def rect(sid, x0, y0, x1, y1):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return Sketch(id=sid, entities=[_pt(f"p{i}", x, y) for i, (x, y) in enumerate(pts)]
                      + [{"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % 4}"} for i in range(4)])
    it = Intent()
    from cad_assist.intent.models import DesignSpace, Frame
    it.objects.append(DesignSpace(id="DS-01", frame=Frame(origin=(0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)), min=(-50, -50, -10), max=(50, 50, 10)))
    d = Design(sketches=[rect("S1", 0, 0, 40, 30), rect("S2", 10, 5, 30, 25)], features=[
        FI(id="F1", kind="pad", params={"z_min": -5.0, "z_max": 5.0}, refs={"sketch": "S1"}),
        FI(id="F2", kind="sketch_pocket", params={"side": "-z", "depth": 3.0}, refs={"base": "F1", "sketch": "S2"}),
    ])
    res = build_design(it, d)
    assert [f.error for f in res.features] == [None, None]
    assert res.volume == pytest.approx(40 * 30 * 10 - 20 * 20 * 3, rel=1e-9)
    bb = res.local.bounding_box()
    assert (bb.min.Z, bb.max.Z) == pytest.approx((-5, 5))


def test_spline_ends_follow_the_curves_they_join():
    """A spline with tangent ends leaves a lug arc along it: no kink where the outline turns from arc to curve."""
    s = Sketch(id="S1", entities=[
        _pt("c", 0, 0), _pt("a", 10, 0), _pt("b", 0, 10), _pt("m", 20, 20),
        {"id": "arc", "kind": "arc", "c": "c", "start": "a", "end": "b"},
        {"id": "sp", "kind": "spline", "points": ["a", "m", "b"], "tangent_start": True, "tangent_end": True},
    ])
    from cad_assist.sketch.geometry import edges
    sp = [e for e in edges(s) if e.geom_type.name == "BEZIER"]
    assert tuple(sp[0].tangent_at(0)) == pytest.approx((0, 1, 0))  # the arc's tangent at (10, 0), heading toward m
    assert tuple(sp[-1].tangent_at(1)) == pytest.approx((-1, 0, 0))  # arriving at (0, 10) along the arc
    assert sum(f.area for f in regions(s)) > math.pi * 100 / 4  # a closed region: quarter arc plus the bulge


def test_offset_pocket_keeps_a_rim_and_clears_the_lugs():
    from cad_assist.topo.sketch_seed import offset_pocket

    R, rim = 14.0, 3.0
    tri = Face(Wire.make_polygon([Vector(0, 0, 0), Vector(100, 0, 0), Vector(0, 60, 0)], close=True))
    pins = {"IF-00": (0, 0), "IF-01": (100, 0), "IF-02": (0, 60)}
    profile = solve(faces_to_sketch(_offset([tri], R), "S1", "Profile", pins))
    pocket = solve(offset_pocket(profile, "S2", "Web", rim, 3.175))
    assert pocket.report.status == "solved" and not pocket.report.redundant and not pocket.report.conflicting
    # Every boss arc is concentric with its pin (shares the pin's held point) at the lug radius.
    bosses = [e for e in pocket.entities if e.kind == "arc" and e.c.startswith("pin_")]
    assert len(bosses) == 3
    for a in bosses:
        c, s0 = pocket.point(a.c), pocket.point(a.start)
        assert math.hypot(s0.x - c.x, s0.y - c.y) == pytest.approx(R, abs=1e-6)
    # The pocket stays the rim inside the outline, and clear of each lug circle.
    from cad_assist.sketch.geometry import edges
    outer = regions(profile)[0].outer_wire()
    pts = [e @ (i / 10) for e in edges(pocket) for i in range(11)]
    assert min(outer.distance_to(q) for q in pts) == pytest.approx(rim, abs=1e-3)
    assert all(math.hypot(q.X - x, q.Y - y) >= R - 1e-6 for q in pts for x, y in pins.values())
