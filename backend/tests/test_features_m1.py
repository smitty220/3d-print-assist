"""Extrude, Revolve and Shell, checked against hand-calculated volumes (formula beside each)."""

import math

import pytest

from print_assist.build.engine import Engine
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.rules import PrintRules
from print_assist.sketch.model import FacePick, Placement, Sketch

A_HEX = 3 * math.sqrt(3) / 2 * (40 / math.sqrt(3)) ** 2  # regular hexagon, 40 across flats: 1385.64 mm²


def poly(sid, pts, plane="XY", **placement):
    ents = [{"id": f"p{i}", "kind": "point", "x": x, "y": y} for i, (x, y) in enumerate(pts)]
    ents += [{"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % len(pts)}"} for i in range(len(pts))]
    return Sketch(id=sid, entities=ents, placement=Placement(plane=plane, **placement))


def hexagon(sid="S1"):
    a = 40 / math.sqrt(3)
    return poly(sid, [(a * math.cos(math.radians(60 * i)), a * math.sin(math.radians(60 * i))) for i in range(6)])


def rect(sid, x0, y0, x1, y1, plane="XY", **kw):
    return poly(sid, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], plane, **kw)


def build(sketches, features):
    r = Engine().build(Design(sketches=sketches, features=features), PrintRules())
    assert [f.error for f in r.features] == [None] * len(features), [f.error for f in r.features]
    return r


def ext(fid="F1", **params):
    return FI(id=fid, kind="extrude", refs={"sketch": "S1"}, params=params)


def test_plain_extrude_of_a_hexagon():
    r = build([hexagon()], [ext(distance=100.0)])
    assert r.volume == pytest.approx(A_HEX * 100, rel=1e-6)  # A·h
    bb = r.world.bounding_box()
    assert (bb.min.Z, bb.max.Z) == pytest.approx((0, 100))
    assert r.face_names.count("F1.start") == 1 and r.face_names.count("F1.end") == 1 and r.face_names.count("F1.side") == 6


def test_twist_keeps_the_volume():
    r = build([hexagon()], [ext(distance=100.0, twist=60.0)])
    assert r.volume == pytest.approx(A_HEX * 100, rel=1e-5)  # turning a section does not change its area


def test_twist_with_end_scale_is_a_twisted_frustum():
    r = build([hexagon()], [ext(distance=100.0, twist=60.0, end_scale=0.8)])
    s = 0.8
    assert r.volume == pytest.approx(A_HEX * 100 * (1 + s + s * s) / 3, rel=1e-5)  # A·h·(1 + s + s²)/3 = 112698.77


def test_draft_tapers_a_square():
    r = build([rect("S1", -10, -10, 10, 10)], [ext(distance=10.0, draft=5.0)])
    top = 20 - 2 * 10 * math.tan(math.radians(5))
    assert r.volume == pytest.approx(10 / 3 * (20**2 + top**2 + 20 * top), rel=1e-6)  # frustum h/3·(A1 + A2 + √(A1·A2))


def test_sketch_planes_and_extents():
    r = build([rect("S1", 0, 0, 20, 10, plane="XZ")], [ext(distance=5.0)])
    bb = r.world.bounding_box()  # XZ's normal is −Y: the 20 × 10 rectangle stands up and extrudes toward −Y
    assert r.volume == pytest.approx(1000)
    assert (bb.min.Y, bb.max.Y, bb.max.Z) == pytest.approx((-5, 0, 10))
    r = build([rect("S1", 0, 0, 20, 10, offset=3.0)], [ext(distance=4.0, extent="symmetric")])
    bb = r.world.bounding_box()
    assert (bb.min.Z, bb.max.Z) == pytest.approx((1, 5))  # sketch at Z = 3, ±2


def test_cut_and_intersect():
    sk = [rect("S1", -20, -15, 20, 15), rect("S2", -5, -5, 5, 5)]
    r = build(sk, [ext(distance=10.0), FI(id="F2", kind="extrude", refs={"sketch": "S2"}, params={"op": "cut", "extent": "through_all"})])
    assert r.volume == pytest.approx(40 * 30 * 10 - 10 * 10 * 10)  # plate − through hole
    r = build(sk, [ext(distance=10.0), FI(id="F2", kind="extrude", refs={"sketch": "S2"}, params={"op": "intersect", "distance": 4.0})])
    assert r.volume == pytest.approx(10 * 10 * 4)  # only the overlap stays


def test_revolve_a_rectangle_into_a_ring():
    r = build([rect("S1", 10, 0, 20, 10, plane="XZ")], [FI(id="F1", kind="revolve", refs={"sketch": "S1"}, params={"axis": "Y"})])
    assert r.volume == pytest.approx(math.pi * (20**2 - 10**2) * 10, rel=1e-6)  # π(R² − r²)h = 9424.78
    bb = r.world.bounding_box()
    assert (bb.min.Z, bb.max.Z, bb.max.X) == pytest.approx((0, 10, 20))
    half = build([rect("S1", 10, 0, 20, 10, plane="XZ")], [FI(id="F1", kind="revolve", refs={"sketch": "S1"}, params={"angle": 180.0})])
    assert half.volume == pytest.approx(r.volume / 2, rel=1e-6)


def test_revolve_refuses_a_profile_across_its_axis():
    r = Engine().build(Design(sketches=[rect("S1", -5, 0, 5, 10, plane="XZ")],
                              features=[FI(id="F1", kind="revolve", refs={"sketch": "S1"})]), PrintRules())
    assert "crosses" in r.features[0].error


def test_shell_an_open_box():
    top = FacePick(name="F1.end", at=[0, 0, 20], normal=[0, 0, 1])
    r = build([rect("S1", -20, -15, 20, 15)], [ext(distance=20.0), FI(id="F2", kind="shell", params={"thickness": 2.0}, faces=[top])])
    assert r.volume == pytest.approx(40 * 30 * 20 - 36 * 26 * 18, rel=1e-6)  # outer − inner (open top) = 7152
    assert r.face_names.count("F2.inner") == 5  # four walls and the floor inside
    closed = build([rect("S1", -20, -15, 20, 15)], [ext(distance=20.0), FI(id="F2", kind="shell", params={"thickness": 2.0})])
    assert closed.volume == pytest.approx(40 * 30 * 20 - 36 * 26 * 16, rel=1e-6)  # a closed hollow box


def test_shell_a_twisted_tapered_hexagon_and_fillet_its_rim():
    top = FacePick(name="F1.end", at=[0, 0, 120], normal=[0, 0, 1])
    feats = [ext(distance=120.0, twist=60.0, end_scale=1.4),
             FI(id="F2", kind="shell", params={"thickness": 1.6}, faces=[top]),
             FI(id="F3", kind="fillet", params={"radius": 0.5}, refs={"faces_a": "F1.end", "faces_b": "F1.side"})]
    r = build([hexagon()], feats)
    outer = A_HEX * 120 * (1 + 1.4 + 1.4**2) / 3
    assert 0.1 * outer < r.volume < 0.25 * outer
    assert r.features[1].status == "ok" and r.features[2].status == "ok"
