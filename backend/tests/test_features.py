import math

import pytest

from print_assist.features.context import FeatureError
from print_assist.features.design import build_design
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.rules import PrintRules


def test_plate_volume_matches_hand_calc():
    r = build_design(Design(features=[FI(id="F1", kind="plate")]))
    # 60 x 40 x 5 with four 3 mm corner radii: (L·W − (4 − π)·r²)·t
    expected = (60 * 40 - (4 - math.pi) * 3**2) * 5
    assert r.volume == pytest.approx(expected, rel=1e-6)
    assert r.face_names.count(None) == 0
    bb = r.world.bounding_box()
    assert bb.min.Z == pytest.approx(0) and bb.max.Z == pytest.approx(5)
    # PLA at 1240 kg/m³
    assert r.mass == pytest.approx(expected * 1e-9 * 1240, rel=1e-6)


def test_thin_plate_fails_the_wall_check():
    r = build_design(Design(features=[FI(id="F1", kind="plate", params={"thickness": 0.6})]), PrintRules())
    (c,) = [c for c in r.checks if c.name == "Thickness"]
    assert c.status == "fail"
    r = build_design(Design(features=[FI(id="F1", kind="plate", params={"thickness": 1.0})]))
    assert next(c for c in r.checks if c.name == "Thickness").status == "warn"


def test_feature_errors_are_reported_not_raised():
    d = Design(features=[FI(id="F1", kind="plate"), FI(id="F2", kind="fillet", params={"radius": 80.0}),
                         FI(id="F3", kind="nope")])
    r = build_design(d)
    errors = {f.id: f.error for f in r.features if f.error}
    assert "pick the edges" in errors["F2"]
    assert "unknown feature kind" in errors["F3"]
    assert r.world is not None and r.world.is_valid


L_SHAPE = [[0, 0], [40, 0], [40, 10], [12, 10], [12, 30], [0, 30]]  # mm, area 640, one inside corner at (12, 10)


def test_profile_plate_rounds_only_inside_corners():
    r = 3.0
    d = Design(features=[FI(id="F1", kind="profile_plate", params={"outline": L_SHAPE, "z_min": 0.0, "z_max": 5.0, "corner_radius": r})])
    res = build_design(d)
    assert res.features[0].error is None
    # The inside corner gains a fillet of r²(1 − π/4); the five outside corners stay sharp.
    assert res.volume == pytest.approx((640 + r * r * (1 - math.pi / 4)) * 5, rel=1e-6)


def test_profile_cut_pocket_and_island():
    rect = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    r = 2.0
    plate = FI(id="F1", kind="profile_plate", params={"outline": rect(0, 0, 40, 30), "z_min": -5.0, "z_max": 5.0, "corner_radius": r})
    pocket = FI(id="F2", kind="profile_cut", refs={"base": "F1"}, params={
        "outline": rect(5, 5, 35, 25), "islands": [rect(15, 10, 25, 20)], "side": "+z", "depth": 4.0, "corner_radius": r})
    res = build_design(Design(features=[plate, pocket]))
    assert [f.error for f in res.features] == [None, None]
    # Pocket 30 x 20 with its four corners rounded, minus a 10 x 10 island left standing (sharp: material corners), 4 deep.
    cut = (30 * 20 - 4 * r * r * (1 - math.pi / 4) - 10 * 10) * 4
    assert res.volume == pytest.approx(40 * 30 * 10 - cut, rel=1e-6)
    through = FI(id="F3", kind="profile_cut", refs={"base": "F1"}, params={"outline": rect(36, 2, 39, 28), "through": "yes", "corner_radius": 1.0})
    res = build_design(Design(features=[plate, through]))
    assert res.volume == pytest.approx(40 * 30 * 10 - (3 * 26 - 4 * (1 - math.pi / 4)) * 10, rel=1e-6)


def _box(w=40.0):
    return FI(id="F1", kind="profile_plate", params={"outline": [[0, 0], [w, 0], [w, 30], [0, 30]], "z_min": 0.0, "z_max": 10.0, "corner_radius": 0.0})


def _edge_at(shape, x, y, z):
    from build123d import Vector

    return min(range(len(shape.edges())), key=lambda i: (shape.edges()[i].position_at(0.5) - Vector(x, y, z)).length)


def test_fillet_on_a_picked_edge_follows_upstream_edits():
    from print_assist.features.design import describe_edge

    base = build_design(Design(features=[_box()]))
    pick = describe_edge(base.local, base.face_names, _edge_at(base.local, 40, 30, 5))  # the vertical edge at (40, 30)
    assert pick.length == pytest.approx(10) and pick.faces == ["F1.sides", "F1.sides"]
    r = 4.0
    fillet = FI(id="F2", kind="fillet", params={"radius": r}, edges=[pick])
    res = build_design(Design(features=[_box(), fillet]))
    assert res.features[1].error is None
    # One 10 mm vertical edge rounded: r²(1 − π/4) off the cross-section.
    assert res.volume == pytest.approx((40 * 30 - r * r * (1 - math.pi / 4)) * 10, rel=1e-6)
    # Widen the plate upstream: the picked edge moves 10 mm with it and is found again (same faces, clearly nearest).
    res = build_design(Design(features=[_box(50.0), fillet]))
    assert res.features[1].error is None
    assert res.volume == pytest.approx((50 * 30 - r * r * (1 - math.pi / 4)) * 10, rel=1e-6)
    # A blend has nothing to pick on a tangent joint: the fillet's own boundary edges are refused.
    with pytest.raises(FeatureError, match="tangent"):
        describe_edge(res.local, res.face_names, _edge_at(res.local, 50 - r, 30, 5))


def test_picked_edge_blend_runs_on_along_tangent_edges():
    from print_assist.features.design import describe_edge

    plate = FI(id="F1", kind="profile_plate", params={"outline": L_SHAPE, "z_min": 0.0, "z_max": 5.0, "corner_radius": 3.0})
    base = build_design(Design(features=[plate]))
    # The top rim along y = 10, the inside-corner arc and the rim along x = 12 are one tangent chain.
    chain = [describe_edge(base.local, base.face_names, _edge_at(base.local, *xyz)) for xyz in ((28, 10, 5), (12.88, 10.88, 5), (12, 22, 5))]
    one = build_design(Design(features=[plate, FI(id="F2", kind="fillet", params={"radius": 1.0}, edges=chain[:1])]))
    three = build_design(Design(features=[plate, FI(id="F2", kind="fillet", params={"radius": 1.0}, edges=chain)]))
    assert one.features[1].error is None and three.features[1].error is None
    assert one.volume == pytest.approx(three.volume, rel=1e-9) and one.volume < base.volume


def test_fillet_without_picks_or_faces_says_what_to_do():
    res = build_design(Design(features=[_box(), FI(id="F2", kind="fillet")]))
    assert "pick the edges" in res.features[1].error
