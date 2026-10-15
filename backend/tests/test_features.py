import math

import pytest
from build123d import IN, Axis

from cad_assist.features.context import FeatureError
from cad_assist.features.design import build_design
from cad_assist.features.model import Design
from cad_assist.features.model import FeatureInstance as FI
from cad_assist.features.templates import TEMPLATES
from cad_assist.geometry.samples import actuator_clevis_end, frame_rib, rod_end, side_panel
from cad_assist.intent import builders
from cad_assist.intent.models import Intent


def _base_intent():
    rib = frame_rib()
    top = next(i for i, f in enumerate(rib.faces()) if f.is_same(rib.faces().sort_by(Axis.Z)[-1]))
    it = Intent()
    for k in ("planar_mount", "bolt_pattern"):
        it.objects.append(getattr(builders, k)(it, rib, "rib", top))
    return it


def _cyl(shape, r_in):
    return next(i for i, f in enumerate(shape.faces()) if f.geom_type.name == "CYLINDER" and abs(f.radius - r_in * IN) < 1e-3)


@pytest.fixture
def clevis_intent():
    it = _base_intent()
    re = rod_end()
    it.objects.append(builders.bore(it, re, "rod", _cyl(re, 0.1885)))
    return it


@pytest.fixture
def fork_intent():
    it = _base_intent()
    fk = actuator_clevis_end()
    it.objects.append(builders.bore(it, fk, "fork", _cyl(fk, 0.1885)))
    return it


@pytest.fixture
def angle_intent():
    it = _base_intent()
    pn = side_panel()
    face = next(i for i, x in enumerate(pn.faces()) if x.geom_type.name == "PLANE" and x.normal_at().X < -0.99)
    it.objects.append(builders.bolt_pattern(it, pn, "panel", face))
    return it


def test_plate_volume_matches_hand_calc(clevis_intent):
    r = build_design(clevis_intent, Design(features=[FI(id="F1", kind="plate")]))
    # 2.8 x 2.8 x 0.25 with four 0.3 corner radii
    expected = (2.8 * 2.8 - (4 - math.pi) * 0.3**2) * 0.25
    assert r.volume / IN**3 == pytest.approx(expected, rel=1e-6)
    assert r.face_names.count(None) == 0


def test_bolt_holes_cut_close_fit_holes(clevis_intent):
    d = Design(features=[FI(id="F1", kind="plate"), FI(id="F2", kind="bolt_holes", refs={"pattern": "IF-01", "target": "F1"})])
    r = build_design(clevis_intent, d)
    plate = (2.8 * 2.8 - (4 - math.pi) * 0.3**2) * 0.25
    holes = 4 * math.pi * (0.201 / 2) ** 2 * 0.25  # NAS1352-3 close-fit hole
    assert r.volume / IN**3 == pytest.approx(plate - holes, rel=1e-6)
    assert r.face_names.count("F2.holes") == 4
    ed = next(c for c in r.checks if c.name == "Edge distance e/D")
    assert ed.value == pytest.approx((1.4 - 0.95) / 0.190, rel=1e-3)
    assert ed.status == "ok"


@pytest.mark.parametrize("template,fixture", [
    ("clevis_bracket", "clevis_intent"),
    ("single_lug_fitting", "fork_intent"),
    ("angle_bracket", "angle_intent"),
])
def test_reference_brackets_build_clean(template, fixture, request):
    intent = request.getfixturevalue(fixture)
    r = build_design(intent, TEMPLATES[template][2](intent))
    assert [f.error for f in r.features if f.error] == []
    assert r.world.is_valid and len(r.world.solids()) == 1
    assert r.face_names.count(None) == 0
    assert all(c.status == "ok" for c in r.checks), [(c.name, c.status) for c in r.checks]
    assert 0.2 < r.mass / 0.45359237 < 0.6


def test_clevis_named_faces_and_driven_values(clevis_intent):
    r = build_design(clevis_intent, TEMPLATES["clevis_bracket"][2](clevis_intent))
    bores = [i for i, n in enumerate(r.face_names) if n == "F3.bore"]
    assert len(bores) == 2
    assert all(r.local.faces()[i].radius / IN == pytest.approx(0.1885) for i in bores)
    clevis = next(f for f in r.features if f.id == "F3")
    assert clevis.driven["gap"] / IN == pytest.approx(0.625)  # 0.600 rod-end width + 0.025 clearance


def test_single_lug_respects_fork_gap(fork_intent):
    r = build_design(fork_intent, TEMPLATES["single_lug_fitting"][2](fork_intent))
    lug = next(f for f in r.features if f.id == "F3")
    assert lug.driven["side_clearance"] / IN == pytest.approx(0.015)
    too_thick = Design(features=[FI(id="F1", kind="plate"), FI(id="F3", kind="lug", refs={"pin": "IF-02", "base": "F1"}, params={"thickness": 0.45 * IN})])
    assert "less than the clevis gap" in build_design(fork_intent, too_thick).features[1].error


def test_rule_checks_flag_bad_values(clevis_intent):
    d = TEMPLATES["clevis_bracket"][2](clevis_intent)
    d.features[2].params["crown_radius"] = 0.45 * IN  # e/D = 1.19: reduced bearing
    d.features.append(FI(id="F7", kind="pocket", refs={"base": "F1"}, params={"floor": 0.03 * IN, "corner_radius": 0.03 * IN, "center_y": 1.0 * IN, "width_y": 0.3 * IN}))
    r = build_design(clevis_intent, d)
    status = {(c.feature, c.name): c.status for c in r.checks}
    assert status[("F3", "Edge margin e/D")] == "warn"
    assert status[("F7", "Floor thickness")] == "fail"
    assert status[("F7", "Depth / tool Ø")] == "fail"


def test_feature_errors_are_reported_not_raised(clevis_intent):
    d = TEMPLATES["clevis_bracket"][2](clevis_intent)
    d.features[4].params["radius"] = 2.0 * IN  # absurd fillet
    d.features[1].refs["pattern"] = "IF-99"
    r = build_design(clevis_intent, d)
    errors = {f.id: f.error for f in r.features if f.error}
    assert "not found" in errors["F2"]
    assert "fillet" in errors["F5"]
    assert r.world is not None and r.world.is_valid


def test_templates_explain_missing_intent():
    with pytest.raises(FeatureError, match="bore"):
        TEMPLATES["clevis_bracket"][2](_base_intent())
    with pytest.raises(FeatureError, match="upright"):
        TEMPLATES["angle_bracket"][2](_base_intent())


L_SHAPE = [[0, 0], [40, 0], [40, 10], [12, 10], [12, 30], [0, 30]]  # mm, area 640, one inside corner at (12, 10)


def test_profile_plate_rounds_only_inside_corners(clevis_intent):
    r = 3.0
    d = Design(features=[FI(id="F1", kind="profile_plate", params={"outline": L_SHAPE, "z_min": 0.0, "z_max": 5.0, "corner_radius": r})])
    res = build_design(clevis_intent, d)
    assert res.features[0].error is None
    # The inside corner gains a fillet of r²(1 − π/4); the five outside corners stay sharp.
    assert res.volume == pytest.approx((640 + r * r * (1 - math.pi / 4)) * 5, rel=1e-6)


def test_profile_cut_pocket_and_island(clevis_intent):
    rect = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    r = 2.0
    plate = FI(id="F1", kind="profile_plate", params={"outline": rect(0, 0, 40, 30), "z_min": -5.0, "z_max": 5.0, "corner_radius": r})
    pocket = FI(id="F2", kind="profile_cut", refs={"base": "F1"}, params={
        "outline": rect(5, 5, 35, 25), "islands": [rect(15, 10, 25, 20)], "side": "+z", "depth": 4.0, "corner_radius": r})
    res = build_design(clevis_intent, Design(features=[plate, pocket]))
    assert [f.error for f in res.features] == [None, None]
    # Pocket 30 x 20 with its four corners rounded, minus a 10 x 10 island left standing (sharp: material corners), 4 deep.
    cut = (30 * 20 - 4 * r * r * (1 - math.pi / 4) - 10 * 10) * 4
    assert res.volume == pytest.approx(40 * 30 * 10 - cut, rel=1e-6)
    through = FI(id="F3", kind="profile_cut", refs={"base": "F1"}, params={"outline": rect(36, 2, 39, 28), "through": "yes", "corner_radius": 1.0})
    res = build_design(clevis_intent, Design(features=[plate, through]))
    assert res.volume == pytest.approx(40 * 30 * 10 - (3 * 26 - 4 * (1 - math.pi / 4)) * 10, rel=1e-6)


def _box(w=40.0):
    return FI(id="F1", kind="profile_plate", params={"outline": [[0, 0], [w, 0], [w, 30], [0, 30]], "z_min": 0.0, "z_max": 10.0, "corner_radius": 0.0})


def _edge_at(shape, x, y, z):
    from build123d import Vector

    return min(range(len(shape.edges())), key=lambda i: (shape.edges()[i].position_at(0.5) - Vector(x, y, z)).length)


def test_fillet_on_a_picked_edge_follows_upstream_edits(clevis_intent):
    from cad_assist.features.design import describe_edge

    base = build_design(clevis_intent, Design(features=[_box()]))
    pick = describe_edge(base.local, base.face_names, _edge_at(base.local, 40, 30, 5))  # the vertical edge at (40, 30)
    assert pick.length == pytest.approx(10) and pick.faces == ["F1.sides", "F1.sides"]
    r = 4.0
    fillet = FI(id="F2", kind="fillet", params={"radius": r}, edges=[pick])
    res = build_design(clevis_intent, Design(features=[_box(), fillet]))
    assert res.features[1].error is None
    # One 10 mm vertical edge rounded: r²(1 − π/4) off the cross-section.
    assert res.volume == pytest.approx((40 * 30 - r * r * (1 - math.pi / 4)) * 10, rel=1e-6)
    # Widen the plate upstream: the picked edge moves 10 mm with it and is found again (same faces, clearly nearest).
    res = build_design(clevis_intent, Design(features=[_box(50.0), fillet]))
    assert res.features[1].error is None
    assert res.volume == pytest.approx((50 * 30 - r * r * (1 - math.pi / 4)) * 10, rel=1e-6)
    # A blend has nothing to pick on a tangent joint: the fillet's own boundary edges are refused.
    with pytest.raises(FeatureError, match="tangent"):
        describe_edge(res.local, res.face_names, _edge_at(res.local, 50 - r, 30, 5))


def test_picked_edge_blend_runs_on_along_tangent_edges(clevis_intent):
    from cad_assist.features.design import describe_edge

    plate = FI(id="F1", kind="profile_plate", params={"outline": L_SHAPE, "z_min": 0.0, "z_max": 5.0, "corner_radius": 3.0})
    base = build_design(clevis_intent, Design(features=[plate]))
    # The top rim along y = 10, the inside-corner arc and the rim along x = 12 are one tangent chain.
    chain = [describe_edge(base.local, base.face_names, _edge_at(base.local, *xyz)) for xyz in ((28, 10, 5), (12.88, 10.88, 5), (12, 22, 5))]
    one = build_design(clevis_intent, Design(features=[plate, FI(id="F2", kind="fillet", params={"radius": 1.0}, edges=chain[:1])]))
    three = build_design(clevis_intent, Design(features=[plate, FI(id="F2", kind="fillet", params={"radius": 1.0}, edges=chain)]))
    assert one.features[1].error is None and three.features[1].error is None
    assert one.volume == pytest.approx(three.volume, rel=1e-9) and one.volume < base.volume


def test_fillet_without_picks_or_faces_says_what_to_do(clevis_intent):
    res = build_design(clevis_intent, Design(features=[_box(), FI(id="F2", kind="fillet")]))
    assert "pick the edges" in res.features[1].error
