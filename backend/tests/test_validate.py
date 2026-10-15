import pytest
from build123d import IN, Axis, Box, Cylinder, Pos, Rot

from print_assist.features.design import build_design
from print_assist.features.model import FeatureInstance as FI
from print_assist.features.templates import TEMPLATES
from print_assist.geometry.samples import actuator_clevis_end, frame_rib, hyd_line_envelope, rod_end, side_panel
from print_assist.intent import builders
from print_assist.intent.models import Intent, KeepOut
from print_assist.validate import _check_bore, check_part, preflight

SHAPES = {
    "rib": frame_rib(),
    "rod": rod_end(),
    "fork": actuator_clevis_end(),
    "panel": side_panel(),
    "hyd": hyd_line_envelope(),
}


def shape_of(name):
    return SHAPES[name]


def _cyl(shape, r_in):
    return next(i for i, f in enumerate(shape.faces()) if f.geom_type.name == "CYLINDER" and abs(f.radius - r_in * IN) < 1e-3)


def _intent(mate: str | None = "rod", keepout: bool = True) -> Intent:
    rib = SHAPES["rib"]
    top = next(i for i, f in enumerate(rib.faces()) if f.is_same(rib.faces().sort_by(Axis.Z)[-1]))
    it = Intent()
    it.objects.append(builders.planar_mount(it, rib, "rib", top))
    it.objects.append(builders.bolt_pattern(it, rib, "rib", top))
    it.objects[-1].fastener = "NAS1352-3"
    if mate in ("rod", "fork"):
        it.objects.append(builders.bore(it, SHAPES[mate], mate, _cyl(SHAPES[mate], 0.1885)))
    elif mate == "panel":
        pn = SHAPES["panel"]
        face = next(i for i, x in enumerate(pn.faces()) if x.geom_type.name == "PLANE" and x.normal_at().X < -0.99)
        it.objects.append(builders.bolt_pattern(it, pn, "panel", face))
        it.objects[-1].fastener = "NAS1352-3"
    if keepout:
        it.objects.append(KeepOut(id="KO-01", part="hyd"))
    it.objects.append(builders.default_design_space(it))
    it.objects.append(builders.new_load_case(it, it.objects[2].id))
    it.objects[-1].force = (0.0, -2800.0, -7740.0)
    it.rules.mass_target = 0.36 * 0.45359237
    return it


def _check(it, design):
    # Only the parts this intent uses are in the project.
    used = {"rib", "hyd"} | {getattr(o, "face").part for o in it.objects if getattr(o, "face", None)}
    roles = {n: ("keepout" if n == "hyd" else "mating") for n in used}
    return {f.title: f for f in check_part(it, build_design(it, design), shape_of, roles)}


def test_preflight_complete_intent_is_ready():
    findings = preflight(_intent(), shape_of)
    assert [f.title for f in findings if f.status != "ok"] == []
    ko = next(f for f in findings if f.title == "KO-01 vs interfaces")
    # Nearest connection is a bolt head: hole at (0.95, 0.95) with a Ø0.438 washer, tube surface near x = 1.0
    assert 0.5 * IN < ko.value < 1.0 * IN
    assert len(ko.points) == 2


def test_preflight_flags_unbuildable_intent():
    it = _intent()
    it.objects[1].fastener = "NAS1352-4"  # 0.250 shank in Ø0.201 holes
    it.objects[3].clearance = 1.0 * IN
    it.objects[-1].target = None
    status = {f.title: f.status for f in preflight(it, shape_of)}
    assert status["IF-01 fastener"] == "fail"
    assert status["KO-01 vs interfaces"] == "fail"
    assert status["LC-01 load"] == "fail"
    assert status["IF-02 pin position"] == "ok"


def test_preflight_without_mount_or_space():
    status = {f.title: f.status for f in preflight(Intent(), shape_of)}
    assert status["Primary mount"] == "fail"
    assert status["Design space"] == "warn"
    assert status["Load cases"] == "warn"


@pytest.mark.parametrize("template,mate", [
    ("clevis_bracket", "rod"),
    ("single_lug_fitting", "fork"),
    ("angle_bracket", "panel"),
])
def test_reference_brackets_pass_part_checks(template, mate):
    # The hydraulic line runs through where the angle bracket's wall goes (see the next test).
    it = _intent(mate, keepout=mate != "panel")
    res = _check(it, TEMPLATES[template][2](it))
    bad = {k: (f.status, f.detail) for k, f in res.items() if f.status != "ok" and f.code != "mass"}
    assert bad == {}
    assert {"Clear of rib", f"Clear of {mate}"} <= set(res)


def test_angle_bracket_wall_enters_hydraulic_keepout():
    it = _intent("panel")
    f = _check(it, TEMPLATES["angle_bracket"][2](it))["Clearance to KO-01"]
    assert f.status == "fail" and "enters" in f.detail


def test_clevis_part_values():
    it = _intent()
    res = _check(it, TEMPLATES["clevis_bracket"][2](it))
    assert res["Minimum wall"].value / IN == pytest.approx(0.200, abs=1e-3)  # gusset thickness
    assert res["IF-02 pin bore"].value / IN == pytest.approx(0.500, abs=1e-3)  # two 0.25 ears
    assert res["IF-00 seated"].value / IN**2 == pytest.approx(7.0858, rel=1e-3)  # plate base minus 4 holes, minus chamfer
    assert 0.12 * IN < res["Clearance to KO-01"].value < 0.4 * IN
    assert res["Mass"].status == "warn"  # 0.393 lb against 0.36


def test_keepout_clearance_fails_when_margin_grows():
    it = _intent()
    it.objects[3].clearance = 0.3 * IN
    f = _check(it, TEMPLATES["clevis_bracket"][2](it))["Clearance to KO-01"]
    assert f.status == "fail" and f.value / IN == pytest.approx(0.241, abs=2e-3)
    it.objects[3].strength = "soft"
    assert _check(it, TEMPLATES["clevis_bracket"][2](it))["Clearance to KO-01"].status == "warn"


def test_missing_and_blocked_bolt_holes():
    it = _intent()
    d = TEMPLATES["clevis_bracket"][2](it)
    d.features[1].suppressed = True
    assert "4 of 4 holes have no" in _check(it, d)["IF-01 bolt holes"].detail
    d = TEMPLATES["clevis_bracket"][2](it)
    # A wall on the +X edge (inner face at x = 1.15) crowds the washers of the x = 0.95 holes (radius 0.219).
    d.features.insert(2, FI(id="F9", kind="wall", refs={"base": "F1"}, params={"side": "+x", "height": 1.0 * IN, "thickness": 0.25 * IN}))
    f = _check(it, d)["IF-01 bolt holes"]
    assert f.status == "fail" and "2 of 4" in f.detail


def test_clevis_on_wrong_mating_end_interferes():
    it = _intent("fork")
    it.objects[2].gap = None  # pretend the fork is a male end: the clevis ears land on its tines
    res = _check(it, TEMPLATES["clevis_bracket"][2](it))
    assert res["Clear of fork"].status == "fail"
    assert res["Clear of rib"].status == "ok"


def test_blocked_pin_path():
    # A bar along the pin axis with blind bores from each end: both bores exist, but solid metal sits between them.
    it = _intent()
    bore = it.objects[2]
    bar = Pos(0, 0, 3 * IN) * Box(2 * IN, 1 * IN, 1 * IN)
    for s in (1, -1):
        bar -= Pos(s * 0.85 * IN, 0, 3 * IN) * Rot(0, 90, 0) * Cylinder(bore.diameter / 2, 0.3 * IN)
    f = _check_bore(bore, bar)[0]
    assert f.status == "fail" and "blocks the pin path" in f.detail


def test_design_space_and_min_wall():
    it = _intent()
    ds = next(o for o in it.objects if o.kind == "design_space")
    ds.max = (ds.max[0], 1.0 * IN, ds.max[2])
    d = TEMPLATES["clevis_bracket"][2](it)
    d.features[3].params["thickness"] = 0.04 * IN  # gussets below the 0.045 in min wall
    res = _check(it, d)
    assert res["Inside DS-01"].status == "fail" and "+Y" in res["Inside DS-01"].detail
    wall = res["Minimum wall"]
    assert wall.status == "fail" and wall.value / IN == pytest.approx(0.04, abs=1e-3)


def test_wall_check_ignores_acute_corners_but_finds_thin_walls():
    from build123d import Box, BuildPart, BuildSketch, Polygon, extrude

    from print_assist.intent.models import Intent
    from print_assist.validate import _check_wall

    it = Intent()
    with BuildPart() as wedge:  # a 30° wedge, 20 mm long, 10 mm tall: thick everywhere but its tip
        with BuildSketch():
            Polygon((0, 0), (20, 0), (20, 20 * 0.57735), align=None)
        extrude(amount=10)
    (f,) = _check_wall(it, wedge.part)
    assert f.status == "ok", f.detail
    thin = Box(20, 20, 0.5)  # 0.020 in: below the 0.045 in minimum everywhere
    (g,) = _check_wall(it, thin)
    assert g.status == "fail" and abs(g.value - 0.5) < 1e-3
