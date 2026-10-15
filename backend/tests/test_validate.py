import math

import pytest
from build123d import Box, BuildPart, BuildSketch, Polygon, extrude

from print_assist.features.design import build_design
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.printers import profiles
from print_assist.rules import PrintRules
from print_assist.validate import check_wall, validate

A1 = profiles()["bambu_a1_ams_lite"]


def _findings(design, printer=A1):
    rules = PrintRules()
    rep = validate(build_design(design, rules), rules, printer)
    return {f.title: f for f in rep.part}


def test_plate_passes_and_reports_mass():
    res = _findings(Design(features=[FI(id="F1", kind="plate")]))
    assert res["Build"].status == "ok"
    assert res["Fits the bed"].status == "ok"
    assert res["Minimum wall"].status == "ok"
    vol = (60 * 40 - (4 - math.pi) * 9) * 5
    assert res["Mass"].value == pytest.approx(vol * 1e-9 * 1240, rel=1e-6)


def test_plate_too_big_for_the_bed():
    res = _findings(Design(features=[FI(id="F1", kind="plate", params={"length": 300.0})]))
    bed = res["Fits the bed"]
    assert bed.status == "fail" and "X 300.0 > 256" in bed.detail


def test_thin_plate_fails_rules_and_wall():
    res = _findings(Design(features=[FI(id="F1", kind="plate", params={"thickness": 0.5})]))
    assert res["F1 Thickness"].status == "fail"
    assert res["Minimum wall"].status == "fail" and res["Minimum wall"].value == pytest.approx(0.5, abs=1e-3)


def test_wall_check_ignores_acute_corners_but_finds_thin_walls():
    with BuildPart() as wedge:  # a 30° wedge, 20 mm long, 10 mm tall: thick everywhere but its tip
        with BuildSketch():
            Polygon((0, 0), (20, 0), (20, 20 * 0.57735), align=None)
        extrude(amount=10)
    (f,) = check_wall(PrintRules(), wedge.part)
    assert f.status == "ok", f.detail
    (g,) = check_wall(PrintRules(), Box(20, 20, 0.5))  # below the 0.8 mm minimum everywhere
    assert g.status == "fail" and abs(g.value - 0.5) < 1e-3


def test_inlays_are_checked_as_printed_not_as_pockets():
    from print_assist.build.ops import apply

    d = Design()
    apply(d, {"op": "add_feature", "kind": "plate", "params": {"length": 60, "width": 30, "thickness": 1.6, "corner_radius": 0}})
    top = {"name": "F1.top", "at": [0, 0, 1.6], "normal": [0, 0, 1], "area": 1800}
    apply(d, {"op": "add_feature", "kind": "text", "faces": [top],
              "params": {"content": "HI", "size": 14, "mode": "inlay", "depth": 0.8, "into": "new"}})
    res = _findings(d)
    # The plate alone is 0.8 mm under its letter pockets, but the letters fill them: printed, it is 1.6 mm throughout.
    assert res["Minimum wall"].status == "ok", res["Minimum wall"].detail
