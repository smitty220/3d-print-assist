"""The Text feature: emboss, deboss and cut on flat faces, wrapping on curved ones, fonts and print checks."""

import math

import pytest
from build123d import Text, TextAlign

from print_assist.build.engine import Engine
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.fonts import families
from print_assist.params.model import Parameter
from print_assist.rules import PrintRules
from print_assist.sketch.model import FacePick, Placement, Sketch
from test_features_m1 import rect

TOP = FacePick(name="F1.end", at=[0, 0, 10], normal=[0, 0, 1])


def glyph_area(text, size=10.0, font="Bebas Neue", bold=False):
    f = families()[font].file(bold)
    return Text(text, font_size=size, font_path=str(f), text_align=(TextAlign.CENTER, TextAlign.CENTER)).area


def box_with(text_params, faces=(TOP,), params=None):
    feats = [FI(id="F1", kind="extrude", refs={"sketch": "S1"}, params={"distance": 10.0}),
             FI(id="F2", kind="text", params=text_params, faces=list(faces))]
    return Engine().build(Design(sketches=[rect("S1", -30, -15, 30, 15)], features=feats, parameters=params or []), PrintRules())


def test_emboss_and_deboss_add_and_remove_the_letters_area_times_depth():
    a = glyph_area("SAGE")
    base = 60 * 30 * 10
    up = box_with({"content": "SAGE", "depth": 1.0})
    assert up.features[1].error is None
    assert up.volume == pytest.approx(base + a * 1.0, rel=1e-4)  # box + letter area × depth
    down = box_with({"content": "SAGE", "depth": 1.0, "mode": "deboss"})
    assert down.volume == pytest.approx(base - a * 1.0, rel=1e-4)
    cut = box_with({"content": "SAGE", "mode": "cut_through"})
    assert cut.volume == pytest.approx(base - a * 10.0, rel=1e-4)  # through the whole 10 mm


def test_text_alone_stands_on_the_bed():
    r = Engine().build(Design(features=[FI(id="F1", kind="text", params={"content": "HI", "depth": 2.0})]), PrintRules())
    assert r.volume == pytest.approx(glyph_area("HI") * 2.0, rel=1e-6)
    assert r.world.bounding_box().min.Z == pytest.approx(0)


def test_bold_is_heavier_and_fonts_are_checked():
    reg = box_with({"content": "Abc", "font": "Lato", "depth": 1.0})
    bold = box_with({"content": "Abc", "font": "Lato", "bold": True, "depth": 1.0})
    assert bold.volume > reg.volume
    bad = box_with({"content": "Abc", "font": "Comic Sans"})
    assert "no font 'Comic Sans'" in bad.features[1].error and "Lato" in bad.features[1].error


def test_small_or_thin_letters_are_flagged():
    small = box_with({"content": "SAGE", "size": 3.0})
    assert small.features[1].status == "warn"
    assert next(c for c in small.checks if c.name == "Letter height").status != "ok"
    ok = box_with({"content": "SAGE", "size": 12.0})
    assert ok.features[1].status == "ok", ok.features[1].message


def test_content_from_a_text_parameter():
    r = box_with({"depth": 1.0}, params=[Parameter(name="label", quantity="text", expr="SAGE")])
    feats = [FI(id="F1", kind="extrude", refs={"sketch": "S1"}, params={"distance": 10.0}),
             FI(id="F2", kind="text", params={"depth": 1.0}, exprs={"content": "label"}, faces=[TOP])]
    r = Engine().build(Design(sketches=[rect("S1", -30, -15, 30, 15)], features=feats,
                              parameters=[Parameter(name="label", quantity="text", expr="SAGE")]), PrintRules())
    assert r.features[1].params["content"] == "SAGE"
    assert r.volume == pytest.approx(60 * 30 * 10 + glyph_area("SAGE"), rel=1e-4)


def test_text_wraps_onto_a_cylinder():
    circle = Sketch(id="S1", entities=[{"id": "c", "kind": "point", "x": 0, "y": 0},
                                       {"id": "k", "kind": "circle", "c": "c", "r": 30}], placement=Placement())
    side = FacePick(name="F1.side", at=[30, 0, 20], normal=[1, 0, 0])
    feats = [FI(id="F1", kind="extrude", refs={"sketch": "S1"}, params={"distance": 40.0}),
             FI(id="F2", kind="text", params={"content": "SAGE", "depth": 1.0}, faces=[side])]
    r = Engine().build(Design(sketches=[circle], features=feats), PrintRules())
    assert r.features[1].error is None, r.features[1].error
    cyl = math.pi * 30**2 * 40
    # Wrapped letters keep their area; thickened outward 1 mm they gain a little (radius 30 → 31).
    assert r.volume - cyl == pytest.approx(glyph_area("SAGE") * 1.0 * (1 + 0.5 / 30), rel=0.03)
    bb = r.world.bounding_box()
    assert bb.max.X == pytest.approx(31.0, abs=0.05)  # standing 1 mm proud of the surface at the pick
