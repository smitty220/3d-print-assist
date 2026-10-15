"""Colours and variants (M2): Split and Combine between bodies, inlaid and separate-body text, the filament palette and
its printer checks, variants and colorways, and coloured 3MF export. Volumes are hand-checked."""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.build.ops import OpError, apply
from print_assist.export.bambu import read_project
from print_assist.features.design import build_design
from print_assist.features.model import Design
from print_assist.printers import profiles
from print_assist.variants import check_filaments, with_colorway, with_variant


def block(d: Design, h=40.0, w=20.0) -> None:
    """A w x w x h block standing on the bed (F1, body B1)."""
    apply(d, {"op": "add_feature", "kind": "plate", "params": {"length": w, "width": w, "thickness": h, "corner_radius": 0}})


def vols(result) -> dict[str, float]:
    return {b: round(s.volume, 3) for b, s in result.parts.items()}


def test_split_moves_the_side_above_the_plane_into_a_new_body():
    d = Design()
    block(d)
    made = apply(d, {"op": "add_feature", "kind": "split", "params": {"offset": 30}})
    assert made == "F2" and [b.id for b in d.bodies] == ["B1", "B2"] and d.get("F2").params["into"] == "B2"
    r = build_design(d)
    assert r.complete, [f.error for f in r.features]
    assert vols(r) == {"B1": 12000.0, "B2": 4000.0}  # 20 x 20 x 30 and 20 x 20 x 10
    assert "F2.cut" in r.body_names["B1"] and "F2.cut" in r.body_names["B2"]


def test_two_splits_make_a_band_and_combine_joins_it_back():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "split", "params": {"offset": 30}})  # B2 = z 30..40
    apply(d, {"op": "add_feature", "kind": "split", "body": "B2", "params": {"offset": 35}})  # B3 = z 35..40
    r = build_design(d)
    assert vols(r) == {"B1": 12000.0, "B2": 2000.0, "B3": 2000.0}
    apply(d, {"op": "add_feature", "kind": "combine", "params": {"tool": "B3"}})
    r = build_design(d)
    assert vols(r) == {"B1": 14000.0, "B2": 2000.0}  # B3 used up
    apply(d, {"op": "edit_feature", "id": "F4", "params": {"keep_tool": True, "op": "cut"}})
    assert vols(build_design(d)) == {"B1": 12000.0, "B2": 2000.0, "B3": 2000.0}  # nothing to cut: B3 sits above B1


def test_split_keeps_the_side_asked_for_and_refuses_a_plane_that_misses():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "split", "params": {"offset": 10, "keep": "above"}})
    assert vols(build_design(d)) == {"B1": 12000.0, "B2": 4000.0}
    apply(d, {"op": "edit_feature", "id": "F2", "params": {"offset": 50}})
    r = build_design(d)
    assert r.features[1].status == "fail" and "misses" in r.features[1].error


def test_inlaid_text_fills_its_pocket_flush_in_its_own_body():
    d = Design()
    apply(d, {"op": "add_feature", "kind": "plate",
              "params": {"length": 80, "width": 30, "thickness": 5, "corner_radius": 0}})
    top = {"name": "F1.top", "at": [0, 0, 5], "normal": [0, 0, 1], "area": 2400}
    apply(d, {"op": "add_feature", "kind": "text", "faces": [top],
              "params": {"content": "MTG", "size": 14, "mode": "inlay", "depth": 1, "into": "new"}})
    r = build_design(d)
    assert r.complete, [f.error for f in r.features]
    v = vols(r)
    assert v["B1"] + v["B2"] == pytest.approx(80 * 30 * 5, rel=1e-6)  # the letters fill the pocket exactly
    assert 50 < v["B2"] < 400
    bb = r.parts["B2"].bounding_box()
    assert bb.max.Z == pytest.approx(5, abs=1e-6) and bb.min.Z == pytest.approx(4, abs=1e-6)  # flush, 1 deep


def test_raised_text_in_its_own_body_leaves_the_face_body_alone():
    d = Design()
    apply(d, {"op": "add_feature", "kind": "plate", "params": {"length": 80, "width": 30, "thickness": 5, "corner_radius": 0}})
    apply(d, {"op": "add_body", "name": "Letters"})
    top = {"name": "F1.top", "at": [0, 0, 5], "normal": [0, 0, 1], "area": 2400}
    apply(d, {"op": "add_feature", "kind": "text", "faces": [top],
              "params": {"content": "MTG", "size": 14, "mode": "emboss", "depth": 0.8, "into": "B2"}})
    r = build_design(d)
    assert vols(r)["B1"] == 12000.0 and r.parts["B2"].bounding_box().min.Z == pytest.approx(5, abs=1e-6)


def test_text_body_rules():
    d = Design()
    block(d)
    with pytest.raises(OpError, match="no body"):
        apply(d, {"op": "add_feature", "kind": "text", "params": {"into": "B9"}})
    apply(d, {"op": "add_feature", "kind": "text", "params": {"mode": "inlay"}})
    r = build_design(d)
    assert "own body" in r.features[1].error
    apply(d, {"op": "edit_feature", "id": "F2", "params": {"into": "B1"}})
    assert "different body" in build_design(d).features[1].error


def test_filament_palette_and_body_assignment():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "split", "params": {"offset": 30, "into": "new"}, "new_body_name": "Rim"})
    assert d.body("B2").name == "Rim"
    with pytest.raises(OpError, match="no filament in slot 1"):
        apply(d, {"op": "edit_body", "id": "B1", "filament": 1})
    with pytest.raises(OpError, match="#rrggbb"):
        apply(d, {"op": "set_filament", "slot": 1, "color": "gold"})
    apply(d, {"op": "set_filament", "slot": 2, "name": "Gold", "color": "#C9A227"})
    apply(d, {"op": "set_filament", "slot": 1, "name": "Black", "color": "#111111"})
    assert [f.slot for f in d.filaments] == [1, 2] and d.filament(2).color == "#c9a227"
    apply(d, {"op": "edit_body", "id": "B1", "filament": 1})
    apply(d, {"op": "edit_body", "id": "B2", "filament": 2, "color": "#ff0000"})
    assert d.body_color(d.body("B2")) == "#c9a227"  # the filament wins over the body's own colour
    apply(d, {"op": "delete_filament", "slot": 2})
    assert d.body("B2").filament is None and d.body_color(d.body("B2")) == "#ff0000"
    with pytest.raises(OpError, match="used by F2"):
        apply(d, {"op": "delete_body", "id": "B2"})


def test_filament_checks_against_the_printer():
    a1 = profiles()["bambu_a1_ams_lite"]
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "split", "params": {"offset": 30}})
    built = ["B1", "B2"]
    assert check_filaments(d, built, a1)[0].status == "warn"  # two bodies, no palette
    apply(d, {"op": "set_filament", "slot": 1, "color": "#111111"})
    apply(d, {"op": "set_filament", "slot": 5, "color": "#c9a227"})
    apply(d, {"op": "edit_body", "id": "B1", "filament": 1})
    f = {x.title: x for x in check_filaments(d, built, a1)}
    assert f["Filament for every body"].status == "warn" and "B2" in f["Filament for every body"].refs
    apply(d, {"op": "edit_body", "id": "B2", "filament": 5})
    f = {x.title: x for x in check_filaments(d, built, a1)}
    assert f["Filaments fit the printer"].status == "fail" and "slot 5" in f["Filaments fit the printer"].detail
    apply(d, {"op": "set_filament", "slot": 2, "color": "#c9a227", "nozzle": 1})
    apply(d, {"op": "edit_body", "id": "B2", "filament": 2})
    assert "nozzle 2" in check_filaments(d, built, a1)[0].detail
    apply(d, {"op": "set_filament", "slot": 2, "nozzle": 0})
    assert check_filaments(d, built, a1)[0].status == "ok"


def test_variants_and_colorways():
    d = Design()
    apply(d, {"op": "set_parameter", "name": "h", "expr": "40"})
    apply(d, {"op": "add_feature", "kind": "plate", "params": {"length": 20, "width": 20, "corner_radius": 0},
              "exprs": {"thickness": "h"}})
    apply(d, {"op": "set_filament", "slot": 1, "name": "Black", "color": "#111111"})
    with pytest.raises(OpError, match="no parameter 'w'"):
        apply(d, {"op": "set_variant", "name": "Wide", "overrides": {"w": "3"}})
    vid = apply(d, {"op": "set_variant", "name": "Short", "overrides": {"h": "20"}})
    cid = apply(d, {"op": "set_colorway", "name": "Gold", "colors": {"1": "#C9A227"}, "names": {"1": "Silk Gold"}})
    assert (vid, cid) == ("V1", "C1")
    vd = with_variant(d, d.variant(vid))
    assert build_design(vd).volume == pytest.approx(8000) and d.parameter("h").expr == "40"
    cd = with_colorway(d, d.colorway(cid))
    assert (cd.filaments[0].color, cd.filaments[0].name) == ("#c9a227", "Silk Gold") and d.filaments[0].color == "#111111"


def test_coloured_3mf_and_batch_export_through_the_api():
    c = TestClient(app_module.app)
    assert c.post("/api/projects", json={"name": "Band Block"}).status_code == 200
    c.post("/api/parameters", json={"name": "h", "expr": "40"})
    ops = [
        {"op": "add_feature", "kind": "plate", "params": {"length": 20, "width": 20, "corner_radius": 0},
         "exprs": {"thickness": "h"}},
        {"op": "add_feature", "kind": "split", "params": {"offset": 30}, "new_body_name": "Band"},
        {"op": "set_filament", "slot": 1, "name": "Black", "color": "#111111"},
        {"op": "set_filament", "slot": 2, "name": "Gold", "color": "#c9a227"},
        {"op": "edit_body", "id": "B1", "filament": 1},
        {"op": "edit_body", "id": "B2", "filament": 2},
        {"op": "set_variant", "name": "Tall", "overrides": {"h": "60"}},
        {"op": "set_colorway", "name": "Silver", "colors": {"2": "#c0c0c0"}},
    ]
    assert c.post("/api/ops", json={"ops": ops}).status_code == 200
    tree = c.post("/api/update").json()
    assert [(b["id"], b["color"], b["filament"]) for b in tree["bodies"]] == [("B1", "#111111", 1), ("B2", "#c9a227", 2)]
    assert tree["bodies"][1]["fed_by"] == ["F2"] and len(tree["variants"]) == 1 and len(tree["colorways"]) == 1
    assert [b["color"] for b in c.get("/api/part").json()["bodies"]] == ["#111111", "#c9a227"]
    checks = {f["title"]: f for f in c.get("/api/checks").json()["part"]}
    assert checks["Filaments fit the printer"]["status"] == "ok"

    r = c.get("/api/export/3mf")  # a Bambu Studio project: one object, a part per body in its filament
    assert r.status_code == 200
    parts, settings, meta = read_project(r.headers["X-Saved-Path"])
    assert sorted((p.name, p.filament, round(p.volume)) for p in parts) == [("Band", 2, 4000), ("Body", 1, 12000)]
    assert settings["filament_colour"] == ["#111111", "#C9A227"] and meta["Title"] == "Band Block"

    r = c.post("/api/export/batch", json={"fmt": "3mf"})  # every variant x every colorway
    assert r.status_code == 200, r.text
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == ["band-block-tall-silver.3mf"]
    r = c.post("/api/export/batch", json={"fmt": "stl", "rows": [{}, {"variant": "V1"}]})
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == ["band-block.stl", "band-block-tall.stl"]
    assert c.post("/api/export/batch", json={"rows": [{"variant": "V9"}]}).status_code == 422
