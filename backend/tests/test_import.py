"""Import (STEP files as exact solids) and references (STEP, STL, 3MF beside the design, to look at, measure and slice).
Sizes and volumes are hand-checked."""

import json
import math
import zipfile

import pytest
from build123d import Box, Cylinder, Location, export_step, export_stl
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.export.bambu import Part, load_template, palette, write_project
from print_assist.features.model import Filament
from print_assist.geometry.meshio import ImportError_, read_mesh, read_stl, section
from print_assist.geometry.render import png_size
from print_assist.printers import profiles

A1 = profiles()["bambu_a1_ams_lite"]
CUBE_V = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)]
CUBE_T = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
          (3, 0, 4), (3, 4, 7)]  # a 10 mm cube, triangles facing outward


def plain_3mf(path, transform: str, unit: str = "millimeter"):
    vs = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in CUBE_V)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in CUBE_T)
    model = (f'<?xml version="1.0"?><model unit="{unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources><object id="1" type="model" name="Cube"><mesh><vertices>{vs}</vertices><triangles>{ts}'
             f'</triangles></mesh></object></resources><build><item objectid="1" transform="{transform}"/></build></model>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", model)
    return path


def test_3mf_placement_units_and_names(tmp_path):
    (cube,) = read_mesh(plain_3mf(tmp_path / "a.3mf", "2 0 0 0 1 0 0 0 1 5 6 7")).parts  # stretched x2 in X, moved
    assert cube.name == "Cube" and len(cube.tris) == 12 and cube.volume() == pytest.approx(2000)
    lo, hi = cube.bounds()
    assert tuple(lo) == pytest.approx((5, 6, 7)) and hi[0] == pytest.approx(25)
    (cube,) = read_mesh(plain_3mf(tmp_path / "b.3mf", "1 0 0 0 1 0 0 0 1 0 0 0", unit="centimeter")).parts
    assert cube.volume() == pytest.approx(1e6)  # a 100 mm cube


def test_bambu_project_parts_plates_and_names(tmp_path):
    base = Location((128, 128, 5)) * Box(40, 40, 10)
    peg = Location((128, 128, 15)) * Cylinder(5, 10)
    path = write_project([Part("Base", base, 1), Part("Peg", peg, 2)], palette([Filament(slot=1), Filament(slot=2)]),
                         A1, load_template(A1), tmp_path / "p.3mf", "t")
    m = read_mesh(path)
    assert m.plates == ["t"] and m.plate == 1 and [p.name for p in m.parts] == ["Base", "Peg"]
    assert m.parts[0].volume() == pytest.approx(16000, rel=1e-9)
    assert m.parts[1].volume() == pytest.approx(math.pi * 25 * 10, rel=0.002)
    lo, hi = m.bounds()
    assert (lo[2], hi[2], lo[0]) == pytest.approx((0, 20, 108), abs=1e-6)  # where they were exported
    with pytest.raises(ImportError_, match="1 plates"):
        read_mesh(path, plate=2)


def test_stl_reading_and_sections(tmp_path):
    export_stl(Box(20, 10, 4) - Cylinder(3, 10), str(tmp_path / "b.stl"))  # a 20 x 10 x 4 slab with a 6 mm hole
    part = read_stl(tmp_path / "b.stl")
    assert part.volume() == pytest.approx(800 - math.pi * 9 * 4, rel=0.01)
    outer, hole = section([part], "Z", 0.5)
    assert outer["area"] == pytest.approx(200, rel=1e-6) and "circle" not in outer
    assert hole["circle"]["radius"] == pytest.approx(3, rel=0.01) and hole["circle"]["center"] == pytest.approx([0, 0], abs=0.01)
    halves = section([part], "X", 0)  # through the hole: the slab's two sides, y 3..5 and -5..-3, z -2..2
    assert [h["area"] for h in halves] == pytest.approx([8, 8], rel=0.01)  # the STL's hole is faceted


def test_step_import_feature_placement(tmp_path):
    from print_assist.build.ops import apply
    from print_assist.features.design import build_design
    from print_assist.features.model import Design

    folder = tmp_path / "references"
    folder.mkdir()
    export_step(Box(20, 10, 4), str(folder / "slab.step"))
    d = Design()
    apply(d, {"op": "add_feature", "kind": "import",
              "params": {"file": "slab.step", "rotate_z": 90, "scale": 2, "x": 100, "z": 3}})
    r = build_design(d, references_dir=folder)
    assert r.complete, [f.error for f in r.features]
    bb = r.parts["B1"].bounding_box()
    assert (bb.size.X, bb.size.Y, bb.size.Z) == pytest.approx((20, 40, 8))  # doubled, turned a quarter
    assert (bb.min.Z, (bb.min.X + bb.max.X) / 2) == pytest.approx((3, 100))
    assert r.parts["B1"].volume == pytest.approx(800 * 8)  # exact, not faceted
    apply(d, {"op": "edit_feature", "id": "F1", "params": {"file": "../secrets.step"}})
    assert "references folder" in build_design(d, references_dir=folder).features[0].error
    export_stl(Box(1, 1, 1), str(folder / "m.stl"))
    apply(d, {"op": "edit_feature", "id": "F1", "params": {"file": "m.stl"}})
    assert "references" in build_design(d, references_dir=folder).features[0].error


def test_import_and_references_through_the_api(tmp_path):
    c = TestClient(app_module.app)
    step = tmp_path / "Peg.step"
    export_step(Location((50, 50, 20)) * Box(20, 10, 4), str(step))
    r = c.post("/api/import", params={"new_design": True}, files={"file": ("Peg.step", step.read_bytes())})
    assert r.status_code == 200, r.text
    assert [b["name"] for b in r.json()["tree"]["bodies"]] == ["Peg"]  # the empty design's body takes the file's name
    tree = c.post("/api/update").json()
    assert tree["exportable"] and c.get("/api/part").json()["volume"] == pytest.approx(800)
    r = c.post("/api/import", files={"file": ("lid.stl", b"solid x\nendsolid x\n")})
    assert r.status_code == 422 and "reference" in r.text  # meshes are references

    stl = tmp_path / "Lid.stl"
    export_stl(Box(30, 20, 2), str(stl))
    r = c.post("/api/references", files={"file": ("Lid.stl", stl.read_bytes())})
    assert r.status_code == 200, r.text
    assert [(x["id"], x["name"], x["file"]) for x in r.json()["tree"]["references"]] == [("R1", "Lid", "Lid.stl")]
    assert c.post("/api/update").json()["pending_count"] == 0  # a reference is not part of the build
    (ref,) = c.get("/api/references").json()
    assert ref["bounds"]["size"] == pytest.approx([30, 20, 2]) and ref["bounds"]["min"][2] == pytest.approx(0)
    assert ref["parts"][0]["volume"] == pytest.approx(1200, rel=1e-6)

    c.post("/api/ops", json={"ops": [{"op": "set_reference", "id": "R1", "x": 100, "rotate_z": 90}]})
    (ref,) = c.get("/api/references", params={"parts": False}).json()
    assert ref["bounds"]["size"] == pytest.approx([20, 30, 2]) and ref["bounds"]["min"][0] == pytest.approx(90)
    s = c.get("/api/references/R1/section", params={"axis": "Z", "offset": 1}).json()
    assert s["coords"] == ["x", "y"] and s["outlines"][0]["area"] == pytest.approx(600, rel=1e-6)
    png = c.get("/api/references/R1/render", params={"view": "top", "size": 200, "with_design": True})
    assert png.status_code == 200 and png_size(png.content) == (200, 200)
    assert c.get("/api/references/R1/render", params={"part": "nope"}).status_code == 422
    mesh = c.get("/api/references/R1/mesh").json()
    assert len(mesh["parts"]) == 1 and mesh["parts"][0]["positions"]

    assert c.post("/api/references", files={"file": ("x.obj", b"o")}).status_code == 422
    assert c.post("/api/references", files={"file": ("Broken.3mf", b"not a zip")}).status_code == 422
    refs = sorted(p.name for p in (tmp_path / "designs" / "peg" / "references").iterdir())
    assert refs == ["Lid.stl", "Peg.step"]  # the broken file is not kept
    c.post("/api/ops", json={"ops": [{"op": "delete_reference", "id": "R1"}]})
    assert c.get("/api/references").json() == []


def test_mcp_reference_tools(tmp_path):
    from print_assist_mcp import server

    c = TestClient(app_module.app)
    server.set_client(c)
    try:
        _mcp_reference_tools(c, server, tmp_path)
    finally:
        server.set_client(None)


def _mcp_reference_tools(c, server, tmp_path):
    c.post("/api/projects", json={"name": "Ref tools"})
    stl = tmp_path / "Tray.stl"
    export_stl(Box(40, 30, 6) - Location((0, 0, 3)) * Cylinder(5, 10), str(stl))
    added = json.loads(server.add_reference(str(stl)))
    assert added["made"] == ["R1"]
    listed = json.loads(server.list_references())
    assert listed[0]["parts"][0]["size"] == pytest.approx([40, 30, 6])
    cut = json.loads(server.section_reference("R1", "Z", 4.5))
    assert any(o.get("circle", {}).get("radius") == pytest.approx(5, rel=0.01) for o in cut["outlines"])
    img = server.look_at_reference("R1", "iso", with_design=True, region=[-25, -20, 0, 25, 20, 6], size=160)
    assert not isinstance(img, str)
    assert "error" in json.loads(server.add_reference(str(tmp_path / "missing.stl")))
