"""Import: STEP, STL and 3MF files become solids in a body of their own, placed as asked. Volumes are hand-checked."""

import io
import math
import zipfile

import pytest
from build123d import Box, Cylinder, Location, export_step, export_stl
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.export.bambu import Part, load_template, palette, write_project
from print_assist.features.model import Filament
from print_assist.geometry.meshio import ImportError_, load, read_3mf
from print_assist.printers import profiles

A1 = profiles()["bambu_a1_ams_lite"]
CUBE_V = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)]
CUBE_T = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
          (3, 0, 4), (3, 4, 7)]  # a 10 mm cube, triangles facing outward


def plain_3mf(path, transform: str, unit: str = "millimeter", drop: int = 0):
    vs = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in CUBE_V)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in CUBE_T[drop:])
    model = (f'<?xml version="1.0"?><model unit="{unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources><object id="1" type="model"><mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh>'
             f'</object></resources><build><item objectid="1" transform="{transform}"/></build></model>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", model)
    return path


def test_3mf_item_transforms_units_and_mirroring(tmp_path):
    shape, tris = load(plain_3mf(tmp_path / "a.3mf", "2 0 0 0 1 0 0 0 1 5 6 7"))  # stretched x2 in X, moved
    assert tris == 12 and shape.volume == pytest.approx(2000)
    bb = shape.bounding_box()
    assert (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X) == pytest.approx((5, 6, 7, 25))
    shape, _ = load(plain_3mf(tmp_path / "b.3mf", "-1 0 0 0 1 0 0 0 1 0 0 0"))  # mirrored: triangles turn inward
    assert shape.volume == pytest.approx(1000) and shape.bounding_box().min.X == pytest.approx(-10)
    shape, _ = load(plain_3mf(tmp_path / "c.3mf", "1 0 0 0 1 0 0 0 1 0 0 0", unit="centimeter"))
    assert shape.volume == pytest.approx(1e6)  # 100 mm cube
    with pytest.raises(ImportError_, match="not closed"):
        load(plain_3mf(tmp_path / "d.3mf", "1 0 0 0 1 0 0 0 1 0 0 0", drop=2))


def test_bambu_project_parts_come_back_where_they_were(tmp_path):
    base = Location((128, 128, 5)) * Box(40, 40, 10)
    peg = Location((128, 128, 15)) * Cylinder(5, 10)
    path = write_project([Part("Base", base, 1), Part("Peg", peg, 2)], palette([Filament(slot=1), Filament(slot=2)]),
                         A1, load_template(A1), tmp_path / "p.3mf", "t")
    assert len(read_3mf(path)) == 2  # one mesh per part, not sewn together
    shape, _ = load(path)
    assert len(shape.solids()) == 2
    assert shape.volume == pytest.approx(16000 + math.pi * 25 * 10, rel=0.001)
    bb = shape.bounding_box()
    assert (bb.min.Z, bb.max.Z, bb.min.X) == pytest.approx((0, 20, 108), abs=1e-6)


def test_stl_and_step(tmp_path):
    export_stl(Box(20, 10, 5), str(tmp_path / "b.stl"))
    assert load(tmp_path / "b.stl")[0].volume == pytest.approx(1000)
    export_step(Cylinder(10, 4), str(tmp_path / "c.step"))
    shape, tris = load(tmp_path / "c.step")
    assert tris == 0 and shape.volume == pytest.approx(math.pi * 100 * 4)  # exact, not faceted
    with pytest.raises(ImportError_, match="formats"):
        load(tmp_path / "x.obj")


def test_import_through_the_api(tmp_path):
    c = TestClient(app_module.app)
    buf = tmp_path / "Peg.stl"
    export_stl(Location((50, 50, 20)) * Box(20, 10, 4), str(buf))
    r = c.post("/api/import", params={"new_design": True}, files={"file": ("Peg.stl", buf.read_bytes())})
    assert r.status_code == 200, r.text
    tree = r.json()["tree"]
    assert [b["name"] for b in tree["bodies"]] == ["Peg"]  # the empty design's body takes the file's name
    tree = c.post("/api/update").json()
    assert tree["exportable"] and c.get("/api/part").json()["volume"] == pytest.approx(800)
    bb = app_module.ws().last_good.parts["B1"].bounding_box()
    assert bb.min.Z == pytest.approx(0) and (bb.min.X + bb.max.X) / 2 == pytest.approx(0)  # on the bed, centred

    # The same file again goes into a body of its own, reusing the stored copy.
    r = c.post("/api/import", files={"file": ("Peg.stl", buf.read_bytes())})
    assert r.status_code == 200 and [b["id"] for b in r.json()["tree"]["bodies"]] == ["B1", "B2"]
    feats = [f for f in r.json()["tree"]["features"] if f["kind"] == "import"]
    assert [f["params"]["file"] for f in feats] == ["Peg.stl", "Peg.stl"]

    assert c.post("/api/import", files={"file": ("x.obj", b"o")}).status_code == 422
    r = c.post("/api/import", files={"file": ("Broken.3mf", b"not a zip")})
    assert r.status_code == 422 and "Broken" in r.text
    refs = sorted(p.name for p in (tmp_path / "designs" / "peg" / "references").iterdir())
    assert refs == ["Peg.stl"]  # the broken file is not kept


def test_import_feature_placement(tmp_path):
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
    apply(d, {"op": "edit_feature", "id": "F1", "params": {"file": "../secrets.step"}})
    assert "references folder" in build_design(d, references_dir=folder).features[0].error
