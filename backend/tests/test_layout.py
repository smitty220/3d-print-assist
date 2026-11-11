"""Print layout: items take solids by box, turn and sit on their plates; problems for overlaps, off-plate items and
solids left over; the Bambu project puts each plate's objects where Bambu Studio puts that plate."""

import json
import zipfile

import pytest
from build123d import Align, Box, Location

from print_assist.build.ops import OpError, apply
from print_assist.export.bambu import Obj, Part, PlateInfo, load_template, palette, read_project, write_project
from print_assist.export.layout import plate_origin, resolve
from print_assist.features.model import Body, Design, Filament, PrintItem, PrintPlate
from print_assist.printers import profiles

A1 = profiles()["bambu_a1_ams_lite"]
BED = (256.0, 256.0)


def box(x0, y0, z0, dx, dy, dz):
    return Location((x0, y0, z0)) * Box(dx, dy, dz, align=Align.MIN)


def model():
    """B1 a 10 x 20 x 30 block at the origin, B2 a 2 x 2 x 1 letter on its top (x 4..6, y 9..11, z 30..31), B3 two
    5 mm cubes at x 50 and x 70."""
    d = Design(bodies=[Body(id="B1", name="Block", filament=1), Body(id="B2", name="Letter", filament=2),
                       Body(id="B3", name="Cubes", filament=1)],
               filaments=[Filament(slot=1, color="#222222"), Filament(slot=2, color="#eeeeee")])
    from build123d import Compound

    parts = {"B1": box(0, 0, 0, 10, 20, 30), "B2": box(4, 9, 30, 2, 2, 1),
             "B3": Compound(children=[box(50, 0, 0, 5, 5, 5), box(70, 0, 0, 5, 5, 5)])}
    return d, parts


def test_item_turns_its_solids_together_and_sits_on_its_plate():
    d, parts = model()
    d.layout = [PrintItem(id="P1", name="Block", bodies=["B1", "B2"], rotate=[90, 0, 0], x=5, y=5),
                PrintItem(id="P2", name="Cube", bodies=["B3"], within=[40, -1, -1, 60, 10, 10], plate=2)]
    lay = resolve(d, parts, BED)
    p1, p2 = lay.placed
    # 90 degrees about X turns the 10 x 20 x 31 item into 10 x 31 x 20; its corner goes to (5, 5) on the bed.
    assert p1.box == (5, 5, 0, 15, 36, 20)
    letter = dict(p1.parts)["B2"].bounding_box()
    # The letter was on top (z 30..31): turning +90 about X sends +Z to -Y, so it is the first 1 mm of y, and its
    # y 9..11 (centre 10, the item's centre) becomes z 9..11.
    assert (letter.min.Y, letter.max.Y) == pytest.approx((5, 6)) and (letter.min.Z, letter.max.Z) == pytest.approx((9, 11))
    assert (letter.min.X, letter.max.X) == pytest.approx((9, 11))  # x 4..6 of the 0..10 block, moved by 5
    assert p2.item.plate == 2 and p2.box == (0, 0, 0, 5, 5, 5) and len(p2.parts) == 1
    assert [(u["body"], u["index"], u["center"]) for u in lay.unplaced] == [("B3", 1, [72.5, 2.5, 2.5])]
    assert len(lay.problems) == 1 and "B3#1 at [72.5, 2.5, 2.5]" in lay.problems[0]


def test_overlaps_off_plate_towers_and_double_takes_are_problems():
    d, parts = model()
    d.plates = [PrintPlate(), PrintPlate(name="Cubes", tower=[100, 100])]
    d.layout = [PrintItem(id="P1", bodies=["B1", "B2"], x=250, y=0),  # 10 wide from x 250: off a 256 bed
                PrintItem(id="P2", bodies=["B3"], within=[40, -1, -1, 60, 10, 10], plate=2, x=0, y=0),
                PrintItem(id="P3", bodies=["B3"], within=[60, -1, -1, 80, 10, 10], plate=2, x=5.5, y=0),  # 0.5 gap
                PrintItem(id="P4", bodies=["B3"], within=[60, -1, -1, 80, 10, 10], plate=2, x=110, y=110),
                PrintItem(id="P5", bodies=["B3"], within=[0, 0, 0, 1, 1, 1], plate=2)]
    probs = " | ".join(resolve(d, parts, BED).problems)
    assert "P1 : off plate 1 (spans x 250.0..260.0" in probs
    assert "P2  and P3  overlap on plate 2" in probs
    assert "solid 1 of B3 is also in P3" in probs  # P4 asks for the cube P3 has
    assert "P5 : its box holds no solid of B3" in probs


def test_plate_origins_follow_bambu_studio():
    # Bambu Studio: ceil(sqrt(n)) columns, 1.2 bed sizes apart, rows toward -Y. A six-plate A1 project from
    # MakerWorld has its plate 2 objects at x 128 + 307.2, plate 3 at 128 + 614.4, plate 4 at y 128 - 307.2.
    assert plate_origin(1, 6, BED) == (0, 0)
    assert plate_origin(2, 6, BED) == pytest.approx((307.2, 0)) and plate_origin(3, 6, BED) == pytest.approx((614.4, 0))
    assert plate_origin(4, 6, BED) == pytest.approx((0, -307.2))
    assert plate_origin(3, 4, BED) == pytest.approx((0, -307.2))  # four plates: two columns
    assert plate_origin(2, 2, BED) == pytest.approx((307.2, 0)) and plate_origin(1, 1, BED) == (0, 0)


def test_project_with_plates_places_objects_and_towers(tmp_path):
    t = load_template(A1)
    fs = palette([Filament(slot=1, color="#222222"), Filament(slot=2, color="#eeeeee")])
    objs = [Obj("Block", [Part("Block", box(10, 10, 0, 10, 20, 30), 1), Part("Letter", box(14, 19, 30, 2, 2, 1), 2)], 1),
            Obj("Cube", [Part("Cube", box(0, 0, 0, 5, 5, 5), 1)], 2)]
    path = write_project([], fs, A1, t, tmp_path / "p.3mf", "Laid out", None, objs,
                         [PlateInfo("Two colour"), PlateInfo("Cubes", [100, 120])])
    got, settings, _ = read_project(path)
    assert [(p.object, p.name, p.plate, p.filament) for p in got] == [
        ("Block", "Block", 1, 1), ("Block", "Letter", 1, 2), ("Cube", "Cube", 2, 1)]
    assert all(p.open_edges() == 0 for p in got)
    lo, hi = got[1].bounds()
    assert lo == pytest.approx((14, 19, 30)) and hi == pytest.approx((16, 21, 31))  # the letter stays on its block
    lo, hi = got[2].bounds()
    assert lo == pytest.approx((307.2, 0, 0)) and hi == pytest.approx((312.2, 5, 5))  # plate 2 of 2 is 307.2 to the right
    base = load_template(A1)["wipe_tower_x"][0]
    assert settings["wipe_tower_x"] == [base, "100"] and settings["wipe_tower_y"][1] == "120"
    with zipfile.ZipFile(path) as z:
        ms = z.read("Metadata/model_settings.config").decode()
    assert ms.count("<plate>") == 2 and 'value="Cubes"' in ms and 'value="Two colour"' in ms
    assert json.loads(json.dumps(settings))  # still valid JSON


def test_layout_ops():
    d = Design(bodies=[Body(id="B1"), Body(id="B2")])
    assert apply(d, {"op": "set_print_item", "name": "Lid", "bodies": ["B2"], "plate": 3, "rotate": [180, 0, 0],
                     "x": 4, "y": 5}, "ai") == "P1"
    apply(d, {"op": "set_print_item", "id": "P1", "within": [0, 0, 0, 1, 1, 1]}, "ai")
    assert d.layout[0].name == "Lid" and d.layout[0].within == [0, 0, 0, 1, 1, 1] and d.layout[0].plate == 3
    with pytest.raises(OpError, match="bodies must be"):
        apply(d, {"op": "set_print_item", "bodies": ["B9"]}, "ai")
    with pytest.raises(OpError, match="within is"):
        apply(d, {"op": "set_print_item", "bodies": ["B1"], "within": [1, 0, 0, 0, 1, 1]}, "ai")
    apply(d, {"op": "set_plate", "plate": 2, "name": "Ivory", "tower": [200, 10]}, "ai")
    assert [p.name for p in d.plates] == ["", "Ivory"] and d.plates[1].tower == [200.0, 10.0]
    with pytest.raises(OpError, match="still has objects"):
        apply(d, {"op": "delete_plate", "plate": 3}, "ai")
    apply(d, {"op": "delete_plate", "plate": 1}, "ai")  # empty: the plates after it move up
    assert [p.name for p in d.plates] == ["Ivory"] and d.layout[0].plate == 2
    apply(d, {"op": "delete_print_item", "id": "P1"}, "ai")
    assert d.layout == []


def test_mesh_merges_copies_either_side_of_a_rounding_boundary():
    import numpy as np

    from print_assist.geometry.tessellate import _merge

    # 5e-7 is where rounding to 6 decimals flips: copies 2e-12 apart used to stay two points and leave a seam.
    p = np.array([[5e-7 - 1e-12, 1, 2], [5e-7 + 1e-12, 1, 2], [3, 1, 2], [3 + 2e-6, 1, 2]])
    merged, label = _merge(p)
    assert len(merged) == 3 and label[0] == label[1] and label[2] != label[3]


def test_turned_letters_mesh_closed(tmp_path):
    # A turned solid with many curved faces (a cylinder at odd angles) still meshes closed.
    from build123d import Axis, Cylinder

    from print_assist.export.bambu import mesh

    s = Cylinder(3.7, 1.3).rotate(Axis.X, 37.3).rotate(Axis.Y, 90).rotate(Axis.Z, 12.9)
    points, faces = mesh(s)
    from print_assist.export.bambu import ReadPart

    assert ReadPart("c", 1, points, faces).open_edges() == 0


def test_objects_carry_their_own_slicer_settings(tmp_path):
    import re

    from print_assist.export.bambu import BambuError

    t = load_template(A1)
    fs = palette([Filament(slot=1, color="#222222")])
    objs = [Obj("Sparse", [Part("Sparse", box(10, 10, 0, 10, 10, 10), 1)], 1, {"sparse_infill_density": "8%"}),
            Obj("Plain", [Part("Plain", box(30, 10, 0, 10, 10, 10), 1)], 1)]
    path = write_project([], fs, A1, t, tmp_path / "s.3mf", "Settings", None, objs)
    with zipfile.ZipFile(path) as z:
        ms = z.read("Metadata/model_settings.config").decode()
        ps = json.loads(z.read("Metadata/project_settings.config"))
    sparse, plain = re.findall(r"<object id=.*?</object>", ms, re.S)
    assert '<metadata key="sparse_infill_density" value="8%"/>' in sparse and "sparse_infill" not in plain
    assert ps["sparse_infill_density"] == t["sparse_infill_density"]  # the project keeps its own default
    with pytest.raises(BambuError, match="infill_densty is not a"):
        write_project([], fs, A1, t, tmp_path / "x.3mf", "X", None,
                      [Obj("Typo", [Part("T", box(0, 0, 0, 5, 5, 5), 1)], 1, {"infill_densty": "8%"})])


def test_copies_lay_out_in_rows_each_its_own_object():
    d, parts = model()
    d.layout = [PrintItem(id="P1", name="Block", bodies=["B1", "B2"], x=5, y=5, copies=5, columns=2, spacing=3),
                PrintItem(id="P2", name="Cube", bodies=["B3"], within=[40, -1, -1, 80, 10, 10], plate=2)]
    lay = resolve(d, parts, BED)
    blocks = [p for p in lay.placed if p.item.id == "P1"]
    # The block is 10 x 20 (x 0..10, y 0..20); copy k sits at column k % 2, row k // 2: x 5 + col * (10 + 3),
    # y 5 + row * (20 + 3).
    assert [p.box[:2] for p in blocks] == [(5, 5), (18, 5), (5, 28), (18, 28), (5, 51)]
    assert [p.copy for p in blocks] == [1, 2, 3, 4, 5] and all(len(p.parts) == 2 for p in blocks)
    assert not lay.problems


def test_copies_that_run_off_the_plate_are_problems():
    d, parts = model()
    d.layout = [PrintItem(id="P1", name="Block", bodies=["B1", "B2"], x=230, y=0, copies=3, spacing=0.5),
                PrintItem(id="P2", name="Cube", bodies=["B3"], plate=2)]
    lay = resolve(d, parts, BED)
    # Copies at x 230, 240.5, 251: the third spans 251..261 on a 256 bed. They are 0.5 apart, under the 1 mm gap.
    assert any("P1 Block (copy 3): off plate 1" in t for t in lay.problems)
    assert any("P1 Block (copy 1) and P1 Block (copy 2) overlap" in t for t in lay.problems)


def test_set_print_item_takes_copies():
    d, _ = model()
    apply(d, {"op": "set_print_item", "bodies": ["B1"], "copies": 3, "columns": 2, "spacing": 4}, "ai")
    assert (d.layout[0].copies, d.layout[0].columns, d.layout[0].spacing) == (3, 2, 4)
    with pytest.raises(OpError):
        apply(d, {"op": "set_print_item", "id": "P1", "copies": 0}, "ai")
