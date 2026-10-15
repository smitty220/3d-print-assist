"""Bambu Studio project export: the A1 template resized to the design's filaments, parts in their filaments, and closed
meshes on the bed. The template is the project_settings.config of a two-colour A1 project saved by Bambu Studio."""

import math
import zipfile

import pytest
from build123d import Box, Cylinder, Location

from print_assist.export.bambu import (
    BambuError, Part, flush_volume, load_template, palette, project_settings, read_project, write_project,
)
from print_assist.features.model import Filament
from print_assist.printers import Nozzle, profiles

A1 = profiles()["bambu_a1_ams_lite"]


def template():
    t = load_template(A1)
    assert t is not None and t["printer_model"] == "Bambu Lab A1" and len(t["filament_colour"]) == 2
    return t


def test_template_resized_to_three_filaments():
    t = template()
    fs = [Filament(slot=1, color="#111111"), Filament(slot=2, color="#ffffff"), Filament(slot=3, color="#c9a227")]
    s = project_settings(t, fs, A1)
    assert set(s) == set(t)  # no setting added or lost
    assert s["filament_colour"] == ["#111111", "#FFFFFF", "#C9A227"] and s["filament_self_index"] == ["1", "2", "3"]
    for key in ("filament_settings_id", "nozzle_temperature", "filament_type", "hot_plate_temp", "filament_map"):
        assert len(s[key]) == 3 and s[key][2] == t[key][0], key
    assert len(s["filament_dev_ams_drying_time"]) == 12  # four values per filament in this template
    for key in ("machine_max_speed_x", "printable_area", "wipe_tower_x", "extruder_ams_count", "nozzle_diameter"):
        assert s[key] == t[key], key  # not per filament
    assert len(s["flush_volumes_matrix"]) == 9 and s["flush_volumes_matrix"][0] == "0"
    assert len(s["different_settings_to_system"]) == 5  # process, three filaments, printer
    assert s["filament_map_mode"] == t["filament_map_mode"]  # one nozzle: Bambu Studio maps them


def test_flush_volumes_follow_the_colour_change():
    assert flush_volume("#000000", "#ffffff") == 620 and flush_volume("#ffffff", "#000000") == 140
    assert flush_volume("#C9A227", "#c9a227") == 0


def test_palette_fills_skipped_slots_and_materials_must_be_in_the_template():
    fs = palette([Filament(slot=3, color="#c9a227"), Filament(slot=1, color="#111111")])
    assert [(f.slot, f.color) for f in fs] == [(1, "#111111"), (2, "#808080"), (3, "#c9a227")]
    assert [f.color for f in palette([], "#123456")] == ["#123456"]
    project_settings(template(), [Filament(slot=1, material="PLA Silk")], A1)  # PLA settings
    with pytest.raises(BambuError, match="only has PLA"):
        project_settings(template(), [Filament(slot=1, material="PETG")], A1)


def test_dual_nozzle_printer_maps_each_filament_to_its_nozzle():
    x2 = A1.model_copy(update={"nozzles": [Nozzle(), Nozzle()]})
    s = project_settings(template(), [Filament(slot=1), Filament(slot=2, nozzle=1)], x2)
    assert s["filament_map"] == ["1", "2"] and s["filament_map_mode"] == "Manual"


def test_project_parts_meshes_and_placement(tmp_path):
    base = Box(40, 40, 10)  # centred on the origin: z -5..5
    peg = Location((0, 0, 10)) * Cylinder(5, 10)  # z 5..15, standing on the base
    fs = palette([Filament(slot=1, color="#111111"), Filament(slot=3, color="#c9a227")])
    parts = [Part("Base", Location((128, 128, 5)) * base, 1), Part("Peg", Location((128, 128, 5)) * peg, 3)]
    path = write_project(parts, fs, A1, template(), tmp_path / "p.3mf", "Peg & Base", b"\x89PNG fake")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"3D/3dmodel.model", "3D/Objects/object_1.model", "Metadata/model_settings.config",
                "Metadata/project_settings.config", "Metadata/plate_1.png", "_rels/.rels"} <= names
        assert b'name="Application">BambuStudio-' in z.read("3D/3dmodel.model")
    got, settings, meta = read_project(path)
    assert meta["Title"] == "Peg & Base" and len(settings["filament_colour"]) == 3
    base_r, peg_r = got
    assert (base_r.name, base_r.filament, peg_r.name, peg_r.filament) == ("Base", 1, "Peg", 3)
    assert base_r.open_edges() == 0 and peg_r.open_edges() == 0
    assert base_r.volume == pytest.approx(16000, rel=1e-9)
    assert peg_r.volume == pytest.approx(math.pi * 25 * 10, rel=0.002)  # faceted at 0.01 mm
    lo, hi = base_r.bounds()
    assert lo == pytest.approx((108, 108, 0)) and hi == pytest.approx((148, 148, 10))
    assert peg_r.bounds()[1][2] == pytest.approx(20)


def test_bad_filament_is_refused(tmp_path):
    with pytest.raises(BambuError, match="filament 4"):
        write_project([Part("Box", Box(1, 1, 1), 4)], palette([]), A1, template(), tmp_path / "x.3mf", "x")
