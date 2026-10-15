import math

import pytest
from build123d import Align, Box, Cylinder, Pos

from print_assist.geometry.step_io import StepImportError, read_step, write_step
from print_assist.geometry.tessellate import mesh_shape

MIN_Z = (Align.CENTER, Align.CENTER, Align.MIN)


def plate_with_holes():
    """60 x 40 x 5 mm plate on the bed with four 4 mm holes at (±20, ±12)."""
    plate = Box(60, 40, 5, align=MIN_Z)
    for x in (-20, 20):
        for y in (-12, 12):
            plate = plate - Pos(x, y, 0) * Cylinder(2, 5, align=MIN_Z)
    return plate


def _tri_normal(pos, a, b, c):
    p = [pos[3 * i : 3 * i + 3] for i in (a, b, c)]
    u = [p[1][k] - p[0][k] for k in range(3)]
    v = [p[2][k] - p[0][k] for k in range(3)]
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def test_plate_round_trips_through_step(tmp_path):
    shape = plate_with_holes()
    path = tmp_path / "plate.stp"
    write_step(shape, path)
    back = read_step(path)
    assert back.is_valid
    # 60·40·5 − 4·π·2²·5
    assert back.volume == pytest.approx(60 * 40 * 5 - 4 * math.pi * 4 * 5, rel=1e-6)
    mesh = mesh_shape(back, "plate")
    assert len(mesh.faces) == len(back.faces())
    for face in mesh.faces:
        assert face.indices and len(face.indices) % 3 == 0
        assert max(face.indices) < len(face.positions) // 3


def test_holes_are_tagged_cylinders():
    mesh = mesh_shape(plate_with_holes(), "plate")
    holes = [f for f in mesh.faces if f.type == "cylinder"]
    assert len(holes) == 4
    assert all(f.radius == pytest.approx(2.0, abs=1e-6) for f in holes)
    assert all(abs(f.axis_dir[2]) == pytest.approx(1.0) for f in holes)
    for f in holes:  # a full hole's area centroid lies on its axis, mid-depth
        assert f.centroid[:2] == pytest.approx(f.axis_origin[:2], abs=1e-6)
        assert f.centroid[2] == pytest.approx(2.5, abs=1e-6)


def test_planar_face_triangles_wind_outward():
    """Triangle winding must agree with the B-rep face normal, or picking and lighting break."""
    mesh = mesh_shape(plate_with_holes(), "plate")
    for face in (f for f in mesh.faces if f.type == "plane"):
        n = [0.0, 0.0, 0.0]
        for t in range(0, len(face.indices), 3):
            tn = _tri_normal(face.positions, *face.indices[t : t + 3])
            n = [n[k] + tn[k] for k in range(3)]
        assert sum(n[k] * face.normal[k] for k in range(3)) > 0, f"face {face.id} winds inward"


def test_top_face_signature():
    mesh = mesh_shape(plate_with_holes(), "plate")
    top = max((f for f in mesh.faces if f.type == "plane"), key=lambda f: f.centroid[2])
    assert top.normal == pytest.approx([0, 0, 1])
    assert top.centroid[2] == pytest.approx(5.0, abs=1e-6)


def test_bad_step_file_raises(tmp_path):
    bad = tmp_path / "bad.stp"
    bad.write_text("not a step file")
    with pytest.raises(StepImportError):
        read_step(bad)
