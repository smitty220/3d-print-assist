import math

import pytest
from build123d import IN

from print_assist.geometry.samples import SAMPLES, frame_rib, write_samples
from print_assist.geometry.step_io import StepImportError, read_step
from print_assist.geometry.tessellate import mesh_shape


def _tri_normal(pos, a, b, c):
    p = [pos[3 * i : 3 * i + 3] for i in (a, b, c)]
    u = [p[1][k] - p[0][k] for k in range(3)]
    v = [p[2][k] - p[0][k] for k in range(3)]
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


@pytest.mark.parametrize("name", list(SAMPLES))
def test_samples_round_trip_through_step(tmp_path, name):
    (path,) = [p for p in write_samples(tmp_path) if p.stem == name]
    shape = read_step(path)
    assert shape.is_valid
    mesh = mesh_shape(shape, name)
    assert len(mesh.faces) == len(shape.faces())
    for face in mesh.faces:
        assert face.indices and len(face.indices) % 3 == 0
        assert max(face.indices) < len(face.positions) // 3


def test_frame_rib_volume_and_holes():
    rib = frame_rib()
    flange = 6 * 4 * 0.25 - 4 * math.pi * 0.1005**2 * 0.25
    web = 6 * 0.25 * 2
    assert rib.volume / IN**3 == pytest.approx(flange + web, rel=1e-6)
    mesh = mesh_shape(rib, "rib")
    holes = [f for f in mesh.faces if f.type == "cylinder"]
    assert len(holes) == 4
    assert all(f.radius == pytest.approx(0.1005 * IN, abs=1e-6) for f in holes)
    assert all(abs(f.axis_dir[2]) == pytest.approx(1.0) for f in holes)
    # A full hole's area centroid lies on its axis, mid-depth in the flange.
    for f in holes:
        assert f.centroid[:2] == pytest.approx(f.axis_origin[:2], abs=1e-6)
        assert f.centroid[2] == pytest.approx(-0.125 * IN, abs=1e-6)


def test_planar_face_triangles_wind_outward():
    """Triangle winding must agree with the B-rep face normal, or picking and lighting break."""
    mesh = mesh_shape(frame_rib(), "rib")
    for face in (f for f in mesh.faces if f.type == "plane"):
        n = [0.0, 0.0, 0.0]
        for t in range(0, len(face.indices), 3):
            tn = _tri_normal(face.positions, *face.indices[t : t + 3])
            n = [n[k] + tn[k] for k in range(3)]
        assert sum(n[k] * face.normal[k] for k in range(3)) > 0, f"face {face.id} winds inward"


def test_top_flange_face_signature():
    mesh = mesh_shape(frame_rib(), "rib")
    top = max((f for f in mesh.faces if f.type == "plane"), key=lambda f: f.centroid[2])
    assert top.normal == pytest.approx([0, 0, 1])
    assert top.centroid[2] == pytest.approx(0.0, abs=1e-6)


def test_bad_step_file_raises(tmp_path):
    bad = tmp_path / "bad.stp"
    bad.write_text("not a step file")
    with pytest.raises(StepImportError):
        read_step(bad)
