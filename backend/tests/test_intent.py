import pytest
from build123d import IN, Axis

from print_assist.geometry.samples import frame_rib, rod_end
from print_assist.geometry.signature import face_signature, find_face
from print_assist.geometry.step_io import write_step
from print_assist.intent import builders
from print_assist.intent.models import Intent
from print_assist.project import Project


def _top_face_id(shape):
    top = shape.faces().sort_by(Axis.Z)[-1]
    return next(i for i, f in enumerate(shape.faces()) if f.is_same(top))


def _hole_face_id(shape):
    return next(i for i, f in enumerate(shape.faces()) if f.geom_type.name == "CYLINDER")


def test_planar_mount_frame_and_footprint():
    rib = frame_rib()
    m = builders.planar_mount(Intent(), rib, "rib", _top_face_id(rib))
    assert m.id == "IF-00"
    assert m.frame.origin == pytest.approx((0, 0, 0), abs=1e-6)
    assert m.frame.z == pytest.approx((0, 0, 1))
    xs = [p[0] for p in m.footprint]
    ys = [p[1] for p in m.footprint]
    assert (max(xs) - min(xs)) / IN == pytest.approx(6.0)
    assert (max(ys) - min(ys)) / IN == pytest.approx(4.0)


def test_bolt_pattern_from_face_reads_holes_and_suggests_fastener():
    rib = frame_rib()
    intent = Intent()
    mount = builders.planar_mount(intent, rib, "rib", _top_face_id(rib))
    intent.objects.append(mount)
    bp = builders.bolt_pattern(intent, rib, "rib", _top_face_id(rib))
    assert bp.id == "IF-01"
    assert bp.parent == "IF-00"
    assert len(bp.holes) == 4
    centres = sorted((round(abs(h.center[0]) / IN, 4), round(abs(h.center[1]) / IN, 4)) for h in bp.holes)
    assert centres == [(0.95, 0.95)] * 4
    assert all(h.diameter / IN == pytest.approx(0.201) for h in bp.holes)
    assert all(h.depth / IN == pytest.approx(0.25) for h in bp.holes)
    assert bp.fastener == "NAS1352-3"


def test_bolt_pattern_from_picking_one_hole_finds_the_mount_face():
    rib = frame_rib()
    bp = builders.bolt_pattern(Intent(), rib, "rib", _hole_face_id(rib))
    assert bp.face.face_id == _top_face_id(rib)
    assert len(bp.holes) == 4


def test_bore_from_rod_end_eye():
    re = rod_end()
    eye_bore = min((i for i, f in enumerate(re.faces()) if f.geom_type.name == "CYLINDER"), key=lambda i: re.faces()[i].radius)
    b = builders.bore(Intent(), re, "rod", eye_bore)
    assert b.diameter / IN == pytest.approx(0.377)
    assert b.length / IN == pytest.approx(0.6)
    assert abs(b.frame.z[0]) == pytest.approx(1.0)
    assert b.frame.origin[2] / IN == pytest.approx(3.0)


def test_wrong_face_type_is_rejected():
    rib = frame_rib()
    with pytest.raises(builders.BuildError):
        builders.bore(Intent(), rib, "rib", _top_face_id(rib))
    with pytest.raises(builders.BuildError):
        builders.planar_mount(Intent(), rib, "rib", _hole_face_id(rib))


def test_signature_finds_face_on_revised_part():
    old, new = frame_rib(), frame_rib(flange_t=0.30, top_z=0.05)
    sig = face_signature(old.faces()[_top_face_id(old)])
    assert find_face(new, sig) == _top_face_id(new)


def test_project_rematches_and_rebuilds_interfaces_on_revision(tmp_path):
    p = Project(tmp_path)
    p.add_part("rib", write_step(frame_rib(), tmp_path / "a.stp").read_bytes())
    rib = p.shape("rib")
    top = _top_face_id(rib)
    for kind in ("planar_mount", "bolt_pattern"):
        p.data.intent.objects.append(getattr(builders, kind)(p.data.intent, rib, "rib", top))
    p.data.intent.objects[1].fastener = "NAS6603"  # a user choice that must survive the revision

    warnings = p.add_part("rib", write_step(frame_rib(flange_t=0.30, top_z=0.05), tmp_path / "b.stp").read_bytes())
    assert warnings == []
    mount, bp = p.data.intent.objects
    assert mount.id == "IF-00" and bp.id == "IF-01"
    assert mount.frame.origin[2] / IN == pytest.approx(0.05)
    assert bp.fastener == "NAS6603"
    assert len(bp.holes) == 4


def test_keepout_role_creates_and_removes_intent(tmp_path):
    p = Project(tmp_path)
    p.seed_samples()
    kos = [o for o in p.data.intent.objects if o.kind == "keep_out"]
    assert [k.part for k in kos] == ["hyd_line_envelope"]
    p.set_role("hyd_line_envelope", "mating")
    assert not any(o.kind == "keep_out" for o in p.data.intent.objects)


def _sample_intent():
    rib, re = frame_rib(), rod_end()
    intent = Intent()
    top = _top_face_id(rib)
    intent.objects.append(builders.planar_mount(intent, rib, "rib", top))
    intent.objects.append(builders.bolt_pattern(intent, rib, "rib", top))
    eye = min((i for i, f in enumerate(re.faces()) if f.geom_type.name == "CYLINDER"), key=lambda i: re.faces()[i].radius)
    intent.objects.append(builders.bore(intent, re, "rod", eye))
    return intent


def test_default_design_space_encloses_interfaces_with_margin():
    intent = _sample_intent()
    ds = builders.default_design_space(intent)
    assert ds.id == "DS-01"
    lo = [v / IN for v in ds.min]
    hi = [v / IN for v in ds.max]
    # Holes at ±0.95 dominate X and Y; the Ø0.377 x 0.600 bore at Z=3.0 sets the top. Margin 0.5 in.
    assert lo[:2] == pytest.approx([-1.45, -1.45], abs=1e-3)
    assert hi[:2] == pytest.approx([1.45, 1.45], abs=1e-3)
    assert lo[2] == 0.0
    assert hi[2] == pytest.approx(3.0 + 0.1885 + 0.5, abs=1e-3)


def test_design_space_needs_a_mount():
    with pytest.raises(builders.BuildError):
        builders.default_design_space(Intent())


def test_new_load_case_targets_the_bore_by_default():
    intent = _sample_intent()
    lc = builders.new_load_case(intent, None)
    assert (lc.id, lc.target) == ("LC-01", "IF-02")
    with pytest.raises(builders.BuildError):
        builders.new_load_case(intent, "KO-99")
