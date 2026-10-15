import pytest
from fastapi.testclient import TestClient

from cad_assist import app as app_module
from cad_assist.geometry.samples import frame_rib
from cad_assist.geometry.step_io import write_step


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "_project", None)
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_project_seeds_samples(client):
    body = client.get("/api/project").json()
    assert set(body["parts"]) == {"frame_rib_FS412", "actuator_rod_end", "hyd_line_envelope"}
    assert body["parts"]["hyd_line_envelope"]["role"] == "keepout"
    assert [o["id"] for o in body["intent"]["objects"]] == ["KO-01"]
    mesh = client.get("/api/project/parts/frame_rib_FS412/mesh").json()
    assert len(mesh["faces"]) == 14
    assert client.get("/api/project/parts/nope/mesh").status_code == 404


def _top_face(client):
    mesh = client.get("/api/project/parts/frame_rib_FS412/mesh").json()
    return max((f for f in mesh["faces"] if f["type"] == "plane"), key=lambda f: f["centroid"][2])["id"]


def test_create_patch_delete_intent(client):
    top = _top_face(client)
    r = client.post("/api/project/intent/from-face", json={"kind": "planar_mount", "part": "frame_rib_FS412", "face_id": top})
    assert r.status_code == 200 and r.json()["created"] == "IF-00"
    r = client.post("/api/project/intent/from-face", json={"kind": "bolt_pattern", "part": "frame_rib_FS412", "face_id": top})
    bp = next(o for o in r.json()["project"]["intent"]["objects"] if o["id"] == "IF-01")
    assert bp["parent"] == "IF-00" and bp["fastener"] == "NAS1352-3"
    dup = client.post("/api/project/intent/from-face", json={"kind": "bolt_pattern", "part": "frame_rib_FS412", "face_id": top})
    assert dup.status_code == 409
    mesh = client.get("/api/project/parts/frame_rib_FS412/mesh").json()
    hole = next(f["id"] for f in mesh["faces"] if f["type"] == "cylinder")
    via_hole = client.post("/api/project/intent/from-face", json={"kind": "bolt_pattern", "part": "frame_rib_FS412", "face_id": hole})
    assert via_hole.status_code == 409

    r = client.patch("/api/project/intent/IF-01", json={"fastener": "NAS6603", "fit": "fixed"})
    bp = next(o for o in r.json()["intent"]["objects"] if o["id"] == "IF-01")
    assert (bp["fastener"], bp["fit"]) == ("NAS6603", "fixed")
    assert client.patch("/api/project/intent/IF-01", json={"fastener": "BOGUS"}).status_code == 422
    assert client.patch("/api/project/intent/IF-00", json={"fastener": "NAS6603"}).status_code == 422

    r = client.delete("/api/project/intent/IF-00")
    bp = next(o for o in r.json()["intent"]["objects"] if o["id"] == "IF-01")
    assert bp["parent"] is None


def test_from_face_rejects_wrong_face(client):
    top = _top_face(client)
    r = client.post("/api/project/intent/from-face", json={"kind": "bore", "part": "frame_rib_FS412", "face_id": top})
    assert r.status_code == 422


def test_upload_and_role(client, tmp_path):
    path = write_step(frame_rib(), tmp_path / "my rib.step")
    with path.open("rb") as fh:
        r = client.post("/api/project/parts", files={"file": ("my rib.step", fh, "application/step")})
    assert r.status_code == 200
    assert "my_rib" in r.json()["project"]["parts"]
    r = client.patch("/api/project/parts/my_rib", json={"role": "keepout"})
    assert any(o["kind"] == "keep_out" and o["part"] == "my_rib" for o in r.json()["intent"]["objects"])
    r = client.delete("/api/project/parts/my_rib")
    assert "my_rib" not in r.json()["parts"]


def test_upload_rejects_bad_files(client):
    assert client.post("/api/project/parts", files={"file": ("p.stl", b"x", "model/stl")}).status_code == 415
    assert client.post("/api/project/parts", files={"file": ("p.stp", b"garbage", "application/step")}).status_code == 422


def test_design_space_loads_and_rules(client):
    assert client.post("/api/project/intent/design-space").status_code == 422  # no mount yet
    top = _top_face(client)
    client.post("/api/project/intent/from-face", json={"kind": "planar_mount", "part": "frame_rib_FS412", "face_id": top})
    r = client.post("/api/project/intent/design-space")
    assert r.json()["created"] == "DS-01"
    assert client.post("/api/project/intent/design-space").status_code == 409
    assert client.patch("/api/project/intent/DS-01", json={"min": [0, 0, 0], "max": [0, 10, 10]}).status_code == 422
    assert client.patch("/api/project/intent/DS-01", json={"max": [50, 50, 100]}).status_code == 200

    r = client.post("/api/project/intent/load-case", json={})
    lc = r.json()["created"]
    r = client.patch(f"/api/project/intent/{lc}", json={"force": [0, -2800, -7740], "factor": 1.5})
    obj = next(o for o in r.json()["intent"]["objects"] if o["id"] == lc)
    assert obj["target"] == "IF-00" and obj["force"] == [0, -2800, -7740]
    assert client.patch(f"/api/project/intent/{lc}", json={"target": "KO-01"}).status_code == 422
    r = client.delete("/api/project/intent/IF-00")
    obj = next(o for o in r.json()["intent"]["objects"] if o["id"] == lc)
    assert obj["target"] is None

    rules = client.get("/api/project").json()["intent"]["rules"]
    assert rules["bolt_edge_min"] == 1.5
    rules["mass_target"] = 0.163
    assert client.put("/api/project/rules", json=rules).json()["intent"]["rules"]["mass_target"] == 0.163
    rules["material"] = "unobtainium"
    assert client.put("/api/project/rules", json=rules).status_code == 422


def test_design_api(client):
    top = _top_face(client)
    for kind in ("planar_mount", "bolt_pattern"):
        client.post("/api/project/intent/from-face", json={"kind": kind, "part": "frame_rib_FS412", "face_id": top})
    mesh = client.get("/api/project/parts/actuator_rod_end/mesh").json()
    eye = min((f for f in mesh["faces"] if f["type"] == "cylinder"), key=lambda f: f["radius"])["id"]
    client.post("/api/project/intent/from-face", json={"kind": "bore", "part": "actuator_rod_end", "face_id": eye})

    assert client.get("/api/project/design").json()["features"] == []
    assert len(client.get("/api/features/catalog").json()) >= 9
    assert client.post("/api/project/design/template", json={"name": "angle_bracket"}).status_code == 422

    body = client.post("/api/project/design/template", json={"name": "clevis_bracket"}).json()
    assert [f["id"] for f in body["features"]] == ["F1", "F2", "F3", "F4", "F5", "F6"]
    assert body["mesh"] and len(body["face_names"]) == len(body["mesh"]["faces"])
    m0 = body["mass"]

    body = client.patch("/api/project/design/features/F3", json={"params": {"ear_thickness": 5.0}, "locked": True}).json()
    f3 = next(f for f in body["features"] if f["id"] == "F3")
    assert f3["params"]["ear_thickness"] == 5.0 and f3["locked"] is True
    assert body["mass"] < m0
    assert client.patch("/api/project/design/features/F3", json={"params": {"bogus": 1}}).status_code == 422

    body = client.delete("/api/project/design/features/F6").json()
    assert "F6" not in [f["id"] for f in body["features"]]
    body = client.post("/api/project/design/features", json={"kind": "chamfer", "refs": {"faces_a": "F1.top", "faces_b": "F1.sides"}}).json()
    assert body["created"] == "F6"

    # A fillet by picking edges on the part: it needs edges first; a pick adds one, picking it again takes it off.
    body = client.post("/api/project/design/features", json={"kind": "fillet", "params": {"radius": 1.0}}).json()
    assert body["created"] == "F7" and "pick the edges" in body["features"][-1]["error"]
    m1 = body["mass"]
    n_edges = len(body["mesh"]["edges"])
    for i in range(n_edges):  # the first edge with a corner to blend (tangent joints are refused with a 422)
        r = client.post("/api/project/design/features/F7/edges", json={"edge": i})
        if r.status_code == 200:
            break
        assert "tangent" in r.json()["detail"] or "later" in r.json()["detail"]
    body = r.json()
    f7 = body["features"][-1]
    assert f7["error"] is None and len(f7["edges"]) == 1 and len(f7["edges"][0]["world"]) == 9
    assert body["mass"] < m1
    assert client.post("/api/project/design/features/F1/edges", json={"edge": i}).status_code == 422
    # Taking the pick off (the list's x): the blend goes and the part is as it was.
    body = client.patch("/api/project/design/features/F7", json={"edges": []}).json()
    assert body["features"][-1]["edges"] == [] and body["mass"] == pytest.approx(m1)


def test_validation_api(client):
    body = client.get("/api/project/validation").json()
    pre = {f["title"]: f["status"] for f in body["preflight"]}
    assert pre["Primary mount"] == "fail" and body["preflight_ready"] is False
    assert body["part"] == []

    top = _top_face(client)
    for kind in ("planar_mount", "bolt_pattern"):
        client.post("/api/project/intent/from-face", json={"kind": kind, "part": "frame_rib_FS412", "face_id": top})
    mesh = client.get("/api/project/parts/actuator_rod_end/mesh").json()
    eye = min((f for f in mesh["faces"] if f["type"] == "cylinder"), key=lambda f: f["radius"])["id"]
    client.post("/api/project/intent/from-face", json={"kind": "bore", "part": "actuator_rod_end", "face_id": eye})
    client.post("/api/project/design/template", json={"name": "clevis_bracket"})

    body = client.get("/api/project/validation").json()
    assert body["preflight_ready"] is True
    part = {f["title"]: f for f in body["part"]}
    assert part["IF-02 pin bore"]["status"] == "ok"
    assert part["Clearance to KO-01"]["status"] == "ok" and len(part["Clearance to KO-01"]["points"]) == 2
    assert body["counts"]["fail"] == 0


def test_ai_generate_gates(client, monkeypatch, tmp_path):
    from cad_assist.ai import generate as gen

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Never read the developer's real backend/.env in tests.
    monkeypatch.setattr(gen.load_env_key, "__defaults__", (tmp_path / "no.env",))
    status = client.get("/api/ai/status").json()
    assert status["key"] is False and status["job"] is None and status["model"]
    # Empty intent: pre-flight blocks generation before any API call.
    r = client.post("/api/ai/generate", json={})
    assert r.status_code == 422 and "Primary mount" in r.json()["detail"]

    top = _top_face(client)
    for kind in ("planar_mount", "bolt_pattern"):
        client.post("/api/project/intent/from-face", json={"kind": kind, "part": "frame_rib_FS412", "face_id": top})
    r = client.post("/api/ai/generate", json={})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.post("/api/ai/revert").status_code == 409


def test_topo_setup_api(client, monkeypatch):
    monkeypatch.setattr(app_module, "_setup_cache", None)
    assert client.get("/api/topo/setup").status_code == 422  # nothing defined yet
    status = client.get("/api/topo/status").json()
    assert status["job"] is None and status["result"] is None and status["default_budget"] > 0
    assert client.get("/api/topo/mesh?step=0").status_code == 404

    top = _top_face(client)
    for kind in ("planar_mount", "bolt_pattern"):
        client.post("/api/project/intent/from-face", json={"kind": kind, "part": "frame_rib_FS412", "face_id": top})
    rod = client.get("/api/project/parts/actuator_rod_end/mesh").json()
    pin = next(f["id"] for f in rod["faces"] if f["type"] == "cylinder" and abs(f["radius"] - 0.1885 * 25.4) < 1e-3)
    client.post("/api/project/intent/from-face", json={"kind": "bore", "part": "actuator_rod_end", "face_id": pin})
    client.post("/api/project/intent/design-space")
    lc = client.post("/api/project/intent/load-case", json={}).json()["created"]
    client.patch(f"/api/project/intent/{lc}", json={"force": [0, -2800, -7740]})

    body = client.get("/api/topo/setup?budget=8000").json()
    assert 0.85 * 8000 <= body["summary"]["elements"] <= 8000
    assert body["summary"]["supports"][0]["id"] == "IF-01"
    assert len(body["preview"]["solid"]) % 3 == 0 and body["preview"]["loads"][0]["id"] == lc

    assert client.patch("/api/project/intent/IF-00", json={"support": "fixed"}).status_code == 422
    assert client.patch("/api/project/intent/IF-01", json={"support": "free"}).status_code == 200
    r = client.get("/api/topo/setup?budget=8000")
    assert r.status_code == 422 and "Nothing holds" in r.json()["detail"]


def test_sketch_put_solves_and_rebuilds(client, monkeypatch):
    """Editing a sketch: the backend solves it (pins follow the intent), saves it and rebuilds the part."""
    monkeypatch.setenv("CAD_ASSIST_PROJECT", "bell_crank")
    app_module._project = None
    from cad_assist.features.model import Design, FeatureInstance
    from cad_assist.sketch.model import Sketch

    p = app_module.project()
    assert "pivot_fitting" in p.data.parts  # seeded from the example
    sq = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    sk = Sketch(id="S1", name="Square", entities=[{"id": f"p{i}", "kind": "point", "x": x, "y": y} for i, (x, y) in enumerate(sq)]
                + [{"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % 4}"} for i in range(4)]
                + [{"id": "pin", "kind": "point", "x": 5.0, "y": 5.0, "ref": "IF-00"}])  # off its pin: solving puts it back
    p.data.design = Design(sketches=[sk], features=[FeatureInstance(id="F1", kind="pad", params={"z_min": -5.0, "z_max": 5.0}, refs={"sketch": "S1"})])
    p.save()
    body = sk.model_dump(include={"name", "entities", "constraints"})
    body["constraints"] = [{"id": "k1", "kind": "distance", "refs": ["p0", "p1"], "value": 50.0},
                           {"id": "k2", "kind": "horizontal", "refs": ["l0"]}]
    r = client.put("/api/project/design/sketches/S1", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    pin = next(e for e in out["sketch"]["entities"] if e["id"] == "pin")
    assert (pin["x"], pin["y"]) == (0.0, 0.0)
    assert out["sketch"]["report"]["status"] == "solved"
    pts = {e["id"]: e for e in out["sketch"]["entities"] if e["kind"] == "point"}
    assert abs(pts["p1"]["x"] - pts["p0"]["x"]) == pytest.approx(50.0, abs=1e-6)
    assert out["features"][0]["error"] is None and out["volume"] > 0
    assert client.put("/api/project/design/sketches/S9", json=body).status_code == 404


def test_fem_mesh_setup_api(client):
    top = _top_face(client)
    for kind in ("planar_mount", "bolt_pattern"):
        client.post("/api/project/intent/from-face", json={"kind": kind, "part": "frame_rib_FS412", "face_id": top})
    mesh = client.get("/api/project/parts/actuator_rod_end/mesh").json()
    eye = min((f for f in mesh["faces"] if f["type"] == "cylinder"), key=lambda f: f["radius"])["id"]
    client.post("/api/project/intent/from-face", json={"kind": "bore", "part": "actuator_rod_end", "face_id": eye})
    assert client.post("/api/fem/setup/propose", json={"preset": "coarse"}).status_code == 422  # no part yet
    design = client.post("/api/project/design/template", json={"name": "clevis_bracket"}).json()

    body = client.get("/api/fem/setup").json()
    assert body["setup"]["source"] == "none" and "Pin bores" in body["default_practices"]
    body = client.post("/api/fem/setup/propose", json={"preset": "coarse"}).json()
    s = body["setup"]
    assert s["source"] == "rules" and s["size"] == 5.0 and [sp["label"] for sp in s["spheres"]] == ["IF-02 lug"]
    assert body["resolved"]["edges"] == [e["edge"] for e in s["edges"]]

    # Edit: keep the existing controls (they carry their refs), add an edge and a sphere by hand.
    line = next(e["id"] for e in design["mesh"]["edges"] if len(e["points"]) == 6)
    s["edges"].append({"edge": line, "n": 7, "label": "by hand"})
    s["spheres"].append({"center": [0, 0, 0], "radius": 5, "size": 1, "label": "mine"})
    body = client.put("/api/fem/setup", json=s).json()
    assert body["setup"]["source"] == "rules, edited"
    assert body["setup"]["edges"][-1]["n"] == 7 and body["setup"]["edges"][-1]["length"] > 0
    assert body["resolved"]["edges"][-1] == line
    # A face with a hole in it cannot be mapped.
    holed = next(f["id"] for f in design["mesh"]["faces"] if f["type"] == "plane" and f["area"] == max(
        g["area"] for g in design["mesh"]["faces"] if g["type"] == "plane"))
    r = client.put("/api/fem/setup", json=body["setup"] | {"faces": [{"face": holed}]})
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert "cannot be mapped" in r.json()["detail"]
    assert client.get("/api/fem/mesh").status_code == 404  # nothing meshed yet
