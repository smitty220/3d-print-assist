import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_designs_are_git_tracked_folders(client, tmp_path):
    first = client.get("/api/project").json()  # nothing yet: an Untitled design is made
    assert first["slug"] == "untitled" and first["design"]["features"] == []
    body = client.post("/api/projects", json={"name": "Twisted Hex Vase"}).json()
    assert body["slug"] == "twisted-hex-vase"
    assert [p["slug"] for p in client.get("/api/projects").json()] == ["twisted-hex-vase", "untitled"]
    root = tmp_path / "designs" / "twisted-hex-vase"
    assert (root / ".git").exists() and (root / "fonts").is_dir() and (root / "exports").is_dir()
    assert [h["message"] for h in client.get("/api/history").json()] == ["Create design"]
    assert client.post("/api/projects/untitled/open").json()["slug"] == "untitled"
    assert client.post("/api/projects/../etc/open").status_code == 404
    assert client.post("/api/projects", json={"name": "!!!"}).status_code == 422
    assert client.get("/api/printers").json()[0]["bed"] == [256, 256, 256]


def test_parameters_features_update_and_part(client):
    assert {f["kind"] for f in client.get("/api/features/catalog").json()} >= {"plate", "pad", "fillet", "chamfer"}
    client.post("/api/parameters", json={"name": "t", "expr": "5"})
    body = client.post("/api/features", json={"kind": "plate", "params": {"corner_radius": 0.0}, "exprs": {"thickness": "t * 2"}}).json()
    assert body["made"] == ["F1"]
    tree = body["tree"]
    assert tree["pending_count"] == 1 and tree["features"][0]["status"] == "stale"
    assert client.get("/api/part").json()["bodies"] == []
    tree = client.post("/api/update").json()
    assert tree["pending_count"] == 0 and tree["features"][0]["status"] == "ok" and tree["exportable"]
    part = client.get("/api/part").json()
    (b,) = part["bodies"]
    assert part["volume"] == pytest.approx(60 * 40 * 10) and len(b["mesh"]["face_names"]) == len(b["mesh"]["faces"])
    assert client.post("/api/expr/eval", json={"expr": "t * 3"}).json() == {"value": 15.0}
    assert "unknown name" in client.post("/api/expr/eval", json={"expr": "q"}).json()["error"]
    assert client.post("/api/features", json={"kind": "nope"}).status_code == 422
    r = client.patch("/api/features/F1", json={"params": {"thickness": 0.5}}, headers={"X-Actor": "claude-code"})
    assert r.status_code == 200 and r.json()["tree"]["features"][0]["status"] == "warn"  # built with a failing check

    # Picking: an edge for a fillet, a face back as a pick record.
    client.post("/api/features", json={"kind": "fillet", "params": {"radius": 0.2}})
    client.post("/api/update")
    for i in range(12):
        r = client.post("/api/features/F2/edges", json={"body": "B1", "index": i})
        if r.status_code == 200:
            break
    tree = client.post("/api/update").json()
    f2 = tree["features"][1]
    assert len(f2["edges"]) == 1 and f2["status"] == "ok", f2
    face = client.get("/api/pick/face", params={"body": "B1", "face": 0}).json()
    assert set(face) == {"name", "at", "normal", "area"}
    assert client.get("/api/pick/face", params={"body": "B9", "face": 0}).status_code == 422


def test_settings_and_events(client):
    assert client.put("/api/settings", json={"update_mode": "auto"}).json()["update_mode"] == "auto"
    assert client.put("/api/settings", json={"printer": "nope"}).status_code == 422
    with client.stream("GET", "/api/events", params={"once": True}) as r:
        line = next(r.iter_lines())
    assert line.startswith("data: ") and '"rev"' in line


def test_ops_endpoint_and_refusals(client):
    r = client.post("/api/ops", json={"ops": [{"op": "set_parameter", "name": "L", "expr": "80"},
                                              {"op": "add_feature", "kind": "plate", "exprs": {"length": "L"}}], "message": "start"})
    assert r.status_code == 200 and r.json()["made"] == ["L", "F1"]
    client.post("/api/update")
    r = client.post("/api/ops", json={"ops": [{"op": "set_parameter", "name": "L", "expr": "2000"}]}, headers={"X-Actor": "claude-code"})
    assert r.status_code == 409 and "F1" in r.json()["detail"]  # would fail: the AI's change is not kept
    assert client.post("/api/ops", json={"ops": [{"op": "nope"}]}).status_code == 422
    assert client.post("/api/ops", json={"ops": [{"op": "set_parameter", "name": "L", "expr": "90"}], "base_rev": 0}).status_code == 409
    acts = client.get("/api/activity").json()
    assert acts[-1]["ok"] is False and acts[-1]["actor"] == "claude-code"


def test_sketch_put_solves_and_rebuilds(client):
    sq = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    ents = [{"id": f"p{i}", "kind": "point", "x": x, "y": y} for i, (x, y) in enumerate(sq)] + \
           [{"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % 4}"} for i in range(4)]
    r = client.post("/api/sketches", json={"name": "Square", "entities": ents})
    assert r.status_code == 200 and r.json()["made"] == ["S1"]
    client.post("/api/parameters", json={"name": "w", "expr": "50"})
    client.post("/api/features", json={"kind": "pad", "params": {"z_min": 0.0, "z_max": 5.0}, "refs": {"sketch": "S1"}})
    cons = [{"id": "k1", "kind": "distance", "refs": ["p0", "p1"], "expr": "w"}, {"id": "k2", "kind": "horizontal", "refs": ["l0"]}]
    r = client.put("/api/sketches/S1", json={"constraints": cons})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["sketch"]["report"]["status"] == "solved"
    pts = {e["id"]: e for e in out["sketch"]["entities"] if e["kind"] == "point"}
    assert abs(pts["p1"]["x"] - pts["p0"]["x"]) == pytest.approx(50.0, abs=1e-6)
    tree = client.post("/api/update").json()
    assert tree["features"][0]["status"] == "ok" and tree["sketches"][0]["inputs"] == ["w"]
    assert client.put("/api/sketches/S9", json={"constraints": cons}).status_code == 422
    assert client.delete("/api/sketches/S1").status_code == 422  # F1 uses it


def test_ai_generate_gates(client, monkeypatch, tmp_path):
    from print_assist.ai import generate as gen

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(gen.load_env_key, "__defaults__", (tmp_path / "no.env",))
    status = client.get("/api/ai/status").json()
    assert status["key"] is False and status["job"] is None and status["model"]
    r = client.post("/api/ai/generate", json={})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.post("/api/ai/revert").status_code == 409


def test_export_only_when_up_to_date(client, tmp_path):
    from build123d import Mesher, import_step

    client.post("/api/features", json={"kind": "plate", "params": {"corner_radius": 0.0}})
    assert client.get("/api/export/3mf").status_code == 409  # not updated yet
    client.post("/api/update")
    r = client.get("/api/export/3mf")
    assert r.status_code == 200
    out = tmp_path / "designs" / "untitled" / "exports" / "untitled.3mf"
    assert out.exists()
    (shape,) = Mesher().read(str(out))
    assert shape.volume == pytest.approx(60 * 40 * 5, rel=1e-6)
    bb = shape.bounding_box()
    assert bb.min.Z == pytest.approx(0) and (bb.min.X + bb.max.X) / 2 == pytest.approx(128)  # on the A1 plate, centred
    assert client.get("/api/export/step").status_code == 200
    assert import_step(str(out.with_suffix(".step"))).volume == pytest.approx(60 * 40 * 5, rel=1e-6)
    assert client.get("/api/export/stl").status_code == 200
    assert client.get("/api/export/obj").status_code == 404
    fonts = {f["name"]: f for f in client.get("/api/fonts").json()}
    assert "Lato" in fonts and fonts["Lato"]["license"].startswith("SIL")


def test_render_faces_and_revert_an_ai_change(client):
    from print_assist.geometry.render import png_size

    client.post("/api/ops", json={"ops": [{"op": "add_feature", "kind": "plate"}]}, headers={"X-Actor": "claude-code"})
    r = client.get("/api/render", params={"view": "iso", "size": 200})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and png_size(r.content) == (200, 200)
    assert client.get("/api/render", params={"view": "sideways"}).status_code == 422
    faces = client.get("/api/faces").json()
    top = max((f for f in faces if f["type"] == "plane"), key=lambda f: f["center"][2])
    assert top["name"] == "F1.top" and top["normal"] == [0, 0, 1]
    client.post("/api/ops", json={"ops": [{"op": "edit_feature", "id": "F1", "params": {"thickness": 8.0}}],
                                  "message": "thicker"}, headers={"X-Actor": "claude-code"})
    assert client.get("/api/part").json()["volume"] > 60 * 40 * 7
    body = client.post("/api/revert-ai").json()
    assert body["reverted"] == "AI: thicker"
    assert client.get("/api/part").json()["volume"] < 60 * 40 * 6
