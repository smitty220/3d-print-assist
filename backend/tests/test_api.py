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
    assert [h["message"] for h in client.get("/api/project/history").json()] == ["Create design"]
    assert client.post("/api/projects/untitled/open").json()["slug"] == "untitled"
    assert client.post("/api/projects/../etc/open").status_code == 404
    assert client.post("/api/projects", json={"name": "!!!"}).status_code == 422
    assert client.get("/api/printers").json()[0]["bed"] == [256, 256, 256]


def test_design_api(client):
    assert client.get("/api/project/design").json()["features"] == []
    assert {f["kind"] for f in client.get("/api/features/catalog").json()} >= {"plate", "pad", "fillet", "chamfer"}

    body = client.post("/api/project/design/features", json={"kind": "plate", "params": {"corner_radius": 0.0}}).json()
    assert body["created"] == "F1" and body["mesh"] and len(body["face_names"]) == len(body["mesh"]["faces"])
    assert body["volume"] == pytest.approx(60 * 40 * 5)
    body = client.patch("/api/project/design/features/F1", json={"params": {"thickness": 10.0}, "locked": True}).json()
    assert body["features"][0]["locked"] is True and body["volume"] == pytest.approx(60 * 40 * 10)
    assert client.patch("/api/project/design/features/F1", json={"params": {"bogus": 1}}).status_code == 422
    assert client.post("/api/project/design/features", json={"kind": "nope"}).status_code == 422

    # A fillet by picking edges on the part: it needs edges first; a pick adds one, picking it again takes it off.
    body = client.post("/api/project/design/features", json={"kind": "fillet", "params": {"radius": 2.0}}).json()
    assert body["created"] == "F2" and "pick the edges" in body["features"][-1]["error"]
    v0 = body["volume"]
    r = client.post("/api/project/design/features/F2/edges", json={"edge": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    f2 = body["features"][-1]
    assert f2["error"] is None and len(f2["edges"]) == 1 and len(f2["edges"][0]["world"]) == 9
    assert body["volume"] < v0
    assert client.post("/api/project/design/features/F1/edges", json={"edge": 0}).status_code == 422
    body = client.patch("/api/project/design/features/F2", json={"edges": []}).json()
    assert body["features"][-1]["edges"] == [] and body["volume"] == pytest.approx(v0)
    body = client.delete("/api/project/design/features/F2").json()
    assert [f["id"] for f in body["features"]] == ["F1"]


def test_validation_api(client):
    assert client.get("/api/project/validation").json()["part"] == []
    client.post("/api/project/design/features", json={"kind": "plate"})
    part = {f["title"]: f for f in client.get("/api/project/validation").json()["part"]}
    assert part["Fits the bed"]["status"] == "ok" and part["Minimum wall"]["status"] == "ok"


def test_ai_generate_gates(client, monkeypatch, tmp_path):
    from print_assist.ai import generate as gen

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Never read the developer's real backend/.env in tests.
    monkeypatch.setattr(gen.load_env_key, "__defaults__", (tmp_path / "no.env",))
    status = client.get("/api/ai/status").json()
    assert status["key"] is False and status["job"] is None and status["model"]
    r = client.post("/api/ai/generate", json={})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.post("/api/ai/revert").status_code == 409


def test_sketch_put_solves_and_rebuilds(client):
    """Editing a sketch: the backend solves it, saves it and rebuilds the part."""
    from print_assist.features.model import Design, FeatureInstance
    from print_assist.sketch.model import Sketch

    p = app_module.project()
    sq = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    sk = Sketch(id="S1", name="Square", entities=[{"id": f"p{i}", "kind": "point", "x": x, "y": y} for i, (x, y) in enumerate(sq)]
                + [{"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % 4}"} for i in range(4)])
    p.data.design = Design(sketches=[sk], features=[FeatureInstance(id="F1", kind="pad", params={"z_min": 0.0, "z_max": 5.0}, refs={"sketch": "S1"})])
    p.save()
    body = sk.model_dump(include={"name", "entities", "constraints"})
    body["constraints"] = [{"id": "k1", "kind": "distance", "refs": ["p0", "p1"], "value": 50.0},
                           {"id": "k2", "kind": "horizontal", "refs": ["l0"]}]
    r = client.put("/api/project/design/sketches/S1", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["sketch"]["report"]["status"] == "solved"
    pts = {e["id"]: e for e in out["sketch"]["entities"] if e["kind"] == "point"}
    assert abs(pts["p1"]["x"] - pts["p0"]["x"]) == pytest.approx(50.0, abs=1e-6)
    assert out["features"][0]["error"] is None and out["volume"] > 0
    assert client.put("/api/project/design/sketches/S9", json=body).status_code == 404
