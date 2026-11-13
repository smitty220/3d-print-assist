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
    assert client.put("/api/settings", json={"slicer": "prusaslicer"}).json()["slicer"] == "prusaslicer"
    assert client.put("/api/settings", json={"printer": "nope"}).status_code == 422
    assert client.put("/api/settings", json={"slicer": "nope"}).status_code == 422
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


def test_export_only_when_up_to_date(client, tmp_path):
    from build123d import import_step

    from print_assist.export.bambu import read_project

    client.post("/api/features", json={"kind": "plate", "params": {"corner_radius": 0.0}})
    assert client.get("/api/export/3mf").status_code == 409  # not updated yet
    client.post("/api/update")
    r = client.get("/api/export/3mf")
    assert r.status_code == 200
    out = tmp_path / "designs" / "untitled" / "exports" / "untitled.3mf"
    assert out.exists()
    (part,), _, _ = read_project(out)
    assert part.volume == pytest.approx(60 * 40 * 5, rel=1e-6) and part.open_edges() == 0
    lo, hi = part.bounds()
    assert lo[2] == pytest.approx(0, abs=1e-6) and (lo[0] + hi[0]) / 2 == pytest.approx(128)  # on the A1 plate, centred
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


def test_print_layout_is_worked_out_on_request(client):
    import base64

    import numpy as np

    client.post("/api/features", json={"kind": "plate", "params": {"length": 60, "width": 40, "thickness": 10,
                                                                    "corner_radius": 0.0}})
    tree = client.post("/api/update").json()
    assert tree["print_layout"] == {"items": 0, "plates": 0}
    rep = client.get("/api/layout").json()
    assert rep["plates"] == [] and rep["unplaced"][0]["size"] == [60, 40, 10]
    r = client.post("/api/ops", json={"ops": [
        {"op": "set_plate", "plate": 2, "name": "Second", "tower": [200, 200]},
        {"op": "set_print_item", "name": "Plate on its side", "bodies": ["B1"], "plate": 2, "rotate": [90, 0, 0],
         "x": 10, "y": 20}]})
    assert r.json()["tree"]["print_layout"] == {"items": 1, "plates": 2}
    rep = client.get("/api/layout").json()
    assert [p["name"] for p in rep["plates"]] == ["Plate 1", "Second"] and rep["problems"] == []
    # 90 degrees about X stands the 60 x 40 x 10 plate on its long edge: 60 x 10 x 40, corner at (10, 20).
    assert rep["plates"][1]["objects"][0]["box"] == [10, 20, 0, 70, 30, 40]
    m = client.get("/api/layout/plates/2/mesh").json()
    assert m["bed"] == [256, 256] and m["tower"] == [200, 200] and m["tower_size"] == 35
    (part,) = m["objects"][0]["parts"]
    pts = np.frombuffer(base64.b64decode(part["positions"]), dtype="<f4").reshape(-1, 3)
    assert pts.min(0) == pytest.approx([10, 20, 0], abs=1e-4) and pts.max(0) == pytest.approx([70, 30, 40], abs=1e-4)
    assert client.get("/api/layout/plates/1/mesh").json()["objects"] == []


def test_upload_reference_image(client):
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    r = client.post("/api/images/upload", files={"file": ("calipers.png", png, "image/png")})
    assert r.status_code == 200
    img = r.json()
    assert img["id"] == "I1" and img["file"] == "I1.png" and img["model"] == "upload"
    assert client.get("/api/images/I1").content == png
    assert client.post("/api/images/upload", files={"file": ("bad.txt", b"hi", "text/plain")}).status_code == 422
    assert client.delete("/api/images/I1").json() == {"deleted": "I1"}
    assert client.get("/api/images/I1").status_code == 404
    assert client.delete("/api/images/I1").status_code == 404
