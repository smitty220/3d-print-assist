"""The MCP server's tools, run against the backend in-process (no network, no Claude)."""

import json

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist_mcp import server


@pytest.fixture(autouse=True)
def wired():
    server.set_client(TestClient(app_module.app))
    yield
    server.set_client(None)


def test_guide_and_catalog():
    assert "apply_changes" in server.get_guide() and "millimetres" in server.get_guide()
    kinds = {f["kind"] for f in json.loads(server.get_feature_catalog())}
    assert {"extrude", "revolve", "shell", "text", "fillet"} <= kinds


def test_claude_designs_a_box_and_sees_it():
    ops = [
        {"op": "set_parameter", "name": "w", "expr": "40"},
        {"op": "add_sketch", "name": "Base", "entities": [
            {"id": "p0", "kind": "point", "x": 0, "y": 0}, {"id": "p1", "kind": "point", "x": 40, "y": 0},
            {"id": "p2", "kind": "point", "x": 40, "y": 30}, {"id": "p3", "kind": "point", "x": 0, "y": 30},
            {"id": "l0", "kind": "line", "p1": "p0", "p2": "p1"}, {"id": "l1", "kind": "line", "p1": "p1", "p2": "p2"},
            {"id": "l2", "kind": "line", "p1": "p2", "p2": "p3"}, {"id": "l3", "kind": "line", "p1": "p3", "p2": "p0"}]},
        {"op": "add_feature", "kind": "extrude", "refs": {"sketch": "S1"}, "params": {"distance": "w / 2"}},
    ]
    out = json.loads(server.apply_changes(ops, "a 40 x 30 box, half as tall as it is wide"))
    assert out["made"] == ["w", "S1", "F1"] and out["tree"]["features"][0]["status"] == "ok"
    img = server.render_preview("iso", 256)
    assert img.data[:8] == b"\x89PNG\r\n\x1a\n"
    model = json.loads(server.get_model())
    assert model["tree"]["features"][0]["inputs"] == ["w"] and model["sketches"][0]["id"] == "S1"
    faces = json.loads(server.list_faces("B1"))
    top = next(f for f in faces if f["name"] == "F1.end")
    pick = json.loads(server.pick_face("B1", top["index"]))
    out = json.loads(server.apply_changes([{"op": "add_feature", "kind": "shell", "params": {"thickness": 2}, "faces": [pick]}], "open box"))
    assert out["tree"]["features"][1]["status"] == "ok"
    checks = json.loads(server.run_checks())
    assert any(f["title"] == "Fits the bed" and f["status"] == "ok" for f in checks["part"])
    saved = json.loads(server.export("3mf"))
    assert saved["saved"].endswith(".3mf") and saved["bytes"] > 0


def test_refusals_come_back_as_errors_not_exceptions():
    server.apply_changes([{"op": "add_feature", "kind": "plate"}], "plate")
    out = json.loads(server.apply_changes([{"op": "edit_feature", "id": "F1", "params": {"corner_radius": 500}}], "too round"))
    assert out["status"] == 409 and "F1" in out["error"]
    out = json.loads(server.apply_changes([{"op": "bogus"}], "?"))
    assert out["status"] == 422
    client = TestClient(app_module.app)
    client.patch("/api/features/F1", json={"locked": True})  # the user locks it
    out = json.loads(server.apply_changes([{"op": "edit_feature", "id": "F1", "params": {"thickness": 9}}], "thicker"))
    assert "locked" in out["error"]
    assert json.loads(server.revert_last_ai_change())["reverted"] == "AI: plate"
