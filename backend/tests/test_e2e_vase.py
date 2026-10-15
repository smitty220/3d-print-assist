"""End to end, through the HTTP API (the same path the MCP server takes): the twisted hex vase.

Parameters drive a constrained hexagon sketch and a twisted, tapered extrude; the vase is shelled open at the top and
its outer rim filleted. Then a parameter changes in manual mode (out of date, export blocked), and an update
rebuilds exactly the features downstream of it.
"""

import math

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module


def hexagon_sketch(r=40.0):
    ents = [{"id": "c", "kind": "point", "x": 0, "y": 0, "fixed": True},
            {"id": "k", "kind": "circle", "c": "c", "r": r, "construction": True}]
    cons = [{"id": "d", "kind": "diameter", "refs": ["k"], "expr": "af * 2 / sqrt(3)"}]
    for i in range(6):
        a = math.radians(60 * i)
        ents.append({"id": f"p{i}", "kind": "point", "x": r * math.cos(a), "y": r * math.sin(a)})
    for i in range(6):
        ents.append({"id": f"l{i}", "kind": "line", "p1": f"p{i}", "p2": f"p{(i + 1) % 6}"})
        cons.append({"id": f"on{i}", "kind": "point_on", "refs": [f"p{i}", "k"]})
    cons += [{"id": f"eq{i}", "kind": "equal", "refs": ["l0", f"l{i}"]} for i in range(1, 6)]
    cons.append({"id": "h", "kind": "horizontal", "refs": ["l1"]})
    return ents, cons


def test_twisted_hex_vase_end_to_end(tmp_path):
    c = TestClient(app_module.app)
    assert c.post("/api/projects", json={"name": "Twisted Hex Vase"}).status_code == 200
    for name, expr, q in [("af", "60", "length"), ("h", "120", "length"), ("twist", "60", "angle"),
                          ("scale", "0.8", "ratio"), ("wall", "1.6", "length"), ("rim_r", "0.6", "length")]:
        assert c.post("/api/parameters", json={"name": name, "expr": expr, "quantity": q}).status_code == 200
    ents, cons = hexagon_sketch()
    r = c.post("/api/sketches", json={"name": "Base hexagon", "entities": ents, "constraints": cons})
    assert r.status_code == 200, r.text
    c.post("/api/features", json={"kind": "extrude", "refs": {"sketch": "S1"},
                                  "params": {"distance": "h", "twist": "twist", "end_scale": "scale"}})
    c.post("/api/update")
    top = next(f for f in c.get("/api/faces").json() if f["name"] == "F1.end")
    pick = c.get("/api/pick/face", params={"body": "B1", "face": top["index"]}).json()
    c.post("/api/features", json={"kind": "shell", "params": {"thickness": "wall"}, "faces": [pick]})
    c.post("/api/features", json={"kind": "fillet", "params": {"radius": "rim_r"},
                                  "refs": {"faces_a": "F1.end", "faces_b": "F1.side"}})
    tree = c.post("/api/update").json()
    assert [f["status"] for f in tree["features"]] == ["ok", "ok", "ok"], tree["features"]
    assert tree["sketches"][0]["status"] == "ok" and tree["sketches"][0]["dof"] == 0

    part = c.get("/api/part").json()
    a = 3 * math.sqrt(3) / 2 * (60 / math.sqrt(3)) ** 2  # hexagon 60 across flats
    outer = a * 120 * (1 + 0.8 + 0.64) / 3  # twisted frustum
    # A thin shell: about wall × (side area + bottom). Check the order of magnitude, then the wall itself.
    assert 0.05 * outer < part["volume"] < 0.15 * outer
    bb = part["bodies"][0]["mesh"]["bbox_max"][2] - part["bodies"][0]["mesh"]["bbox_min"][2]
    assert bb == pytest.approx(120, abs=1e-3)
    checks = {f["title"]: f for f in c.get("/api/checks").json()["part"]}
    assert checks["Minimum wall"]["value"] >= 1.5 and checks["Fits the bed"]["status"] == "ok"

    # Manual mode: change the twist; the extrude and everything after it need an update; export waits.
    eng = app_module.ws().engine
    builds = eng.builds
    tree = c.post("/api/parameters", json={"name": "twist", "expr": "90"}).json()["tree"]
    assert tree["pending_count"] == 3 and {f["status"] for f in tree["features"]} == {"stale"}
    assert tree["sketches"][0]["status"] == "ok"  # the sketch does not use the twist
    assert c.get("/api/export/3mf").status_code == 409
    tree = c.post("/api/update").json()
    assert tree["pending_count"] == 0 and eng.builds == builds + 3

    r = c.get("/api/export/3mf")
    assert r.status_code == 200
    from print_assist.export.bambu import read_project

    (part,), _, _ = read_project(r.headers["x-saved-path"])  # a Bambu Studio project for the A1
    assert part.open_edges() == 0 and part.volume == pytest.approx(c.get("/api/part").json()["volume"], rel=0.01)
    png = c.get("/api/render", params={"view": "iso", "size": 320})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert c.get("/api/file").json()["dirty"] is True
    c.post("/api/file/save", json={"message": "First vase"})
    hist = [h["message"] for h in c.get("/api/history").json()]
    assert hist == ["First vase", "Create design"]
