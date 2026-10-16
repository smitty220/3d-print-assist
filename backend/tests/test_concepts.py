"""Concepts: the AI shows what it understood before building, and its changes wait for the user's answer."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.concepts import ConceptError, Concepts, check_svg

SVG = '<svg viewBox="0 0 100 50"><rect x="10" y="10" width="80" height="30"/><text x="50" y="30">80 mm</text></svg>'
AI = {"X-Actor": "claude-code"}
BOX = {"op": "add_feature", "kind": "plate"}


def test_drawings_must_be_one_svg_document():
    assert check_svg(SVG).startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert check_svg('<?xml version="1.0"?>\n' + SVG).startswith("<svg")
    with pytest.raises(ConceptError, match="one <svg>"):
        check_svg("<html><svg></svg></html>")


def test_concepts_persist_and_a_new_one_replaces_a_waiting_one(tmp_path):
    cs = Concepts(tmp_path / "concepts")
    a = cs.propose("Spinner, 0 to 9", SVG, "Two dials", ["Lato Bold digits"], ["Thumb side?"])
    b = cs.propose("Spinner, 0 to 9 (v2)", SVG)
    assert (a.id, b.id) == ("K1", "K2") and cs.get("K1").status == "revise" and cs.pending().id == "K2"
    cs.respond("K2", "approve", "go ahead")
    with pytest.raises(ConceptError, match="already answered"):
        cs.respond("K2", "revise")
    again = Concepts(tmp_path / "concepts")
    assert [(c.id, c.status) for c in again.items] == [("K1", "revise"), ("K2", "approved")]
    assert again.get("K1").assumptions == ["Lato Bold digits"] and again.get("K2").note == "go ahead"


def test_ai_changes_wait_for_the_users_answer():
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Concept flow"})
    r = c.post("/api/concepts", json={"title": "A plate", "svg": SVG, "summary": "60 x 40 x 5",
                                      "assumptions": ["PLA"], "questions": ["Rounded corners?"]}, headers=AI)
    assert r.status_code == 200 and r.json()["id"] == "K1" and r.json()["status"] == "pending"
    refused = c.post("/api/ops", json={"ops": [BOX]}, headers=AI)
    assert refused.status_code == 409 and "K1" in refused.text
    assert c.post("/api/ops", json={"ops": [BOX]}).status_code == 200  # the user is never held up
    listed = c.get("/api/concepts").json()
    assert listed[0]["svg"].startswith("<svg") and "svg" not in c.get("/api/concepts", params={"drawings": False}).json()[0]
    assert c.post("/api/concepts/K1/answer", json={"decision": "approve", "note": "sharp corners"}).status_code == 200
    assert c.post("/api/ops", json={"ops": [BOX]}, headers=AI).status_code == 200
    log = [a["text"] for a in c.get("/api/activity").json()]
    assert any("Concept K1" in t and "waiting" in t for t in log) and any("K1 approved: sharp corners" in t for t in log)
    assert c.post("/api/concepts", json={"title": "x", "svg": "not svg"}, headers=AI).status_code == 422


def test_mcp_tools_propose_and_wait():
    from print_assist_mcp import server

    c = TestClient(app_module.app)
    server.set_client(c)
    try:
        c.post("/api/projects", json={"name": "Concept tools"})
        made = json.loads(server.propose_concept("Hook", SVG, "A wall hook", ["2 screws"], ["Load?"]))
        assert made["id"] == "K1"

        def answer():
            time.sleep(0.5)
            c.post("/api/concepts/K1/answer", json={"decision": "revise", "note": "make it 20 mm deeper"})

        threading.Thread(target=answer).start()
        got = json.loads(server.wait_for_concept("K1", timeout_s=30))
        assert got == {"id": "K1", "title": "Hook", "status": "revise", "note": "make it 20 mm deeper"}
        server.propose_concept("Hook v2", SVG)
        assert json.loads(server.wait_for_concept("K2", timeout_s=5))["status"] == "pending"  # timed out, still waiting
    finally:
        server.set_client(None)


def test_the_ai_can_look_at_a_picture_before_and_after_proposing():
    from print_assist.concepts import picture
    from print_assist.geometry.render import png_size

    png = picture(SVG, 400)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png_size(png)[0] == 400
    sneaky = SVG.replace("</svg>", '<image href="/etc/hosts" width="10" height="10"/></svg>')
    assert png_size(picture(sneaky, 400))[0] == 400  # other files are never read into the picture
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Concept pictures"})
    r = c.post("/api/concepts/picture", json={"svg": SVG, "width": 300})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert c.post("/api/concepts/picture", json={"svg": "nope"}).status_code == 422
    c.post("/api/concepts", json={"title": "A plate", "svg": SVG}, headers=AI)
    assert c.get("/api/concepts/K1/picture", params={"width": 500}).headers["content-type"] == "image/png"


def test_mcp_look_at_concept_returns_an_image():
    from print_assist_mcp import server
    from mcp.server.mcpserver import Image

    c = TestClient(app_module.app)
    server.set_client(c)
    try:
        c.post("/api/projects", json={"name": "Concept look"})
        assert isinstance(server.look_at_concept(svg=SVG, width=300), Image)
        server.propose_concept("Hook", SVG)
        assert isinstance(server.look_at_concept(id="K1"), Image)
    finally:
        server.set_client(None)
