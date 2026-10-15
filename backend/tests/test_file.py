"""Traditional file handling: unsaved edits, Save / Save As / Revert, undo and redo, recovery, versions, Close,
Rename and Delete."""

import json

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module


@pytest.fixture
def c():
    return TestClient(app_module.app)


def plate(c, t=5.0):
    return c.post("/api/features", json={"kind": "plate", "params": {"thickness": t}})


def on_disk(tmp_path, slug):
    return json.loads((tmp_path / "designs" / slug / "project.json").read_text(encoding="utf-8"))["design"]


def test_edits_stay_unsaved_until_save(c, tmp_path):
    c.post("/api/projects", json={"name": "Box"})
    plate(c)
    f = c.get("/api/file").json()
    assert f["dirty"] and f["undo"] == "Add plate"
    assert on_disk(tmp_path, "box")["features"] == []  # nothing written yet
    assert (tmp_path / "designs" / "box" / ".recovery.json").exists()  # but a recovery copy is
    c.post("/api/update")
    assert c.post("/api/file/save", json={"message": "First plate"}).json()["dirty"] is False
    assert len(on_disk(tmp_path, "box")["features"]) == 1
    assert not (tmp_path / "designs" / "box" / ".recovery.json").exists()
    assert c.get("/api/projects/box/thumbnail").headers["content-type"] == "image/png"
    assert [h["message"] for h in c.get("/api/history").json()] == ["First plate", "Create design"]


def test_undo_redo_and_revert(c):
    c.post("/api/projects", json={"name": "Box"})
    plate(c)
    c.patch("/api/features/F1", json={"params": {"thickness": 8.0}})
    assert c.post("/api/undo").json()["undone"] == "Edit F1"
    assert c.get("/api/tree").json()["features"][0]["params"]["thickness"] == 5.0
    assert c.post("/api/redo").json()["redone"] == "Edit F1"
    assert c.get("/api/tree").json()["features"][0]["params"]["thickness"] == 8.0
    c.post("/api/file/save", json={})
    c.delete("/api/features/F1")
    assert c.get("/api/file").json()["dirty"]
    c.post("/api/file/revert")
    f = c.get("/api/file").json()
    assert not f["dirty"] and f["undo"] == "Revert to saved"  # revert itself can be undone
    assert len(c.get("/api/tree").json()["features"]) == 1
    assert c.post("/api/redo").status_code == 409


def test_recovery_after_a_crash(c, tmp_path):
    c.post("/api/projects", json={"name": "Box"})
    plate(c)  # unsaved, then the app stops
    app_module.reset()
    c.post("/api/projects/box/open")
    f = c.get("/api/file").json()
    assert f["recovery"] and not f["dirty"] and c.get("/api/tree").json()["features"] == []
    c.post("/api/file/recover", json={"keep": True})
    f = c.get("/api/file").json()
    assert not f["recovery"] and f["dirty"] and len(c.get("/api/tree").json()["features"]) == 1


def test_save_as_rename_versions_close_and_delete(c, tmp_path):
    c.post("/api/projects", json={"name": "Deck Box"})
    plate(c)
    c.post("/api/file/save", json={"message": "v1"})
    v1 = c.get("/api/history").json()[0]["id"]
    c.patch("/api/features/F1", json={"params": {"thickness": 9.0}})
    f = c.post("/api/file/save-as", json={"name": "Deck Box 60"}).json()
    assert f["slug"] == "deck-box-60" and not f["dirty"]
    assert on_disk(tmp_path, "deck-box-60")["features"][0]["params"]["thickness"] == 9.0
    assert on_disk(tmp_path, "deck-box")["features"][0]["params"]["thickness"] == 5.0  # the original keeps its save
    c.post("/api/projects/deck-box/open")
    c.post(f"/api/history/{v1}/restore")
    assert c.get("/api/tree").json()["features"][0]["params"]["thickness"] == 5.0
    f = c.post("/api/file/rename", json={"name": "Deck Box 100"}).json()
    assert f["slug"] == "deck-box-100" and (tmp_path / "designs" / "deck-box-100").is_dir()
    assert c.post("/api/file/close").json() == {"open": False}
    assert c.get("/api/file").json() == {"open": False}
    assert c.get("/api/tree").status_code == 409
    with c.stream("GET", "/api/events", params={"once": True}) as r:
        assert '"slug": null' in next(r.iter_lines())


def test_delete_moves_to_the_trash(c, tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    c.post("/api/projects", json={"name": "Scrap"})
    r = c.delete("/api/projects/scrap").json()
    assert "/.Trash/scrap" in r["trashed"] and not (tmp_path / "designs" / "scrap").exists()
    assert c.get("/api/file").json() == {"open": False}
