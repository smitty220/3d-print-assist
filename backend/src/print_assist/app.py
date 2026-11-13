"""3D Print Assist API.

Run with:  uv run uvicorn print_assist.app:app --reload

Every change to a design is a list of ops (`build/ops.py`) applied by the open design's Workspace; the endpoints
below are thin wrappers. The `X-Actor` header names who is changing it ("user" for the app, "claude-code" for the
MCP server, "claude" for the in-app chat); changes by anyone but the user are built and checked before they are kept.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .ai import client as ai
from .build.ops import USER, OpError
from .build.workspace import ChangeRefused, Workspace
from .features.context import FeatureError
from .features.design import describe_edge, describe_face
from .features.library import FEATURES
from .params.expr import ExprError, evaluate
from .params.graph import resolve
from .printers import profiles
from .project import Projects

app = FastAPI(title="3D Print Assist", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5183", "http://127.0.0.1:5183"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_projects: Projects | None = None
_ws: Workspace | None = None
_closed = False  # File › Close: no design open until one is opened or created


def projects() -> Projects:
    global _projects
    if _projects is None:
        _projects = Projects()
    return _projects


def ws() -> Workspace:
    """The open design. At start-up the most recently changed one (or a new one) opens, as an editor reopens its last
    document; after File › Close there is none until one is opened."""
    global _ws
    if _ws is None:
        if _closed:
            raise HTTPException(409, "No design is open; open or create one")
        listed = projects().list()
        p = projects().open(listed[0]["slug"]) if listed else projects().create("Untitled")
        _ws = Workspace(p)
    return _ws


def project():
    return ws().project


def reset(root=None) -> None:
    """Forget the open design (tests point the registry at a temporary folder)."""
    global _projects, _ws, _closed
    _projects = Projects(root) if root is not None else None
    _ws = None
    _closed = False


def _switch(w: Workspace | None) -> None:
    global _ws, _closed
    _ws, _closed = w, w is None


def _change(ops: list[dict], actor: str, message: str = "", base_rev: int | None = None) -> dict:
    try:
        made = ws().change(ops, actor=actor or USER, message=message, base_rev=base_rev)
    except OpError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"made": made, "tree": ws().tree()}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------- designs


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return projects().list()


class NewProject(BaseModel):
    name: str


@app.post("/api/projects")
def create_project(body: NewProject) -> dict:
    try:
        _switch(Workspace(projects().create(body.name)))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _ws.project.summary()


@app.post("/api/projects/{slug}/open")
def open_project(slug: str) -> dict:
    """Open a design. Opening the one already open keeps its live state: reloading it from disk would turn its unsaved
    changes (for example an AI build in progress) into a "recover or discard" question."""
    if _ws is not None and _ws.project.slug == slug:
        return _ws.project.summary()
    try:
        _switch(Workspace(projects().open(slug)))
    except KeyError:
        raise HTTPException(404, f"No design {slug!r}") from None
    return _ws.project.summary()


@app.delete("/api/projects/{slug}")
def delete_project(slug: str) -> dict:
    """Move a design to the Trash. Closes it first if it is open."""
    if _ws is not None and _ws.project.slug == slug:
        _switch(None)
    try:
        dest = projects().delete(slug)
    except KeyError:
        raise HTTPException(404, f"No design {slug!r}") from None
    return {"trashed": str(dest)}


@app.get("/api/projects/{slug}/thumbnail")
def project_thumbnail(slug: str) -> Response:
    try:
        p = projects().open(slug)
    except KeyError:
        raise HTTPException(404, f"No design {slug!r}") from None
    f = p.root / "thumbnail.png"
    if not f.exists():
        raise HTTPException(404, "No thumbnail yet (saved designs get one)")
    return Response(f.read_bytes(), media_type="image/png")


# ---------------------------------------------------------------- File menu


@app.get("/api/file")
def file_state() -> dict:
    if _ws is None and _closed:
        return {"open": False}
    return ws().file_state()


class SaveIn(BaseModel):
    message: str = ""


@app.post("/api/file/save")
def file_save(body: SaveIn) -> dict:
    ws().save(body.message)
    return ws().file_state()


class NameIn(BaseModel):
    name: str


@app.post("/api/file/save-as")
def file_save_as(body: NameIn) -> dict:
    """Save the working copy as a new design and switch to it; the original keeps its last save."""
    w = ws()
    try:
        p = projects().save_as(w.project, w.working, body.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    (w.project.root / ".recovery.json").unlink(missing_ok=True)
    nw = Workspace(p)
    nw.save("")  # thumbnail
    _switch(nw)
    return nw.file_state()


@app.post("/api/file/rename")
def file_rename(body: NameIn) -> dict:
    w = ws()
    if not body.name.strip():
        raise HTTPException(422, "The name needs at least one letter or digit")
    try:
        p = projects().rename(w.project.slug, body.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    w.project = p
    return w.file_state()


@app.post("/api/file/revert")
def file_revert() -> dict:
    ws().revert_to_saved()
    return ws().file_state()


class RecoverIn(BaseModel):
    keep: bool


@app.post("/api/file/recover")
def file_recover(body: RecoverIn) -> dict:
    ws().recover(body.keep)
    return ws().file_state()


@app.post("/api/file/close")
def file_close() -> dict:
    """Close the open design (the app asks about unsaved changes first; the recovery copy stays until then)."""
    _switch(None)
    return {"open": False}


@app.post("/api/file/reveal")
def file_reveal() -> dict:
    """Show the design's folder in Finder (on the machine running the app)."""
    import subprocess
    import sys

    root = ws().project.root
    if sys.platform == "darwin":
        subprocess.run(["open", str(root)], check=False)
    return {"path": str(root)}


@app.post("/api/undo")
def post_undo(x_actor: str = Header(USER)) -> dict:
    try:
        label = ws().undo(x_actor)
    except ChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"undone": label, "tree": ws().tree()}


@app.post("/api/redo")
def post_redo(x_actor: str = Header(USER)) -> dict:
    try:
        label = ws().redo(x_actor)
    except ChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"redone": label, "tree": ws().tree()}


@app.post("/api/history/{commit}/restore")
def restore_version(commit: str) -> dict:
    if not commit.isalnum():
        raise HTTPException(422, "Not a version id")
    try:
        ws().restore_version(commit)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"tree": ws().tree()}


@app.get("/api/project")
def get_project() -> dict:
    """The design as it is being edited (the working copy, saved or not)."""
    w = ws()
    return w.project.summary() | {"design": json.loads(w.working.model_dump_json()), "dirty": w.dirty}


@app.get("/api/history")
def get_history() -> list[dict]:
    return project().history()


@app.get("/api/printers")
def get_printers() -> list[dict]:
    return [p.model_dump() for p in profiles().values()]


class SettingsPatch(BaseModel):
    update_mode: Literal["manual", "auto"] | None = None
    printer: str | None = None
    ai_mode: Literal["mcp", "api"] | None = None
    ai_provider: Literal["auto", "agy", "gemini", "anthropic"] | None = None
    ai_model: str | None = None


@app.put("/api/settings")
def put_settings(body: SettingsPatch) -> dict:
    w = ws()
    s = w.project.data.settings
    if body.printer is not None:
        if body.printer not in profiles():
            raise HTTPException(422, f"No printer profile {body.printer!r}")
        s.printer = body.printer
    if body.ai_mode is not None:
        s.ai_mode = body.ai_mode
    if body.ai_provider is not None:
        s.ai_provider = body.ai_provider
    if body.ai_model is not None:
        s.ai_model = body.ai_model
    w.project.save()
    if body.update_mode is not None:
        w.set_mode(body.update_mode)
    return s.model_dump()


@app.get("/api/features/catalog")
def feature_catalog() -> list[dict]:
    return [f.describe() for f in FEATURES.values()]


# ---------------------------------------------------------------- the model and its build


@app.get("/api/tree")
def get_tree() -> dict:
    return ws().tree()


@app.get("/api/part")
def get_part() -> dict:
    return ws().part()


@app.post("/api/update")
def post_update() -> dict:
    ws().update()
    return ws().tree()


@app.get("/api/checks")
def get_checks() -> dict:
    return ws().checks()


@app.get("/api/activity")
def get_activity(since: float = 0.0) -> list[dict]:
    return [a.to_dict() for a in ws().activity if a.at > since]


@app.get("/api/events")
async def events(request: Request, once: bool = False) -> StreamingResponse:
    """Server-sent events: one message whenever the design, its build or the activity log changes."""

    async def stream():
        last = None
        while True:
            if _ws is None and _closed:
                state = {"slug": None, "rev": 0, "built_rev": 0, "activity": 0}
            else:
                w = ws()
                state = {"slug": w.project.slug, "rev": w.rev, "built_rev": w.built_rev, "activity": len(w.activity),
                         "dirty": w.dirty}
            if state != last:
                yield f"data: {json.dumps(state)}\n\n"
                last = state
                if once:
                    return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------- changes


class OpsRequest(BaseModel):
    ops: list[dict[str, Any]]
    message: str = ""
    base_rev: int | None = None


@app.post("/api/ops")
def post_ops(body: OpsRequest, x_actor: str = Header(USER)) -> dict:
    return _change(body.ops, x_actor, body.message, body.base_rev)


class ParameterIn(BaseModel):
    name: str
    expr: str
    quantity: Literal["length", "angle", "ratio", "count", "bool", "text"] | None = None
    comment: str | None = None


@app.post("/api/parameters")
def set_parameter(body: ParameterIn, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "set_parameter", **body.model_dump(exclude_none=True)}], x_actor, f"Set {body.name} = {body.expr}")


@app.delete("/api/parameters/{name}")
def delete_parameter(name: str, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "delete_parameter", "name": name}], x_actor, f"Delete parameter {name}")


@app.post("/api/features")
def add_feature(body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "add_feature", **body}], x_actor, f"Add {body.get('kind')}")


@app.patch("/api/features/{fid}")
def edit_feature(fid: str, body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "edit_feature", **body, "id": fid}], x_actor, f"Edit {fid}")


@app.delete("/api/features/{fid}")
def delete_feature(fid: str, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "delete_feature", "id": fid}], x_actor, f"Delete {fid}")


@app.post("/api/sketches")
def add_sketch(body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "add_sketch", **body}], x_actor, "Add sketch")


@app.put("/api/sketches/{sid}")
def edit_sketch(sid: str, body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    out = _change([{"op": "edit_sketch", **body, "id": sid}], x_actor, f"Edit {sid}")
    sk = ws().design.sketch(sid)
    return out | {"sketch": sk.model_dump() if sk else None}


@app.get("/api/sketches/{sid}")
def get_sketch(sid: str) -> dict:
    sk = ws().design.sketch(sid)
    if sk is None:
        raise HTTPException(404, f"No sketch {sid!r}")
    return sk.model_dump()


@app.delete("/api/sketches/{sid}")
def delete_sketch(sid: str, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "delete_sketch", "id": sid}], x_actor, f"Delete {sid}")


class RollbackIn(BaseModel):
    index: int | None


@app.put("/api/rollback")
def set_rollback(body: RollbackIn, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "set_rollback", "index": body.index}], x_actor, "Move the end-of-part bar")


@app.post("/api/bodies")
def add_body(body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "add_body", **body}], x_actor, "Add body")


@app.patch("/api/bodies/{bid}")
def edit_body(bid: str, body: dict[str, Any], x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "edit_body", **body, "id": bid}], x_actor, f"Edit {bid}")


@app.delete("/api/bodies/{bid}")
def delete_body(bid: str, x_actor: str = Header(USER)) -> dict:
    return _change([{"op": "delete_body", "id": bid}], x_actor, f"Delete {bid}")


# ---------------------------------------------------------------- picking on the part


def _shown(body: str):
    good = ws().last_good
    shape = good.parts.get(body) if good else None
    if shape is None:
        raise HTTPException(422, f"Nothing built in body {body} to pick on")
    return shape, good.body_names.get(body, [])


@app.get("/api/pick/face")
def pick_face(body: str, face: int) -> dict:
    shape, names = _shown(body)
    try:
        return describe_face(shape, names, face).model_dump()
    except FeatureError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/pick/edge")
def pick_edge(body: str, edge: int) -> dict:
    shape, names = _shown(body)
    try:
        return describe_edge(shape, names, edge).model_dump()
    except FeatureError as exc:
        raise HTTPException(422, str(exc)) from exc


class PickIn(BaseModel):
    body: str = "B1"
    index: int


@app.post("/api/features/{fid}/edges")
def toggle_edge(fid: str, pick: PickIn, x_actor: str = Header(USER)) -> dict:
    """Add an edge picked on the part to a fillet or chamfer, or take it off again if it is already picked."""
    from build123d import Vector

    fi = ws().design.get(fid)
    if fi is None:
        raise HTTPException(404, f"No feature {fid!r}")
    e = pick_edge(pick.body, pick.index)
    at = Vector(*e["at"])
    same = [i for i, x in enumerate(fi.edges) if (Vector(*x.at) - at).length < 0.05]
    edges = [x.model_dump() for i, x in enumerate(fi.edges) if i not in same]
    if not same:
        edges.append(e)
    return _change([{"op": "edit_feature", "id": fid, "edges": edges}], x_actor, f"{'Unpick' if same else 'Pick'} an edge for {fid}")


@app.post("/api/features/{fid}/faces")
def toggle_face(fid: str, pick: PickIn, x_actor: str = Header(USER)) -> dict:
    """Add a face picked on the part to a feature (shell openings, text target), or take it off again."""
    from build123d import Vector

    fi = ws().design.get(fid)
    if fi is None:
        raise HTTPException(404, f"No feature {fid!r}")
    f = pick_face(pick.body, pick.index)
    at = Vector(*f["at"])
    same = [i for i, x in enumerate(fi.faces) if (Vector(*x.at) - at).length < 0.05 or (x.name and x.name == f["name"] and x.normal == f["normal"])]
    faces = [x.model_dump() for i, x in enumerate(fi.faces) if i not in same]
    if not same:
        faces.append(f)
    return _change([{"op": "edit_feature", "id": fid, "faces": faces}], x_actor, f"{'Unpick' if same else 'Pick'} a face for {fid}")


# ---------------------------------------------------------------- concepts


class ConceptIn(BaseModel):
    title: str
    svg: str
    summary: str = ""
    assumptions: list[str] = []
    questions: list[str] = []


class ConceptAnswer(BaseModel):
    decision: Literal["approve", "revise"]
    note: str = ""


@app.get("/api/concepts")
def list_concepts(drawings: bool = True) -> list[dict]:
    """The design's concepts, newest first; `drawings=false` leaves out the SVGs."""
    from .images import inline

    out = [c.model_dump() for c in reversed(ws().concepts.items)]
    for c in out:
        if drawings:
            c["svg"] = inline(c["svg"], ws().images)
        else:
            c.pop("svg")
    return out


@app.get("/api/concepts/{cid}")
def get_concept(cid: str) -> dict:
    c = ws().concepts.get(cid)
    if c is None:
        raise HTTPException(404, f"No concept {cid!r}")
    from .images import inline

    return c.model_dump() | {"svg": inline(c.svg, ws().images)}


@app.post("/api/concepts")
def propose_concept(body: ConceptIn, x_actor: str = Header(default=USER)) -> dict:
    """Show the user what the AI understood (a 2D drawing, summary, assumptions, questions) before it builds; the AI's
    changes are refused until the user answers."""
    from .concepts import ConceptError

    try:
        c = ws().propose_concept(x_actor, title=body.title, svg=body.svg, summary=body.summary,
                                 assumptions=body.assumptions, questions=body.questions)
    except ConceptError as exc:
        raise HTTPException(422, str(exc)) from exc
    return c.model_dump(exclude={"svg"})


class PictureIn(BaseModel):
    svg: str
    width: int = 1200


@app.post("/api/concepts/picture")
def concept_picture_draft(body: PictureIn) -> Response:
    """A drawing (not yet proposed) as a PNG: how the AI checks its picture before showing it to the user."""
    from .concepts import ConceptError, picture
    from .images import inline

    try:
        return Response(picture(inline(body.svg, ws().images), body.width), media_type="image/png")
    except ConceptError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/concepts/{cid}/picture")
def concept_picture(cid: str, width: int = 1200) -> Response:
    """A proposed concept's drawing as a PNG."""
    from .concepts import picture

    c = ws().concepts.get(cid)
    if c is None:
        raise HTTPException(404, f"No concept {cid!r}")
    from .images import inline

    return Response(picture(inline(c.svg, ws().images), width), media_type="image/png")


@app.post("/api/concepts/{cid}/answer")
def answer_concept(cid: str, body: ConceptAnswer) -> dict:
    from .concepts import ConceptError

    try:
        c = ws().answer_concept(cid, body.decision, body.note)
    except ConceptError as exc:
        raise HTTPException(422, str(exc)) from exc
    return c.model_dump(exclude={"svg"})


# ---------------------------------------------------------------- generated images


class ImageIn(BaseModel):
    prompt: str
    quality: Literal["draft", "standard", "best"] = "standard"
    aspect: str = "1:1"
    size: Literal["1K", "2K", "4K"] = "1K"
    from_images: list[str] = []


@app.get("/api/images")
def list_images() -> list[dict]:
    """The design's generated images (how each was made), newest first."""
    return [i.model_dump() for i in reversed(ws().images.items)]


@app.post("/api/images")
def generate_image(body: ImageIn, x_actor: str = Header(default=USER)) -> dict:
    """Make a picture from a prompt (and earlier images to start from) and keep it in the design's images folder."""
    from .images import ImageError

    try:
        return ws().generate_image(x_actor, prompt=body.prompt, quality=body.quality, aspect=body.aspect,
                                   size=body.size, from_images=body.from_images).model_dump()
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/images/{iid}")
def image_file(iid: str) -> FileResponse:
    from .images import ImageError

    try:
        return FileResponse(ws().images.path(iid))
    except ImageError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- import and references

IMPORT_MAX = 200 * 1024 * 1024  # bytes


async def _store(file: UploadFile, kinds: tuple[str, ...], new_design: bool):
    """Save an uploaded file in the open design's references folder (a new design named after it first, if asked),
    reusing the stored copy when the same file is there already. Returns the workspace, the path, the file's stem and
    whether the file is new (so a file that turns out unreadable can be removed again)."""
    import re
    from pathlib import Path

    original = Path(file.filename or "file").name
    stem, ext = Path(original).stem, Path(original).suffix.lower()
    if ext not in kinds:
        raise HTTPException(422, f"Can't use {ext or 'that'} files here; formats: "
                                 + ", ".join(sorted({k.strip('.').upper() for k in kinds})))
    data = await file.read(IMPORT_MAX + 1)
    if len(data) > IMPORT_MAX:
        raise HTTPException(413, f"{original} is over {IMPORT_MAX // 2**20} MB")
    if new_design:
        try:
            _switch(Workspace(projects().create(stem)))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    w = ws()
    folder = w.project.root / "references"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "file"
    path, n = folder / f"{safe}{ext}", 1
    while path.exists() and path.read_bytes() != data:
        n += 1
        path = folder / f"{safe}-{n}{ext}"
    fresh = not path.exists()
    if fresh:
        path.write_bytes(data)
    return w, path, stem, fresh


def _unreadable(path, exc: Exception, fresh: bool) -> HTTPException:
    from .geometry.meshio import ImportError_

    if fresh:
        path.unlink(missing_ok=True)
    return HTTPException(422, str(exc) if isinstance(exc, ImportError_) else f"can't read {path.name}: {exc}")


@app.post("/api/import")
async def import_file(file: UploadFile, new_design: bool = False, x_actor: str = Header(default=USER)) -> dict:
    """Bring a STEP file in as an exact solid: it is kept in the design's references folder and added as an Import
    feature in a body of its own (the first body, when the design is still empty). `new_design` starts a design named
    after the file first. Meshes (STL, 3MF) are references instead: POST /api/references."""
    from .geometry.meshio import STEP, load_step

    if (file.filename or "").lower().endswith((".stl", ".3mf")):
        raise HTTPException(422, "STL and 3MF files are meshes: add them as references (File › Add reference) to look "
                                 "at, measure and copy features from; import STEP files to edit")
    w, path, stem, fresh = await _store(file, STEP, new_design)
    try:
        await asyncio.to_thread(load_step, path)  # read it once here, so a bad file is refused before the tree
    except Exception as exc:
        raise _unreadable(path, exc, fresh) from exc
    d = w.design
    if not d.features and len(d.bodies) == 1:
        ops: list[dict] = [{"op": "edit_body", "id": d.bodies[0].id, "name": stem}]
        body = d.bodies[0].id
    else:
        body = d.next_body_id()
        ops = [{"op": "add_body", "name": stem}]
    ops.append({"op": "add_feature", "kind": "import", "body": body, "params": {"file": path.name}})
    return _change(ops, x_actor, f"Import {path.name}")


@app.post("/api/references")
async def add_reference(file: UploadFile, plate: int | None = None, new_design: bool = False,
                        x_actor: str = Header(default=USER)) -> dict:
    """Add a STEP, STL or 3MF file as a reference: shown beside the design (never printed) for the user and the AI to
    look at, measure and copy features from."""
    from .geometry.meshio import FORMATS, read_mesh

    w, path, stem, fresh = await _store(file, FORMATS, new_design)
    try:
        await asyncio.to_thread(read_mesh, path, plate)
    except Exception as exc:
        raise _unreadable(path, exc, fresh) from exc
    op = {"op": "set_reference", "file": path.name, "name": stem}
    if plate:
        op["plate"] = plate
    return _change([op], x_actor, f"Add reference {path.name}")


def _ref(rid: str):
    w = ws()
    r = w.design.reference(rid)
    if r is None:
        raise HTTPException(404, f"No reference {rid!r}")
    return w, r


@app.get("/api/references")
def list_references(parts: bool = True) -> list[dict]:
    """Every reference with its size and parts (name, triangles, bounding box, volume when closed), as placed."""
    from .references import summary

    w = ws()
    return [summary(r, w.project.root / "references", parts) for r in w.design.references]


@app.get("/api/references/{rid}/mesh")
def reference_mesh(rid: str) -> dict:
    """The placed reference's triangles for the view: per part, base64 little-endian float32 positions and uint32
    indices."""
    import base64

    import numpy as np

    from .references import placed

    w, r = _ref(rid)
    try:
        mesh = placed(r, w.project.root / "references")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, str(exc)) from exc
    b64 = lambda a: base64.b64encode(a.tobytes()).decode()  # noqa: E731
    return {"id": r.id, "color": r.color, "parts": [
        {"name": p.name, "positions": b64(p.points.astype("<f4")), "indices": b64(p.tris.astype("<u4"))}
        for p in mesh.parts]}


def _region(region: str | None):
    if not region:
        return None
    try:
        v = tuple(float(x) for x in region.split(","))
    except ValueError:
        v = ()
    if len(v) != 6:
        raise HTTPException(422, "region is xmin,ymin,zmin,xmax,ymax,zmax in mm")
    return v


@app.get("/api/references/{rid}/render")
def reference_render(rid: str, view: str = "iso", size: int = 768, part: str | None = None,
                     with_design: bool = False, region: str | None = None) -> Response:
    """A picture of the reference (or one part of it) as placed; `with_design` draws the design's last good build in
    with it, to compare; `region` (xmin,ymin,zmin,xmax,ymax,zmax) zooms to what lies inside a box."""
    from .geometry.render import render
    from .references import pick_parts, placed

    w, r = _ref(rid)
    try:
        mesh = placed(r, w.project.root / "references")
        parts = pick_parts(mesh, part)
    except KeyError as exc:
        raise HTTPException(422, exc.args[0]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, str(exc)) from exc
    bodies = []
    if with_design and w.last_good:
        bodies = [(s, w.design.body_color(b)) for b in w.design.bodies if (s := w.last_good.parts.get(b.id)) is not None]
    try:
        png = render(bodies, view, size, meshes=[(p.points, p.tris, r.color) for p in parts], region=_region(region))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(png, media_type="image/png")


@app.get("/api/references/{rid}/section")
def reference_section(rid: str, axis: Literal["X", "Y", "Z", "x", "y", "z"] = "Z", offset: float = 0.0,
                      part: str | None = None, tol: float = 0.05) -> dict:
    """Where the plane axis = offset cuts the reference: its closed outlines in the plane's other two coordinates
    (Z: x, y; Y: x, z; X: y, z), largest first, with area, length, and centre and radius for circles."""
    from .geometry.meshio import section
    from .references import pick_parts, placed

    w, r = _ref(rid)
    try:
        parts = pick_parts(placed(r, w.project.root / "references"), part)
    except KeyError as exc:
        raise HTTPException(422, exc.args[0]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, str(exc)) from exc
    loops = section(parts, axis.upper(), offset, max(tol, 0.001))
    coords = {"X": ["y", "z"], "Y": ["x", "z"], "Z": ["x", "y"]}[axis.upper()]
    return {"reference": rid, "axis": axis.upper(), "offset": offset, "coords": coords, "outlines": loops[:60],
            "more": max(0, len(loops) - 60)}


# ---------------------------------------------------------------- export and fonts


@app.get("/api/export/{fmt}")
def export_design(fmt: str) -> FileResponse:
    """The last update's part as 3MF, STL or STEP (also saved in the design's exports/ folder). Refused while the
    model is out of date or a feature failed."""
    from .export import FORMATS, ExportError, export, filename

    if fmt not in FORMATS:
        raise HTTPException(404, f"Unknown format {fmt!r}; formats: {', '.join(FORMATS)}")
    w = ws()
    t = w.tree()
    if not t["exportable"]:
        why = "update it first" if t["stale"] else "a feature failed" if t["failed"] else "nothing is built"
        raise HTTPException(409, f"Can't export: the model is not ready ({why})")
    good = w.last_good
    names = {b.id: b.name for b in w.design.bodies}
    colors = {b.id: w.design.body_color(b) for b in w.design.bodies}
    path = w.project.root / "exports" / filename(w.project.data.name, fmt)
    try:
        export(good.parts, names, w.project.printer, fmt, path, w.project.data.name, colors, w.design)
    except ExportError as exc:
        raise HTTPException(422, str(exc)) from exc
    w.log(USER, f"Exported {path.name}")
    return FileResponse(path, filename=path.name, headers={"X-Saved-Path": str(path)})


class BatchRow(BaseModel):
    variant: str | None = None  # None: the design as it is
    colorway: str | None = None  # None: the palette's colours


class BatchIn(BaseModel):
    fmt: Literal["3mf", "stl", "step"] = "3mf"
    rows: list[BatchRow] = []  # empty: every variant (or the design) in every colorway (or the palette)


@app.post("/api/export/batch")
def export_batch(body: BatchIn) -> Response:
    """Several variants and colorways in one zip (each file is also saved in exports/<design>/). Each variant is built
    first; a variant that does not build stops the batch and says which."""
    import io
    import zipfile

    from .export import ExportError, export, filename
    from .variants import with_colorway

    w = ws()
    d = w.design
    rows = body.rows or [BatchRow(variant=v, colorway=c) for v in ([x.id for x in d.variants] or [None])
                         for c in ([x.id for x in d.colorways] or [None])]
    for r in rows:
        if r.variant and d.variant(r.variant) is None:
            raise HTTPException(422, f"No variant {r.variant!r}")
        if r.colorway and d.colorway(r.colorway) is None:
            raise HTTPException(422, f"No colorway {r.colorway!r}")
    folder = w.project.root / "exports" / filename(w.project.data.name, "d")[:-2]
    buf = io.BytesIO()
    built: dict[str | None, tuple] = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            if r.variant not in built:
                built[r.variant] = w.build_variant(r.variant)
            vd, result = built[r.variant]
            vname = d.variant(r.variant).name if r.variant else ""
            if not result.complete:
                bad = next(f for f in result.features if f.status == "fail")
                raise HTTPException(422, f"{vname or 'The design'} does not build: {bad.id}: {bad.error}")
            cd = with_colorway(vd, d.colorway(r.colorway) if r.colorway else None)
            label = " ".join(x for x in (w.project.data.name, vname, d.colorway(r.colorway).name if r.colorway else "") if x)
            path = folder / filename(label, body.fmt)
            try:
                export(result.parts, {b.id: b.name for b in cd.bodies}, w.project.printer, body.fmt, path, label,
                       {b.id: cd.body_color(b) for b in cd.bodies}, cd)
            except ExportError as exc:
                raise HTTPException(422, f"{label}: {exc}") from exc
            z.write(path, path.name)
    w.log(USER, f"Exported {len(rows)} file(s) to exports/{folder.name}")
    name = filename(w.project.data.name, "zip")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "X-Saved-Path": str(folder)})


@app.get("/api/layout")
def print_layout() -> dict:
    """The print layout as the export would lay it out: plates, their objects' boxes, problems, unplaced solids."""
    return ws().layout()


@app.get("/api/layout/plates/{n}/mesh")
def plate_mesh(n: int) -> dict:
    """One plate of the print layout for the view (worked out when first asked, then kept until the part or the
    layout changes)."""
    try:
        return ws().plate_mesh(n)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/render")
def render_view(view: str = "iso", size: int = 640) -> Response:
    """The last good part as a PNG from a standard view (iso, front, back, left, right, top, bottom), or a plate of
    the print layout ("plate1", "plate1-top")."""
    try:
        png = ws().render(view, size)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(png, media_type="image/png")


@app.get("/api/faces")
def list_faces(body: str = "B1") -> list[dict]:
    return ws().faces(body)


@app.post("/api/revert-ai")
def revert_ai(x_actor: str = Header(USER)) -> dict:
    try:
        msg = ws().revert_last_ai(x_actor)
    except ChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"reverted": msg, "tree": ws().tree()}


@app.get("/api/fonts")
def get_fonts() -> list[dict]:
    from .fonts import families

    return [f.to_dict() for f in families(project().root / "fonts").values()]


# ---------------------------------------------------------------- formulas


class ExprIn(BaseModel):
    expr: str


@app.post("/api/expr/eval")
def eval_expr(body: ExprIn) -> dict:
    values = {k: v for k, v in resolve(ws().design).values.items() if not isinstance(v, str)}
    try:
        return {"value": evaluate(body.expr, values)}
    except ExprError as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------- AI


@app.get("/api/ai/guide")
def ai_guide() -> dict:
    from .ai.guide import GUIDE

    from .build.ops import OP_NAMES

    return {"guide": GUIDE, "ops": OP_NAMES}


# ---------------------------------------------------------------- AI chat (the panel's API mode)


def chat():
    w = ws()
    if not hasattr(w, "chat"):
        from .ai.chat import Chat

        w.chat = Chat()
    return w.chat


AVAILABLE_MODELS = {
    "gemini": [
        {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview (Recommended, Free Tier)"},
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (Free Tier)"},
        {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
        {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
        {"id": "gemini-4-argon", "label": "Gemini 4.0 Argon (Reasoning, Paid Tier)"},
    ],
    "anthropic": [
        {"id": "claude-opus-5-5", "label": "Claude Opus 5.5"},
        {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
        {"id": "claude-haiku-5-5", "label": "Claude Haiku 5.5"},
    ],
}


@app.get("/api/ai/status")
def ai_status(refresh: bool = False) -> dict:
    s = project().data.settings
    chosen_provider = s.ai_provider if s.ai_provider != "auto" else None
    active_provider = ai.provider(chosen_provider)
    model = ai.model_name(s.ai_model or None, provider_override=active_provider)
    has_key = ai.has_api_key(active_provider)
    available = {
        "agy": ai.DEFAULT_AGY_MODELS,
        "gemini": ai.list_gemini_models(force_refresh=refresh),
        "anthropic": AVAILABLE_MODELS["anthropic"],
    }
    return {
        "model": model,
        "provider": active_provider,
        "key": has_key,
        "mode": s.ai_mode,
        "settings_provider": s.ai_provider,
        "settings_model": s.ai_model,
        "available_models": available,
        "chat": chat().to_dict(),
    }


class ChatIn(BaseModel):
    message: str


@app.post("/api/ai/chat")
def ai_chat(body: ChatIn) -> dict:
    if not body.message.strip():
        raise HTTPException(422, "Say what you want changed")
    s = project().data.settings
    chosen_provider = s.ai_provider if s.ai_provider != "auto" else None
    active_provider = ai.provider(chosen_provider)
    model = ai.model_name(s.ai_model or None, provider_override=active_provider)
    try:
        client = ai.make_client(active_provider)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        chat().send(body.message.strip(), client, model=model)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return chat().to_dict()


@app.post("/api/ai/chat/cancel")
def ai_chat_cancel() -> dict:
    chat().cancel.set()
    ws().concepts.dismiss_pending("cancelled by user")
    return chat().to_dict()


@app.post("/api/ai/chat/reset")
def ai_chat_reset() -> dict:
    try:
        chat().reset()
        ws().concepts.dismiss_pending("superseded by new conversation")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return chat().to_dict()
