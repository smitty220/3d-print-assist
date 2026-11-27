"""3D Print Assist API.

Run with:  uv run uvicorn print_assist.app:app --reload

Every change to a design is a list of ops (`build/ops.py`) applied by the open design's Workspace; the endpoints
below are thin wrappers. The `X-Actor` header names who is changing it ("user" for the app, "claude-code" for the
MCP server, "claude" for the in-app chat); changes by anyone but the user are built and checked before they are kept.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .ai import client as ai
from .build.ops import USER, OpError
from .build.workspace import ChangeRefused, Workspace
from .features.context import FeatureError
from .features.design import describe_edge, describe_face
from .features.library import FEATURES
from .params.expr import ExprError, evaluate
from .params.graph import resolve
from .printers import profiles
from .project import Projects, save_user_prefs

@asynccontextmanager
async def _lifespan(_: FastAPI):
    from print_assist_mcp.server import mcp as _mcp
    from .ml.installer import installer

    if not os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("PRINT_ASSIST_DISABLE_ML_AUTO_INSTALL") != "1":
        st = installer.status()
        if not st["ready"] and not st["installing"]:
            installer.start_install()

    try:
        async with _mcp.session_manager.run():
            yield
    finally:
        installer.cancel()
        _stop_all_chats()


app = FastAPI(title="3D Print Assist", version="0.2.0", lifespan=_lifespan)
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


def _defer_open() -> bool:
    return not bool(os.environ.get("PYTEST_CURRENT_TEST"))


def ws() -> Workspace:
    """The open design. At start-up the most recently changed one (or a new one) opens, as an editor reopens its last
    document; after File › Close there is none until one is opened."""
    global _ws
    if _ws is None:
        if _closed:
            raise HTTPException(409, "No design is open; open or create one")
        listed = projects().list()
        p = projects().open(listed[0]["slug"]) if listed else projects().create("Untitled")
        _ws = Workspace(p, defer_build=_defer_open())
    return _ws


def project():
    return ws().project


def reset(root=None) -> None:
    """Forget the open design (tests point the registry at a temporary folder)."""
    global _projects, _ws, _closed
    _stop_all_chats()
    _projects = Projects(root) if root is not None else None
    _ws = None
    _closed = False
    _chats.clear()


def _switch(w: Workspace | None) -> None:
    global _ws, _closed
    _ws, _closed = w, w is None


def _change(ops: list[dict], actor: str, message: str = "", base_rev: int | None = None) -> dict:
    try:
        made = ws().change(ops, actor=actor or USER, message=message, base_rev=base_rev)
    except ChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OpError, ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface CAD/kernel failures cleanly instead of opaque 500
        raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc
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
    creation_mode: Literal["engineer", "character"] = "engineer"


@app.post("/api/projects")
def create_project(body: NewProject) -> dict:
    try:
        _switch(Workspace(projects().create(body.name, creation_mode=body.creation_mode)))
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
        _switch(Workspace(projects().open(slug), defer_build=_defer_open()))
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
    """Show the design's folder in Finder / Explorer (on the machine running the app)."""
    import subprocess
    import sys

    root = ws().project.root
    if sys.platform == "darwin":
        subprocess.run(["open", str(root)], check=False)
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", str(root)])
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


class PrinterIn(BaseModel):
    name: str
    bed: tuple[float, float, float]
    nozzle_dia: float = 0.4
    slots: int = 1


@app.post("/api/printers")
def create_printer(body: PrinterIn) -> dict:
    from .printers import Nozzle, PrinterProfile, save_custom_profile
    from .project import slugify

    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Printer name is required")
    if any(v <= 0 for v in body.bed):
        raise HTTPException(422, "Bed dimensions must be positive")
    if body.nozzle_dia <= 0 or body.slots < 1:
        raise HTTPException(422, "Nozzle diameter must be positive and slots >= 1")
    pid = f"custom_{slugify(name)}"
    prof = save_custom_profile(PrinterProfile(id=pid, name=name, bed=body.bed, nozzles=[Nozzle(dia=body.nozzle_dia, slots=body.slots)]))
    return prof.model_dump()


class SettingsPatch(BaseModel):
    update_mode: Literal["manual", "auto"] | None = None
    printer: str | None = None
    slicer: str | None = None
    ai_mode: Literal["mcp", "api"] | None = None
    ai_provider: Literal["auto", "agy", "gemini", "anthropic"] | None = None
    ai_model: str | None = None
    creation_mode: Literal["engineer", "character"] | None = None
    direct_build: bool | None = None
    character_multicolor: bool | None = None


@app.put("/api/settings")
def put_settings(body: SettingsPatch) -> dict:
    w = ws()
    s = w.project.data.settings
    if body.printer is not None:
        if body.printer not in profiles():
            raise HTTPException(422, f"No printer profile {body.printer!r}")
        s.printer = body.printer
    if body.slicer is not None:
        if body.slicer and body.slicer not in SLICERS:
            raise HTTPException(422, f"Unknown slicer {body.slicer!r}")
        s.slicer = body.slicer
    if body.ai_mode is not None:
        s.ai_mode = body.ai_mode
    if body.ai_provider is not None:
        s.ai_provider = body.ai_provider
    if body.ai_model is not None:
        s.ai_model = body.ai_model
    if body.creation_mode is not None:
        s.creation_mode = body.creation_mode
    if body.direct_build is not None:
        s.direct_build = body.direct_build
    if body.character_multicolor is not None:
        s.character_multicolor = body.character_multicolor
    w.project.save()
    save_user_prefs(
        projects().root,
        printer=body.printer,
        slicer=body.slicer,
        ai_mode=body.ai_mode,
        ai_provider=body.ai_provider,
        ai_model=body.ai_model,
    )
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
                state = {"slug": None, "rev": 0, "built_rev": 0, "building": False, "activity": 0}
            else:
                w = ws()
                state = {"slug": w.project.slug, "rev": w.rev, "built_rev": w.built_rev, "building": w._building,
                         "activity": len(w.activity), "dirty": w.dirty}
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
    views: list[str] = ["3/4 iso", "front", "side (Z=0)", "top"]


class ConceptAnswer(BaseModel):
    decision: Literal["approve", "revise"]
    note: str = ""


@app.get("/api/concepts")
def list_concepts(drawings: bool = True) -> list[dict]:
    """The design's concepts, newest first; `drawings=false` leaves out the SVGs."""
    from .images import inline

    ws().concepts.sync()
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
        ws().concepts.sync()
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
                                 assumptions=body.assumptions, questions=body.questions, views=body.views)
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
        ws().concepts.sync()
        c = ws().concepts.get(cid)
    if c is None:
        available = ", ".join(x.id for x in ws().concepts.items) or "none yet"
        raise HTTPException(404, f"No concept {cid!r} (available: {available})")
    img_path = ws().concepts.image_path(cid)
    if img_path is not None:
        return FileResponse(img_path)
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


IMAGE_MAX = 25 * 1024 * 1024  # bytes


@app.post("/api/images/upload")
async def upload_image(file: UploadFile, note: str = "", x_actor: str = Header(default=USER)) -> dict:
    """Upload a reference photo (PNG, JPEG, WebP) into the design's images folder."""
    from pathlib import Path
    from .images import EXT, ImageError

    data = await file.read(IMAGE_MAX + 1)
    if len(data) > IMAGE_MAX:
        raise HTTPException(413, "Image is over 25 MB")
    ext = Path(file.filename or "").suffix.lower() or {**EXT, "image/heic": ".heic", "image/heif": ".heif"}.get(file.content_type or "", "")
    try:
        return ws().upload_image(data=data, ext=ext, note=note or (file.filename or ""), actor=x_actor).model_dump()
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/images/{iid}")
def image_file(iid: str) -> FileResponse:
    from .images import ImageError

    try:
        return FileResponse(ws().images.path(iid))
    except ImageError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/images/{iid}")
def delete_image(iid: str, x_actor: str = Header(default=USER)) -> dict:
    from .images import ImageError

    try:
        ws().delete_image(iid, actor=x_actor)
        return {"deleted": iid}
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
    path = w.project.root / "exports" / filename(w.project.data.name, fmt)
    try:
        w.export_current(fmt, path)
    except (ExportError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    w.log(USER, f"Exported {path.name}")
    return FileResponse(path, filename=path.name, headers={"X-Saved-Path": str(path)})


SLICERS: dict[str, tuple[str, list[str]]] = {
    "prusaslicer": ("PrusaSlicer", ["PrusaSlicer", "prusa-slicer"]),
    "bambu": ("Bambu Studio", ["Bambu Studio", "BambuStudio", "bambu-studio"]),
    "orca": ("OrcaSlicer", ["OrcaSlicer", "orca-slicer"]),
    "cura": ("UltiMaker Cura", ["UltiMaker Cura", "UltiMaker-Cura", "cura"]),
    "creality": ("Creality Print", ["Creality Print", "CrealityPrint"]),
}


def _find_slicer_app(candidates: list[str]) -> str | None:
    import shutil
    import sys
    from pathlib import Path

    if sys.platform == "darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            if not base.is_dir():
                continue
            for name in candidates:
                app_path = base / f"{name}.app"
                if app_path.exists():
                    return str(app_path)
            for sub in base.iterdir():
                if sub.is_dir() and not sub.name.endswith(".app"):
                    for name in candidates:
                        app_path = sub / f"{name}.app"
                        if app_path.exists():
                            return str(app_path)
    if sys.platform == "win32":
        win_roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Programs",
        ]
        win_subdirs = ("", "Prusa3D/PrusaSlicer", "Bambu Studio", "OrcaSlicer", "UltiMaker Cura", "Creality Print")
        for base in win_roots:
            if not base.is_dir():
                continue
            for sub in win_subdirs:
                folder = base / sub if sub else base
                if not folder.is_dir():
                    continue
                for name in candidates:
                    exe = folder / f"{name}.exe"
                    if exe.is_file():
                        return str(exe)
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


@app.get("/api/slicers")
def get_slicers() -> list[dict]:
    return [
        {"id": sid, "name": label, "installed": _find_slicer_app(cands) is not None}
        for sid, (label, cands) in SLICERS.items()
    ]


class SendToIn(BaseModel):
    slicer: str | None = None
    fmt: Literal["3mf", "stl", "step"] = "3mf"


@app.post("/api/export/send-to")
def export_send_to(body: SendToIn) -> dict:
    import subprocess
    import sys

    from .export import ExportError, export, filename

    w = ws()
    slicer_id = body.slicer or w.project.data.settings.slicer
    if not slicer_id:
        slicer_id = next((sid for sid, (_, cands) in SLICERS.items() if _find_slicer_app(cands)), "prusaslicer")
    entry = SLICERS.get(slicer_id)
    if entry is None:
        raise HTTPException(404, f"Unknown slicer {slicer_id!r}")
    label, cands = entry
    app_target = _find_slicer_app(cands)
    if not app_target:
        raise HTTPException(422, f"{label} was not found in /Applications or PATH")

    t = w.tree()
    if not t["exportable"]:
        why = "update it first" if t["stale"] else "a feature failed" if t["failed"] else "nothing is built"
        raise HTTPException(409, f"Can't export: the model is not ready ({why})")
    path = w.project.root / "exports" / filename(w.project.data.name, body.fmt)
    try:
        w.export_current(body.fmt, path)
    except (ExportError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    if sys.platform == "darwin" and app_target.endswith(".app"):
        subprocess.Popen(["open", "-a", app_target, str(path)])
    else:
        subprocess.Popen([app_target, str(path)])
    w.log(USER, f"Sent {path.name} to {label}")
    return {"slicer": label, "path": str(path)}


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


@app.get("/api/web/search")
def api_web_search(query: str, max_results: int = 5) -> dict:
    from .ai.web import web_search

    return web_search(query, max_results=max_results)


@app.get("/api/web/fetch")
def api_web_fetch(url: str) -> dict:
    from .ai.web import fetch_url

    return fetch_url(url)


# ---------------------------------------------------------------- Character Mode & Local ML Engine


class CharacterDesignIn(BaseModel):
    title: str
    prompt: str
    summary: str = ""
    reference_photos: list[str] = []
    assumptions: list[str] = []
    questions: list[str] = []
    turnaround: bool = True  # Default 4-view turnaround sheet for every design
    style: Literal["auto", "figurine", "flexi", "bust", "keychain", "magnet", "planter", "scene"] = "auto"
    from_concept_id: str = ""
    repaint: bool = False
    flexi_segments: int = 0
    flexi_cuts_pct: list[float] = []
    steps: int = 14
    seed: int = -1
    wait_s: float = -1.0  # -1 = block until done; >= 0 = async job waiting up to wait_s seconds


class CharacterSculptIn(BaseModel):
    concept_id: str = ""
    replace_sculpt_id: str = ""
    name: str = ""
    prompt: str = ""
    reference_photos: list[str] = []
    target_height_mm: float = 80.0
    flatten_bottom_mm: float = 0.4
    base_style: Literal["none", "round", "oval", "square"] = "none"
    base_height_mm: float = 3.0
    base_margin_mm: float = 4.0
    keychain_loop: bool = False
    keychain_hole_mm: float = 4.5
    flexi_segments: int = 0
    flexi_axis: Literal["auto", "x", "y"] = "auto"
    flexi_start_pct: float = 35.0
    flexi_end_pct: float = 88.0
    flexi_clearance_mm: float = 0.45
    flexi_angle_deg: float = 24.0
    flexi_cuts_pct: list[float] = []
    cut_bodies: list[str] = []
    union_bodies: list[str] = []
    magnet_hole: Literal["none", "bottom", "back"] = "none"
    magnet_dia_mm: float = 8.2
    magnet_depth_mm: float = 3.2
    hollow_cavity: bool = False
    hollow_wall_mm: float = 3.0
    multicolor: bool = False
    max_colors: int = 0  # 0 = auto (up to 5 when multicolor + 3D paint, or 2 clean sweeping belly)
    color_mode: Literal["auto", "single", "multicolor"] = "auto"
    color: str | None = "#b8c2cc"
    wait_s: float = -1.0


class SculptPrepPatch(BaseModel):
    name: str | None = None
    target_height_mm: float | None = None
    flatten_bottom_mm: float | None = None
    base_style: Literal["none", "round", "oval", "square"] | None = None
    base_height_mm: float | None = None
    base_margin_mm: float | None = None
    keychain_loop: bool | None = None
    keychain_hole_mm: float | None = None
    flexi_segments: int | None = None
    flexi_axis: Literal["auto", "x", "y"] | None = None
    flexi_start_pct: float | None = None
    flexi_end_pct: float | None = None
    flexi_clearance_mm: float | None = None
    flexi_angle_deg: float | None = None
    flexi_cuts_pct: list[float] | None = None
    cut_bodies: list[str] | None = None
    union_bodies: list[str] | None = None
    magnet_hole: Literal["none", "bottom", "back"] | None = None
    magnet_dia_mm: float | None = None
    magnet_depth_mm: float | None = None
    hollow_cavity: bool | None = None
    hollow_wall_mm: float | None = None
    rotate_x: float | None = None
    rotate_y: float | None = None
    rotate_z: float | None = None
    max_colors: int | None = None
    color_mode: Literal["auto", "single", "multicolor"] | None = None
    color: str | None = None
    filament: int | None = None


@app.get("/api/character/job")
def get_character_job(wait_s: float = 0.0) -> dict:
    from .ml import poll_character_job

    return poll_character_job(wait_s=max(0.0, min(float(wait_s), 45.0)))


@app.get("/api/ml/status")
def ml_status() -> dict:
    from .ml.installer import installer

    return installer.status()


class MlSetupIn(BaseModel):
    components: list[Literal["painter", "sculptor", "tex_painter"]] = ["painter", "sculptor"]


@app.post("/api/ml/setup")
def ml_setup(body: MlSetupIn) -> dict:
    from .ml.installer import installer

    return installer.start_install(tuple(body.components))


@app.post("/api/ml/cancel")
def ml_cancel() -> dict:
    from .ml.installer import installer

    installer.cancel()
    return installer.status()


@app.post("/api/character/design")
def create_character_design(body: CharacterDesignIn, x_actor: str = Header(default=USER)) -> dict:
    """Generate a Character Mode 2D Design File using the local 2D Painter (never saved to Photos)."""
    from .concepts import ConceptError
    from .images import ImageError
    from .ml import start_character_job
    from .ml.painter import (
        PainterError,
        generate_character_design_image,
        prepare_quad_turnaround_sheet,
        render_flexi_concept_preview,
        resolve_character_style,
    )

    w = ws()
    ref_paths = []
    for iid in body.reference_photos:
        try:
            ref_paths.append(w.images.path(iid))
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc

    src_cid = body.from_concept_id.strip()
    src_c = None
    src_clean_path = None
    if src_cid:
        w.concepts.sync()
        if src_cid.lower() == "latest" and w.concepts.items:
            src_c = w.concepts.items[-1]
        else:
            src_c = w.concepts.get(src_cid)
        if src_c is not None:
            src_clean_path = w.concepts.clean_image_path(src_c.id)

    combined_text = f"{body.title} {body.prompt} {body.summary}"
    resolved_style = (
        "flexi"
        if (body.flexi_segments > 0 or len(body.flexi_cuts_pct) > 0)
        else resolve_character_style(combined_text, body.style)
    )

    def _work() -> dict:
        if src_c is not None and src_clean_path is not None and not body.repaint:
            raw_png_bytes = src_clean_path.read_bytes()
            raw_views = ["front"]
        else:
            eff_refs = list(ref_paths)
            if not eff_refs and src_clean_path is not None:
                eff_refs = [src_clean_path]
            raw_png_bytes, raw_views = generate_character_design_image(
                prompt=body.prompt,
                ref_paths=eff_refs or None,
                turnaround=(False if resolved_style == "flexi" else body.turnaround),
                style=resolved_style,
                steps=body.steps,
                seed=body.seed,
            )

        if resolved_style == "flexi":
            # Ensure clean_png_bytes is a single hero character (extract top-left if source happened to be a 4-grid)
            _, clean_png_bytes, _ = prepare_quad_turnaround_sheet(raw_png_bytes, raw_views)
            eff_segs = body.flexi_segments if body.flexi_segments >= 3 else 6
            preview_png_bytes, cuts_pct = render_flexi_concept_preview(
                clean_png_bytes,
                flexi_segments=eff_segs,
                flexi_cuts_pct=body.flexi_cuts_pct or None,
            )
            final_segs = (len(cuts_pct) + 1) if cuts_pct else eff_segs
            views = ["3/4 flexi (Z=0)", "side (Z=0 & hinges)", "top (fin-safe)", "bottom (Z=0 pads)"]
        else:
            preview_png_bytes, clean_png_bytes, views = prepare_quad_turnaround_sheet(raw_png_bytes, raw_views)
            cuts_pct = []
            final_segs = 0

        c = w.propose_concept(
            x_actor,
            title=body.title,
            summary=body.summary or body.prompt,
            assumptions=body.assumptions,
            questions=body.questions,
            mode="character",
            image_bytes=preview_png_bytes,
            clean_image_bytes=clean_png_bytes,
            image_ext=".png",
            reference_photos=body.reference_photos,
            views=views,
            style=resolved_style,
            flexi_segments=final_segs,
            flexi_cuts_pct=cuts_pct,
        )
        return c.model_dump(exclude={"svg"})

    if body.wait_s < 0 or (src_c is not None and src_clean_path is not None and not body.repaint):
        try:
            return _work()
        except (PainterError, ConceptError) as exc:
            raise HTTPException(422, str(exc)) from exc

    res = start_character_job("painting", "Creating design preview", body.title, _work, wait_s=body.wait_s)
    if res["status"] == "error":
        raise HTTPException(422, res["error"])
    if res["status"] == "done" and res["result"] is not None:
        return res["result"]
    return res


@app.post("/api/character/sculpt")
def create_character_sculpt(body: CharacterSculptIn, x_actor: str = Header(default=USER)) -> dict:
    """Sculpt a Character Mode 3D mesh + precision CAD print mechanics (flexi hinges, cutouts, base, loop)."""
    from .features.model import SculptPrep
    from .geometry.mesh_prep import MeshPrepError
    from .images import ImageError
    from .ml import ml_job_update, start_character_job
    from .ml.painter import PainterError, generate_character_design_image, render_flexi_concept_preview
    from .ml.sculptor import SculptorError, sculpt_mesh_from_image

    w = ws()
    img_path = None
    views: list[str] = []
    cid = body.concept_id.strip()
    sculpt_name = body.name.strip()
    ref_paths = []
    for iid in body.reference_photos:
        try:
            ref_paths.append(w.images.path(iid))
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc

    eff_flexi_segments = body.flexi_segments
    eff_flatten_bottom_mm = body.flatten_bottom_mm
    if cid:
        c = w.concepts.get(cid)
        if c is None:
            w.concepts.sync()
            c = w.concepts.get(cid)
        if c is None:
            raise HTTPException(404, f"No concept {cid!r}")
        if c.status == "pending" and x_actor != USER and not w.project.data.settings.direct_build:
            raise HTTPException(409, f"Concept {c.id} is still waiting for the user's approval")
        img_path = (
            w.concepts.image_path(c.id)
            if (len(c.views) >= 4 and "back" in " ".join(v.lower() for v in c.views) and "bottom" not in " ".join(v.lower() for v in c.views))
            else w.concepts.clean_image_path(c.id)
        )
        if img_path is None:
            img_path = w.concepts.clean_image_path(c.id)
        views = list(c.views)
        sculpt_name = sculpt_name or c.title
        if img_path is None:
            raise HTTPException(422, f"Concept {c.id} does not have a Character Mode design image")
        if eff_flexi_segments == 0 and (
            c.flexi_segments > 0 or c.style == "flexi" or "flexi" in c.title.lower()
        ):
            eff_flexi_segments = c.flexi_segments if c.flexi_segments >= 3 else 6
            eff_flatten_bottom_mm = max(eff_flatten_bottom_mm, 0.4)
    elif not body.prompt.strip() and not ref_paths:
        raise HTTPException(422, "Provide concept_id, prompt, or reference_photos to sculpt")
    elif not body.prompt.strip() and ref_paths:
        img_path = ref_paths[0]
        sculpt_name = sculpt_name or body.reference_photos[0]

    prep = SculptPrep(
        target_height_mm=body.target_height_mm,
        flatten_bottom_mm=eff_flatten_bottom_mm,
        base_style=body.base_style,
        base_height_mm=body.base_height_mm,
        base_margin_mm=body.base_margin_mm,
        keychain_loop=body.keychain_loop,
        keychain_hole_mm=body.keychain_hole_mm,
        flexi_segments=eff_flexi_segments,
        flexi_axis=body.flexi_axis,
        flexi_start_pct=body.flexi_start_pct,
        flexi_end_pct=body.flexi_end_pct,
        flexi_clearance_mm=body.flexi_clearance_mm,
        flexi_angle_deg=body.flexi_angle_deg,
        flexi_cuts_pct=body.flexi_cuts_pct,
        cut_bodies=body.cut_bodies,
        union_bodies=body.union_bodies,
        magnet_hole=body.magnet_hole,
        magnet_dia_mm=body.magnet_dia_mm,
        magnet_depth_mm=body.magnet_depth_mm,
        hollow_cavity=body.hollow_cavity,
        hollow_wall_mm=body.hollow_wall_mm,
        max_colors=(
            max(1, min(8, int(body.max_colors)))
            if int(body.max_colors) > 0
            else (
                max(2, min(5, w.project.printer.slots if w.project.printer.slots > 1 else 5))
                if (body.multicolor or body.color_mode == "multicolor")
                else 1
            )
        ),
        color_mode="multicolor" if body.multicolor else body.color_mode,
    )

    def _work() -> dict:
        nonlocal img_path, views, cid, sculpt_name
        if img_path is None:
            # Direct Build bypass: synthesize clean hero image with matching archetype framing, then sculpt
            auto_style = (
                "flexi" if prep.flexi_segments > 0
                else "magnet" if body.magnet_hole != "none"
                else "planter" if body.hollow_cavity
                else "keychain" if body.keychain_loop
                else "auto"
            )
            clean_png_bytes, views = generate_character_design_image(
                prompt=body.prompt, ref_paths=ref_paths or None, turnaround=False, style=auto_style
            )
            if auto_style == "flexi" or prep.flexi_segments > 0:
                preview_png_bytes, cuts_pct = render_flexi_concept_preview(
                    clean_png_bytes,
                    flexi_segments=prep.flexi_segments or 6,
                    flexi_cuts_pct=prep.flexi_cuts_pct or None,
                )
            else:
                preview_png_bytes, cuts_pct = clean_png_bytes, []
            c_new = w.propose_concept(
                x_actor,
                title=sculpt_name or body.prompt[:48],
                summary=body.prompt,
                mode="character",
                image_bytes=preview_png_bytes,
                clean_image_bytes=clean_png_bytes,
                reference_photos=body.reference_photos,
                views=views,
                style=auto_style,
                flexi_segments=prep.flexi_segments,
                flexi_cuts_pct=cuts_pct,
            )
            w.answer_concept(c_new.id, "approve", "Direct Build")
            cid = c_new.id
            img_path = w.concepts.clean_image_path(c_new.id)
            sculpt_name = sculpt_name or c_new.title
        assert img_path is not None
        raw_glb = sculpt_mesh_from_image(img_path, views=views)
        # If multi-color (>=3 colors) is requested and local 3D Texture Painter (`Hunyuan3D-Paint`) is ready,
        # bake real geometry-locked 3D UV colors directly onto the sculpt (zero 2D slide-projector misalignment!).
        want_3d_paint = (
            prep.color_mode != "single"
            and (prep.max_colors >= 3 or (prep.color_mode == "auto" and w.project.printer.slots > 1 and int(body.max_colors) != 2))
        )
        if want_3d_paint:
            from .ml.installer import tex_painter_ready
            if tex_painter_ready():
                import tempfile
                from .ml.tex_painter import paint_mesh_from_image
                try:
                    ml_job_update("Painting 3D surface colors", 84, "Baking 3D UV colors onto sculpt…")
                    with tempfile.TemporaryDirectory(prefix="pa_sculpt_paint_") as _td:
                        tmp_mesh = Path(_td) / "raw.glb"
                        tmp_mesh.write_bytes(raw_glb)
                        raw_glb = paint_mesh_from_image(tmp_mesh, img_path)
                except Exception:  # noqa: BLE001
                    pass
        ml_job_update("Finishing for 3D printing", 92, "Repairing solid & cutting print mechanics…")
        sm = w.add_sculpt(
            actor=x_actor, name=sculpt_name or "Sculpt", raw_bytes=raw_glb, ext=".glb",
            concept_id=cid, prep=prep, color=body.color,
            replace_id=body.replace_sculpt_id.strip(),
        )
        return {"sculpt": sm.model_dump(), "tree": w.tree()}

    if body.wait_s < 0:
        try:
            return _work()
        except (PainterError, SculptorError, MeshPrepError) as exc:
            raise HTTPException(422, str(exc)) from exc

    res = start_character_job("sculpting", "Sculpting 3D model", sculpt_name or "Character sculpt", _work, wait_s=body.wait_s)
    if res["status"] == "error":
        raise HTTPException(422, res["error"])
    if res["status"] == "done" and res["result"] is not None:
        return res["result"]
    return res


class CharacterBlenderIn(BaseModel):
    name: str
    script: str
    concept_id: str = ""
    replace_sculpt_id: str = ""
    target_height_mm: float = 0.0  # 0 keeps exact mm dimensions produced by the Blender script
    flatten_bottom_mm: float = 0.0
    base_style: Literal["none", "round", "oval", "square"] = "none"
    base_height_mm: float = 3.0
    base_margin_mm: float = 4.0
    keychain_loop: bool = False
    keychain_hole_mm: float = 4.5
    flexi_segments: int = 0  # Leave 0 when script calls pa_flexi_articulate(...) directly
    flexi_axis: Literal["auto", "x", "y"] = "auto"
    flexi_clearance_mm: float = 0.45
    flexi_angle_deg: float = 24.0
    flexi_cuts_pct: list[float] = []
    cut_bodies: list[str] = []
    union_bodies: list[str] = []
    magnet_hole: Literal["none", "bottom", "back"] = "none"
    magnet_dia_mm: float = 8.2
    magnet_depth_mm: float = 3.2
    hollow_cavity: bool = False
    hollow_wall_mm: float = 3.0
    color_mode: Literal["auto", "single", "multicolor"] = "auto"
    color: str | None = "#4b6e91"
    timeout_s: int = 120


@app.post("/api/character/blender")
def create_blender_character(body: CharacterBlenderIn, x_actor: str = Header(default=USER)) -> dict:
    """Build a glass-smooth Catmull-Clark SubSurf, multi-color, print-in-place Flexi-Factory character
    directly inside the bundled headless Blender 4.5 LTS engine (~2-4 seconds, zero cloud cost)."""
    from .features.model import SculptPrep
    from .geometry.mesh_prep import MeshPrepError, analyze_sculpt_printability
    from .ml.blender import BlenderError, run_blender_character_script

    if not body.script.strip():
        raise HTTPException(422, "Provide a non-empty Blender Python `script`")

    w = ws()
    try:
        raw_glb, report = run_blender_character_script(
            body.script,
            timeout_s=max(15, min(int(body.timeout_s), 240)),
        )
    except BlenderError as exc:
        raise HTTPException(422, str(exc)) from exc

    prep = SculptPrep(
        target_height_mm=body.target_height_mm,
        flatten_bottom_mm=body.flatten_bottom_mm,
        base_style=body.base_style,
        base_height_mm=body.base_height_mm,
        base_margin_mm=body.base_margin_mm,
        keychain_loop=body.keychain_loop,
        keychain_hole_mm=body.keychain_hole_mm,
        flexi_segments=body.flexi_segments,
        flexi_axis=body.flexi_axis,
        flexi_clearance_mm=body.flexi_clearance_mm,
        flexi_angle_deg=body.flexi_angle_deg,
        flexi_cuts_pct=body.flexi_cuts_pct,
        cut_bodies=body.cut_bodies,
        union_bodies=body.union_bodies,
        magnet_hole=body.magnet_hole,
        magnet_dia_mm=body.magnet_dia_mm,
        magnet_depth_mm=body.magnet_depth_mm,
        hollow_cavity=body.hollow_cavity,
        hollow_wall_mm=body.hollow_wall_mm,
        max_colors=5 if body.color_mode != "single" else 1,
        color_mode=body.color_mode,
    )
    try:
        sm = w.add_sculpt(
            actor=x_actor,
            name=body.name.strip() or "Blender Character",
            raw_bytes=raw_glb,
            ext=".glb",
            concept_id=body.concept_id.strip(),
            prep=prep,
            color=body.color,
            replace_id=body.replace_sculpt_id.strip(),
            source_script=body.script,
        )
    except (MeshPrepError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    cached = w._sculpt_cache.get(sm.id)
    prepared = cached[1] if cached else None
    part_summary = (
        [{"name": p.name, "color": p.color, "filament": p.filament, "triangles": p.triangles} for p in prepared.parts]
        if prepared
        else []
    )
    printability = analyze_sculpt_printability(prepared.parts) if prepared else {}
    return {
        "sculpt": sm.model_dump(),
        "blender": report,
        "parts": part_summary,
        "printability": printability,
        "notes": prepared.notes if prepared else [],
        "tree": w.tree(),
    }


@app.get("/api/sculpts")
def list_sculpts() -> list[dict]:
    return [s.model_dump() for s in ws().design.sculpts]


@app.post("/api/sculpts/upload")
async def upload_sculpt(file: UploadFile, name: str = "", target_height_mm: float = 80.0,
                        x_actor: str = Header(default=USER)) -> dict:
    """Import an organic mesh file (.glb/.obj/.stl/.3mf) into Character Mode with watertight print-prep."""
    from pathlib import Path
    from .features.model import SculptPrep
    from .geometry.mesh_prep import MeshPrepError

    data = await file.read(100 * 1024 * 1024 + 1)
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "Mesh file is over 100 MB")
    ext = Path(file.filename or "mesh.glb").suffix.lower() or ".glb"
    try:
        sm = ws().add_sculpt(
            actor=x_actor,
            name=name or Path(file.filename or "Sculpt").stem,
            raw_bytes=data,
            ext=ext,
            prep=SculptPrep(target_height_mm=target_height_mm),
        )
    except MeshPrepError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"sculpt": sm.model_dump(), "tree": ws().tree()}


@app.post("/api/sculpts/{sid}/prep")
def update_sculpt_prep(sid: str, body: SculptPrepPatch, x_actor: str = Header(default=USER)) -> dict:
    from .geometry.mesh_prep import MeshPrepError

    try:
        sm = ws().update_sculpt_prep(x_actor, sid, body.model_dump(exclude_none=True))
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    except (MeshPrepError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"sculpt": sm.model_dump(), "tree": ws().tree()}


@app.get("/api/sculpts/{sid}/anatomy")
def get_sculpt_anatomy(sid: str) -> dict:
    from .geometry.mesh_prep import MeshPrepError

    try:
        return ws().inspect_sculpt_anatomy(sid)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    except (MeshPrepError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


class SculptPaintStrokeIn(BaseModel):
    mode: Literal["smart_fill", "brush", "bucket"] = "smart_fill"
    hit_point: list[float]
    filament: int = Field(default=3, ge=1, le=8)
    crease_deg: float = Field(default=26.0, ge=4.0, le=85.0)
    radius_mm: float = Field(default=4.0, ge=0.5, le=40.0)


class SculptPaletteIn(BaseModel):
    palette: dict[str, str]


@app.get("/api/sculpts/{sid}/paint")
def get_sculpt_paint(sid: str) -> dict:
    try:
        return ws().get_sculpt_paint(sid)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/sculpts/{sid}/paint/stroke")
def paint_sculpt_stroke(sid: str, body: SculptPaintStrokeIn, x_actor: str = Header(default=USER)) -> dict:
    if len(body.hit_point) != 3:
        raise HTTPException(422, "hit_point must have 3 coordinates [x, y, z]")
    try:
        state = ws().apply_sculpt_paint_stroke(
            x_actor, sid,
            mode=body.mode,
            hit_point=body.hit_point,
            filament=body.filament,
            crease_deg=body.crease_deg,
            radius_mm=body.radius_mm,
        )
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"paint": state, "part": ws().part()}


@app.post("/api/sculpts/{sid}/paint/palette")
def paint_sculpt_palette(sid: str, body: SculptPaletteIn, x_actor: str = Header(default=USER)) -> dict:
    try:
        state = ws().update_sculpt_palette(x_actor, sid, body.palette)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"paint": state, "part": ws().part()}


@app.post("/api/sculpts/{sid}/paint/undo")
def undo_sculpt_paint(sid: str, x_actor: str = Header(default=USER)) -> dict:
    try:
        state = ws().undo_sculpt_paint(x_actor, sid)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"paint": state, "part": ws().part()}


@app.post("/api/sculpts/{sid}/paint/reset")
def reset_sculpt_paint(sid: str, x_actor: str = Header(default=USER)) -> dict:
    try:
        state = ws().reset_sculpt_paint(x_actor, sid)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"paint": state, "part": ws().part()}


@app.post("/api/sculpts/{sid}/paint/ai")
def ai_paint_existing_sculpt(sid: str, wait_s: float = 0.0, x_actor: str = Header(default=USER)) -> dict:
    """Run local 3D Texture Painter (`Hunyuan3D-Paint`) on an existing sculpt using its approved concept image."""
    from .ml import ml_job_update, start_character_job
    from .ml.tex_painter import TexPainterError, paint_mesh_from_image

    w = ws()
    sm = w.design.sculpt(sid)
    if sm is None:
        raise HTTPException(404, f"No sculpt {sid!r}")
    raw_path = w.sculpts_dir / sm.source_file
    c_img = w._sculpt_concept_image(sm)
    if not raw_path.is_file():
        raise HTTPException(404, f"Sculpt mesh missing for {sid}")
    if c_img is None or not c_img.is_file():
        raise HTTPException(422, f"No concept image linked to sculpt {sid} to paint from")

    def _work() -> dict:
        painted_glb = paint_mesh_from_image(raw_path, c_img)
        ml_job_update("Finishing for 3D printing", 94, "Quantizing 3D colors for MMU3…")
        new_prep = sm.prep.model_copy(deep=True)
        new_prep.color_mode = "multicolor"
        new_prep.max_colors = max(5, min(5, w.project.printer.slots if w.project.printer.slots > 1 else 5))
        # Clear any stale manual paint override so the newly baked 3D texture takes effect immediately
        pp = w._sculpt_paint_path(sid)
        if pp.is_file():
            pp.unlink()
        sm_new = w.add_sculpt(
            actor=x_actor, name=sm.name, raw_bytes=painted_glb, ext=".glb",
            concept_id=sm.concept_id, prep=new_prep, color=sm.color, filament=sm.filament,
            replace_id=sid,
        )
        return {"sculpt": sm_new.model_dump(), "tree": w.tree()}

    if wait_s < 0:
        try:
            return _work()
        except TexPainterError as exc:
            raise HTTPException(422, str(exc)) from exc

    res = start_character_job("sculpting", "Painting 3D surface colors", sm.name, _work, wait_s=wait_s)
    if res["status"] == "error":
        raise HTTPException(422, res["error"])
    if res["status"] == "done" and res["result"] is not None:
        return res["result"]
    return res


@app.delete("/api/sculpts/{sid}")
def delete_sculpt(sid: str, x_actor: str = Header(default=USER)) -> dict:
    try:
        ws().delete_sculpt(x_actor, sid)
    except ChangeRefused as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": sid, "tree": ws().tree()}


# ---------------------------------------------------------------- AI chat (the panel's API mode)


_chats: dict[str, Any] = {}


def _stop_all_chats() -> None:
    """Terminate all active in-app AI turns and their subprocesses cleanly."""
    for c in list(_chats.values()):
        stop_fn = getattr(c, "stop", None)
        if callable(stop_fn):
            stop_fn()
        else:
            c.cancel.set()


atexit.register(_stop_all_chats)


def chat():
    w = ws()
    slug = w.project.slug
    c = _chats.get(slug)
    if c is None:
        from .ai.chat import Chat

        c = Chat(storage_path=w.project.root / "chat.json")
        _chats[slug] = c
    return c


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
    from .ml import ml_job_status

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
        "creation_mode": s.creation_mode,
        "direct_build": s.direct_build,
        "character_multicolor": s.character_multicolor,
        "settings_provider": s.ai_provider,
        "settings_model": s.ai_model,
        "available_models": available,
        "ml_job": ml_job_status(),
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
    chat().stop()
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


# Expose the MCP server over Streamable HTTP at /mcp so remote clients can connect by URL with no local repo.
from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402
from print_assist_mcp.server import mcp as _mcp_server  # noqa: E402

app.mount(
    "/mcp",
    _mcp_server.streamable_http_app(
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    ),
)


_static_dir = os.environ.get("PRINT_ASSIST_STATIC_DIR")
if _static_dir:
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    _static_root = Path(_static_dir).resolve()
    if _static_root.is_dir():
        _assets_dir = _static_root / "assets"
        if _assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def _spa(full_path: str) -> Response:
            if full_path.startswith(("api/", "mcp")):
                raise HTTPException(404)
            candidate = (_static_root / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(_static_root):
                return FileResponse(candidate)
            return FileResponse(_static_root / "index.html")
