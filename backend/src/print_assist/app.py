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

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .ai import generate as ai
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_projects: Projects | None = None
_ws: Workspace | None = None
generator = ai.Generator()


def projects() -> Projects:
    global _projects
    if _projects is None:
        _projects = Projects()
    return _projects


def ws() -> Workspace:
    """The open design: the most recently changed one, or a new one when there are none."""
    global _ws
    if _ws is None:
        listed = projects().list()
        p = projects().open(listed[0]["slug"]) if listed else projects().create("Untitled")
        _ws = Workspace(p)
    return _ws


def project():
    return ws().project


def reset(root=None) -> None:
    """Forget the open design (tests point the registry at a temporary folder)."""
    global _projects, _ws
    _projects = Projects(root) if root is not None else None
    _ws = None


def _not_generating() -> None:
    if generator.running():
        raise HTTPException(409, "The AI is changing the design; wait for it or cancel it")


def _change(ops: list[dict], actor: str, message: str = "", base_rev: int | None = None) -> dict:
    _not_generating()
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
    global _ws
    _not_generating()
    try:
        _ws = Workspace(projects().create(body.name))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _ws.project.summary()


@app.post("/api/projects/{slug}/open")
def open_project(slug: str) -> dict:
    global _ws
    _not_generating()
    try:
        _ws = Workspace(projects().open(slug))
    except KeyError:
        raise HTTPException(404, f"No design {slug!r}") from None
    return _ws.project.summary()


@app.get("/api/project")
def get_project() -> dict:
    return project().summary()


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
    _not_generating()
    ws().update()
    return ws().tree()


@app.get("/api/checks")
def get_checks() -> dict:
    from .validate import validate

    w = ws()
    return validate(w.last_good, w.project.data.rules, w.project.printer).to_dict() | {"stale": bool(w.pending())}


@app.get("/api/activity")
def get_activity(since: float = 0.0) -> list[dict]:
    return [a.to_dict() for a in ws().activity if a.at > since]


@app.get("/api/events")
async def events(request: Request, once: bool = False) -> StreamingResponse:
    """Server-sent events: one message whenever the design, its build or the activity log changes."""

    async def stream():
        last = None
        while True:
            w = ws()
            state = {"slug": w.project.slug, "rev": w.rev, "built_rev": w.built_rev, "activity": len(w.activity)}
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
    sk = project().data.design.sketch(sid)
    return out | {"sketch": sk.model_dump() if sk else None}


@app.get("/api/sketches/{sid}")
def get_sketch(sid: str) -> dict:
    sk = project().data.design.sketch(sid)
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

    fi = project().data.design.get(fid)
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

    fi = project().data.design.get(fid)
    if fi is None:
        raise HTTPException(404, f"No feature {fid!r}")
    f = pick_face(pick.body, pick.index)
    at = Vector(*f["at"])
    same = [i for i, x in enumerate(fi.faces) if (Vector(*x.at) - at).length < 0.05 or (x.name and x.name == f["name"] and x.normal == f["normal"])]
    faces = [x.model_dump() for i, x in enumerate(fi.faces) if i not in same]
    if not same:
        faces.append(f)
    return _change([{"op": "edit_feature", "id": fid, "faces": faces}], x_actor, f"{'Unpick' if same else 'Pick'} a face for {fid}")


# ---------------------------------------------------------------- formulas


class ExprIn(BaseModel):
    expr: str


@app.post("/api/expr/eval")
def eval_expr(body: ExprIn) -> dict:
    values = {k: v for k, v in resolve(project().data.design).values.items() if not isinstance(v, str)}
    try:
        return {"value": evaluate(body.expr, values)}
    except ExprError as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------- AI generation (one-shot; the chat replaces it)


class GenerateRequest(BaseModel):
    instruction: str = ""
    max_rounds: int = 6


def _job() -> dict | None:
    return generator.job.to_dict() if generator.job else None


@app.get("/api/ai/status")
def ai_status() -> dict:
    model = ai.model_name()  # also re-reads backend/.env
    return {"model": model, "key": bool(os.environ.get("ANTHROPIC_API_KEY")), "job": _job()}


@app.post("/api/ai/generate")
def ai_generate(body: GenerateRequest) -> dict:
    if generator.running():
        raise HTTPException(409, "A generation is already running")
    try:
        client = ai.make_client()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    generator.start(project(), client, body.instruction, max(1, min(body.max_rounds, 12)))
    return {"job": _job()}


@app.post("/api/ai/cancel")
def ai_cancel() -> dict:
    generator.cancel()
    return {"job": _job()}


@app.post("/api/ai/revert")
def ai_revert() -> dict:
    try:
        generator.revert(project())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": _job()}
