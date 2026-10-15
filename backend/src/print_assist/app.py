"""3D Print Assist API.

Run with:  uv run uvicorn print_assist.app:app --reload
"""

from __future__ import annotations

import os

from build123d import Vector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ai import generate as ai
from .features.context import FeatureError
from .features.design import describe_edge
from .features.library import FEATURES
from .features.model import EdgePick, FeatureInstance
from .geometry.tessellate import mesh_shape
from .printers import profiles
from .project import Project, Projects
from .sketch.model import Sketch
from .sketch.solve import SketchSolveError
from .sketch.solve import solve as solve_sketch

app = FastAPI(title="3D Print Assist", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_projects: Projects | None = None
_project: Project | None = None
generator = ai.Generator()


def projects() -> Projects:
    global _projects
    if _projects is None:
        _projects = Projects()
    return _projects


def project() -> Project:
    """The open design: the most recently changed one, or a new one when there are none."""
    global _project
    if _project is None:
        listed = projects().list()
        _project = projects().open(listed[0]["slug"]) if listed else projects().create("Untitled")
    return _project


def reset(root=None) -> None:
    """Forget the open design (tests point the registry at a temporary folder)."""
    global _projects, _project
    _projects = Projects(root) if root is not None else None
    _project = None


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
    global _project
    _not_generating()
    try:
        _project = projects().create(body.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _project.summary()


@app.post("/api/projects/{slug}/open")
def open_project(slug: str) -> dict:
    global _project
    _not_generating()
    try:
        _project = projects().open(slug)
    except KeyError:
        raise HTTPException(404, f"No design {slug!r}") from None
    return _project.summary()


@app.get("/api/project")
def get_project() -> dict:
    return project().summary()


@app.get("/api/project/history")
def get_history() -> list[dict]:
    return project().history()


@app.get("/api/printers")
def get_printers() -> list[dict]:
    return [p.model_dump() for p in profiles().values()]


@app.get("/api/features/catalog")
def feature_catalog() -> list[dict]:
    return [f.describe() for f in FEATURES.values()]


# ---------------------------------------------------------------- design


def _not_generating() -> None:
    if generator.running():
        raise HTTPException(409, "The AI is changing the design; wait for it or cancel it")


def design_payload() -> dict:
    p = project()
    d = p.data.design
    sketches = [sk.model_dump() for sk in d.sketches]
    empty = {"features": [], "checks": [], "mass": 0.0, "volume": 0.0, "mesh": None, "face_names": [], "error": None,
             "sketches": sketches}
    if not d.features:
        return empty
    try:
        r = p.build()
    except FeatureError as exc:
        return empty | {"error": str(exc)}
    features = [fr.to_dict() | {"locked": fi.locked, "refs": fi.refs,
                                "edges": [e.model_dump() | {"world": e.points} for e in fi.edges]}
                for fr, fi in zip(r.features, d.features)]
    return {
        "features": features,
        "checks": [c.to_dict() for c in r.checks],
        "mass": r.mass,
        "volume": r.volume,
        "mesh": mesh_shape(r.world, "part").to_dict() if r.world is not None else None,
        "face_names": r.face_names,
        "error": None,
        "sketches": sketches,
    }


@app.get("/api/project/design")
def get_design() -> dict:
    return design_payload()


@app.get("/api/project/validation")
def get_validation() -> dict:
    return project().validate().to_dict()


class SketchUpdate(BaseModel):
    name: str | None = None
    entities: list[dict]
    constraints: list[dict]


@app.put("/api/project/design/sketches/{sid}")
def put_sketch(sid: str, body: SketchUpdate) -> dict:
    """Replace a sketch's geometry and constraints, solve it and rebuild."""
    _not_generating()
    p = project()
    d = p.data.design
    old = d.sketch(sid)
    if old is None:
        raise HTTPException(404, f"No sketch {sid!r}")
    try:
        sk = Sketch(id=sid, name=body.name if body.name is not None else old.name, entities=body.entities,
                    constraints=body.constraints)
        sk = solve_sketch(sk)
    except (ValueError, SketchSolveError, FeatureError) as exc:
        raise HTTPException(422, str(exc)) from exc
    d.sketches = [sk if s.id == sid else s for s in d.sketches]
    p.save()
    return design_payload() | {"sketch": sk.model_dump()}


class NewFeature(BaseModel):
    kind: str
    refs: dict[str, str] = {}
    params: dict[str, float | str | bool | list[list[float]]] = {}


@app.post("/api/project/design/features")
def add_feature(body: NewFeature) -> dict:
    _not_generating()
    if body.kind not in FEATURES:
        raise HTTPException(422, f"Unknown feature kind {body.kind!r}")
    p = project()
    d = p.data.design
    d.features.append(FeatureInstance(id=d.next_id(), kind=body.kind, refs=body.refs, params=body.params))
    p.save()
    return design_payload() | {"created": d.features[-1].id}


class FeaturePatch(BaseModel):
    params: dict[str, float | str | bool | list[list[float]]] | None = None
    refs: dict[str, str] | None = None
    edges: list[EdgePick] | None = None
    locked: bool | None = None
    suppressed: bool | None = None


@app.patch("/api/project/design/features/{fid}")
def patch_feature(fid: str, body: FeaturePatch) -> dict:
    _not_generating()
    p = project()
    fi = p.data.design.get(fid)
    if fi is None:
        raise HTTPException(404, f"No feature {fid!r}")
    if body.params is not None:
        known = {s.name for s in FEATURES[fi.kind].params}
        bad = sorted(set(body.params) - known)
        if bad:
            raise HTTPException(422, f"{fi.kind} has no parameter(s) {', '.join(bad)}")
        fi.params = fi.params | body.params
    if body.refs is not None:
        fi.refs = fi.refs | body.refs
    if body.edges is not None:
        fi.edges = body.edges
    if body.locked is not None:
        fi.locked = body.locked
    if body.suppressed is not None:
        fi.suppressed = body.suppressed
    p.save()
    return design_payload()


class EdgePickIn(BaseModel):
    edge: int  # index into the displayed part's edges (its mesh's edge ids)


EDGE_KINDS = ("fillet", "chamfer")


@app.post("/api/project/design/features/{fid}/edges")
def pick_feature_edge(fid: str, body: EdgePickIn) -> dict:
    """Add an edge picked on the part to a fillet or chamfer, or take it off again if it is already picked. The edge is
    kept by its geometry, so it is found on the part as it is before this feature, and after upstream edits."""
    _not_generating()
    p = project()
    fi = p.data.design.get(fid)
    if fi is None:
        raise HTTPException(404, f"No feature {fid!r}")
    if fi.kind not in EDGE_KINDS:
        raise HTTPException(422, f"{fid} is a {fi.kind}; only fillets and chamfers take edges")
    r = p.build()
    if r.local is None:
        raise HTTPException(422, "There is no part to pick on yet")
    try:
        pick = describe_edge(r.local, r.face_names, body.edge)
    except FeatureError as exc:
        raise HTTPException(422, str(exc)) from exc
    at = Vector(*pick.at)
    same = [i for i, e in enumerate(fi.edges) if (Vector(*e.at) - at).length < 0.05]
    if same:
        fi.edges = [e for i, e in enumerate(fi.edges) if i not in same]
    else:
        fi.edges = [*fi.edges, pick]
        res = next(x for x in p.build().features if x.id == fid)
        if res.error and "is not on the part" in res.error:
            fi.edges = fi.edges[:-1]
            raise HTTPException(422, f"That edge is made after {fid} in the tree; move the blend later or pick an earlier edge")
    p.save()
    return design_payload() | {"picked": not same}


@app.delete("/api/project/design/features/{fid}")
def delete_feature(fid: str) -> dict:
    _not_generating()
    p = project()
    if p.data.design.get(fid) is None:
        raise HTTPException(404, f"No feature {fid!r}")
    p.data.design.features = [f for f in p.data.design.features if f.id != fid]
    p.save()
    return design_payload()


# ---------------------------------------------------------------- AI generation


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
    p = project()
    if generator.running():
        raise HTTPException(409, "A generation is already running")
    try:
        client = ai.make_client()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    generator.start(p, client, body.instruction, max(1, min(body.max_rounds, 12)))
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
