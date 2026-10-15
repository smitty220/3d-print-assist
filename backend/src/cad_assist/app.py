"""CAD Assist API.

Run with:  uv run uvicorn cad_assist.app:app --reload
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from build123d import Vector
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ai import generate as ai
from .catalog import FASTENERS
from .examples import EXAMPLES
from .features.context import FeatureError
from .features.library import FEATURES
from .features.model import EdgePick, FeatureInstance
from .features.templates import TEMPLATES
from .intent.models import MATERIALS, Rules
from .geometry.step_io import StepImportError
from .geometry.tessellate import mesh_shape
from .intent import builders
from .project import BUILDERS, Project, Role
from .topo import job as topo
from .features.context import BuildContext
from .features.context import frame_plane
from .features.design import describe_edge, design_frame
from .sketch.model import Sketch
from .sketch.solve import SketchSolveError, place_refs
from .sketch.solve import solve as solve_sketch
from .fem.job import FemRunner
from .topo import interpret
from .topo.machining import MachiningSpec
from .topo.setup import DEFAULT_BUDGET, TopoSetupError, build_setup

DATA_DIR = Path(os.environ.get("CAD_ASSIST_DATA", Path(__file__).resolve().parents[2] / "data"))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="CAD Assist", version="0.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_project: Project | None = None
generator = ai.Generator()
topo_runner = topo.Runner()
interpreter = interpret.Interpreter()
fem_runner = FemRunner()


def project() -> Project:
    global _project
    if _project is None:
        name = os.environ.get("CAD_ASSIST_PROJECT", "default")
        _project = Project(DATA_DIR / "projects" / name)
        if name in EXAMPLES and not _project.data.parts:
            EXAMPLES[name](_project)
        _project.seed_samples()
    return _project


def _shape_or_404(name: str):
    try:
        return project().shape(name)
    except KeyError:
        raise HTTPException(404, f"No part named {name!r}") from None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/catalog/fasteners")
def fasteners() -> list[dict]:
    return [f.to_mm_dict() for f in FASTENERS.values()]


@app.get("/api/project")
def get_project() -> dict:
    return project().summary()


@app.get("/api/project/parts/{name}/mesh")
def part_mesh(name: str) -> dict:
    return mesh_shape(_shape_or_404(name), name).to_dict()


@app.post("/api/project/parts")
async def upload_part(file: UploadFile) -> dict:
    filename = file.filename or "upload.stp"
    if Path(filename).suffix.lower() not in {".stp", ".step"}:
        raise HTTPException(415, "Upload a STEP file (.stp or .step)")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "STEP file is larger than 50 MB")
    try:
        warnings = project().add_part(Path(filename).stem, data)
    except (StepImportError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"project": project().summary(), "warnings": warnings}


class RoleUpdate(BaseModel):
    role: Role


@app.patch("/api/project/parts/{name}")
def set_role(name: str, body: RoleUpdate) -> dict:
    _shape_or_404(name)
    project().set_role(name, body.role)
    return project().summary()


@app.delete("/api/project/parts/{name}")
def delete_part(name: str) -> dict:
    _shape_or_404(name)
    project().remove_part(name)
    return project().summary()


class FromFace(BaseModel):
    kind: Literal["planar_mount", "bolt_pattern", "bore"]
    part: str
    face_id: int


@app.post("/api/project/intent/from-face")
def intent_from_face(body: FromFace) -> dict:
    shape = _shape_or_404(body.part)
    p = project()
    try:
        obj = BUILDERS[body.kind](p.data.intent, shape, body.part, body.face_id)
    except builders.BuildError as exc:
        raise HTTPException(422, str(exc)) from exc
    # Compare against the resolved face: picking a hole resolves a bolt pattern to the face it passes through.
    dup = next((o for o in p.data.intent.objects if o.kind == obj.kind and o.face == obj.face), None)
    if dup is not None:
        raise HTTPException(409, f"{dup.id} already uses that face")
    p.data.intent.objects.append(obj)
    p.save()
    return {"project": p.summary(), "created": obj.id}


@app.post("/api/project/intent/design-space")
def add_design_space() -> dict:
    p = project()
    existing = next((o for o in p.data.intent.objects if o.kind == "design_space"), None)
    if existing is not None:
        raise HTTPException(409, f"{existing.id} already defines the design space")
    try:
        obj = builders.default_design_space(p.data.intent)
    except builders.BuildError as exc:
        raise HTTPException(422, str(exc)) from exc
    p.data.intent.objects.append(obj)
    p.save()
    return {"project": p.summary(), "created": obj.id}


class NewLoad(BaseModel):
    target: str | None = None


@app.post("/api/project/intent/load-case")
def add_load_case(body: NewLoad) -> dict:
    p = project()
    try:
        obj = builders.new_load_case(p.data.intent, body.target)
    except builders.BuildError as exc:
        raise HTTPException(422, str(exc)) from exc
    p.data.intent.objects.append(obj)
    p.save()
    return {"project": p.summary(), "created": obj.id}


Vec3 = tuple[float, float, float]


class IntentPatch(BaseModel):
    label: str | None = None
    strength: Literal["hard", "soft"] | None = None
    fastener: str | None = None
    fit: Literal["floating", "fixed"] | None = None
    clearance: float | None = None
    min: Vec3 | None = None
    max: Vec3 | None = None
    target: str | None = None
    force: Vec3 | None = None
    moment: Vec3 | None = None
    category: Literal["limit", "ultimate"] | None = None
    factor: float | None = None
    support: Literal["auto", "fixed", "free", "normal", "pin", "link"] | None = None


SUPPORTS = {"planar_mount": {"auto", "free", "normal"}, "bolt_pattern": {"auto", "fixed", "free"}, "bore": {"auto", "fixed", "free", "pin", "link"}}


@app.patch("/api/project/intent/{obj_id}")
def patch_intent(obj_id: str, body: IntentPatch) -> dict:
    p = project()
    obj = p.data.intent.get(obj_id)
    if obj is None:
        raise HTTPException(404, f"No intent object {obj_id!r}")
    changes = body.model_dump(exclude_none=True)
    bad = [k for k in changes if k not in type(obj).model_fields]
    if bad:
        raise HTTPException(422, f"{obj_id} has no field(s) {', '.join(bad)}")
    if "fastener" in changes and changes["fastener"] not in FASTENERS:
        raise HTTPException(422, f"Unknown fastener {changes['fastener']!r}")
    if "target" in changes:
        t = p.data.intent.get(changes["target"])
        if t is None or t.kind not in ("planar_mount", "bolt_pattern", "bore"):
            raise HTTPException(422, f"{changes['target']} is not an interface")
    if "min" in changes or "max" in changes:
        lo, hi = changes.get("min", obj.min), changes.get("max", obj.max)
        if any(a >= b for a, b in zip(lo, hi)):
            raise HTTPException(422, "Design space min must be below max on every axis")
    if "support" in changes and changes["support"] not in SUPPORTS.get(obj.kind, set()):
        raise HTTPException(422, f"{obj_id} can't be supported as {changes['support']!r}")
    if changes.get("clearance", 0) < 0 or changes.get("factor", 1) <= 0:
        raise HTTPException(422, "Clearance must be ≥ 0 and factor > 0")
    for k, v in changes.items():
        setattr(obj, k, v)
    p.save()
    return p.summary()


@app.delete("/api/project/intent/{obj_id}")
def delete_intent(obj_id: str) -> dict:
    p = project()
    if p.data.intent.get(obj_id) is None:
        raise HTTPException(404, f"No intent object {obj_id!r}")
    p.delete_intent(obj_id)
    return p.summary()


@app.get("/api/materials")
def materials() -> dict[str, float]:
    return MATERIALS


@app.put("/api/project/rules")
def put_rules(rules: Rules) -> dict:
    if rules.material not in MATERIALS:
        raise HTTPException(422, f"Unknown material {rules.material!r}")
    p = project()
    p.data.intent.rules = rules
    p.save()
    return p.summary()


# ---------------------------------------------------------------- design (feature tree)


@app.get("/api/features/catalog")
def feature_catalog() -> list[dict]:
    return [f.describe() for f in FEATURES.values()]


@app.get("/api/templates")
def templates() -> list[dict]:
    return [{"name": k, "label": v[0], "summary": v[1]} for k, v in TEMPLATES.items()]


def _not_generating() -> None:
    if generator.running():
        raise HTTPException(409, "The AI is generating a design; wait for it or cancel it")
    if interpreter.running():
        raise HTTPException(409, "A parametric model is being made from the topology result; wait for it to finish")


def design_payload() -> dict:
    p = project()
    d = p.data.design
    sketches = [sk.model_dump() for sk in d.sketches]
    empty = {"features": [], "checks": [], "mass": 0.0, "volume": 0.0, "mesh": None, "face_names": [], "error": None, "sketches": sketches, "ai_draft": None}
    if not d.features:
        return empty
    try:
        r = p.build()
    except FeatureError as exc:
        return empty | {"error": str(exc)}
    pl = frame_plane(design_frame(p.data.intent))

    def world(pts):
        return [[round(c, 4) for c in tuple(pl.from_local_coords(Vector(*q)))] for q in pts]

    features = [fr.to_dict() | {"locked": fi.locked, "refs": fi.refs,
                                "edges": [e.model_dump() | {"world": world(e.points)} for e in fi.edges]}
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
        "ai_draft": {"summary": d.recipe_summary, "questions": d.recipe_questions} if d.recipe else None,
    }


class SketchUpdate(BaseModel):
    name: str | None = None
    entities: list[dict]
    constraints: list[dict]


@app.put("/api/project/design/sketches/{sid}")
def put_sketch(sid: str, body: SketchUpdate) -> dict:
    """Replace a sketch's geometry and constraints, solve it (points tied to pins follow the intent) and rebuild."""
    _not_generating()
    p = project()
    d = p.data.design
    old = d.sketch(sid)
    if old is None:
        raise HTTPException(404, f"No sketch {sid!r}")
    try:
        sk = Sketch(id=sid, name=body.name if body.name is not None else old.name, entities=body.entities, constraints=body.constraints)
        frame = design_frame(p.data.intent)
        sk = solve_sketch(place_refs(sk, p.data.intent, BuildContext(intent=p.data.intent, frame=frame).to_local))
    except (ValueError, SketchSolveError, FeatureError) as exc:
        raise HTTPException(422, str(exc)) from exc
    d.sketches = [sk if s.id == sid else s for s in d.sketches]
    p.save()
    return design_payload() | {"sketch": sk.model_dump()}


class OffsetRequest(BaseModel):
    source: str  # the profile sketch to offset from
    rim: float = Field(0.15 * 25.4, gt=0)  # mm, wall left inside the profile's outer edge
    boss_gap: float = 0.0  # mm, added to each lug's radius for the boss the pocket keeps clear of
    corner_radius: float = Field(0.125 * 25.4, ge=0)  # mm, the cutter


@app.post("/api/project/design/sketches/{sid}/offset")
def offset_sketch(sid: str, body: OffsetRequest) -> dict:
    """Redraw a pocket sketch as a standard offset inside a profile sketch (rim, bosses at the lugs), then solve and rebuild."""
    from .topo.sketch_seed import offset_pocket

    _not_generating()
    p = project()
    d = p.data.design
    old, src = d.sketch(sid), d.sketch(body.source)
    if old is None or src is None:
        raise HTTPException(404, f"No sketch {sid if old is None else body.source!r}")
    try:
        sk = solve_sketch(offset_pocket(src, sid, old.name, body.rim, body.corner_radius, body.boss_gap))
    except (ValueError, SketchSolveError, FeatureError) as exc:
        raise HTTPException(422, str(exc)) from exc
    d.sketches = [sk if s.id == sid else s for s in d.sketches]
    p.save()
    return design_payload() | {"sketch": sk.model_dump()}


@app.get("/api/project/validation")
def get_validation() -> dict:
    """Pre-flight findings on the intent plus checks of the built part (empty when there is no part)."""
    try:
        return project().validate().to_dict()
    except FeatureError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/project/design")
def get_design() -> dict:
    return design_payload()


class TemplateRequest(BaseModel):
    name: str


@app.post("/api/project/design/template")
def apply_template(body: TemplateRequest) -> dict:
    _not_generating()
    if body.name not in TEMPLATES:
        raise HTTPException(404, f"No template named {body.name!r}")
    p = project()
    try:
        p.data.design = TEMPLATES[body.name][2](p.data.intent)
    except FeatureError as exc:
        raise HTTPException(422, f"{TEMPLATES[body.name][0]}: {exc}") from exc
    p.save()
    return design_payload()


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
    replay = ai.replay_mode()
    model = ai.model_name()  # also re-reads backend/.env
    return {"model": "replay (offline)" if replay else model, "key": replay or bool(os.environ.get("ANTHROPIC_API_KEY")), "job": _job()}


@app.post("/api/ai/generate")
def ai_generate(body: GenerateRequest) -> dict:
    p = project()
    if generator.running():
        raise HTTPException(409, "A generation is already running")
    blocking = [f for f in p.validate().preflight if f.status == "fail"]
    if blocking:
        raise HTTPException(422, "Pre-flight failed: " + "; ".join(f"{f.title}: {f.detail}" for f in blocking))
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


# ---------------------------------------------------------------- topology optimisation

_setup_cache: tuple[str, int, object] | None = None


def _topo_setup(budget: int):
    global _setup_cache
    p = project()
    key = topo.intent_key(p)
    if _setup_cache is None or _setup_cache[:2] != (key, budget):
        roles = {k: v.role for k, v in p.data.parts.items()}
        try:
            _setup_cache = (key, budget, build_setup(p.data.intent, p.shape, roles, budget))
        except TopoSetupError as exc:
            raise HTTPException(422, str(exc)) from exc
    return _setup_cache[2]


@app.get("/api/topo/setup")
def topo_setup(budget: int = DEFAULT_BUDGET) -> dict:
    """How the design space will be meshed, supported and loaded, before running."""
    s = _topo_setup(budget)
    return {"summary": s.summary(), "preview": s.preview()}


def _topo_state() -> dict:
    p = project()
    topo_runner.attach(p.root / "topo")
    r = topo_runner.result
    return {
        "job": topo_runner.job.to_dict() if topo_runner.job else None,
        "result": r.meta() if r else None,
        "stale": r is not None and r.key != topo.intent_key(p),
        "default_budget": DEFAULT_BUDGET,
    }


@app.get("/api/topo/status")
def topo_status() -> dict:
    return _topo_state()


class MachiningRequest(BaseModel):
    axis: Literal["auto", "x", "y", "z"] = "auto"
    depths: int = Field(3, ge=1, le=6)
    free_end: Literal["none", "+", "-"] = "none"
    tool_diameter: float = Field(6.35, gt=0, le=50)


class TopoRunRequest(BaseModel):
    budget: int = DEFAULT_BUDGET
    mass: float | None = None  # kg; defaults to the project's mass target
    sweep: bool = False
    machining: MachiningRequest | None = None  # None: raw (unconstrained) only


@app.post("/api/topo/run")
def topo_run(body: TopoRunRequest) -> dict:
    if topo_runner.running():
        raise HTTPException(409, "A topology optimisation is already running")
    try:
        spec = MachiningSpec(**body.machining.model_dump()) if body.machining else None
        topo_runner.start(project(), body.budget, body.mass, body.sweep, spec)
    except (TopoSetupError, topo.TopoRunError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return _topo_state()


@app.post("/api/topo/cancel")
def topo_cancel() -> dict:
    topo_runner.cancel()
    return _topo_state()


class InterpretRequest(BaseModel):
    step: int | None = None  # defaults to the result's target step
    mode: Literal["rules", "ai"] = "rules"
    instruction: str = Field("", max_length=4000)  # the engineer's request or feedback, for an AI draft
    max_builds: int = Field(5, ge=1, le=10)
    stress: bool = False  # AI drafts: a coarse stress analysis on every build


@app.post("/api/topo/interpret")
def topo_interpret(body: InterpretRequest) -> dict:
    """Read a topology result as machined features and replace the design with them, step by step."""
    _not_generating()
    if topo_runner.running():
        raise HTTPException(409, "A topology optimisation is running")
    p = project()
    topo_runner.attach(p.root / "topo")
    r = topo_runner.result
    if r is None:
        raise HTTPException(404, "No topology results yet")
    step = body.step if body.step is not None else (r.target_step or 0)
    if not 0 <= step < len(r.steps):
        raise HTTPException(404, f"No step {step}")
    if r.key != topo.intent_key(p):
        raise HTTPException(409, "The set-up has changed since this result; rerun the topology optimisation first")
    client, model = None, ""
    if body.mode == "ai":
        try:
            client, model = ai.make_client(), ai.model_name()
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
    interpreter.start(p, r, step, lambda: _topo_setup(r.budget), body.mode, body.instruction, client, body.max_builds, model, body.stress)
    return {"job": interpreter.job.to_dict()}


@app.post("/api/topo/interpret/cancel")
def topo_interpret_cancel() -> dict:
    interpreter.cancel()
    return {"job": interpreter.job.to_dict() if interpreter.job else None}


@app.get("/api/topo/interpret")
def topo_interpret_status() -> dict:
    return {"job": interpreter.job.to_dict() if interpreter.job else None}


@app.get("/api/topo/interpret/preview")
def topo_interpret_preview() -> dict:
    """The part as far as it has been built, while a parametric model is being made."""
    job = interpreter.job
    if job is None or job.preview is None:
        raise HTTPException(404, "Nothing built yet")
    return {"revision": job.revision, "mesh": job.preview}


@app.get("/api/topo/underlay")
def topo_underlay(step: int | None = None) -> dict:
    """Thickness (elements through Z) of a result per grid column, in the design frame: the sketcher's underlay."""
    topo_runner.attach(project().root / "topo")
    r = topo_runner.result
    if r is None:
        raise HTTPException(404, "No topology results yet")
    i = step if step is not None else (r.target_step or 0)
    if not 0 <= i < len(r.x):
        raise HTTPException(404, f"No step {i}")
    th = (r.x[i] >= 0.5).sum(axis=2)
    return {"nx": int(th.shape[0]), "ny": int(th.shape[1]), "nz": int(r.x[i].shape[2]), "h": r.h,
            "x0": r.origin[0], "y0": r.origin[1], "thickness": th.T.ravel().astype(int).tolist()}  # row-major in y


@app.get("/api/topo/mesh")
def topo_mesh(step: str = "0", cutoff: float = 0.5) -> dict:
    """Iso-surface of one sweep step (or the run in progress, step=live) at a density cutoff."""
    topo_runner.attach(project().root / "topo")
    job, r = topo_runner.job, topo_runner.result
    if step == "live":
        live = job.live if job is not None else None
        if live is None or topo_runner.current is None:
            raise HTTPException(404, "No run in progress")
        return topo_runner.current.mesh(live, cutoff) | {"step": "live"}
    if r is None:
        raise HTTPException(404, "No topology results yet")
    try:
        i = int(step)
        x = r.x[i]
    except (ValueError, IndexError):
        raise HTTPException(404, f"No step {step!r}") from None
    return r.mesh(x, cutoff) | {"step": i}


# ---------------------------------------------------------------- stress analysis (FEA)


class FemRunRequest(BaseModel):
    preset: Literal["coarse", "medium", "fine"] = "medium"
    convergence: bool = False
    bearing: Literal["cosine", "contact"] = "cosine"
    peaks: bool = False  # converge the peaks by local refinement
    peaks_contact: bool = False  # with pin contact in every round (slow); else cosine times the contact factors


def _fem_state() -> dict:
    if not fem_runner.running():
        fem_runner.attach(project().root / "fem")
    p = project()
    key = p.data.design.sketch_key() + str(len(p.data.design.features))

    def dump(job):
        d = job.to_dict() if job else None
        if d is not None and job.status == "done" and job.mode in ("analyse", "size"):
            d["stale"] = job.design_key != key
        return d

    # `job`: the latest job of any kind (log, progress); `results`: the latest analysis or sizing (what the results show).
    return {"job": dump(fem_runner.job), "results": dump(fem_runner.results_job)}


@app.post("/api/fem/run")
def fem_run(body: FemRunRequest) -> dict:
    """Mesh and analyse the current part under every load case, in the background."""
    if fem_runner.running():
        raise HTTPException(409, "An analysis is already running")
    if not project().data.design.features:
        raise HTTPException(422, "There is no part to analyse yet")
    fem_runner.start(project(), body.preset, body.convergence, body.bearing, body.peaks, body.peaks_contact)
    return _fem_state()


@app.get("/api/fem/status")
def fem_status() -> dict:
    return _fem_state()


@app.get("/api/fem/field")
def fem_field(case: str, quantity: Literal["vm", "p1", "u"] = "vm") -> dict:
    if not fem_runner.running():
        fem_runner.attach(project().root / "fem")
    try:
        return fem_runner.field(case, quantity)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


class SizeRequest(BaseModel):
    target_ms: float = Field(0.10, ge=-0.5, le=2.0)
    max_evals: int = Field(20, ge=3, le=60)


@app.post("/api/fem/size")
def fem_size(body: SizeRequest) -> dict:
    """Find the lightest version of the drafted part that meets the margin (coarse analysis per trial)."""
    _not_generating()
    if fem_runner.running():
        raise HTTPException(409, "An analysis is already running")
    if not project().data.design.recipe:
        raise HTTPException(422, "Sizing works on a drafted part (Draft with AI); this design has no recipe")
    fem_runner.start_sizing(project(), body.target_ms, body.max_evals)
    return _fem_state()


# ---- mesh controls: set (by hand, by the rules or by Claude), mesh and check, then solve


def _fem_part():
    b = project().build()
    if b.world is None:
        raise HTTPException(422, "There is no part yet")
    return b.world


def _setup_payload() -> dict:
    from .fem.controls import DEFAULT_PRACTICES, resolve

    p = project()
    s = p.data.mesh
    b = p.build()
    found = resolve(s, b.world) if b.world is not None else {"edges": [None] * len(s.edges), "faces": [None] * len(s.faces)}
    return {"setup": json.loads(s.model_dump_json()), "resolved": found, "default_practices": DEFAULT_PRACTICES,
            "mesh_key": fem_runner.mesh_key(p)}


@app.get("/api/fem/setup")
def fem_setup() -> dict:
    return _setup_payload()


class EdgeIn(BaseModel):
    edge: int
    n: int = Field(ge=1, le=500)
    label: str = ""
    ref: list[float] | None = None
    length: float | None = None


class FaceIn(BaseModel):
    face: int
    label: str = ""
    ref: list[float] | None = None
    area: float | None = None


class SetupIn(BaseModel):
    size: float = Field(gt=0.1, le=100)
    curvature: int = Field(ge=0, le=120)
    min_size: float = Field(gt=0.01, le=100)
    edges: list[EdgeIn] = Field(default_factory=list)
    faces: list[FaceIn] = Field(default_factory=list)
    spheres: list[dict] = Field(default_factory=list)
    practices: str | None = None
    notes: list[str] | None = None


@app.put("/api/fem/setup")
def fem_setup_put(body: SetupIn) -> dict:
    """Save edited controls. Items with `ref` keep it (an existing control); new ones are measured on the current part."""
    from .fem.controls import EdgeDivision, MappedFace, MeshSetup, Sphere, edge_ref, face_ref, mappable

    if fem_runner.running():
        raise HTTPException(409, "Wait for the running job")
    p = project()
    old = p.data.mesh
    part = None
    edges, faces = [], []
    for e in body.edges:
        if e.ref is None:
            part = part or _fem_part()
            if not 0 <= e.edge < len(part.edges()):
                raise HTTPException(422, f"No edge {e.edge} on the part")
            edges.append(EdgeDivision(edge=e.edge, n=e.n, label=e.label, **edge_ref(part, e.edge)))
        else:
            edges.append(EdgeDivision(edge=e.edge, n=e.n, label=e.label, ref=e.ref, length=e.length or 0.0))
    for f in body.faces:
        if f.ref is None:
            part = part or _fem_part()
            if not 0 <= f.face < len(part.faces()):
                raise HTTPException(422, f"No face {f.face} on the part")
            why = mappable(part, f.face)
            if why:
                raise HTTPException(422, f"That face cannot be mapped: {why}")
            faces.append(MappedFace(face=f.face, label=f.label, **face_ref(part, f.face)))
        else:
            faces.append(MappedFace(face=f.face, label=f.label, ref=f.ref, area=f.area or 0.0))
    try:
        spheres = [Sphere.model_validate(x) for x in body.spheres]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    new = MeshSetup(size=body.size, curvature=body.curvature, min_size=min(body.min_size, body.size), edges=edges, faces=faces,
                    spheres=spheres, practices=body.practices if body.practices is not None else old.practices,
                    notes=body.notes if body.notes is not None else old.notes, source=old.source)
    if new.key() != old.key():
        base = old.source.removesuffix(", edited")
        new.source = "user" if base in ("none", "user") else f"{base}, edited"
    p.data.mesh = new
    p.save()
    return _setup_payload()


class ProposeRequest(BaseModel):
    preset: Literal["coarse", "medium", "fine"] = "medium"


@app.post("/api/fem/setup/propose")
def fem_setup_propose(body: ProposeRequest) -> dict:
    """Replace the controls with the best-practice rules at a preset density."""
    from .fem.controls import propose

    if fem_runner.running():
        raise HTTPException(409, "Wait for the running job")
    p = project()
    p.data.mesh = propose(p.data.intent, _fem_part(), body.preset, p.data.mesh.practices)
    p.save()
    return _setup_payload()


@app.post("/api/fem/mesh")
def fem_mesh_start() -> dict:
    """Mesh the part with its controls (no solve) for checking."""
    if fem_runner.running():
        raise HTTPException(409, "An analysis is already running")
    _fem_part()
    fem_runner.start_mesh(project())
    return _fem_state()


@app.get("/api/fem/mesh")
def fem_mesh_get() -> dict:
    if not fem_runner.running():
        fem_runner.attach(project().root / "fem")
    m = fem_runner.mesh_payload(project())
    if m is None:
        raise HTTPException(404, "No mesh yet")
    return m


class PlanRequest(BaseModel):
    instruction: str = ""
    max_meshes: int = Field(3, ge=1, le=6)


@app.post("/api/fem/plan")
def fem_plan(body: PlanRequest) -> dict:
    """Claude proposes mesh controls from the best practices, meshing and checking each proposal."""
    if fem_runner.running():
        raise HTTPException(409, "An analysis is already running")
    _fem_part()
    try:
        client, model = ai.make_client(), ai.model_name()
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    fem_runner.start_plan(project(), client, model, body.instruction, body.max_meshes)
    return _fem_state()


@app.post("/api/fem/cancel")
def fem_cancel() -> dict:
    fem_runner.cancel()
    return _fem_state()


@app.post("/api/fem/size/revert")
def fem_size_revert() -> dict:
    try:
        fem_runner.revert_sizing(project())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _fem_state() | {"design": design_payload()}
