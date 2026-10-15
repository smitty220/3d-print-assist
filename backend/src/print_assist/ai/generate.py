"""AI generation loop: Claude composes library features into a part, scored by the validator.

Claude never writes geometry. It proposes a feature tree (the same features a user edits), the backend builds
it and runs the validator, and Claude revises from the findings until the part passes or it runs out of rounds.
Locked features are carried through unchanged whatever Claude proposes.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from ..features.context import FeatureError
from ..features.design import build_design
from ..features.library import FEATURES
from ..features.model import Design, FeatureInstance
from ..validate import Report, validate

DEFAULT_MODEL = "claude-opus-5-5"


def _param_text(v) -> str:
    if isinstance(v, float):
        return str(round(v, 3))
    if isinstance(v, list):
        return f"<outline, {len(v)} points>"
    return str(v)


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic().messages the loop uses (tests pass a scripted fake)."""

    def create(self, **kwargs: Any) -> Any: ...


ENV_FILE = Path(__file__).resolve().parents[3] / ".env"  # backend/.env


def load_env_key(env: Path = ENV_FILE) -> None:
    """Read ANTHROPIC_API_KEY (and PRINT_ASSIST_MODEL, ANTHROPIC_WORKSPACE_ID) from backend/.env unless the shell set them.

    Called before every AI request, so a key added to the file works without restarting the backend.
    """
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        k = k.removeprefix("export ").strip()
        if sep and k in ENV_KEYS and k not in _SHELL_ENV:
            os.environ[k] = v.strip().strip('"').strip("'")


ENV_KEYS = ("ANTHROPIC_API_KEY", "PRINT_ASSIST_MODEL", "ANTHROPIC_WORKSPACE_ID")
_SHELL_ENV = {k for k in ENV_KEYS if os.environ.get(k)}


def model_name() -> str:
    load_env_key()
    return os.environ.get("PRINT_ASSIST_MODEL") or DEFAULT_MODEL


def make_client() -> MessagesClient:
    load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("No Anthropic API key: set ANTHROPIC_API_KEY in your shell or in backend/.env, then restart the backend")
    import anthropic

    # An organisation-level key (not scoped to a workspace) must name the workspace to bill on every request.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None).messages


# ---------------------------------------------------------------- job state


@dataclass
class Step:
    kind: Literal["note", "build", "finish", "error"]
    text: str
    round: int = 0
    counts: dict[str, int] | None = None
    mass: float | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Job:
    id: str
    instruction: str
    model: str
    max_rounds: int
    status: Literal["running", "done", "failed", "cancelled"] = "running"
    steps: list[Step] = field(default_factory=list)
    rounds: int = 0
    design_rev: int = 0  # bumps whenever the project's design is replaced, so the UI knows to refetch
    summary: str = ""
    error: str | None = None
    best: dict | None = None  # {"round", "counts", "mass"}
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "cache_read": 0})
    started: float = field(default_factory=time.time)
    finished: float | None = None
    can_revert: bool = False
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    before: Design | None = field(default=None, repr=False)  # the design when the job started, for revert

    def to_dict(self) -> dict:
        # Not asdict(): it would deep-copy the threading.Event (unpicklable) and the Design.
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("cancel", "before")}
        d["steps"] = [asdict(s) for s in self.steps]
        return json.loads(json.dumps(d))


# ---------------------------------------------------------------- prompt


SYSTEM = """You are the design engine inside 3D Print Assist, a parametric CAD tool for 3D-printed products. You \
design parts by composing parametric library features. You never write geometry yourself: you propose a feature \
tree, the tool builds it with OpenCascade and runs print checks, and you revise from the results.

Units and frame:
- All lengths are millimetres. Z is up and the print bed is Z = 0; parts are built in this world frame.

How the feature tree works:
- A design is an ordered list of features. Each has an id ("F1", "F2", ...), a kind from the catalog, params \
(numbers in mm, or a choice string) and refs. A ref points at an earlier feature id ("F1") or a sketch id. Edge \
features (fillet, chamfer) take face groups such as "F1.top" or "F3" for refs faces_a / faces_b and act on the \
edges between them. A fillet or chamfer shown with "picked edges" uses edges the user picked on the part: keep \
its id and kind and leave faces_a / faces_b out, and the picks carry over.
- Features build in order: add features fuse, cuts subtract, fillet/chamfer modify edges.
- Params outside a spec's min/max are clamped; the build result shows the resolved values.
- Features marked locked were fixed by the user. Keep them exactly as given; the tool restores them if you \
change or drop them.

Working method:
1. Call build_and_check with a complete design. Read every failure and warning.
2. Revise and rebuild. Change a few things per round and say why in one short sentence before the call.
3. Stop when there are no failures and nothing more worth improving, or when you cannot make progress. Then \
call finish.

Priorities, in order: no build errors; no print-check failures (fits the bed, minimum wall, rule checks); then \
fewer warnings; then simplicity (fewer features, printable without supports where possible).

Keep your text short and specific. The user reads every note you write, between builds, as a log."""


def _catalog() -> list[dict]:
    return [f.describe() for f in FEATURES.values()]


TOOLS = [
    {
        "name": "build_and_check",
        "description": "Build a complete design (replacing the current one) and run the print checks. Returns build errors, resolved params, mass and every finding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "One sentence for the user: what this round changes and why."},
                "features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string", "enum": list(FEATURES)},
                            "params": {"type": "object"},
                            "refs": {"type": "object"},
                            "suppressed": {"type": "boolean"},
                        },
                        "required": ["id", "kind"],
                    },
                },
            },
            "required": ["note", "features"],
        },
    },
    {
        "name": "finish",
        "description": "End the session. The best design built so far is kept in the project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "For the user, 2-5 sentences: what the part is, any remaining warnings and why they stand, anything they should decide.",
                }
            },
            "required": ["summary"],
        },
    },
]


# ---------------------------------------------------------------- the loop


def _score(report: Report, mass: float) -> tuple:
    c = report.to_dict()["counts"]
    return (c["fail"], c["warn"], mass)


def _report_text(design: Design, build, report: Report, kept: list[str]) -> str:
    lines = []
    if kept:
        lines.append(f"Locked features restored as the user set them: {', '.join(kept)}.")
    for fr, fi in zip(build.features, design.features):
        state = "ERROR " + fr.error if fr.error else ("suppressed" if fr.suppressed else "ok")
        params = ", ".join(f"{k}={_param_text(v)}" for k, v in fr.params.items())
        driven = f" driven: {fr.driven}" if fr.driven else ""
        picked = f" picked edges: {len(fi.edges)}" if fi.edges else ""
        lines.append(f"{fi.id} {fi.kind} [{state}] params: {params} refs: {fi.refs}{picked}{driven}")
    lines.append(f"Mass {build.mass * 1000:.1f} g, volume {build.volume / 1000:.1f} cm³")
    counts = report.to_dict()["counts"]
    lines.append(f"Validator: {counts['fail']} fail, {counts['warn']} warn, {counts['ok']} ok")
    for f in report.preflight + report.part:
        if f.code == "mass":
            continue
        if f.status != "ok" or f.code in ("bed", "wall"):
            val = f" (value {f.value:.3f} mm)" if f.value is not None and f.quantity == "length" else ""
            lines.append(f"[{f.status}] {f.title}: {f.detail}{val}")
    return "\n".join(lines)


def _restore_locked(proposed: list[FeatureInstance], base: Design) -> tuple[list[FeatureInstance], list[str]]:
    locked = {f.id: f for f in base.features if f.locked}
    before = {f.id: f for f in base.features}
    kept, out, seen = [], [], set()
    for f in proposed:
        if f.id in locked:
            if f.model_dump() != locked[f.id].model_dump():
                kept.append(f.id)
            out.append(locked[f.id].model_copy(deep=True))
            seen.add(f.id)
        else:
            # Picked edges are the user's (Claude cannot pick): they stay with the same fillet / chamfer unless
            # Claude gives it face groups instead.
            old = before.get(f.id)
            update: dict = {"locked": False}
            if old and old.kind == f.kind and old.edges and not f.edges and not (f.refs.get("faces_a") and f.refs.get("faces_b")):
                update["edges"] = [e.model_copy() for e in old.edges]
            out.append(f.model_copy(update=update))
    for fid, f in locked.items():
        if fid not in seen:
            kept.append(fid)
            out.append(f.model_copy(deep=True))
    return out, kept


def run(job: Job, project, client: MessagesClient, lock: threading.Lock) -> None:
    """Run one generation session to completion, writing each candidate into the project as it is built."""
    base = project.data.design.model_copy(deep=True)
    before = base.model_copy(deep=True)
    best: list = [None]  # [(score, design)], shared with the tool calls

    first = {
        "rules": json.loads(project.data.rules.model_dump_json()),
        "printer": json.loads(project.printer.model_dump_json()),
        "current_design": json.loads(base.model_dump_json())["features"],
        "sketches": [s.id for s in base.sketches],
        "max_builds": job.max_rounds,
    }
    ask = job.instruction.strip() or "Improve the design."
    messages: list[dict] = [{
        "role": "user",
        "content": f"Request: {ask}\n\nProject state (mm):\n{json.dumps(first, separators=(',', ':'))}",
    }]
    system = [
        {"type": "text", "text": SYSTEM},
        {"type": "text", "text": "Feature catalog:\n" + json.dumps(_catalog(), separators=(",", ":")), "cache_control": {"type": "ephemeral"}},
    ]

    def log(kind, text, **kw):
        job.steps.append(Step(kind, text, round=job.rounds, **kw))

    try:
        finished = False
        for _turn in range(job.max_rounds * 3 + 4):
            if job.cancel.is_set():
                job.status = "cancelled"
                break
            resp = client.create(model=job.model, max_tokens=8000, system=system, tools=TOOLS, messages=messages)
            u = getattr(resp, "usage", None)
            if u is not None:
                job.tokens["input"] += getattr(u, "input_tokens", 0) or 0
                job.tokens["output"] += getattr(u, "output_tokens", 0) or 0
                job.tokens["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            messages.append({"role": "assistant", "content": [_block_dict(b) for b in resp.content]})
            results = []
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    log("note", b.text.strip())
                elif b.type == "tool_use":
                    text, is_error, done = _call_tool(b, job, project, base, lock, log, best)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": is_error})
                    finished = finished or done
            if finished:
                job.status = "done"
                break
            if not results:
                # Claude stopped without calling finish; treat its last text as the summary.
                job.summary = job.summary or next((s.text for s in reversed(job.steps) if s.kind == "note"), "")
                job.status = "done"
                break
            messages.append({"role": "user", "content": results})
        else:
            job.status = "done"
            log("error", "Stopped: turn limit reached")
    except Exception as exc:  # API or unexpected failure: keep what was built
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        log("error", job.error)
    finally:
        if best[0] is not None:
            with lock:
                project.data.design = best[0][1]
                project.save()
                job.design_rev += 1
            job.can_revert = True
            job.before = before
        job.finished = time.time()


def _block_dict(b) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return b.model_dump() if hasattr(b, "model_dump") else dict(b)


def _call_tool(b, job: Job, project, base: Design, lock, log, best_ref) -> tuple[str, bool, bool]:
    name, args = b.name, b.input or {}
    if name == "build_and_check":
        if job.rounds >= job.max_rounds:
            return "Build limit reached. Call finish now.", True, False
        try:
            proposed = [FeatureInstance.model_validate(f) for f in args.get("features", [])]
        except ValidationError as exc:
            return f"Invalid feature list: {exc}", True, False
        ids = [f.id for f in proposed]
        if len(ids) != len(set(ids)):
            return "Feature ids must be unique.", True, False
        features, kept = _restore_locked(proposed, base)
        design = base.model_copy(update={"features": features})  # keeps the sketches the pads and pockets consume
        job.rounds += 1
        try:
            build = build_design(design, project.data.rules)
        except FeatureError as exc:
            log("build", f"Round {job.rounds}: {exc}")
            return f"Build failed: {exc}", True, False
        report = validate(build, project.data.rules, project.printer)
        score = _score(report, build.mass)
        counts = report.to_dict()["counts"]
        if best_ref[0] is None or score < best_ref[0][0]:
            best_ref[0] = (score, design)
            job.best = {"round": job.rounds, "counts": counts, "mass": build.mass}
        # Show the candidate live; the best one is put back at the end.
        with lock:
            project.data.design = design
            project.save()
            job.design_rev += 1
        log("build", args.get("note", "").strip() or f"Round {job.rounds}", counts=counts, mass=build.mass)
        return _report_text(design, build, report, kept), False, False

    if name == "finish":
        job.summary = (args.get("summary") or "").strip()
        log("finish", job.summary)
        return "Session ended.", False, True

    return f"Unknown tool {name!r}", True, False


# ---------------------------------------------------------------- runner


class Generator:
    """One generation job at a time per project, run on a background thread."""

    def __init__(self) -> None:
        self.job: Job | None = None
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def running(self) -> bool:
        return self.job is not None and self.job.status == "running"

    def start(self, project, client: MessagesClient, instruction: str = "", max_rounds: int = 6, model: str | None = None, wait: bool = False) -> Job:
        if self.running():
            raise RuntimeError("A generation is already running")
        job = Job(id=uuid.uuid4().hex[:8], instruction=instruction, model=model or model_name(), max_rounds=max_rounds)
        self.job = job
        self._thread = threading.Thread(target=run, args=(job, project, client, self.lock), daemon=True)
        self._thread.start()
        if wait:
            self._thread.join()
        return job

    def cancel(self) -> None:
        if self.running():
            self.job.cancel.set()

    def revert(self, project) -> None:
        job = self.job
        if job is None or self.running() or not job.can_revert:
            raise RuntimeError("Nothing to revert")
        with self.lock:
            project.data.design = job.before
            project.save()
            job.design_rev += 1
        job.can_revert = False
