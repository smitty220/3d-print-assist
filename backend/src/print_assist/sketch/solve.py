"""Solve a sketch with planegcs, through the same TypeScript module the browser uses (frontend/sketch-cli.ts, run by
Node), so a sketch solves identically in both places."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..intent.models import Intent
from .model import Point, Sketch, SolveReport

CLI = Path(__file__).resolve().parents[4] / "frontend" / "sketch-cli.ts"


class SketchSolveError(RuntimeError):
    pass


def place_refs(sketch: Sketch, intent: Intent, to_local) -> Sketch:
    """Move points tied to intent objects (pin centres) to where the intent puts them, in the sketch plane."""
    out = sketch.model_copy(deep=True)
    for e in out.entities:
        if isinstance(e, Point) and e.ref:
            obj = intent.get(e.ref)
            if obj is None or not hasattr(obj, "frame"):
                raise SketchSolveError(f"{e.id} is tied to {e.ref}, which no longer exists")
            p = to_local(obj.frame.origin)
            e.x, e.y = float(p.X), float(p.Y)
    return out


def solve(sketch: Sketch) -> Sketch:
    node = shutil.which("node")
    if node is None:
        raise SketchSolveError("Node.js is needed to solve sketches (the solver is WebAssembly) and was not found")
    run = subprocess.run([node, str(CLI)], input=sketch.model_dump_json(exclude={"report"}), capture_output=True,
                         text=True, timeout=60, cwd=CLI.parent)
    try:
        out = json.loads(run.stdout.rsplit("@@SKETCH@@", 1)[1])  # the WebAssembly solver may print ahead of the result
    except (json.JSONDecodeError, IndexError):
        raise SketchSolveError(f"solver failed: {run.stderr.strip()[-400:]}") from None
    if "error" in out:
        raise SketchSolveError(out["error"])
    solved = Sketch.model_validate(out["sketch"])
    solved.report = SolveReport.model_validate(out["report"])
    return solved
