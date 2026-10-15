"""Solve a sketch with planegcs, through the same TypeScript module the browser uses (frontend/sketch-cli.ts, run by
Node), so a sketch solves identically in both places."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .model import Sketch, SolveReport

CLI = Path(__file__).resolve().parents[4] / "frontend" / "sketch-cli.ts"


class SketchSolveError(RuntimeError):
    pass


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
