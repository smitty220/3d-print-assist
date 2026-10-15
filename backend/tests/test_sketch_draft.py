"""Claude drafting a part from a topology result, driven by a scripted stand-in (no API calls)."""

import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cad_assist.examples import bell_crank
from cad_assist.project import Project
from cad_assist.topo import job as topo
from cad_assist.topo.interpret import Interpreter
from cad_assist.topo.setup import build_setup
from cad_assist.topo.simp import Optimizer


@pytest.fixture(scope="module")
def crank_result(tmp_path_factory):
    """The bell crank with a stand-in 'result': a plate filling the gap-thickness band of the design space."""
    p = Project(tmp_path_factory.mktemp("crank"))
    bell_crank(p)
    roles = {k: v.role for k, v in p.data.parts.items()}
    setup = build_setup(p.data.intent, p.shape, roles, 8_000)
    g = setup.grid
    x = (~setup.problem.void).astype(float)
    zc = setup.origin[2] + (np.arange(g.nz) + 0.5) * g.h
    x[:, :, np.abs(zc) > 4.8] = 0.0  # a 0.38 in thick slab, so it fits every fork
    x[setup.problem.void] = 0.0
    comp = Optimizer(setup.problem).solid_compliance(x)
    mass = float((x >= 0.5).sum() * g.h**3 * setup.density)
    step = topo.StepInfo(0, 0.5, mass, comp, 1, True, 0.0, "2.5D", mass, comp)
    r = topo.Result(topo.intent_key(p), setup.budget, [g.nx, g.ny, g.nz], g.h, list(setup.frame.origin), setup.origin.tolist(),
                    setup.axes.tolist(), setup.density, setup.problem.void, [step], [x], 0, setup.summary(), {"tool_diameter": 6.35})
    return p, r, (lambda: setup)


class Scripted:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        n = len(self.calls)
        if n == 1:
            first = kw["messages"][0]["content"]
            self.base = json.loads(first.split("Baseline recipe (rules-based draft, passes the checks):\n")[1].split("\n")[0])
            return NS(content=[NS(type="text", text="Starting from the baseline."),
                               NS(type="tool_use", id="t1", name="build_and_check", input={"note": "baseline", "recipe": self.base})], usage=None)
        if n == 2:
            bad = dict(self.base, outline=list(reversed(self.base["outline"])))  # clockwise: rejected with a reason
            return NS(content=[NS(type="tool_use", id="t2", name="build_and_check", input={"note": "try the other way round", "recipe": bad})], usage=None)
        return NS(content=[NS(type="tool_use", id="t3", name="finish", input={"summary": "A plate crank.", "questions": ["Thinner web?"]})], usage=None)


def test_ai_draft_builds_checks_and_keeps_the_best(crank_result):
    p, r, setup_of = crank_result
    claude = Scripted()
    it = Interpreter()
    job = it.start(p, r, 0, setup_of, "ai", "draft it", claude, 3, "fake")
    deadline = time.time() + 600
    while it.running() and time.time() < deadline:
        time.sleep(0.2)
    assert job.status == "done", job.log
    kinds = [entry["kind"] for entry in job.log]
    assert "ai" in kinds and "build" in kinds and "question" in kinds
    # The bad build came back to Claude as an error with the reason, not a crash.
    msgs = claude.calls[-1]["messages"]  # one list, grown call by call
    results = {c["tool_use_id"]: c for m in msgs if m["role"] == "user" and isinstance(m["content"], list) for c in m["content"]}
    assert results["t2"]["is_error"] and "once round every pin" in results["t2"]["content"]
    # The brief carried the measurements and the map; the first result carried the comparison.
    brief = claude.calls[0]["messages"][0]["content"]
    assert "Pins (x, y in" in brief and "thickness map" in brief and "Outline edge order" in brief
    first_result = results["t1"]["content"]
    assert "Stiffness vs result" in first_result and "Model vs result" in first_result
    d = p.data.design
    assert d.recipe is not None and d.recipe_key == d.sketch_key()
    assert d.recipe_summary == "A plate crank." and d.recipe_questions == ["Thinner web?"]
    # The next round sends Claude its questions with the engineer's reply.
    claude2 = Scripted()
    job2 = it.start(p, r, 0, setup_of, "ai", "1. yes, thinner", claude2, 3, "fake")
    while it.running():
        time.sleep(0.2)
    brief2 = claude2.calls[0]["messages"][0]["content"]
    assert "1. Thinner web?" in brief2 and "1. yes, thinner" in brief2 and job2.status == "done"
    assert [f.kind for f in d.features][:1] == ["pad"] and job.check["stiffness_ratio"] > 0



def test_ai_draft_with_stress_reports_margins(crank_result):
    p, r, setup_of = crank_result
    claude = Scripted()
    it = Interpreter()
    job = it.start(p, r, 0, setup_of, "ai", "draft it", claude, 2, "fake", stress=True)
    while it.running():
        time.sleep(0.2)
    assert job.status == "done", job.log
    msgs = claude.calls[-1]["messages"]
    results = {c["tool_use_id"]: c for m in msgs if m["role"] == "user" and isinstance(m["content"], list) for c in m["content"]}
    assert "MS ultimate" in results["t1"]["content"] and "IF-01 lug" in results["t1"]["content"]
    assert "Stress checking is on" in claude.calls[0]["messages"][0]["content"]
