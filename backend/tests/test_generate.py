"""The generation loop, driven by a scripted stand-in for Claude (no API calls)."""

import json
import threading
from types import SimpleNamespace as NS

import pytest

from print_assist.ai.generate import Generator, Job, run
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.printers import profiles
from print_assist.project import Project
from print_assist.rules import PrintRules


class FakeProject:
    """Just enough of Project for the loop: design, rules, printer, save()."""

    def __init__(self, design=None):
        self.data = NS(design=design or Design(), rules=PrintRules())
        self.printer = profiles()["bambu_a1_ams_lite"]
        self.saves = 0

    def save(self):
        self.saves += 1


def text(t):
    return NS(type="text", text=t)


def tool(name, args, i):
    return NS(type="tool_use", id=f"tu{i}", name=name, input=args)


class ScriptedClaude:
    """Each script entry is a function of the last tool results, returning the next assistant content."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        last = kw["messages"][-1]["content"]
        results = [c["content"] for c in last] if isinstance(last, list) else []
        content = self.script.pop(0)(results)
        return NS(content=content, stop_reason="tool_use", usage=NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))


def _run(proj, script, rounds=6):
    job = Job(id="t", instruction="make it pass", model="fake", max_rounds=rounds)
    claude = ScriptedClaude(script)
    run(job, proj, claude, threading.Lock())
    return job, claude


def plate(t=5.0, **kw):
    return {"id": "F1", "kind": "plate", "params": {"thickness": t}, **kw}


def test_loop_builds_revises_and_keeps_the_best():
    proj = FakeProject()

    def restore(results):
        assert "[fail] Minimum wall" in results[0]
        return [tool("build_and_check", {"note": "Back to 5 mm", "features": [plate(5.0)]}, 3)]

    script = [
        lambda r: [text("Trying a thin plate."), tool("build_and_check", {"note": "Thin plate", "features": [plate(0.5)]}, 1)],
        restore,
        lambda r: [tool("finish", {"summary": "Plate, 5 mm thick."}, 4)],
    ]
    job, claude = _run(proj, script)
    assert job.status == "done" and job.rounds == 2
    assert job.summary.startswith("Plate")
    assert [s.kind for s in job.steps] == ["note", "build", "build", "finish"]
    assert job.steps[1].counts["fail"] >= 1 and job.steps[2].counts["fail"] == 0
    # The best design (round 2) is left in the project, not the last one built.
    assert job.best["round"] == 2
    assert proj.data.design.features[0].params["thickness"] == pytest.approx(5.0)
    assert job.can_revert and job.before.features == []
    d = job.to_dict()
    assert d["status"] == "done" and len(d["steps"]) == 4 and "cancel" not in d and "before" not in d
    assert job.tokens == {"input": 3000, "output": 600, "cache_read": 2400}
    # The catalog block is cached; the tool list is offered on every call.
    assert claude.calls[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_locked_features_survive_whatever_claude_proposes():
    base = Design(features=[FI(id="F1", kind="plate", params={"thickness": 6.0}, locked=True),
                            FI(id="F2", kind="chamfer", params={"distance": 0.5})])
    proj = FakeProject(base)
    proposal = [plate(2.0)]  # Claude tries to thin the locked plate and drops F2

    def check(results):
        assert "Locked features restored" in results[0] and "F1" in results[0]
        return [tool("finish", {"summary": "done"}, 2)]

    _run(proj, [lambda r: [tool("build_and_check", {"note": "thin it", "features": proposal}, 1)], check])
    f1 = proj.data.design.features[0]
    assert f1.locked and f1.params["thickness"] == pytest.approx(6.0)
    assert "F2" not in [f.id for f in proj.data.design.features]  # unlocked features can go


def test_build_limit_and_bad_input_are_reported_to_claude():
    proj = FakeProject()
    seen = []

    def build(results):
        seen.append(results)
        return [tool("build_and_check", {"note": "again", "features": [plate()]}, len(seen))]

    script = [
        lambda r: [tool("build_and_check", {"note": "dupes", "features": [plate(), plate()]}, 0)],
        build,  # sees the duplicate-id error
        build,  # round 1 result; asks for round 2
        lambda r: (seen.append(r), [tool("finish", {"summary": "ok"}, 9)])[1],
    ]
    job, _ = _run(proj, script, rounds=1)
    assert "unique" in seen[0][0]
    assert "Build limit reached" in seen[2][0]
    assert job.rounds == 1


def test_api_failure_keeps_best_and_cancel_stops():
    proj = FakeProject()

    def boom(results):
        raise RuntimeError("overloaded")

    job, _ = _run(proj, [lambda r: [tool("build_and_check", {"note": "first", "features": [plate()]}, 1)], boom])
    assert job.status == "failed" and "overloaded" in job.error
    assert len(proj.data.design.features) == 1

    job = Job(id="c", instruction="", model="fake", max_rounds=3)
    job.cancel.set()
    run(job, FakeProject(), ScriptedClaude([]), threading.Lock())
    assert job.status == "cancelled"


def test_generator_revert(tmp_path):
    p = Project(tmp_path / "p")
    gen = Generator()
    job = gen.start(p, ScriptedClaude([
        lambda r: [tool("build_and_check", {"note": "go", "features": [plate()]}, 1)],
        lambda r: [tool("finish", {"summary": "ok"}, 2)],
    ]), wait=True)
    assert job.status == "done" and len(p.data.design.features) == 1
    gen.revert(p)
    assert p.data.design.features == [] and not job.can_revert
    with pytest.raises(RuntimeError):
        gen.revert(p)


def test_env_file_key(tmp_path, monkeypatch):
    from print_assist.ai import generate

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generate, "_SHELL_ENV", set())
    env = tmp_path / ".env"
    env.write_text('# comment\nexport ANTHROPIC_API_KEY="sk-test-1"\n')
    generate.load_env_key(env)
    assert generate.os.environ["ANTHROPIC_API_KEY"] == "sk-test-1"
    env.write_text("ANTHROPIC_API_KEY=sk-test-2\n")  # edited file wins without a restart
    generate.load_env_key(env)
    assert generate.os.environ["ANTHROPIC_API_KEY"] == "sk-test-2"


def test_candidates_keep_sketches_and_picked_edges():
    """Claude proposes features only: the design's sketches and the user's picked fillet edges stay with every
    candidate."""
    from print_assist.features.context import FeatureError
    from print_assist.features.design import build_design, describe_edge
    from print_assist.sketch.model import Sketch

    base = Design(features=[FI(id="F1", kind="plate", params={"corner_radius": 0.0})])
    built = build_design(base)
    for i in range(len(built.local.edges())):
        try:
            pick = describe_edge(built.local, built.face_names, i)
            break
        except FeatureError:
            continue
    base.features.append(FI(id="F2", kind="fillet", params={"radius": 0.5}, edges=[pick]))
    base.sketches.append(Sketch(id="S9", name="Kept", entities=[], constraints=[]))
    proj = FakeProject(base)
    proposal = json.loads(base.model_dump_json())["features"]
    for f in proposal:
        f.pop("edges", None)  # Claude cannot pick edges; it sends refs and params

    def check(results):
        assert "picked edges: 1" in results[0]
        return [tool("finish", {"summary": "done"}, 2)]

    _run(proj, [lambda r: [tool("build_and_check", {"note": "same again", "features": proposal}, 1)], check])
    assert [s.id for s in proj.data.design.sketches] == ["S9"]
    assert proj.data.design.features[-1].edges == [pick]
