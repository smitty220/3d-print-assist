"""The generation loop, driven by a scripted stand-in for Claude (no API calls)."""

import json
import threading
from types import SimpleNamespace as NS

import pytest
from build123d import IN

from print_assist.ai.generate import Generator, Job, run
from print_assist.features.model import Design
from print_assist.features.templates import TEMPLATES
from print_assist.project import Project
from test_validate import SHAPES, _intent


class FakeProject:
    """Just enough of Project for the loop: intent, design, part roles, shapes, save()."""

    def __init__(self, intent, design=None):
        self.data = NS(intent=intent, design=design or Design(), parts={
            "rib": NS(role="mating"), "rod": NS(role="mating"), "hyd": NS(role="keepout")})
        self.saves = 0

    def shape(self, name):
        return SHAPES[name]

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


def test_loop_builds_revises_and_keeps_the_best():
    proj = FakeProject(_intent())
    template = {}

    def thin_gussets(results):
        feats = json.loads(results[0])
        template["features"] = feats
        bad = json.loads(json.dumps(feats))
        bad[3]["params"]["thickness"] = 0.04 * IN
        return [text("Starting from the clevis template."), tool("build_and_check", {"note": "Thin gussets to save mass", "features": bad}, 2)]

    def restore(results):
        assert "[fail] Minimum wall" in results[0]
        return [tool("build_and_check", {"note": "Back to 0.20 gussets", "features": template["features"]}, 3)]

    script = [
        lambda r: [tool("get_template", {"name": "clevis_bracket"}, 1)],
        thin_gussets,
        restore,
        lambda r: [tool("finish", {"summary": "Clevis bracket, 0.393 lb, 9% over target."}, 4)],
    ]
    job, claude = _run(proj, script)
    assert job.status == "done" and job.rounds == 2
    assert job.summary.startswith("Clevis bracket")
    assert [s.kind for s in job.steps] == ["template", "note", "build", "build", "finish"]
    assert job.steps[2].counts["fail"] >= 1 and job.steps[3].counts["fail"] == 0
    # The best design (round 2) is left in the project, not the last one built.
    assert job.best["round"] == 2
    assert proj.data.design.features[3].params["thickness"] == pytest.approx(0.2 * IN)
    assert job.can_revert and job.before.features == []
    d = job.to_dict()
    assert d["status"] == "done" and len(d["steps"]) == 5 and "cancel" not in d and "before" not in d
    assert job.tokens == {"input": 4000, "output": 800, "cache_read": 3200}
    # The catalog block is cached; the tool list is offered on every call.
    assert claude.calls[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_locked_features_survive_whatever_claude_proposes():
    it = _intent()
    base = TEMPLATES["clevis_bracket"][2](it)
    base.features[0].locked = True
    base.features[0].params["thickness"] = 0.3 * IN
    proj = FakeProject(it, base)
    proposal = json.loads(base.model_dump_json())["features"]
    proposal[0]["params"]["thickness"] = 0.1 * IN  # Claude tries to thin the locked plate
    proposal = [f for f in proposal if f["id"] != "F6"]

    def check(results):
        assert "Locked features restored" in results[0] and "F1" in results[0]
        return [tool("finish", {"summary": "done"}, 2)]

    job, _ = _run(proj, [lambda r: [tool("build_and_check", {"note": "thin it", "features": proposal}, 1)], check])
    f1 = proj.data.design.features[0]
    assert f1.locked and f1.params["thickness"] == pytest.approx(0.3 * IN)
    assert "F6" not in [f.id for f in proj.data.design.features]  # unlocked features can go


def test_build_limit_and_bad_input_are_reported_to_claude():
    proj = FakeProject(_intent())
    feats = json.loads(TEMPLATES["clevis_bracket"][2](proj.data.intent).model_dump_json())["features"]
    seen = []

    def build(results):
        seen.append(results)
        return [tool("build_and_check", {"note": "again", "features": feats}, len(seen))]

    script = [
        lambda r: [tool("build_and_check", {"note": "dupes", "features": [feats[0], feats[0]]}, 0)],
        build,  # sees the duplicate-id error
        build,  # round 1 result; asks for round 2
        lambda r: (seen.append(r), [tool("finish", {"summary": "ok"}, 9)])[1],
    ]
    job, _ = _run(proj, script, rounds=1)
    assert "unique" in seen[0][0]
    assert "Build limit reached" in seen[2][0]
    assert job.rounds == 1


def test_api_failure_keeps_best_and_cancel_stops():
    proj = FakeProject(_intent())
    feats = json.loads(TEMPLATES["clevis_bracket"][2](proj.data.intent).model_dump_json())["features"]

    def boom(results):
        raise RuntimeError("overloaded")

    job, _ = _run(proj, [lambda r: [tool("build_and_check", {"note": "first", "features": feats}, 1)], boom])
    assert job.status == "failed" and "overloaded" in job.error
    assert len(proj.data.design.features) == 6

    gen = Generator()
    gen.job = None
    job = Job(id="c", instruction="", model="fake", max_rounds=3)
    job.cancel.set()
    run(job, FakeProject(_intent()), ScriptedClaude([]), threading.Lock())
    assert job.status == "cancelled"


def test_generator_revert(tmp_path):
    p = Project(tmp_path / "p")
    p.data.intent = _intent()
    p.data.parts = {"rib": NS(role="mating")}
    gen = Generator()
    feats = json.loads(TEMPLATES["clevis_bracket"][2](p.data.intent).model_dump_json())["features"]
    p.shape = SHAPES.__getitem__
    job = gen.start(p, ScriptedClaude([
        lambda r: [tool("build_and_check", {"note": "go", "features": feats}, 1)],
        lambda r: [tool("finish", {"summary": "ok"}, 2)],
    ]), wait=True)
    assert job.status == "done" and len(p.data.design.features) == 6
    gen.revert(p)
    assert p.data.design.features == [] and not job.can_revert
    with pytest.raises(RuntimeError):
        gen.revert(p)


def test_replay_client_runs_the_loop_offline():
    from print_assist.ai.replay import ReplayClient

    proj = FakeProject(_intent())
    job = Job(id="r", instruction="", model="replay", max_rounds=6)
    run(job, proj, ReplayClient(delay=0), threading.Lock())
    assert job.status == "done" and job.rounds == 2
    assert [s.kind for s in job.steps] == ["template", "note", "build", "build", "finish"]
    assert len(proj.data.design.features) == 6


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
    """Claude proposes features only: the design's sketches (what pads and pockets consume) and the engineer's picked
    fillet edges stay with every candidate."""
    from print_assist.features.context import FeatureError
    from print_assist.features.design import build_design, describe_edge
    from print_assist.features.model import FeatureInstance
    from print_assist.sketch.model import Sketch

    it = _intent()
    base = TEMPLATES["clevis_bracket"][2](it)
    built = build_design(it, base)
    for i in range(len(built.local.edges())):
        try:
            pick = describe_edge(built.local, built.face_names, i)
            break
        except FeatureError:
            continue
    base.features.append(FeatureInstance(id="F7", kind="fillet", params={"radius": 0.5}, edges=[pick]))
    base.sketches.append(Sketch(id="S9", name="Kept", entities=[], constraints=[]))
    proj = FakeProject(it, base)
    proposal = json.loads(base.model_dump_json())["features"]
    for f in proposal:
        f.pop("edges", None)  # Claude cannot pick edges; it sends refs and params

    def check(results):
        assert "picked edges: 1" in results[0]
        return [tool("finish", {"summary": "done"}, 2)]

    _run(proj, [lambda r: [tool("build_and_check", {"note": "same again", "features": proposal}, 1)], check])
    assert [s.id for s in proj.data.design.sketches] == ["S9"]
    assert proj.data.design.features[-1].edges == [pick]
