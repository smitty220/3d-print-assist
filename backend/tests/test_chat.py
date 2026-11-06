"""The in-app chat, driven by a scripted stand-in for Claude (no API calls)."""

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.ai.chat import Chat


def text(t):
    return NS(type="text", text=t)


def tool(name, args, i):
    return NS(type="tool_use", id=f"tu{i}", name=name, input=args)


class ScriptedClaude:
    """Each script entry is a function of the last tool results, returning (content, stop_reason)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        last = kw["messages"][-1]["content"]
        results = [c["content"] for c in last] if isinstance(last, list) else []
        content, stop = self.script.pop(0)(results)
        return NS(content=content, stop_reason=stop, usage=NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_chat_designs_through_the_shared_tools_and_sees_the_result(client):
    client.get("/api/project")  # open a design

    def build(results):
        return [text("Adding a plate."), tool("apply_changes", {"ops": [{"op": "add_feature", "kind": "plate"}], "message": "plate"}, 1)], "tool_use"

    def look(results):
        assert json.loads(results[0])["made"] == ["F1"]
        return [tool("render_preview", {"view": "iso", "size": 128}, 2)], "tool_use"

    def done(results):
        assert results[0][0]["type"] == "image" and results[0][0]["source"]["media_type"] == "image/png"
        return [text("A 60 × 40 × 5 mm plate.")], "end_turn"

    chat = Chat()
    claude = ScriptedClaude([build, look, done])
    chat.send("make a plate", claude, model="fake", wait=True)
    roles = [(e.role, e.tool) for e in chat.log]
    assert roles == [("user", None), ("assistant", None), ("tool", "apply_changes"), ("tool", "render_preview"), ("assistant", None)]
    assert not chat.running and chat.tokens == {"input": 3000, "output": 600, "cache_read": 2400}
    first = claude.calls[0]
    assert first["system"][-1]["cache_control"] == {"type": "ephemeral"} and "get_guide" not in {t["name"] for t in first["tools"]}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    acts = client.get("/api/activity").json()
    assert acts[-1]["actor"] == "claude" and client.get("/api/tree").json()["features"][0]["status"] == "ok"


def test_refusals_errors_and_api_failures_are_logged():
    chat = Chat()
    chat.send("x", ScriptedClaude([lambda r: ([], "refusal")]), model="fake", wait=True)
    assert chat.log[-1].role == "error" and "declined" in chat.log[-1].text

    def boom(r):
        raise RuntimeError("overloaded")

    chat.send("again", ScriptedClaude([boom]), model="fake", wait=True)
    assert "overloaded" in chat.log[-1].text and not chat.running
    chat.send("bad tool", ScriptedClaude([lambda r: ([tool("nope", {}, 1)], "tool_use"), lambda r: ([text("ok")], "end_turn")]),
              model="fake", wait=True)
    bad = next(e for e in chat.log if e.tool == "nope")
    assert not bad.ok
    chat.reset()
    assert chat.log == [] and chat.messages == []


def test_chat_endpoints(client, monkeypatch, tmp_path):
    from print_assist.ai import client as ai_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ai_client.load_env_key, "__defaults__", (tmp_path / "no.env",))
    status = client.get("/api/ai/status").json()
    assert status["key"] is False and status["mode"] == "mcp" and status["chat"]["log"] == []
    r = client.post("/api/ai/chat", json={"message": "hi"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.post("/api/ai/chat", json={"message": " "}).status_code == 422


def test_env_file_key(tmp_path, monkeypatch):
    from print_assist.ai import client as ai_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ai_client, "_SHELL_ENV", set())
    env = tmp_path / ".env"
    env.write_text('# comment\nexport ANTHROPIC_API_KEY="sk-test-1"\n', encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.os.environ["ANTHROPIC_API_KEY"] == "sk-test-1"
    env.write_text("ANTHROPIC_API_KEY=sk-test-2\n", encoding="utf-8")  # edited file wins without a restart
    ai_client.load_env_key(env)
    assert ai_client.os.environ["ANTHROPIC_API_KEY"] == "sk-test-2"


class ScriptedGemini:
    """Mock for Gemini interactions client."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.interactions = self

    def create(self, **kw):
        self.calls.append(kw)
        step_fn = self.script.pop(0)
        steps, status = step_fn(kw.get("input"))
        return NS(
            id=f"int_{len(self.calls)}",
            status=status,
            steps=steps,
            output_text=None,
            usage=NS(total_input_tokens=1000, total_output_tokens=200, total_cached_tokens=800),
        )


def test_gemini_chat_designs_through_the_shared_tools_and_sees_the_result(client):
    client.get("/api/project")  # open a design

    def build(inp):
        return [
            NS(type="model_output", content=[NS(type="text", text="Adding a plate.")]),
            NS(type="function_call", id="fc1", name="apply_changes", arguments={"ops": [{"op": "add_feature", "kind": "plate"}], "message": "plate"}),
        ], "requires_action"

    def look(inp):
        assert any(r.get("call_id") == "fc1" for r in inp)
        return [
            NS(type="function_call", id="fc2", name="render_preview", arguments={"view": "iso", "size": 128}),
        ], "requires_action"

    def done(inp):
        assert any(r.get("call_id") == "fc2" for r in inp)
        return [
            NS(type="model_output", content=[NS(type="text", text="A 60 × 40 × 5 mm plate.")]),
        ], "completed"

    chat = Chat()
    gemini = ScriptedGemini([build, look, done])
    chat.send("make a plate with gemini", gemini, model="gemini-3.8-flash", wait=True)
    roles = [(e.role, e.tool) for e in chat.log]
    assert roles == [("user", None), ("assistant", None), ("tool", "apply_changes"), ("tool", "render_preview"), ("assistant", None)]
    assert not chat.running and chat.tokens == {"input": 3000, "output": 600, "cache_read": 2400}
    first = gemini.calls[0]
    assert "get_guide" not in {t["name"] for t in first["tools"]}
    assert first["model"] == "gemini-3.8-flash"
    acts = client.get("/api/activity").json()
    assert acts[-1]["actor"] == "gemini" and client.get("/api/tree").json()["features"][0]["status"] == "ok"


def test_gemini_env_key_and_provider_selection(tmp_path, monkeypatch):
    from print_assist.ai import client as ai_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PRINT_ASSIST_AI_PROVIDER", raising=False)
    monkeypatch.delenv("PRINT_ASSIST_MODEL", raising=False)
    monkeypatch.setattr(ai_client, "_SHELL_ENV", set())

    # Neither key -> default anthropic
    assert ai_client.provider() == "anthropic"
    assert ai_client.model_name() == "claude-opus-5-5"

    # Only GEMINI_API_KEY set -> auto-detects gemini
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=gm-test-1\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "gemini"
    assert ai_client.model_name() == "gemini-3.8-flash"
    assert ai_client.has_api_key() is True

    # Explicit PRINT_ASSIST_AI_PROVIDER override
    env.write_text("GEMINI_API_KEY=gm-test-1\nANTHROPIC_API_KEY=sk-test-1\nPRINT_ASSIST_AI_PROVIDER=gemini\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "gemini"

    env.write_text("GEMINI_API_KEY=gm-test-1\nANTHROPIC_API_KEY=sk-test-1\nPRINT_ASSIST_AI_PROVIDER=anthropic\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "anthropic"
