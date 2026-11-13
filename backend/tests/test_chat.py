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
    monkeypatch.setattr(ai_client.load_env_key, "__defaults__", (tmp_path / "no.env",))
    monkeypatch.setattr(ai_client, "_SHELL_ENV", set())

    # Neither key -> default anthropic
    assert ai_client.provider() == "anthropic"
    assert ai_client.model_name() == "claude-opus-5-5"

    # Only GEMINI_API_KEY set -> auto-detects gemini
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=gm-test-1\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "gemini"
    assert ai_client.model_name() == "gemini-3-flash-preview"
    assert ai_client.has_api_key() is True

    # Explicit PRINT_ASSIST_AI_PROVIDER override
    env.write_text("GEMINI_API_KEY=gm-test-1\nANTHROPIC_API_KEY=sk-test-1\nPRINT_ASSIST_AI_PROVIDER=gemini\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "gemini"

    env.write_text("GEMINI_API_KEY=gm-test-1\nANTHROPIC_API_KEY=sk-test-1\nPRINT_ASSIST_AI_PROVIDER=anthropic\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.provider() == "anthropic"

    # Gemini 4 model alias resolution
    monkeypatch.delenv("PRINT_ASSIST_AI_PROVIDER", raising=False)
    env.write_text("GEMINI_API_KEY=gm-test-1\nPRINT_ASSIST_MODEL=gemini-4.0\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.model_name() == "gemini-4-argon"
    assert ai_client.provider() == "gemini"

    env.write_text("GEMINI_API_KEY=gm-test-1\nPRINT_ASSIST_MODEL=gemini-4-argon\n", encoding="utf-8")
    ai_client.load_env_key(env)
    assert ai_client.model_name() == "gemini-4-argon"


def test_gemini_follow_up_message_continues_the_conversation(client):
    client.get("/api/project")

    def reply(inp):
        return [NS(type="model_output", content=[NS(type="text", text="ok")])], "completed"

    chat = Chat()
    gemini = ScriptedGemini([reply, reply, reply])
    chat.send("first", gemini, model="gemini-3.8-flash", wait=True)
    chat.send("second", gemini, model="gemini-3.8-flash", wait=True)
    assert gemini.calls[0]["previous_interaction_id"] is None
    assert gemini.calls[1]["previous_interaction_id"] == "int_1"  # the follow-up sees the first exchange
    chat.reset()
    chat.send("fresh", gemini, model="gemini-3.8-flash", wait=True)
    assert gemini.calls[2]["previous_interaction_id"] is None


def test_wait_for_concept_cancellation(client):
    import threading
    import time
    from print_assist.ai.chat import _call

    client.get("/api/project")
    r = client.post("/api/concepts", json={
        "title": "Pending Concept",
        "svg": "<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>",
        "summary": "pending",
    }, headers={"X-Actor": "gemini"})
    cid = r.json()["id"]

    cancel = threading.Event()
    # Cancel after 0.1s while wait_for_concept is polling
    threading.Timer(0.1, cancel.set).start()
    t0 = time.time()
    res, ok = _call("wait_for_concept", {"id": cid, "timeout_s": 600}, cancel=cancel)
    t1 = time.time()
    assert ok is False
    assert "Cancelled" in res
    assert t1 - t0 < 2.0  # Must return immediately, never waiting 600s


def test_ai_chat_cancel_and_reset_dismisses_pending_concepts(client):
    client.get("/api/project")
    # Propose a concept to leave it pending
    r = client.post("/api/concepts", json={
        "title": "Test Concept",
        "svg": "<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>",
        "summary": "draft",
    }, headers={"X-Actor": "gemini"})
    assert r.status_code == 200
    cid = r.json()["id"]

    # Verify concept is pending
    concepts = client.get("/api/concepts").json()
    assert any(c["id"] == cid and c["status"] == "pending" for c in concepts)

    # Cancel chat -> dismisses pending concept
    client.post("/api/ai/chat/cancel")
    concepts_after = client.get("/api/concepts").json()
    assert not any(c["id"] == cid and c["status"] == "pending" for c in concepts_after)
    cancelled_c = next(c for c in concepts_after if c["id"] == cid)
    assert cancelled_c["status"] == "revise"
    assert "cancelled" in cancelled_c["note"]


def test_gemini_fallback_on_rate_limit(client):
    client.get("/api/project")

    class RateLimitedGemini:
        def __init__(self):
            self.calls = []
            self.interactions = self

        def create(self, **kw):
            self.calls.append(kw)
            if kw.get("model") == "gemini-2.5-flash":
                raise RuntimeError("Rate limit exceeded for model gemini-2.5-flash: 429 too_many_requests")
            return NS(
                id="int_fallback",
                status="completed",
                steps=[NS(type="model_output", content=[NS(type="text", text="Fallback succeeded")])],
                output_text="Fallback succeeded",
                usage=NS(total_input_tokens=500, total_output_tokens=100, total_cached_tokens=0),
            )

    chat = Chat()
    gemini = RateLimitedGemini()
    chat.send("test fallback", gemini, model="gemini-2.5-flash", wait=True)
    assert not chat.running
    assert len(gemini.calls) == 2
    assert gemini.calls[0]["model"] == "gemini-2.5-flash"
    assert gemini.calls[1]["model"] == "gemini-3-flash-preview"
    assert any("Fallback succeeded" in e.text for e in chat.log)


def test_gemini_timeout_retry_and_recovery(client):
    client.get("/api/project")

    class TimingOutGemini:
        def __init__(self):
            self.calls = []
            self.interactions = self

        def create(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                raise RuntimeError("APITimeoutError: Request timed out. This is a client-side timeout.")
            return NS(
                id="int_recovered",
                status="completed",
                steps=[NS(type="model_output", content=[NS(type="text", text="Recovered after timeout")])],
                output_text="Recovered after timeout",
                usage=NS(total_input_tokens=500, total_output_tokens=100, total_cached_tokens=0),
            )

    chat = Chat()
    gemini = TimingOutGemini()
    chat.send("test timeout", gemini, model="gemini-3-flash-preview", wait=True)
    assert not chat.running
    assert len(gemini.calls) == 2
    assert any("retrying" in e.text for e in chat.log)
    assert any("Recovered after timeout" in e.text for e in chat.log)


def test_list_gemini_models_dynamic_discovery(monkeypatch):
    from print_assist.ai import client as ai_client

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(ai_client, "_GEMINI_MODELS_CACHE", None)

    fake_models = [
        NS(name="models/gemini-3-flash-preview", display_name="Gemini 3 Flash Preview", supported_actions=["generateContent"]),
        NS(name="models/gemini-4-hydrogen", display_name="Gemini 4 Hydrogen", supported_actions=["generateContent"]),
        NS(name="models/gemini-2.5-flash-tts", display_name="TTS Model", supported_actions=["generateContent"]),
        NS(name="models/gemini-embedding-001", display_name="Embed", supported_actions=["embedContent"]),
    ]

    class FakeClient:
        class models:
            @staticmethod
            def list():
                return fake_models

    monkeypatch.setattr(ai_client, "make_gemini_client", lambda: FakeClient())
    result = ai_client.list_gemini_models(force_refresh=True)
    ids = [m["id"] for m in result]
    assert "gemini-4-hydrogen" in ids
    assert "gemini-3-flash-preview" in ids
    assert "gemini-2.5-flash-tts" not in ids
    assert "gemini-embedding-001" not in ids
    assert ids[0] == "gemini-3-flash-preview"


def test_web_search_and_fetch_url_tools(monkeypatch):
    import httpx2
    from print_assist.ai.chat import _call, _tools

    tool_names = {t["name"] for t in _tools()}
    assert "web_search" in tool_names
    assert "fetch_url" in tool_names

    monkeypatch.setattr(
        httpx2,
        "post",
        lambda *a, **kw: NS(
            status_code=200,
            raise_for_status=lambda: None,
            text='<a class="result__a" href="https://example.com/spec">M3 Spec</a><a class="result__snippet">Diameter 3mm</a>',
        ),
    )
    out, ok = _call("web_search", {"query": "M3 screw dimensions"})
    assert ok is True
    data = json.loads(out)
    assert data["results"][0]["title"] == "M3 Spec"
    assert data["results"][0]["url"] == "https://example.com/spec"

    monkeypatch.setattr(
        httpx2,
        "get",
        lambda *a, **kw: NS(
            status_code=200,
            url="https://example.com/spec",
            raise_for_status=lambda: None,
            text="<html><script>ignore()</script><body><p>Body width is 28mm.</p></body></html>",
        ),
    )
    out2, ok2 = _call("fetch_url", {"url": "https://example.com/spec"})
    assert ok2 is True
    data2 = json.loads(out2)
    assert "Body width is 28mm." in data2["content"]
    assert "ignore()" not in data2["content"]


def test_agy_cli_chat_turn(monkeypatch):
    from print_assist.ai import chat as ai_chat, client as ai_client

    calls = []

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kw):
            calls.append(cmd)
            self._polls = 0

        def poll(self):
            self._polls += 1
            return 0 if self._polls >= 1 else None

        def communicate(self):
            payload = json.dumps({
                "conversation_id": "conv-123",
                "status": "SUCCESS",
                "response": "Created the box.",
                "usage": {"input_tokens": 120, "output_tokens": 15, "cache_read_tokens": 30},
            })
            return payload, ""

    monkeypatch.setattr(ai_chat.subprocess, "Popen", lambda cmd, **kw: FakeProc(cmd, **kw))
    monkeypatch.setattr(ai_client, "find_agy_bin", lambda: "/fake/agy")

    assert ai_client.provider("agy") == "agy"
    assert ai_client.has_api_key("agy") is True
    c = ai_client.make_client("agy")
    assert isinstance(c, ai_client.AgyClient)

    ch = Chat()
    ch.send("make a box", c, model="", wait=True)
    assert ch._agy_conv == "conv-123"
    assert ch.tokens == {"input": 120, "output": 15, "cache_read": 30}
    assert ch.log[-1].role == "assistant"
    assert ch.log[-1].text == "Created the box."
    assert "--conversation" not in calls[0]

    ch.send("fillet it", c, model="", wait=True)
    assert "--conversation" in calls[1]
    assert "conv-123" in calls[1]
