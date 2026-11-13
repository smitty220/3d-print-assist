"""The in-app AI chat (the AI panel's API mode): a conversation with Claude that designs through the same tools the MCP
server gives Claude Code, so both modes behave the same. Each turn runs on a background thread; the panel polls the
log. Claude's changes go through the app as the actor "claude": built and checked first, refused if they break the
part, never applied to a locked feature.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from .client import MessagesClient, model_name
from .guide import GUIDE

MAX_TURNS = 40
GEMINI_TIMEOUT = 180  # seconds
SKIP_TOOLS = {"get_guide"}  # the guide is the system prompt


@dataclass
class Entry:
    role: Literal["user", "assistant", "tool", "error"]
    text: str
    tool: str | None = None
    ok: bool = True
    at: float = field(default_factory=time.time)


_WEB_TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for real-world product dimensions, specs, datasheets, standard hardware sizes "
            "(e.g. M3 screw, 608 bearing, device dimensions), or 3D printing tips."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch and read the text content of an HTTP/HTTPS web page (e.g. a product spec page from web_search).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "HTTP or HTTPS URL to fetch"}},
            "required": ["url"],
        },
    },
]

_WEB_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def _web_search(args: dict) -> tuple[str, bool]:
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}), False
    import httpx2

    try:
        r = httpx2.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _WEB_UA},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"web_search failed: {exc}"}), False

    titles = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
    results = []
    for i, (href, t) in enumerate(titles[:6]):
        s = snippets[i] if i < len(snippets) else ""
        results.append({
            "title": html.unescape(re.sub(r"<[^>]+>", "", t)).strip(),
            "url": html.unescape(href),
            "snippet": html.unescape(re.sub(r"<[^>]+>", "", s)).strip(),
        })
    return json.dumps({"query": query, "results": results}), True


def _fetch_url(args: dict) -> tuple[str, bool]:
    url = str(args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "url must start with http:// or https://"}), False
    import httpx2

    try:
        r = httpx2.get(url, headers={"User-Agent": _WEB_UA}, follow_redirects=True, timeout=15)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"fetch_url failed: {exc}"}), False

    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return json.dumps({"url": str(r.url), "content": text[:6000]}), True


def _tools() -> list[dict]:
    from print_assist_mcp import server

    listed = asyncio.run(server.mcp.list_tools())
    base = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in listed if t.name not in SKIP_TOOLS]
    return base + _WEB_TOOLS


def _wire_tools(actor: str = "claude") -> None:
    """Point the shared tools at this backend, in process, acting as the given actor."""
    from fastapi.testclient import TestClient

    from print_assist_mcp import server

    from ..app import app

    server.ACTOR = {"X-Actor": actor}
    server.set_client(TestClient(app))


def _call(name: str, args: dict, cancel: threading.Event | None = None) -> tuple[list[dict] | str, bool]:
    from mcp.server.mcpserver import Image

    from print_assist_mcp import server

    if name == "web_search":
        return _web_search(args)
    if name == "fetch_url":
        return _fetch_url(args)

    if name == "wait_for_concept":
        cid = args.get("id")
        timeout_s = args.get("timeout_s", 600)
        deadline = time.time() + max(5, min(int(timeout_s), 3600))
        while True:
            if cancel is not None and cancel.is_set():
                return "Cancelled by user.", False
            r = server._call("GET", "/api/concepts", params={"drawings": False})
            if isinstance(r, dict):
                return server._text(r), True
            c = next((x for x in r if x["id"] == cid), None)
            if c is None:
                return server._text({"error": f"no concept {cid!r}"}), False
            if c["status"] != "pending" or time.time() >= deadline:
                return server._text({k: c[k] for k in ("id", "title", "status", "note")}), True
            time.sleep(0.5)

    fn = getattr(server, name, None)
    if fn is None or name in SKIP_TOOLS or not callable(fn):
        return f"unknown tool {name!r}", False
    try:
        out = fn(**args)
    except TypeError as exc:
        return f"bad arguments for {name}: {exc}", False
    if isinstance(out, Image):
        data = out.data if out.data is not None else open(out.path, "rb").read()
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(data).decode()}}], True
    ok = True
    try:
        ok = "error" not in json.loads(out) if isinstance(out, str) and out.startswith("{") else True
    except json.JSONDecodeError:
        pass
    return out, ok


def _block(b) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    # Thinking, fallback and other blocks go back unchanged.
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)


class Chat:
    """One conversation per open design."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.log: list[Entry] = []
        self.running = False
        self.cancel = threading.Event()
        self.tokens = {"input": 0, "output": 0, "cache_read": 0}
        self._thread: threading.Thread | None = None
        self._gemini_last: str | None = None  # Gemini keeps the conversation server side; continue from here

    def to_dict(self) -> dict:
        return {"running": self.running, "log": [asdict(e) for e in self.log], "tokens": dict(self.tokens)}

    def reset(self) -> None:
        if self.running:
            raise RuntimeError("the AI is still working; stop it first")
        self.messages, self.log = [], []
        self._gemini_last = None

    def send(self, text: str, client: Any, model: str | None = None, wait: bool = False) -> None:
        if self.running:
            raise RuntimeError("the AI is still working on the last message")
        self.running = True
        self.cancel.clear()
        self.log.append(Entry("user", text))
        self.messages.append({"role": "user", "content": text})
        self._thread = threading.Thread(target=self._run, args=(client, model or model_name()), daemon=True)
        self._thread.start()
        if wait:
            self._thread.join()

    def _run(self, client: Any, model: str) -> None:
        try:
            if hasattr(client, "interactions"):
                self._run_gemini(client, model)
            else:
                self._run_claude(client, model)
        except Exception as exc:  # noqa: BLE001 - API or unexpected failure: report it in the panel
            msg = str(exc)
            if hasattr(client, "interactions") and ("limit: 0" in msg or "too_many_requests" in msg):
                msg = (
                    f"Rate limit for {model}: this model is restricted or has a 0 token/min limit on the Free Tier. "
                    "Please switch to 'Gemini 2.5 Flash' or 'Gemini 3 Flash' in the Model dropdown, "
                    "or upgrade your Google AI Studio tier."
                )
            elif "timed out" in msg.lower() or "timeout" in msg.lower():
                msg = (
                    f"Request timed out for {model}. Google's servers took longer than {GEMINI_TIMEOUT}s to respond. "
                    "Please try sending your message again, or switch models in the dropdown."
                )
            self.log.append(Entry("error", f"{type(exc).__name__}: {msg}", ok=False))
        finally:
            self.running = False

    def _run_claude(self, client: Any, model: str) -> None:
        _wire_tools(actor="claude")
        tools = _tools()
        from ..features.library import FEATURES

        catalog = json.dumps([f.describe() for f in FEATURES.values()], separators=(",", ":"))
        system = [{"type": "text", "text": GUIDE + "\n\nFeature catalog:\n" + catalog, "cache_control": {"type": "ephemeral"}}]
        for _ in range(MAX_TURNS):
            if self.cancel.is_set():
                self.log.append(Entry("error", "Stopped."))
                return
            resp = client.create(
                model=model, max_tokens=16000, system=system, tools=tools, messages=self.messages,
                output_config={"effort": "high"},
                # If a safety classifier declines, the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
            u = getattr(resp, "usage", None)
            if u is not None:
                self.tokens["input"] += getattr(u, "input_tokens", 0) or 0
                self.tokens["output"] += getattr(u, "output_tokens", 0) or 0
                self.tokens["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.messages.append({"role": "assistant", "content": [_block(b) for b in resp.content]})
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    self.log.append(Entry("assistant", b.text.strip()))
            stop = getattr(resp, "stop_reason", "end_turn")
            if stop == "refusal":
                self.log.append(Entry("error", "Claude declined this request.", ok=False))
                return
            if stop == "max_tokens":
                self.log.append(Entry("error", "The reply was cut off (too long); ask again more narrowly.", ok=False))
                return
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                return
            results = []
            for b in uses:
                content, ok = _call(b.name, b.input or {}, cancel=self.cancel)
                summary = content if isinstance(content, str) else "(image)"
                self.log.append(Entry("tool", summary[:600], tool=b.name, ok=ok))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": not ok})
            self.messages.append({"role": "user", "content": results})
        self.log.append(Entry("error", "Stopped: too many steps in one turn.", ok=False))

    def _run_gemini(self, client: Any, model: str) -> None:
        _wire_tools(actor="gemini")
        tools = _tools()
        from ..features.library import FEATURES

        catalog = json.dumps([f.describe() for f in FEATURES.values()], separators=(",", ":"))
        gemini_directives = (
            "\n\nDIRECTIVES FOR IN-APP CHAT (FOLLOW THESE STRICTLY):\n"
            "- Direct building: When the user requests a specific shape, part, or geometry change with explicit dimensions or parameters "
            "(e.g., 'make me a 20mm cube', 'cylinder 30mm diameter and 15mm tall', 'chamfer top edge 1mm', 'change wall thickness to 2mm'), "
            "this is a change they spelled out exactly. You MUST SKIP propose_concept and wait_for_concept entirely! "
            "Call apply_changes directly to build the geometry, then render_preview and run_checks.\n"
            "- When to propose concepts: Only call propose_concept when the user's request is high-level, creative, ambiguous, or asks for open-ended "
            "designs where dimensions and styling must be explored (e.g. 'design a phone stand', 'organizer for my desk', 'a case for my trimmer').\n"
            "- Conversational responses: Always provide clear, concise natural language in your reply describing what changes you made, "
            "how they were checked, and any decisions for the user, just like a helpful CAD engineering colleague.\n"
        )
        system_text = GUIDE + gemini_directives + "\n\nFeature catalog:\n" + catalog

        interaction_tools = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]

        def _safe_create(**kwargs: Any) -> Any:
            nonlocal model
            for attempt in range(3):
                try:
                    return client.interactions.create(**kwargs)
                except Exception as exc:
                    err_str = str(exc).lower()
                    is_rate_limit = "rate limit" in err_str or "too_many_requests" in err_str or "429" in err_str
                    is_timeout = "timeout" in err_str or "timed out" in err_str
                    if is_rate_limit and model != "gemini-3-flash-preview":
                        fallback_model = "gemini-3-flash-preview"
                        self.log.append(Entry("assistant", f"Rate limit reached on {model}. Seamlessly continuing with {fallback_model}..."))
                        model = fallback_model
                        kwargs["model"] = fallback_model
                        return client.interactions.create(**kwargs)
                    if is_rate_limit and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    if is_timeout and attempt < 2:
                        self.log.append(Entry("assistant", f"Gemini response timed out; retrying (attempt {attempt + 2}/3)..."))
                        time.sleep(1)
                        continue
                    raise

        user_prompt = ""
        for m in reversed(self.messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                user_prompt = m["content"]
                break

        create_kwargs: dict[str, Any] = {
            "model": model,
            "previous_interaction_id": self._gemini_last,
            "input": user_prompt,
            "tools": interaction_tools,
            "system_instruction": system_text,
            "timeout": GEMINI_TIMEOUT,
        }

        interaction = _safe_create(**create_kwargs)
        self._gemini_last = getattr(interaction, "id", None) or self._gemini_last

        for _ in range(MAX_TURNS):
            if self.cancel.is_set():
                self.log.append(Entry("error", "Stopped."))
                return

            self._gemini_last = getattr(interaction, "id", None) or self._gemini_last
            u = getattr(interaction, "usage", None)
            if u is not None:
                self.tokens["input"] += getattr(u, "total_input_tokens", 0) or 0
                self.tokens["output"] += getattr(u, "total_output_tokens", 0) or 0
                self.tokens["cache_read"] += getattr(u, "total_cached_tokens", 0) or 0

            steps = getattr(interaction, "steps", None) or []

            found_text = False
            for step in steps:
                stype = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
                if stype == "model_output":
                    content = getattr(step, "content", None) or (step.get("content") if isinstance(step, dict) else [])
                    for item in content:
                        t = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                        if t and t.strip():
                            self.log.append(Entry("assistant", t.strip()))
                            found_text = True
            if not found_text:
                out_text = getattr(interaction, "output_text", None)
                if out_text and out_text.strip():
                    self.log.append(Entry("assistant", out_text.strip()))

            status = getattr(interaction, "status", None)
            if status == "failed":
                err = getattr(interaction, "error", "Gemini declined or failed this request.")
                self.log.append(Entry("error", str(err), ok=False))
                return

            fc_steps = []
            for step in steps:
                stype = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
                if stype == "function_call":
                    fc_steps.append(step)

            if not fc_steps:
                return

            results = []
            for fc in fc_steps:
                if self.cancel.is_set():
                    self.log.append(Entry("error", "Stopped."))
                    return

                fc_name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
                fc_args = getattr(fc, "arguments", None) or (fc.get("arguments") if isinstance(fc, dict) else {})
                fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else "")

                content, ok = _call(fc_name, fc_args or {}, cancel=self.cancel)
                if self.cancel.is_set():
                    self.log.append(Entry("error", "Stopped."))
                    return

                summary = content if isinstance(content, str) else "(image)"
                self.log.append(Entry("tool", summary[:600], tool=fc_name, ok=ok))

                if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "image":
                    src = content[0].get("source", {})
                    fn_result = [{"type": "image", "data": src.get("data", ""), "mime_type": src.get("media_type", "image/png")}]
                else:
                    fn_result = content if isinstance(content, str) else json.dumps(content)

                results.append({
                    "type": "function_result",
                    "call_id": fc_id,
                    "name": fc_name,
                    "result": fn_result,
                })

            interaction = _safe_create(
                model=model,
                previous_interaction_id=getattr(interaction, "id", None),
                input=results,
                tools=interaction_tools,
                system_instruction=system_text,
                timeout=GEMINI_TIMEOUT,
            )
            self._gemini_last = getattr(interaction, "id", None) or self._gemini_last

        self.log.append(Entry("error", "Stopped: too many steps in one turn.", ok=False))
