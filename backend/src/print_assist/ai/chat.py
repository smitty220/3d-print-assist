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
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .client import AgyClient, MessagesClient, model_name
from .guide import GUIDE

MAX_TURNS = 40
GEMINI_TIMEOUT = 180  # seconds
AGY_PRINT_TIMEOUT = "60m"
AGY_IDLE_TIMEOUT = 1800  # seconds (30m) without any stream event before killing a stalled turn
AGY_TOTAL_TIMEOUT = 3600  # hard cap (seconds) for a single active agy turn (60m)
SKIP_TOOLS = {"get_guide", "wait_for_concept"}  # guide is system prompt; UI handles concept approval turns

_IN_APP_DIRECTIVES = (
    "\n\nDIRECTIVES FOR IN-APP CHAT (FOLLOW THESE STRICTLY):\n"
    "- Direct building: When the user requests a specific shape, part, or geometry change with explicit dimensions or parameters "
    "(e.g., 'make me a 20mm cube', 'cylinder 30mm diameter and 15mm tall', 'chamfer top edge 1mm', 'change wall thickness to 2mm'), "
    "this is a change they spelled out exactly. SKIP propose_concept entirely! "
    "Call apply_changes directly to build the geometry, then render_preview and run_checks.\n"
    "- Concept approval workflow: Only call propose_concept for high-level, creative, or open-ended requests where dimensions "
    "or styling must be explored. Immediately after propose_concept, END YOUR TURN with a brief summary so the user can review "
    "and click Approve & build or reply — do NOT call wait_for_concept and do NOT poll or wait.\n"
    "- Efficiency & error recovery: Batch related ops into a single apply_changes call; inspect list_faces before pick_face. "
    "If an approach fails twice, switch to a simpler construction or explain the blocker instead of looping.\n"
    "- Conversational responses: Always provide clear, concise natural language describing what changes you made, "
    "how they were checked, and any decisions for the user.\n"
)

_AGY_SYSTEM_PREAMBLE = (
    "You are the CAD assistant inside 3D Print Assist.\n"
    "STRICT TOOL RULES:\n"
    "1. ONLY call tools from the `3d-print-assist` MCP server (get_guide, get_model, get_feature_catalog, "
    "apply_changes, update, render_preview, run_checks, print_layout, list_faces, pick_face, list_fonts, "
    "list_concepts, propose_concept, look_at_concept, set_creation_mode, propose_character_design, "
    "sculpt_character, wait_for_character_job, update_mesh_prep, generate_image, list_images, look_at_image, "
    "list_references, look_at_reference, section_reference, add_reference, open_design, create_design, "
    "list_designs, export).\n"
    "2. NEVER inspect or edit files, and NEVER use run_command, shell scripts, python/uv commands, "
    "invoke_subagent, schedule, timers, or background tasks.\n"
    "3. CONCEPT APPROVAL: After calling propose_concept or once propose_character_design finishes, NEVER call "
    "wait_for_concept and NEVER run scripts or background tasks to wait for approval. End your turn immediately with "
    "a concise summary — the app resumes you automatically when the user clicks Approve & build or replies.\n"
    "4. Respect the design's active mode: in Engineer Mode build parametric CAD (`propose_concept` → `apply_changes`). "
    "In Character Mode (hybrid clay + CAD engineering): ALWAYS paint and sculpt the organic character as a clean SOLID "
    "shape first via `propose_character_design` (do NOT fight the 2D painter to draw tiny flexi hinges or magnet holes!). "
    "When the user clicks Approve and build (or Direct Build is active), YOU translate EVERYTHING they asked for into "
    "`sculpt_character(...)` parameters automatically — e.g. for any flexi/articulated toy pass `flexi_segments=5` and "
    "`flatten_bottom_mm=8` (so every segment rests flat on the bed); for magnets pass `magnet_hole='bottom'|'back'`; for "
    "planters/cups pass `hollow_cavity=True`; for keychains pass `keychain_loop=True`; for figurines on a stand pass "
    "`base_style='round'`. If `propose_character_design` or `sculpt_character` returns `status: \"running\"`, call "
    "`wait_for_character_job(wait_s=40)` until `status == \"done\"` (never start a duplicate job). ALWAYS inspect "
    "`render_preview(view='iso')` and `render_preview(view='top')` after sculpting to verify the 3D result truly matches "
    "what the user asked for (e.g., visible flexi joints & flat bottom); if not, call `update_mesh_prep` and re-check "
    "before handing it over!\n"
    "5. Batch related ops in one apply_changes call. If the user refers to a concept or you want to check existing "
    "concepts, call list_concepts first, then look_at_concept(id=...) on an existing ID. If an approach fails twice, "
    "switch strategy instead of looping.\n\n"
    "User request: "
)

_AGY_TURN_REMINDER = (
    "\n\n[System rule: Use ONLY `3d-print-assist` MCP tools. Never use shell/run_command, python, "
    "subagents, background tasks, or wait_for_concept — end your turn right after propose_concept or edits.]"
)


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


def _kill_proc(proc: Any) -> None:
    """Terminate a subprocess and its process group cleanly so background tasks cannot linger."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:  # noqa: BLE001
        pass
    pid = getattr(proc, "pid", None)
    if os.name == "posix" and isinstance(pid, int) and pid > 0:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                return
        except OSError:
            pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _tool_output_ok(out: str) -> bool:
    """Determine whether an MCP tool output represents success (avoiding false positives on `'error': null`)."""
    s = out.strip()
    if not s:
        return True
    if s.startswith("{"):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                if data.get("error"):
                    return False
                status = data.get("status")
                if isinstance(status, int) and status >= 400:
                    return False
                return True
        except json.JSONDecodeError:
            pass
    return not (s.startswith("Error") or s.startswith("Refused:"))


def _format_agy_error(raw: str) -> str:
    """Turn internal CLI timeout / background-task noise into a clean user-facing message."""
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw or "").strip()
    low = cleaned.lower()
    if "print timeout" in low or ("background task" in low and ("terminated" in low or "waiting" in low)):
        return (
            "The assistant timed out while waiting on a background step, so the stalled task was "
            "automatically cleaned up. Please resend your instruction (or try switching models)."
        )
    if len(cleaned) > 600:
        cleaned = cleaned[:600].rstrip() + "…"
    return cleaned or "The assistant encountered an unexpected error."


def _block(b) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    # Thinking, fallback and other blocks go back unchanged.
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)


class Chat:
    """One conversation per open design (persisted to `chat.json` inside the design folder)."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = storage_path
        self.messages: list[dict] = []
        self.log: list[Entry] = []
        self.running = False
        self.cancel = threading.Event()
        self.tokens = {"input": 0, "output": 0, "cache_read": 0}
        self.started_at: float | None = None
        self.last_event_at: float | None = None
        self.current_step: str | None = None
        self.steps = 0
        self._lock = threading.Lock()
        self._queue: list[tuple[str, Any, str]] = []
        self._thread: threading.Thread | None = None
        self._gemini_last: str | None = None  # Gemini keeps the conversation server side; continue from here
        self._agy_conv: str | None = None  # Antigravity CLI conversation_id
        self._active_proc: Any = None  # currently running subprocess (for immediate kill on stop/shutdown)
        self._load()

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.is_file():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            self.log = [Entry(**e) for e in raw.get("log", []) if isinstance(e, dict) and "role" in e and "text" in e]
            self._agy_conv = raw.get("agy_conv") or None
            self._gemini_last = raw.get("gemini_last") or None
            if isinstance(raw.get("messages"), list):
                self.messages = raw["messages"][-40:]
        except Exception:  # noqa: BLE001 - corrupt chat file should never break opening a design
            pass

    def _save(self) -> None:
        if self._storage_path is None:
            return
        try:
            # Keep compact serializable text-only summary of messages so Claude/Gemini fallback retains context
            compact_msgs = []
            for m in self.messages[-40:]:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    compact_msgs.append({"role": m.get("role", "user"), "content": m["content"]})
            payload = {
                "log": [asdict(e) for e in self.log[-200:]],
                "agy_conv": self._agy_conv,
                "gemini_last": self._gemini_last,
                "messages": compact_msgs,
            }
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._storage_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._storage_path)
        except Exception:  # noqa: BLE001
            pass

    def to_dict(self) -> dict:
        now = time.time()
        elapsed_s = max(0, int(now - self.started_at)) if self.running and self.started_at else 0
        idle_s = max(0, int(now - self.last_event_at)) if self.running and self.last_event_at else 0
        with self._lock:
            queued = len(self._queue)
        return {
            "running": self.running,
            "log": [asdict(e) for e in self.log],
            "tokens": dict(self.tokens),
            "elapsed_s": elapsed_s,
            "idle_s": idle_s,
            "current_step": self.current_step,
            "steps": self.steps,
            "queued": queued,
        }

    def _touch(self, step: str | None = None) -> None:
        self.last_event_at = time.time()
        if step is not None:
            self.current_step = step

    def _record_tool(self, tool: str, text: str, ok: bool = True) -> None:
        self.steps += 1
        self.last_event_at = time.time()
        self.current_step = None
        self.log.append(Entry("tool", text, tool=tool, ok=ok))
        self._save()

    def stop(self) -> None:
        """Cancel the active turn and immediately terminate any active subprocess."""
        with self._lock:
            self.cancel.set()
            self._queue.clear()
            proc = self._active_proc
        if proc is not None:
            _kill_proc(proc)

    def reset(self) -> None:
        if self.running:
            raise RuntimeError("the AI is still working; stop it first")
        with self._lock:
            self._queue.clear()
        self.messages, self.log = [], []
        self.started_at = None
        self.last_event_at = None
        self.current_step = None
        self.steps = 0
        self._gemini_last = None
        self._agy_conv = None
        self._save()

    def _drain_queue(self) -> list[str]:
        with self._lock:
            items = [t for t, _, _ in self._queue]
            self._queue.clear()
        return items

    def send(self, text: str, client: Any, model: str | None = None, wait: bool = False) -> None:
        chosen_model = model or model_name()
        with self._lock:
            self.log.append(Entry("user", text))
            if self.running:
                self._queue.append((text, client, chosen_model))
                return
            self.running = True
            now = time.time()
            self.started_at = now
            self.last_event_at = now
            self.current_step = "Thinking"
            self.steps = 0
            self.cancel.clear()
            self.messages.append({"role": "user", "content": text})
            self._thread = threading.Thread(target=self._run, args=(client, chosen_model), daemon=True)
            self._thread.start()
        if wait and self._thread is not None:
            self._thread.join()

    def _run(self, client: Any, model: str) -> None:
        while True:
            try:
                if isinstance(client, AgyClient):
                    self._run_agy(client, model)
                elif hasattr(client, "interactions"):
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
            with self._lock:
                self._save()
                if self.cancel.is_set() or not self._queue:
                    self._queue.clear()
                    self.running = False
                    self.current_step = None
                    return
                next_text, client, model = self._queue.pop(0)
                self.messages.append({"role": "user", "content": next_text})
                self.last_event_at = time.time()
                self.current_step = "Thinking"

    def _run_agy(self, client: AgyClient, model: str) -> None:
        user_text = str(self.messages[-1]["content"])
        if self._agy_conv:
            prompt = f"{user_text}{_AGY_TURN_REMINDER}"
        else:
            prompt = f"{_AGY_SYSTEM_PREAMBLE}{user_text}"
        cmd = [
            client.binary,
            "-p", prompt,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--print-timeout", AGY_PRINT_TIMEOUT,
        ]
        if self._agy_conv:
            cmd.extend(["--conversation", self._agy_conv])
        if model:
            cmd.extend(["--model", model])

        port = os.environ.get("PRINT_ASSIST_PORT", "8010")
        from ..project import data_root

        # Always run in an isolated workspace dir (not the repo root) so agy only sees MCP tools, not repo files/venv.
        work_dir = data_root() / "ai_workspace"
        work_dir.mkdir(parents=True, exist_ok=True)
        mcp_cfg = {
            "mcpServers": {
                "3d-print-assist": {
                    "type": "http",
                    "url": f"http://127.0.0.1:{port}/mcp/",
                }
            }
        }
        (work_dir / ".mcp.json").write_text(json.dumps(mcp_cfg, indent=2), encoding="utf-8")

        popen_kw: dict[str, Any] = {
            "cwd": str(work_dir),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        env = os.environ.copy()
        env["PRINT_ASSIST_PORT"] = port
        env["PRINT_ASSIST_URL"] = f"http://127.0.0.1:{port}"
        if sys.platform == "win32":
            popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kw["start_new_session"] = True
            extra = [str(Path.home() / ".local" / "bin"), "/opt/homebrew/bin", "/usr/local/bin"]
            env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
        popen_kw["env"] = env
        proc = subprocess.Popen(cmd, **popen_kw)
        with self._lock:
            self._active_proc = proc

        import queue

        result_data: dict | None = None
        last_text = ""
        lines_q: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                if proc.stdout is not None:
                    for raw in proc.stdout:
                        lines_q.put(raw)
            finally:
                lines_q.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        def _handle_line(raw_line: str) -> None:
            nonlocal result_data, last_text
            s = raw_line.strip()
            if not s.startswith("{"):
                return
            try:
                ev = json.loads(s)
            except json.JSONDecodeError:
                return
            if ev.get("conversation_id"):
                self._agy_conv = str(ev["conversation_id"])
            self._touch()
            kind = ev.get("event")
            if kind == "step_update":
                su = ev.get("step_update") or {}
                if su.get("step_type") == "tool":
                    info = su.get("tool_info") or {}
                    tname = su.get("tool_name") or "tool"
                    params = info.get("parameters") or {}
                    if tname == "call_mcp_tool" and params.get("ToolName"):
                        tname = str(params["ToolName"])
                    elif tname == "read_file":
                        return  # skip noisy MCP schema reads
                    if su.get("state") == "DONE":
                        out = str(info.get("output") or "").strip()
                        self._record_tool(tname, out[:400], ok=_tool_output_ok(out))
                    else:
                        self._touch(tname)
                elif su.get("step_type") == "agent_response":
                    if su.get("state") == "DONE":
                        td = (su.get("text_delta") or "").strip()
                        if td:
                            last_text = td
                        self._touch("Thinking")
                    else:
                        self._touch("Writing response")
            elif kind == "result":
                result_data = ev.get("result") or {}

        try:
            turn_started = time.time()
            last_stream_at = turn_started
            while True:
                if self.cancel.is_set():
                    _kill_proc(proc)
                    self.log.append(Entry("error", "Stopped.", ok=False))
                    return
                now = time.time()
                if now - last_stream_at >= AGY_IDLE_TIMEOUT:
                    _kill_proc(proc)
                    self.log.append(Entry(
                        "error",
                        "Stopped automatically after 30 minutes of inactivity so background tasks would not hang. "
                        "Please resend your instruction (or try another model).",
                        ok=False,
                    ))
                    return
                if now - turn_started >= AGY_TOTAL_TIMEOUT:
                    _kill_proc(proc)
                    self.log.append(Entry(
                        "error",
                        "Stopped automatically after the 60-minute active turn limit. "
                        "Any completed changes were kept — please continue with a follow-up instruction.",
                        ok=False,
                    ))
                    return
                try:
                    line = lines_q.get(timeout=0.25)
                except queue.Empty:
                    if proc.poll() is not None and lines_q.empty():
                        break
                    continue
                if line is None:
                    break
                last_stream_at = time.time()
                _handle_line(line)

            while not lines_q.empty():
                rem = lines_q.get_nowait()
                if rem is not None:
                    _handle_line(rem)
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            proc.wait()
            if self.cancel.is_set():
                self.log.append(Entry("error", "Stopped.", ok=False))
                return

            if result_data is not None:
                if result_data.get("conversation_id"):
                    self._agy_conv = str(result_data["conversation_id"])
                u = result_data.get("usage") or {}
                self.tokens["input"] += int(u.get("input_tokens") or 0)
                self.tokens["output"] += int(u.get("output_tokens") or 0)
                self.tokens["cache_read"] += int(u.get("cache_read_tokens") or 0)
                if proc.returncode == 0 and result_data.get("status") == "SUCCESS":
                    reply = (result_data.get("response") or last_text or "").strip() or "(Done)"
                    self.log.append(Entry("assistant", reply))
                    return
                err = result_data.get("error") or result_data.get("response") or stderr or f"agy exited with {proc.returncode}"
                self.log.append(Entry("error", _format_agy_error(str(err)), ok=False))
                return

            if proc.returncode != 0:
                err = stderr or f"agy exited with {proc.returncode}"
                self.log.append(Entry("error", _format_agy_error(str(err)), ok=False))
                return
            self.log.append(Entry("assistant", last_text or "(Done)"))
        finally:
            with self._lock:
                if self._active_proc is proc:
                    self._active_proc = None
            _kill_proc(proc)

    def _run_claude(self, client: Any, model: str) -> None:
        _wire_tools(actor="claude")
        tools = _tools()
        from ..features.library import FEATURES

        catalog = json.dumps([f.describe() for f in FEATURES.values()], separators=(",", ":"))
        system = [{
            "type": "text",
            "text": GUIDE + _IN_APP_DIRECTIVES + "\n\nFeature catalog:\n" + catalog,
            "cache_control": {"type": "ephemeral"},
        }]
        for _ in range(MAX_TURNS):
            if self.cancel.is_set():
                self.log.append(Entry("error", "Stopped.", ok=False))
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
                if self.cancel.is_set():
                    self.log.append(Entry("error", "Stopped.", ok=False))
                    return
                self._touch(b.name)
                content, ok = _call(b.name, b.input or {}, cancel=self.cancel)
                if self.cancel.is_set():
                    self.log.append(Entry("error", "Stopped.", ok=False))
                    return
                summary = content if isinstance(content, str) else "(image)"
                self._record_tool(b.name, summary[:600], ok=ok)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": not ok})
            self._touch("Thinking")
            self.messages.append({"role": "user", "content": results})
            for q_text in self._drain_queue():
                self.messages.append({"role": "user", "content": q_text})
        self.log.append(Entry("error", "Stopped: too many steps in one turn.", ok=False))

    def _run_gemini(self, client: Any, model: str) -> None:
        _wire_tools(actor="gemini")
        tools = _tools()
        from ..features.library import FEATURES

        catalog = json.dumps([f.describe() for f in FEATURES.values()], separators=(",", ":"))
        system_text = GUIDE + _IN_APP_DIRECTIVES + "\n\nFeature catalog:\n" + catalog

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
                self.log.append(Entry("error", "Stopped.", ok=False))
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
                    self.log.append(Entry("error", "Stopped.", ok=False))
                    return

                fc_name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "")
                fc_args = getattr(fc, "arguments", None) or (fc.get("arguments") if isinstance(fc, dict) else {})
                fc_id = getattr(fc, "id", None) or (fc.get("id") if isinstance(fc, dict) else "")

                self._touch(fc_name)
                content, ok = _call(fc_name, fc_args or {}, cancel=self.cancel)
                if self.cancel.is_set():
                    self.log.append(Entry("error", "Stopped.", ok=False))
                    return

                summary = content if isinstance(content, str) else "(image)"
                self._record_tool(fc_name, summary[:600], ok=ok)
                self._touch("Thinking")

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

            for q_text in self._drain_queue():
                results.append({"type": "text", "text": q_text})

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
