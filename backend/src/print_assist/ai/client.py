"""The Claude API client for the in-app chat: the key (from the shell or backend/.env), the model, and the client.

The chat is the API mode of the AI panel; the preferred mode is Claude Code driving the app over MCP, which needs no
key here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL = "claude-opus-5-5"


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic().messages the loop uses (tests pass a scripted fake)."""

    def create(self, **kwargs: Any) -> Any: ...


ENV_FILE = Path(__file__).resolve().parents[3] / ".env"  # backend/.env


def load_env_key(env: Path = ENV_FILE) -> None:
    """Read ANTHROPIC_API_KEY (and PRINT_ASSIST_MODEL, ANTHROPIC_WORKSPACE_ID, GEMINI_API_KEY for images) from backend/.env unless the shell set them.

    Called before every AI request, so a key added to the file works without restarting the backend.
    """
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        k = k.removeprefix("export ").strip()
        if sep and k in ENV_KEYS and k not in _SHELL_ENV:
            os.environ[k] = v.strip().strip('"').strip("'")


ENV_KEYS = ("ANTHROPIC_API_KEY", "PRINT_ASSIST_MODEL", "ANTHROPIC_WORKSPACE_ID", "GEMINI_API_KEY")
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
    # The beta namespace: the chat sends the server-side refusal fallback (see chat.py).
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None).beta.messages
