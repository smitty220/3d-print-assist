"""AI API client for the in-app chat: supporting both Anthropic (Claude) and Google (Gemini).

The chat is the API mode of the AI panel; the preferred mode is Claude Code driving the app over MCP, which needs no
key here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol


DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

GEMINI_MODEL_ALIASES = {
    "gemini-4": "gemini-4-argon",
    "gemini-4.0": "gemini-4-argon",
    "gemini-4-pro": "gemini-4-argon",
    "4": "gemini-4-argon",
    "4.0": "gemini-4-argon",
}


class MessagesClient(Protocol):
    """The slice of anthropic.Anthropic().messages the loop uses (tests pass a scripted fake)."""

    def create(self, **kwargs: Any) -> Any: ...


ENV_FILE = Path(__file__).resolve().parents[3] / ".env"  # backend/.env


def load_env_key(env: Path = ENV_FILE) -> None:
    """Read API keys and configuration from backend/.env unless the shell set them.

    Called before every AI request, so a key added to the file works without restarting the backend.
    """
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        k = k.removeprefix("export ").strip()
        if sep and k in ENV_KEYS and k not in _SHELL_ENV:
            os.environ[k] = v.strip().strip('"').strip("'")


ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "PRINT_ASSIST_AI_PROVIDER",
    "PRINT_ASSIST_MODEL",
    "ANTHROPIC_WORKSPACE_ID",
)
_SHELL_ENV = {k for k in ENV_KEYS if os.environ.get(k)}


def provider(override: str | None = None) -> str:
    """Return active provider: 'gemini' or 'anthropic'.

    Can be explicitly set with override, PRINT_ASSIST_AI_PROVIDER ('gemini' or 'anthropic').
    Otherwise auto-detects based on available API keys and model name, defaulting
    to 'anthropic' for backwards compatibility.
    """
    load_env_key()
    p = (override or os.environ.get("PRINT_ASSIST_AI_PROVIDER") or "").strip().lower()
    if p in ("gemini", "google"):
        return "gemini"
    if p in ("anthropic", "claude"):
        return "anthropic"

    model = (os.environ.get("PRINT_ASSIST_MODEL") or "").lower()
    if model.startswith("gemini") or model in GEMINI_MODEL_ALIASES:
        return "gemini"

    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_gemini and not has_anthropic:
        return "gemini"
    if has_anthropic and not has_gemini:
        return "anthropic"

    return "anthropic"


def model_name(model_override: str | None = None, provider_override: str | None = None) -> str:
    load_env_key()
    p = provider_override or provider()
    custom = (model_override or os.environ.get("PRINT_ASSIST_MODEL") or "").strip()
    if custom:
        if p == "gemini":
            return GEMINI_MODEL_ALIASES.get(custom.lower(), custom)
        return custom
    return DEFAULT_GEMINI_MODEL if p == "gemini" else DEFAULT_ANTHROPIC_MODEL


def has_api_key(provider_override: str | None = None) -> bool:
    load_env_key()
    p = provider_override or provider()
    if p == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_anthropic_client() -> MessagesClient:
    load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("No Anthropic API key: set ANTHROPIC_API_KEY in your shell or in backend/.env, then restart the backend")
    import anthropic

    # An organisation-level key (not scoped to a workspace) must name the workspace to bill on every request.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    # The beta namespace: the chat sends the server-side refusal fallback (see chat.py).
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None).beta.messages


def make_gemini_client() -> Any:
    load_env_key()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key: set GEMINI_API_KEY in your shell or in backend/.env, then restart the backend")
    from google import genai

    client = genai.Client(api_key=key)
    try:
        from google.genai._gaos.utils import RetryConfig

        if hasattr(client, "interactions") and hasattr(client.interactions, "sdk_configuration"):
            # Disable automatic long backoff sleeps on 429 so errors report immediately to the UI
            client.interactions.sdk_configuration.retry_config = RetryConfig("none", None, False)
    except Exception:
        pass
    return client


def make_client(provider_override: str | None = None) -> Any:
    load_env_key()
    p = provider_override or provider()
    if p == "gemini":
        return make_gemini_client()
    return make_anthropic_client()

