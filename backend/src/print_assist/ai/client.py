"""AI API client for the in-app chat: supporting both Anthropic (Claude) and Google (Gemini).

The chat is the API mode of the AI panel; the preferred mode is Claude Code driving the app over MCP, which needs no
key here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol


DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

DEFAULT_GEMINI_MODELS: list[dict[str, str]] = [
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview (Recommended, Free Tier)"},
    {"id": "gemini-4-hydrogen", "label": "Gemini 4 Hydrogen"},
    {"id": "gemini-4-carbon", "label": "Gemini 4 Carbon"},
    {"id": "gemini-4-argon", "label": "Gemini 4 Argon"},
    {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
]

_GEMINI_SKIP_SUBSTRINGS = (
    "-tts", "-image", "nano-banana", "omni", "transcribe",
    "robotics", "computer-use", "embedding", "-live", "decision",
    "customtools", "-latest", "flash-lite",
)
_GEMINI_MODELS_CACHE: tuple[float, list[dict[str, str]]] | None = None
_GEMINI_MODELS_TTL = 60.0  # seconds

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

    client = genai.Client(api_key=key, http_options={"timeout": 180000})
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


def list_gemini_models(force_refresh: bool = False) -> list[dict[str, str]]:
    """Fetch available Gemini chat models from the API, caching briefly for fast UI status polls."""
    global _GEMINI_MODELS_CACHE
    load_env_key()
    if not os.environ.get("GEMINI_API_KEY"):
        return list(DEFAULT_GEMINI_MODELS)

    now = time.time()
    if not force_refresh and _GEMINI_MODELS_CACHE is not None:
        cached_at, cached_models = _GEMINI_MODELS_CACHE
        if now - cached_at < _GEMINI_MODELS_TTL:
            return list(cached_models)

    try:
        client = make_gemini_client()
        fetched: list[dict[str, str]] = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            actions = getattr(m, "supported_actions", None) or []
            if not name.startswith("models/gemini-") or "generateContent" not in actions:
                continue
            mid = name.removeprefix("models/")
            if any(s in mid for s in _GEMINI_SKIP_SUBSTRINGS):
                continue
            label = getattr(m, "display_name", None) or mid
            if mid == "gemini-3-flash-preview":
                label = f"{label} (Recommended, Free Tier)"
            fetched.append({"id": mid, "label": label})

        if fetched:
            # Put recommended model first, then newest models first
            fetched.sort(key=lambda x: (x["id"] != "gemini-3-flash-preview", x["id"]), reverse=False)
            # Re-order non-recommended items in reverse (newest versions at the top)
            rec = [x for x in fetched if x["id"] == "gemini-3-flash-preview"]
            others = list(reversed([x for x in fetched if x["id"] != "gemini-3-flash-preview"]))
            result = rec + others
            _GEMINI_MODELS_CACHE = (now, result)
            return list(result)
    except Exception:
        pass

    return list(DEFAULT_GEMINI_MODELS)

