"""Shared test setup. No test may call the real Anthropic or Gemini APIs, and no test touches the user's designs folder."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch):
    import anthropic

    def refuse(*a, **k):
        raise AssertionError("tests must not call the real Anthropic API; pass a scripted client")

    monkeypatch.setattr(anthropic, "Anthropic", refuse)

    def refuse_gemini(*a, **k):
        raise AssertionError("tests must not call the real Gemini API; pass a scripted post")

    monkeypatch.setattr("print_assist.images._post", refuse_gemini)
    monkeypatch.setattr("print_assist.images.load_env_key", lambda: None)  # never read the user's backend/.env key

    try:
        from google import genai

        def refuse_genai(*a, **k):
            raise AssertionError("tests must not call the real Gemini API; pass a scripted client")

        monkeypatch.setattr(genai, "Client", refuse_genai)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PRINT_ASSIST_DATA", str(tmp_path / "designs"))
    from print_assist import app

    app.reset()
    yield
    app.reset()
