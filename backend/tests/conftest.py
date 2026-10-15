"""Shared test setup. No test may call the real Anthropic API, and no test touches the user's designs folder."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch):
    import anthropic

    def refuse(*a, **k):
        raise AssertionError("tests must not call the real Anthropic API; pass a scripted client")

    monkeypatch.setattr(anthropic, "Anthropic", refuse)


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PRINT_ASSIST_DATA", str(tmp_path / "designs"))
    from print_assist import app

    app.reset()
    yield
    app.reset()
