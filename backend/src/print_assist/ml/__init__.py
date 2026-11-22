"""Local, cross-platform (macOS Apple Silicon + Windows) Character Mode ML engine.

All binaries, Python ML environments, and open-weight models live in an on-demand folder outside the
app bundle (`ml_root()`), keeping the desktop installer lightweight and zero-cost for every LLM harness.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..project import data_root


_GPU_LOCK = threading.Lock()
_JOB_LOCK = threading.Lock()
_JOB_SEQ = 0
_JOB_DONE = threading.Event()
_JOB_DONE.set()
_JOB_STATE: dict[str, Any] = {
    "id": "",
    "active": False,
    "status": "idle",  # idle | running | done | error
    "kind": "",
    "stage": "",
    "progress_pct": 0,
    "detail": "",
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
}
_JOB_RESULT: dict[str, Any] | None = None


def ml_job_status() -> dict[str, Any]:
    with _JOB_LOCK:
        return dict(_JOB_STATE)


def ml_job_update(stage: str = "", progress_pct: int | None = None, detail: str = "") -> None:
    with _JOB_LOCK:
        if not _JOB_STATE["active"]:
            return
        if stage:
            _JOB_STATE["stage"] = stage
        if progress_pct is not None:
            _JOB_STATE["progress_pct"] = max(0, min(100, int(progress_pct)))
        if detail:
            _JOB_STATE["detail"] = detail


@contextmanager
def gpu_job(kind: str, stage: str, detail: str = "") -> Iterator[None]:
    """Serialize local GPU workloads so two ML subprocesses never thrash Metal/CUDA VRAM simultaneously."""
    with _GPU_LOCK:
        with _JOB_LOCK:
            managed = bool(_JOB_STATE["active"] and _JOB_STATE["id"])
            if not managed:
                _JOB_STATE.update({
                    "active": True,
                    "status": "running",
                    "kind": kind,
                    "stage": stage,
                    "progress_pct": 5,
                    "detail": detail,
                    "started_at": time.time(),
                    "finished_at": 0.0,
                    "error": "",
                })
            else:
                _JOB_STATE["stage"] = stage
                _JOB_STATE["detail"] = detail
                if _JOB_STATE["progress_pct"] < 5:
                    _JOB_STATE["progress_pct"] = 5
        try:
            yield
        finally:
            with _JOB_LOCK:
                if not _JOB_STATE["id"]:
                    _JOB_STATE.update({
                        "active": False,
                        "status": "idle",
                        "kind": "",
                        "stage": "",
                        "progress_pct": 0,
                        "detail": "",
                        "started_at": 0.0,
                        "finished_at": time.time(),
                    })


def start_character_job(
    kind: str,
    stage: str,
    detail: str,
    fn: Callable[[], dict[str, Any]],
    wait_s: float = 0.0,
) -> dict[str, Any]:
    """Run a Character Mode GPU job on a background thread (deduplicating retries) and wait up to `wait_s`."""
    global _JOB_SEQ, _JOB_RESULT

    with _JOB_LOCK:
        if _JOB_STATE["active"]:
            # Already running a GPU job: never spawn a second competing process — attach & wait instead!
            ev = _JOB_DONE
        else:
            _JOB_SEQ += 1
            jid = f"J{_JOB_SEQ}"
            _JOB_RESULT = None
            _JOB_DONE.clear()
            ev = _JOB_DONE
            _JOB_STATE.update({
                "id": jid,
                "active": True,
                "status": "running",
                "kind": kind,
                "stage": stage,
                "progress_pct": 3,
                "detail": detail,
                "started_at": time.time(),
                "finished_at": 0.0,
                "error": "",
            })

            def _runner() -> None:
                global _JOB_RESULT
                try:
                    res = fn()
                    with _JOB_LOCK:
                        _JOB_RESULT = res
                        _JOB_STATE["active"] = False
                        _JOB_STATE["status"] = "done"
                        _JOB_STATE["stage"] = "Done"
                        _JOB_STATE["progress_pct"] = 100
                        _JOB_STATE["finished_at"] = time.time()
                        _JOB_STATE["error"] = ""
                except Exception as exc:  # noqa: BLE001
                    with _JOB_LOCK:
                        _JOB_RESULT = None
                        _JOB_STATE["active"] = False
                        _JOB_STATE["status"] = "error"
                        _JOB_STATE["stage"] = "Error"
                        _JOB_STATE["finished_at"] = time.time()
                        _JOB_STATE["error"] = str(exc)
                finally:
                    _JOB_DONE.set()

            threading.Thread(target=_runner, name=f"ml-job-{jid}", daemon=True).start()

    if wait_s < 0:
        ev.wait()
    elif wait_s > 0:
        ev.wait(timeout=min(float(wait_s), 55.0))
    return poll_character_job(wait_s=0.0)


def poll_character_job(wait_s: float = 0.0) -> dict[str, Any]:
    """Wait up to `wait_s` (capped at 45s) for the current Character Mode GPU job and return its status/result."""
    if wait_s > 0:
        _JOB_DONE.wait(timeout=min(float(wait_s), 45.0))
    with _JOB_LOCK:
        st = dict(_JOB_STATE)
        res = dict(_JOB_RESULT) if isinstance(_JOB_RESULT, dict) else _JOB_RESULT
    return {
        "status": st["status"],
        "job": st,
        "result": res if st["status"] == "done" else None,
        "error": st["error"] if st["status"] == "error" else "",
    }


def ml_root() -> Path:
    """Shared folder for local 2D/3D engines and weights (outside the PyInstaller bundle)."""
    if env := os.environ.get("PRINT_ASSIST_ML_DIR"):
        return Path(env).expanduser()
    # Keep beside the `designs/` folder (e.g. ~/Projects/3DPrinting/ml-engine)
    return data_root().parent / "ml-engine"
