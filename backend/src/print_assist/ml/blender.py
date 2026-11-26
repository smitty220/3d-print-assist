"""Cross-platform Headless Blender 4.5 LTS Engine for stylized organic & Flexi-Factory 3D modeling.

Discovery priority:
  1. `PRINT_ASSIST_BLENDER` environment variable override
  2. Bundled inside the packaged desktop app (`3D Print Assist.app/Contents/Resources/blender` on macOS,
     `3DPrintAssist/blender` on Windows/Linux)
  3. Repo-staged vendor copy (`backend/vendor/blender`) for local development
  4. Shared ML engine folder (`ml_root()/blender`) for on-demand installs
  5. System-installed Blender (`/Applications/Blender.app`, `Program Files\\Blender Foundation`, or `PATH`)

Runs Blender strictly as an isolated `--background --factory-startup --python` subprocess (zero Python version
conflicts with the app's Python 3.12 / OpenCascade runtime, full GPLv3 process isolation).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ml_root

BLENDER_VERSION = "4.5.9"
BLENDER_SERIES = "Blender4.5"
BLENDER_BASE_URL = f"https://download.blender.org/release/{BLENDER_SERIES}"


class BlenderError(RuntimeError):
    pass


@dataclass
class BlenderInstallation:
    path: Path
    source: str  # env | bundled | vendor | ml_engine | system
    version: str = ""


_CACHED_INFO: dict[str, Any] | None = None
_CACHE_LOCK = threading.Lock()


def _exec_in_dir(root: Path) -> Path | None:
    """Locate the Blender executable inside a staged/bundled directory on macOS, Windows, or Linux."""
    if not root.exists():
        return None
    if sys.platform == "darwin":
        candidates = (
            root / "Blender.app" / "Contents" / "MacOS" / "Blender",
            root / "Contents" / "MacOS" / "Blender",
            root / "Blender",
        )
    elif sys.platform == "win32":
        candidates = (
            root / "blender.exe",
            root / "Blender" / "blender.exe",
        )
    else:
        candidates = (
            root / "blender",
            root / "Blender" / "blender",
        )
    for c in candidates:
        if c.is_file() and (sys.platform == "win32" or os.access(c, os.X_OK)):
            return c
    return None


def find_blender(root: Path | None = None) -> BlenderInstallation | None:
    """Locate standalone Blender across bundled, staged vendor, ml-engine, and system paths."""
    # 1. Explicit env override
    env_path = os.environ.get("PRINT_ASSIST_BLENDER", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return BlenderInstallation(p, "env")
        hit = _exec_in_dir(p)
        if hit is not None:
            return BlenderInstallation(hit, "env")

    # 2. Bundled inside frozen PyInstaller desktop app (macOS .app & Windows/Linux dist)
    bundled_roots: list[Path] = []
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        exe_dir = Path(sys.executable).resolve().parent
        if sys.platform == "darwin":
            bundled_roots.append(exe_dir.parent / "Resources" / "blender")
        bundled_roots.append(exe_dir / "blender")
        if hasattr(sys, "_MEIPASS"):
            bundled_roots.append(Path(sys._MEIPASS) / "blender")
    for br in bundled_roots:
        hit = _exec_in_dir(br)
        if hit is not None:
            return BlenderInstallation(hit, "bundled")

    # 3. Repo-staged vendor directory (`backend/vendor/blender`) during dev/testing
    repo_backend = Path(__file__).resolve().parents[3]
    vendor_hit = _exec_in_dir(repo_backend / "vendor" / "blender")
    if vendor_hit is not None:
        return BlenderInstallation(vendor_hit, "vendor")

    # 4. Shared ML engine folder (`ml_root()/blender`)
    ml_hit = _exec_in_dir((root or ml_root()) / "blender")
    if ml_hit is not None:
        return BlenderInstallation(ml_hit, "ml_engine")

    # 5. System-installed Blender
    sys_candidates: list[Path] = []
    if sys.platform == "darwin":
        sys_candidates.extend([
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications" / "Blender.app" / "Contents" / "MacOS" / "Blender",
        ])
    elif sys.platform == "win32":
        for pf in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not pf:
                continue
            bf = Path(pf) / "Blender Foundation"
            if bf.is_dir():
                for sub in sorted(bf.iterdir(), reverse=True):
                    if sub.is_dir():
                        sys_candidates.append(sub / "blender.exe")
    for sc in sys_candidates:
        if sc.is_file() and (sys.platform == "win32" or os.access(sc, os.X_OK)):
            return BlenderInstallation(sc, "system")

    which = shutil.which("blender")
    if which:
        return BlenderInstallation(Path(which), "system")
    return None


def find_blender_executable(root: Path | None = None) -> Path | None:
    inst = find_blender(root)
    return inst.path if inst is not None else None


def _probe_version(exe: Path) -> str:
    try:
        res = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        for line in (res.stdout or "").splitlines():
            if line.strip().lower().startswith("blender"):
                return line.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def blender_status(root: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    """Return ready status, executable path, source, and version for `/api/ml/status`."""
    global _CACHED_INFO
    with _CACHE_LOCK:
        if _CACHED_INFO is not None and not refresh and root is None:
            return dict(_CACHED_INFO)
    inst = find_blender(root)
    if inst is None:
        info = {
            "ready": False,
            "path": None,
            "source": "missing",
            "version": "",
        }
    else:
        ver = _probe_version(inst.path)
        info = {
            "ready": True,
            "path": str(inst.path),
            "source": inst.source,
            "version": ver or f"Blender {BLENDER_VERSION} LTS",
        }
    if root is None:
        with _CACHE_LOCK:
            _CACHED_INFO = dict(info)
    return info


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _target_download_url() -> tuple[str, str]:
    mach = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
        fname = f"blender-{BLENDER_VERSION}-macos-{arch}.dmg"
    elif sys.platform == "win32":
        arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
        fname = f"blender-{BLENDER_VERSION}-windows-{arch}.zip"
    else:
        fname = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
    return f"{BLENDER_BASE_URL}/{fname}", fname


def ensure_blender_installed(root: Path | None = None) -> Path:
    """Return a working Blender executable, downloading official Blender 4.5 LTS into `ml_root()/blender` if missing."""
    existing = find_blender_executable(root)
    if existing is not None:
        return existing

    if os.environ.get("PRINT_ASSIST_DISABLE_ML_AUTO_INSTALL") == "1":
        raise BlenderError("Headless Blender is not installed and auto-install is disabled.")

    dest_dir = (root or ml_root()) / "blender"
    url, fname = _target_download_url()
    with tempfile.TemporaryDirectory(prefix="pa_blender_dl_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / fname
        req = urllib.request.Request(url, headers={"User-Agent": "3DPrintAssist/1.0"})
        try:
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=600) as resp:
                with open(archive, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            raise BlenderError(f"Failed to download Blender {BLENDER_VERSION} LTS: {exc}") from exc

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            mnt = tmp_path / "mnt"
            mnt.mkdir()
            subprocess.run(
                ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mnt), str(archive)],
                check=True,
                capture_output=True,
            )
            try:
                src_app = mnt / "Blender.app"
                if not src_app.is_dir():
                    raise BlenderError(f"Blender.app not found inside {fname}")
                subprocess.run(["ditto", str(src_app), str(dest_dir / "Blender.app")], check=True)
            finally:
                subprocess.run(["hdiutil", "detach", str(mnt), "-quiet"], check=False, capture_output=True)
        elif sys.platform == "win32":
            extract_dir = tmp_path / "extract"
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_dir)
            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            src = roots[0] if len(roots) == 1 and (roots[0] / "blender.exe").is_file() else extract_dir
            for item in src.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))
        else:
            import tarfile

            extract_dir = tmp_path / "extract"
            with tarfile.open(archive, "r:xz") as tf:
                tf.extractall(extract_dir)
            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            src = roots[0] if len(roots) == 1 and (roots[0] / "blender").is_file() else extract_dir
            for item in src.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))

    exe = _exec_in_dir(dest_dir)
    if exe is None:
        raise BlenderError(f"Downloaded Blender archive did not contain an executable in {dest_dir}")
    blender_status(refresh=True)
    return exe


def _lib_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "pa_blender_lib.py").is_file():
        return here
    if hasattr(sys, "_MEIPASS"):
        frozen_dir = Path(sys._MEIPASS) / "print_assist" / "ml"
        if (frozen_dir / "pa_blender_lib.py").is_file():
            return frozen_dir
    return here


def _terminate_proc(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except Exception:  # noqa: BLE001
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass


def run_blender_script(
    script: str,
    out_glb: Path,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: int = 120,
    blender_exe: Path | None = None,
) -> dict[str, Any]:
    """Execute a Python modeling script inside headless Blender 4.5 LTS and return the exported `.glb` report.

    Inside `script`, `pa_blender_lib` is importable directly, `OUT_GLB` is the target `.glb` path, and
    `PA_PARAMS` is the JSON-decoded `params` dict. If the script doesn't explicitly call `pa_export_glb(OUT_GLB)`,
    the harness automatically exports all visible mesh objects in the scene to `OUT_GLB`!
    """
    exe = blender_exe or ensure_blender_installed()
    out_glb = Path(out_glb).resolve()
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    if out_glb.exists():
        out_glb.unlink()

    with tempfile.TemporaryDirectory(prefix="pa_blender_run_") as tmp:
        tmp_path = Path(tmp)
        user_script = tmp_path / "user_model.py"
        runner_script = tmp_path / "runner.py"
        report_json = tmp_path / "report.json"
        params_json = tmp_path / "params.json"
        user_script.write_text(script, encoding="utf-8")
        params_json.write_text(json.dumps(params or {}), encoding="utf-8")

        runner_code = f"""\
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, {str(_lib_dir())!r})
import bpy
import pa_blender_lib as pa
from pa_blender_lib import *  # noqa: F401,F403

OUT_GLB = Path({str(out_glb)!r})
REPORT_JSON = Path({str(report_json)!r})
PA_PARAMS = json.loads(Path({str(params_json)!r}).read_text(encoding="utf-8"))

pa_clear_scene()
ns = {{
    "__name__": "__main__",
    "bpy": bpy,
    "pa": pa,
    "OUT_GLB": OUT_GLB,
    "PA_PARAMS": PA_PARAMS,
}}
for _k in dir(pa):
    if not _k.startswith("_"):
        ns[_k] = getattr(pa, _k)

try:
    code = Path({str(user_script)!r}).read_text(encoding="utf-8")
    exec(compile(code, "blender_character_script.py", "exec"), ns)
    if not OUT_GLB.is_file() or OUT_GLB.stat().st_size == 0:
        pa.pa_export_glb(OUT_GLB)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_get()]
    REPORT_JSON.write_text(json.dumps({{
        "ok": True,
        "out_glb": str(OUT_GLB),
        "bytes": OUT_GLB.stat().st_size if OUT_GLB.is_file() else 0,
        "objects": [o.name for o in meshes],
    }}), encoding="utf-8")
except Exception as exc:
    tb = traceback.format_exc()
    REPORT_JSON.write_text(json.dumps({{"ok": False, "error": str(exc), "traceback": tb}}), encoding="utf-8")
    print(tb, file=sys.stderr, flush=True)
    sys.exit(1)
"""
        runner_script.write_text(runner_code, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [
            str(exe),
            "--background",
            "--factory-startup",
            "--python-exit-code", "1",
            "--python", str(runner_script),
        ]
        t0 = time.monotonic()
        popen_kw: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env,
        }
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        elif sys.platform == "win32":
            popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        proc = subprocess.Popen(cmd, **popen_kw)
        try:
            stdout, stderr = proc.communicate(timeout=max(10, int(timeout_s)))
        except subprocess.TimeoutExpired as exc:
            _terminate_proc(proc)
            raise BlenderError(f"Blender script timed out after {timeout_s}s") from exc

        elapsed = round(time.monotonic() - t0, 2)
        report: dict[str, Any] = {}
        if report_json.is_file():
            try:
                report = json.loads(report_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}

        if proc.returncode != 0 or not report.get("ok"):
            err_msg = report.get("error") or ""
            tb = report.get("traceback") or ""
            tail = "\n".join((stderr or stdout or "").splitlines()[-12:])
            detail = f"{err_msg}\n{tb}".strip() if err_msg else tail
            raise BlenderError(f"Blender execution failed (exit {proc.returncode}): {detail}")

        if not out_glb.is_file() or out_glb.stat().st_size < 100:
            raise BlenderError("Blender finished without producing a valid .glb mesh file.")

        report["elapsed_s"] = elapsed
        return report


def run_blender_character_script(
    script: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: int = 120,
    blender_exe: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Run a Blender character script and return `(glb_bytes, report)`."""
    with tempfile.TemporaryDirectory(prefix="pa_blender_glb_") as tmp:
        out_glb = Path(tmp) / "character.glb"
        report = run_blender_script(
            script,
            out_glb,
            params=params,
            timeout_s=timeout_s,
            blender_exe=blender_exe,
        )
        return out_glb.read_bytes(), report
