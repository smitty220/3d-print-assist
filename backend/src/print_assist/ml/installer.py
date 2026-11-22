"""On-demand installer for the local Character Mode ML engine (macOS Apple Silicon + Windows).

Downloads everything into `ml_root()` (outside the desktop bundle):
  - `bin/sd-cli` (prebuilt `stable-diffusion.cpp` binary for Metal / CUDA / Vulkan / CPU)
  - `models/painter/` (non-gated FLUX.1-Kontext Q4_K_M GGUF + VAE + CLIP-L + T5-XXL Q4_K_M GGUF)
  - `venv/` (isolated Python environment with PyTorch + `hy3dgen` for Hunyuan3D-2mv / Hunyuan3D-2mini)
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import ml_job_status, ml_root
from .hardware import HardwareProfile, detect_hardware

SD_CPP_RELEASES_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"

# Verified non-gated Hugging Face downloads for FLUX.1-Kontext Q4_K_M (reference-image editing & turnarounds)
# Tuple: (hf_repo, remote_name, local_name, approx_mb_for_progress_weight)
PAINTER_FILES: tuple[tuple[str, str, str, int], ...] = (
    ("QuantStack/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q4_K_M.gguf", "flux1-kontext-dev-Q4_K_M.gguf", 6611),
    ("second-state/FLUX.1-dev-GGUF", "ae.safetensors", "ae.safetensors", 320),
    ("second-state/FLUX.1-dev-GGUF", "clip_l.safetensors", "clip_l.safetensors", 235),
    ("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf", 2765),
)

HUNYUAN3D_GIT = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"


@dataclass
class InstallState:
    installing: bool = False
    stage: str = ""
    progress_pct: int = 0
    detail: str = ""
    error: str = ""
    started_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sd_cli_path(root: Path | None = None) -> Path | None:
    base = (root or ml_root()) / "bin"
    if not base.is_dir():
        return None
    names = ("sd-cli.exe", "sd.exe") if sys.platform == "win32" else ("sd-cli", "sd")
    for n in names:
        direct = base / n
        if direct.is_file():
            return direct
        for hit in base.rglob(n):
            if hit.is_file():
                return hit
    return None


def painter_dir(root: Path | None = None) -> Path:
    return (root or ml_root()) / "models" / "painter"


def sculptor_venv(root: Path | None = None) -> Path:
    return (root or ml_root()) / "venv"


def sculptor_python(root: Path | None = None) -> Path:
    v = sculptor_venv(root)
    return v / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def painter_ready(root: Path | None = None) -> bool:
    r = root or ml_root()
    if sd_cli_path(r) is None:
        return False
    pdir = painter_dir(r)
    return all((pdir / local_name).is_file() and (pdir / local_name).stat().st_size > 1024 for _, _, local_name, _ in PAINTER_FILES)


def sculptor_ready(root: Path | None = None) -> bool:
    r = root or ml_root()
    marker = r / "sculptor_ready.json"
    return sculptor_python(r).is_file() and marker.is_file()


def _dir_size_gb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / (1024 ** 3), 2)


def pick_sd_assets(assets: list[dict[str, Any]], hw: HardwareProfile) -> list[dict[str, Any]]:
    """Select the exact prebuilt `stable-diffusion.cpp` release zip(s) for macOS or Windows."""
    by_name = [(str(a.get("name", "")), a) for a in assets]
    if hw.os_name == "darwin":
        for name, a in by_name:
            low = name.lower()
            if "darwin" in low and "arm64" in low and low.endswith(".zip"):
                return [a]
    elif hw.os_name == "win32":
        if hw.accelerator == "cuda":
            picked = []
            for name, a in by_name:
                low = name.lower()
                if "win-cuda" in low and low.endswith(".zip"):
                    picked.append(a)
                elif low.startswith("cudart-") and "win" in low and low.endswith(".zip"):
                    picked.append(a)
            if picked:
                return picked
        if hw.accelerator == "vulkan":
            for name, a in by_name:
                if "win-vulkan" in name.lower() and name.lower().endswith(".zip"):
                    return [a]
        for name, a in by_name:
            if "win-cpu" in name.lower() and name.lower().endswith(".zip"):
                return [a]
    else:
        for name, a in by_name:
            low = name.lower()
            if "linux" in low and ("vulkan" if hw.accelerator == "vulkan" else "ubuntu") in low and low.endswith(".zip"):
                return [a]
    return []


def _find_tool(name: str, extra_paths: tuple[Path, ...] = ()) -> str | None:
    """Locate an external CLI tool even when launched from a macOS .app bundle with a minimal PATH."""
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates: list[Path] = list(extra_paths)
    if sys.platform == "win32":
        exe = f"{name}.exe" if not name.endswith(".exe") else name
        candidates.extend([
            home / ".local" / "bin" / exe,
            home / ".cargo" / "bin" / exe,
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / exe,
            Path("C:/Program Files/Git/cmd") / exe,
        ])
    else:
        candidates.extend([
            home / ".local" / "bin" / name,
            home / ".cargo" / "bin" / name,
            Path("/opt/homebrew/bin") / name,
            Path("/usr/local/bin") / name,
            Path("/usr/bin") / name,
        ])
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _find_system_python() -> str | None:
    if not getattr(sys, "frozen", False) and Path(sys.executable).is_file():
        return sys.executable
    for candidate in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        found = _find_tool(candidate)
        if found:
            return found
    return None


class EngineInstaller:
    """Manages on-demand local Character Mode engine installation with progress and cancellation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = InstallState()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def status(self, root: Path | None = None) -> dict[str, Any]:
        r = root or ml_root()
        hw = detect_hardware()
        p_ok = painter_ready(r)
        s_ok = sculptor_ready(r)
        with self._lock:
            st = self._state.to_dict()
        return {
            "ready": p_ok and s_ok,
            "painter_ready": p_ok,
            "sculptor_ready": s_ok,
            "engine_dir": str(r),
            "disk_gb": _dir_size_gb(r),
            "hardware": hw.to_dict(),
            "job": ml_job_status(),
            **st,
        }

    def cancel(self) -> None:
        self._cancel.set()

    def start_install(self, components: tuple[str, ...] = ("painter", "sculptor")) -> dict[str, Any]:
        with self._lock:
            if self._state.installing:
                return self.status()
            self._cancel.clear()
            self._state = InstallState(
                installing=True,
                stage="Preparing Character Engine",
                progress_pct=1,
                detail="Checking hardware and disk space…",
                started_at=time.time(),
            )
            self._thread = threading.Thread(target=self._run, args=(components,), daemon=True)
            self._thread.start()
        return self.status()

    def _update(self, stage: str, pct: int, detail: str = "") -> None:
        with self._lock:
            self._state.stage = stage
            self._state.progress_pct = max(0, min(100, int(pct)))
            self._state.detail = detail

    def ensure_ready(
        self,
        component: str = "painter",
        on_progress: Callable[[str], None] | None = None,
        root: Path | None = None,
    ) -> None:
        """Ensure the requested local ML component is installed, waiting seamlessly if a background download is running."""
        r = root or ml_root()
        is_ready = (lambda: painter_ready(r)) if component == "painter" else (lambda: sculptor_ready(r))
        if is_ready():
            return
        if root is not None or os.environ.get("PRINT_ASSIST_DISABLE_ML_AUTO_INSTALL") == "1":
            raise RuntimeError(f"Character Mode {component} assets are not available yet.")
        self.start_install()
        last_msg = ""
        while not is_ready():
            st = self.status(r)
            msg = f"{st['stage']} ({st['progress_pct']}%)" + (f" — {st['detail']}" if st["detail"] else "")
            if on_progress and msg != last_msg:
                last_msg = msg
                on_progress(msg)
            if not st["installing"]:
                if is_ready():
                    return
                raise RuntimeError(st["error"] or f"Character Mode {component} failed to initialize.")
            time.sleep(0.5)

    def _download_file(self, url: str, dest: Path, label: str, pct_start: float, pct_end: float) -> None:
        """Download `url` to `dest` with HTTP Range resume, retries, and live byte-accurate progress."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        for attempt in range(3):
            existing = tmp.stat().st_size if tmp.is_file() else 0
            headers = {"User-Agent": "3d-print-assist/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    status = getattr(resp, "status", 200)
                    content_len = int(resp.headers.get("Content-Length") or 0)
                    if status == 206 and existing > 0:
                        mode = "ab"
                        got = existing
                        total = existing + content_len if content_len > 0 else 0
                    else:
                        mode = "wb"
                        got = 0
                        total = content_len
                    last_t = 0.0
                    with tmp.open(mode) as fh:
                        while True:
                            if self._cancel.is_set():
                                raise InterruptedError("Setup paused.")
                            chunk = resp.read(1024 * 512)
                            if not chunk:
                                break
                            fh.write(chunk)
                            got += len(chunk)
                            now = time.time()
                            if now - last_t >= 0.4:
                                last_t = now
                                frac = (got / total) if total > 0 else 0.0
                                pct = int(round(pct_start + (pct_end - pct_start) * frac))
                                mb_got = got / (1024 * 1024)
                                mb_tot = total / (1024 * 1024) if total > 0 else 0.0
                                info = f"{label}: {mb_got:.0f} / {mb_tot:.0f} MB" if total else f"{label}: {mb_got:.0f} MB"
                                self._update(f"Downloading {label}", pct, info)
                    if total > 0 and got < total:
                        raise ConnectionResetError(f"Short read for {label} ({got}/{total} bytes)")
                    self._update(f"Downloaded {label}", int(round(pct_end)), f"{label} ready")
                    tmp.replace(dest)
                    return
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and existing > 1024:
                    # .part already holds the complete file
                    tmp.replace(dest)
                    return
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
            except InterruptedError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))

    def _install_painter(self, r: Path, hw: HardwareProfile) -> None:
        if sd_cli_path(r) is None:
            self._update("Downloading 2D engine binary", 2, f"Fetching stable-diffusion.cpp for {hw.os_name}/{hw.accelerator}…")
            req = urllib.request.Request(SD_CPP_RELEASES_API, headers={"User-Agent": "3d-print-assist/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                rel = json.loads(resp.read().decode("utf-8"))
            chosen = pick_sd_assets(rel.get("assets") or [], hw)
            if not chosen:
                raise RuntimeError(f"No prebuilt stable-diffusion.cpp binary found for {hw.os_name}/{hw.arch}")
            bin_dir = r / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for asset in chosen:
                zpath = bin_dir / asset["name"]
                self._download_file(asset["browser_download_url"], zpath, asset["name"], 2, 5)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(bin_dir)
                zpath.unlink(missing_ok=True)
            cli = sd_cli_path(r)
            if cli is not None and sys.platform != "win32":
                cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        pdir = painter_dir(r)
        pdir.mkdir(parents=True, exist_ok=True)
        # Allocate 5% -> 68% proportionally by actual file size so large files smoothly advance the bar
        total_mb = sum(weight for _, _, _, weight in PAINTER_FILES) or 1
        cursor = 5.0
        span = 63.0
        for repo, remote_name, local_name, weight_mb in PAINTER_FILES:
            p0 = cursor
            p1 = cursor + span * (weight_mb / total_mb)
            cursor = p1
            target = pdir / local_name
            if target.is_file() and target.stat().st_size > 1024:
                self._update("Preparing 2D model", int(round(p1)), f"{local_name} already downloaded")
                continue
            url = f"https://huggingface.co/{repo}/resolve/main/{remote_name}"
            self._download_file(url, target, local_name, p0, p1)

    def _run_cmd(self, cmd: list[str], stage: str, pct: int, detail: str) -> None:
        self._update(stage, pct, detail)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        out_tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            if self._cancel.is_set():
                proc.terminate()
                raise InterruptedError("Setup paused.")
            s = line.strip()
            if s:
                out_tail.append(s)
                del out_tail[:-8]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{stage} failed (exit {rc}): {' | '.join(out_tail[-3:])}")

    def _install_sculptor(self, r: Path, hw: HardwareProfile) -> None:
        venv_dir = sculptor_venv(r)
        py = sculptor_python(r)
        uv = _find_tool("uv")
        if not py.is_file():
            if uv:
                self._run_cmd([uv, "venv", str(venv_dir), "--python", "3.12"], "Preparing 3D sculptor runtime", 70, "Setting up Python environment…")
            else:
                sys_py = _find_system_python()
                if not sys_py:
                    raise RuntimeError("Python 3.10+ or uv is required to install the local 3D Sculptor runtime.")
                self._run_cmd([sys_py, "-m", "venv", str(venv_dir)], "Preparing 3D sculptor runtime", 70, "Setting up Python environment…")

        pip_prefix = [uv, "pip", "install", "--python", str(py)] if uv else [str(py), "-m", "pip", "install"]
        if hw.os_name == "win32" and hw.accelerator == "cuda":
            torch_cmd = [*pip_prefix, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
        else:
            torch_cmd = [*pip_prefix, "torch", "torchvision"]
        self._run_cmd(torch_cmd, "Installing PyTorch runtime", 73, f"Installing PyTorch for {hw.accelerator.upper()}…")

        deps = [
            "huggingface-hub>=0.27,<1", "diffusers>=0.31,<0.36", "transformers>=4.46,<5", "accelerate", "einops",
            "omegaconf", "safetensors", "Pillow", "PyYAML", "tqdm", "scipy",
            "scikit-image", "opencv-python-headless", "trimesh", "pygltflib",
            "pymeshlab", "rembg", "onnxruntime",
        ]
        self._run_cmd([*pip_prefix, *deps], "Installing 3D mesh packages", 77, "Installing mesh & diffusion dependencies…")

        # Install Hunyuan3D-2 source package without forcing CUDA-only rasterizer builds
        repo_dir = r / "Hunyuan3D-2"
        if not repo_dir.is_dir():
            git = _find_tool("git")
            if not git:
                raise RuntimeError("git is required to fetch Hunyuan3D-2 runtime")
            self._run_cmd([git, "clone", "--depth", "1", HUNYUAN3D_GIT, str(repo_dir)],
                          "Downloading Hunyuan3D-2 engine", 79, "Cloning Hunyuan3D-2 shape generator…")

        # Download config.yaml + fp16 safetensors directly with HTTP Range resume & live MB progress
        sub_dir = r / "models" / "sculptor" / hw.sculptor_repo / hw.sculptor_subfolder
        sub_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = sub_dir / "config.yaml"
        if not (cfg_file.is_file() and cfg_file.stat().st_size > 100):
            cfg_url = f"https://huggingface.co/{hw.sculptor_repo}/resolve/main/{hw.sculptor_subfolder}/config.yaml"
            self._download_file(cfg_url, cfg_file, f"{hw.sculptor_subfolder}/config.yaml", 80, 81)
        weights_file = sub_dir / "model.fp16.safetensors"
        if not (weights_file.is_file() and weights_file.stat().st_size > 1024):
            w_url = f"https://huggingface.co/{hw.sculptor_repo}/resolve/main/{hw.sculptor_subfolder}/model.fp16.safetensors"
            self._download_file(w_url, weights_file, f"{hw.sculptor_label} weights", 81, 99)

        (r / "sculptor_ready.json").write_text(json.dumps({
            "installed_at": time.time(),
            "repo": hw.sculptor_repo,
            "subfolder": hw.sculptor_subfolder,
            "accelerator": hw.accelerator,
            "tier": hw.tier,
        }, indent=2), encoding="utf-8")

    def _run(self, components: tuple[str, ...]) -> None:
        r = ml_root()
        r.mkdir(parents=True, exist_ok=True)
        hw = detect_hardware()
        try:
            if "painter" in components:
                self._install_painter(r, hw)
            if "sculptor" in components:
                self._install_sculptor(r, hw)
            with self._lock:
                self._state.installing = False
                self._state.stage = "Ready"
                self._state.progress_pct = 100
                self._state.detail = f"Character Engine ready ({hw.gpu_name}, {hw.tier} tier)"
                self._state.error = ""
        except InterruptedError as exc:
            with self._lock:
                self._state.installing = False
                self._state.stage = "Cancelled"
                self._state.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state.installing = False
                self._state.stage = "Error"
                self._state.error = str(exc)


installer = EngineInstaller()
