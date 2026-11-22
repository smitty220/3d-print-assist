"""Local 2D Painter for Character Mode Design Files (powered by `stable-diffusion.cpp` + FLUX.1-Kontext GGUF).

Takes a user prompt + optional uploaded reference photos (`Photos` tab) and produces a clean,
neutral-background Front / Side / Back turnaround PNG stored directly inside the Design File
(`concepts/K#.png`), never cluttering the user's `Photos` tab.
"""

from __future__ import annotations

import io
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from . import gpu_job, ml_job_update, ml_root
from .hardware import detect_hardware
from .installer import PAINTER_FILES, installer, painter_dir, painter_ready, sd_cli_path

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


class PainterError(RuntimeError):
    pass


def _make_reference_collage(ref_paths: list[Path], out_path: Path, max_side: int = 1024) -> Path:
    """Combine 1+ uploaded reference photos into a clean side-by-side RGB PNG for FLUX.1-Kontext."""
    imgs: list[Image.Image] = []
    for p in ref_paths:
        if p.is_file():
            with Image.open(p) as im:
                imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
    if not imgs:
        raise PainterError("none of the selected reference photo files exist")
    if len(imgs) == 1:
        im = imgs[0]
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        im.save(out_path, format="PNG")
        return out_path
    # Side-by-side strip with matched height
    target_h = min(max_side // 2, max(256, min(im.height for im in imgs)))
    resized = []
    for im in imgs[:4]:
        w = max(64, int(im.width * (target_h / max(1, im.height))))
        resized.append(im.resize((w, target_h), Image.Resampling.LANCZOS))
    total_w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (total_w, target_h), (240, 240, 240))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    canvas.save(out_path, format="PNG")
    return out_path


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=2)
            return
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except Exception:  # noqa: BLE001
                pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def build_turnaround_prompt(prompt: str, turnaround: bool = False, has_refs: bool = False) -> str:
    base = prompt.strip()
    if turnaround:
        prefix = (
            "3D printable figurine character turnaround sheet showing three complete uncropped views "
            "side by side (FRONT view on left, SIDE profile in middle, BACK view on right) with generous white margin "
            "around all edges so tails and fins are never cut off, clean studio lighting, plain solid light-gray background, "
            "chunky sculpted details suitable for 3D printing, no text labels, no watermarks. "
        )
    else:
        prefix = (
            "3D printable figurine studio product photo, complete uncropped centered hero three-quarter view with generous "
            "margin around all edges so the entire silhouette is fully visible, clean studio lighting, plain solid "
            "light-gray background, smooth sculpted vinyl toy finish suitable for 3D printing, no text, no watermarks. "
        )
    ref_hint = "Preserve the likeness, faces, hairstyles, and distinct features of the people/subjects in the reference photo. " if has_refs else ""
    return f"{prefix}{ref_hint}Subject: {base}"


def generate_character_design_image(
    prompt: str,
    ref_paths: list[Path] | None = None,
    turnaround: bool = False,
    steps: int = 14,
    seed: int = -1,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
) -> tuple[bytes, list[str]]:
    """Run local `stable-diffusion.cpp` to create a Character Mode Design File PNG."""
    r = root or ml_root()
    if not painter_ready(r):
        try:
            installer.ensure_ready("painter", on_progress=on_progress, root=root)
        except Exception as exc:  # noqa: BLE001
            raise PainterError(str(exc)) from exc
    cli = sd_cli_path(r)
    assert cli is not None
    pdir = painter_dir(r)
    hw = detect_hardware()

    diff_model = pdir / PAINTER_FILES[0][2]
    vae = pdir / PAINTER_FILES[1][2]
    clip_l = pdir / PAINTER_FILES[2][2]
    t5xxl = pdir / PAINTER_FILES[3][2]

    n_steps = max(8, min(int(steps), 30))
    width, height = (960, 384) if turnaround else (640, 640)
    full_prompt = build_turnaround_prompt(prompt, turnaround=turnaround, has_refs=bool(ref_paths))

    with gpu_job("painting", "Painting character design", f"Loading 2D model on {hw.gpu_name}…"):
        with tempfile.TemporaryDirectory(prefix="pa_painter_") as td:
            td_path = Path(td)
            out_png = td_path / "design.png"
            cmd = [
                str(cli),
                "--diffusion-model", str(diff_model),
                "--vae", str(vae),
                "--clip_l", str(clip_l),
                "--t5xxl", str(t5xxl),
                "-p", full_prompt,
                "-W", str(width),
                "-H", str(height),
                "--steps", str(n_steps),
                "--cfg-scale", "1.0",
                "--sampling-method", "euler",
                "--flash-attn", "on",
                "--cache-mode", "easycache",
                "-o", str(out_png),
            ]
            # Keep text encoders on GPU for high-memory Apple Silicon / 12 GB+ NVIDIA cards; offload only on lower tiers.
            if hw.tier != "high":
                cmd.extend(["--clip-on-cpu", "--offload-to-cpu"])
            if seed >= 0:
                cmd.extend(["-s", str(seed)])
            if ref_paths:
                collage = _make_reference_collage(ref_paths, td_path / "ref.png")
                cmd.extend(["-r", str(collage)])

            popen_kw: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "bufsize": 0,
                "cwd": str(cli.parent),
            }
            if sys.platform == "win32":
                popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                popen_kw["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kw)
            tail: list[str] = []
            buf = b""
            try:
                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(256)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\r" in buf or b"\n" in buf:
                        i_r = buf.find(b"\r")
                        i_n = buf.find(b"\n")
                        cut = min(x for x in (i_r, i_n) if x >= 0)
                        raw_line = buf[:cut]
                        buf = buf[cut + 1:]
                        s = raw_line.decode("utf-8", errors="replace").strip()
                        if not s:
                            continue
                        tail.append(s)
                        del tail[:-12]
                        low = s.lower()
                        if "|" in s and (m := _STEP_RE.search(s)):
                            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
                            if tot <= 50:
                                pct = 15 + int(75 * (cur / tot))
                                detail = f"Step {cur}/{tot}"
                                ml_job_update("Painting character design", pct, detail)
                                if on_progress:
                                    on_progress(detail)
                        elif "decoding" in low:
                            ml_job_update("Finalizing image", 92, "Decoding image…")
                            if on_progress:
                                on_progress("Decoding image…")
                        elif "loading tensors" in low:
                            ml_job_update("Loading 2D model", 10, "Loading weights onto GPU…")
                rc = proc.wait(timeout=600)
            except Exception:
                _kill_proc(proc)
                raise
            finally:
                _kill_proc(proc)

            if rc != 0 or not out_png.is_file():
                raise PainterError(f"Local 2D painter failed (exit {rc}): {' | '.join(tail[-3:])}")
            data = out_png.read_bytes()
            # Verify readable PNG
            with Image.open(io.BytesIO(data)) as im:
                im.load()
            ml_job_update("Done", 100, "Design image ready")
            return data, (["front", "side", "back"] if turnaround else ["front"])
