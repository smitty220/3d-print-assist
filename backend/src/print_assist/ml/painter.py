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
from typing import Callable, Literal

from PIL import Image, ImageOps

from . import gpu_job, ml_job_update, ml_root
from .hardware import detect_hardware
from .installer import PAINTER_FILES, installer, painter_dir, painter_ready, sd_cli_path

CharacterStyle = Literal["auto", "figurine", "flexi", "bust", "keychain", "magnet", "planter", "scene"]
_VALID_STYLES = frozenset({"figurine", "flexi", "bust", "keychain", "magnet", "planter", "scene"})

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_STYLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("flexi", re.compile(r"\b(flexi|articulated|wiggle|wiggly|print[- ]in[- ]place|segmented|fidget)\b", re.I)),
    ("keychain", re.compile(r"\b(keychain|bag charm|pendant|ornament)\b", re.I)),
    ("magnet", re.compile(r"\b(fridge magnet|magnet)\b", re.I)),
    ("planter", re.compile(r"\b(planter|succulent pot|flower pot|pencil holder|pen cup|candy dish)\b", re.I)),
    ("bust", re.compile(r"\b(portrait bust|head sculpt|bust)\b", re.I)),
    ("scene", re.compile(r"\b(diorama|miniature scene|couple|family portrait|group scene)\b", re.I)),
)


def resolve_character_style(prompt: str, style: str = "auto") -> str:
    """Resolve an explicit archetype or auto-detect it from the user's prompt."""
    clean = (style or "auto").strip().lower()
    if clean in _VALID_STYLES:
        return clean
    text = prompt or ""
    for name, pat in _STYLE_PATTERNS:
        if pat.search(text):
            return name
    return "figurine"


def style_canvas_size(style: str, turnaround: bool = False) -> tuple[int, int]:
    """Choose the ideal canvas aspect ratio for FLUX so elongated flexi toys don't inflate into spheres."""
    if turnaround:
        return (960, 384)
    if style == "flexi":
        return (768, 448)
    if style == "scene":
        return (768, 512)
    return (640, 640)


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


_STYLE_PREFIXES: dict[str, str] = {
    "flexi": (
        "3D printable desk toy studio product photo, sleek elongated horizontal body resting completely flat on its "
        "belly on a smooth horizontal ground plane (length about 2.5 to 3 times its height, NOT chubby or spherical), "
        "smooth solid unsegmented sculpted body with front fins or legs positioned forward near the head and a long "
        "gently tapering tail extending straight back along the floor, clean side three-quarter angle with generous "
        "margin around nose, fins, and tail so nothing is cropped, plain solid light-gray background, smooth matte "
        "vinyl sculpt finish, no hinge cuts or seams drawn on the surface, no text, no watermarks. "
    ),
    "bust": (
        "3D printable sculpted character bust from chest up resting on a clean flat bottom edge, centered "
        "three-quarter portrait view with generous margin around all edges, crisp facial features and hair volumes "
        "suitable for 3D printing, plain solid light-gray background, matte clay sculpt finish, no text, no watermarks. "
    ),
    "keychain": (
        "3D printable chunky pocket keychain charm sculpt, compact sturdy silhouette with thick features and no "
        "fragile thin overhangs, centered front three-quarter view with generous margin, plain solid light-gray "
        "background, smooth matte vinyl toy finish, no metal chains or rings drawn, no text, no watermarks. "
    ),
    "magnet": (
        "3D printable chunky fridge magnet character sculpt with a clean flat back surface and bold sculpted front "
        "details, centered front three-quarter view with generous margin, plain solid light-gray background, smooth "
        "matte vinyl finish, no text, no watermarks. "
    ),
    "planter": (
        "3D printable cute character planter pot with a sturdy wide flat bottom and rounded hollow-ready body, "
        "centered three-quarter studio product photo with generous margin, plain solid light-gray background, smooth "
        "ceramic matte sculpt finish, empty interior with no soil or plants, no text, no watermarks. "
    ),
    "scene": (
        "3D printable miniature character scene resting together on a single unified ground base, complete uncropped "
        "three-quarter studio product photo with generous margin around all edges, sturdy connected forms suitable "
        "for 3D printing, plain solid light-gray background, smooth matte sculpted finish, no text, no watermarks. "
    ),
    "figurine": (
        "3D printable figurine studio product photo, complete uncropped centered hero three-quarter view standing "
        "stably with generous margin around all edges so the entire silhouette is fully visible, clean studio lighting, "
        "plain solid light-gray background, smooth sculpted vinyl toy finish suitable for 3D printing, no text, no watermarks. "
    ),
}


def build_turnaround_prompt(
    prompt: str,
    turnaround: bool = False,
    has_refs: bool = False,
    style: str = "auto",
) -> str:
    base = prompt.strip()
    resolved = resolve_character_style(base, style)
    if turnaround:
        shape_note = (
            "sleek elongated horizontal flat-bellied solid body (length 2.5 to 3x height, no hinge seams drawn), "
            if resolved == "flexi"
            else ""
        )
        prefix = (
            f"3D printable character turnaround sheet showing three complete uncropped views {shape_note}"
            "side by side (FRONT view on left, SIDE profile in middle, BACK view on right) with generous white margin "
            "around all edges so tails and fins are never cut off, clean studio lighting, plain solid light-gray background, "
            "chunky sculpted details suitable for 3D printing, no text labels, no watermarks. "
        )
    else:
        prefix = _STYLE_PREFIXES.get(resolved, _STYLE_PREFIXES["figurine"])
    ref_hint = "Preserve the likeness, faces, hairstyles, and distinct features of the people/subjects in the reference photo. " if has_refs else ""
    return f"{prefix}{ref_hint}Subject: {base}"


def generate_character_design_image(
    prompt: str,
    ref_paths: list[Path] | None = None,
    turnaround: bool = False,
    style: str = "auto",
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
    resolved_style = resolve_character_style(prompt, style)
    width, height = style_canvas_size(resolved_style, turnaround=turnaround)
    full_prompt = build_turnaround_prompt(prompt, turnaround=turnaround, has_refs=bool(ref_paths), style=resolved_style)

    with gpu_job("painting", "Creating design preview", "Starting 2D generator…"):
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
                                detail = f"Step {cur} of {tot}"
                                ml_job_update("Creating design preview", pct, detail)
                                if on_progress:
                                    on_progress(detail)
                        elif "decoding" in low:
                            ml_job_update("Finishing image", 92, "Almost done…")
                            if on_progress:
                                on_progress("Almost done…")
                        elif "loading tensors" in low:
                            ml_job_update("Creating design preview", 10, "Warming up GPU…")
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
