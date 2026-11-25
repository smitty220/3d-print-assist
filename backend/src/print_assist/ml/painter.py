"""Local 2D Painter for Character Mode Design Files (powered by `stable-diffusion.cpp` + FLUX.1-Kontext GGUF).

Takes a user prompt + optional uploaded reference photos (`Photos` tab) and produces a clean,
neutral-background Front / Side / Back turnaround PNG stored directly inside the Design File
(`concepts/K#.png`), never cluttering the user's `Photos` tab.
"""

from __future__ import annotations

import io
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Literal

from PIL import Image, ImageFilter, ImageOps

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
    if style == "flexi":
        return (768, 448)
    if turnaround:
        return (896, 672)
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
        "3D printable flat-bottomed desk toy sculpt resting 100% flush along its entire underside on a flat "
        "horizontal build plate from chin to tail base, sleek elongated horizontal body (length about 2.5 to 3 times "
        "its height, NOT chubby or spherical), flat coplanar lower jaw, flat belly, horizontal side fins resting flush "
        "on the floor (NOT angled downward below the belly), and rear tail tapering along the floor (NOT curving up in "
        "mid-air), self-supporting closed or low-relief grin with no downward-hanging overhangs, smooth solid "
        "unsegmented sculpted body with front fins near the head, clean side three-quarter angle with generous margin "
        "around nose, fins, and tail so nothing is cropped, plain solid light-gray background, smooth matte vinyl "
        "sculpt finish, no hinge cuts or seams drawn on the surface, no text, no watermarks. "
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
    if turnaround and resolved != "flexi":
        prefix = (
            "2x2 four-view 3D printable character turnaround sheet divided into four clean equal quadrants on a "
            "plain solid light-gray background with generous margin around every view so tails, ears, and limbs are "
            "never cropped: TOP-LEFT quadrant shows 3/4 hero view, TOP-RIGHT quadrant shows front orthographic view, "
            "BOTTOM-LEFT quadrant shows side profile view resting flat on the horizontal ground plane, BOTTOM-RIGHT "
            "quadrant shows back orthographic view. Consistent identical character across all four views, chunky "
            "self-supporting sculpted details suitable for 3D printing, no text labels, no watermarks. "
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
            views = ["3/4 hero", "front", "side (Z=0)", "back"] if (turnaround and resolved_style != "flexi") else ["front"]
            return data, views


def _smoothstep(e0: float | "np.ndarray", e1: float | "np.ndarray", x: "np.ndarray") -> "np.ndarray":
    import numpy as np

    t = np.clip((x - e0) / np.maximum(1e-5, e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _longest_run(mask_1d: "np.ndarray") -> tuple[int, int]:
    import numpy as np

    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return 0, 0
    splits = np.where(np.diff(idx) > 6)[0] + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    return int(best[0]), int(best[-1])


def _extract_foreground_profile(arr: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray", int, int]:
    """Segment foreground character from light studio backdrop, excluding neutral floor shadow."""
    import numpy as np

    H, W, _ = arr.shape
    Y_grid, _ = np.mgrid[0:H, 0:W].astype(np.float32)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg, axis=-1)
    sat = arr.max(axis=-1) - arr.min(axis=-1)
    mean_rgb = arr.mean(axis=-1)
    rb_diff = arr[:, :, 0] - arr[:, :, 2]

    is_shadow = (Y_grid > 0.58 * H) & (rb_diff > -8.0) & (
        ((sat < 12.0) & (mean_rgb < bg.mean() - 4.0)) | (mean_rgb < 165.0)
    )
    fg_raw = ((dist > 15.0) | (np.abs(rb_diff) > 7.5)) & ~is_shadow
    fg_eroded = Image.fromarray((fg_raw * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
    fg_alpha = np.asarray(fg_eroded.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0
    fg = fg_alpha > 0.35

    xs = np.where(fg.any(axis=0))[0]
    if len(xs) < 16:
        fg = np.zeros((H, W), dtype=bool)
        fg[int(0.22 * H):int(0.78 * H), int(0.12 * W):int(0.88 * W)] = True
        fg_alpha = fg.astype(np.float32)
        xs = np.where(fg.any(axis=0))[0]
    return bg, fg, fg_alpha, int(xs[0]), int(xs[-1])


def _draw_panel_badge(dr: "ImageDraw.ImageDraw", ox: int, oy: int, text: str) -> None:
    dr.rectangle([ox + 10, oy + 10, ox + 18 + len(text) * 6, oy + 30], fill=(20, 24, 31))
    dr.text((ox + 15, oy + 15), text, fill=(226, 232, 240))


def prepare_quad_turnaround_sheet(raw_png_bytes: bytes, views: list[str]) -> tuple[bytes, bytes, list[str]]:
    """Ensure every non-flexi Character Mode design shows a 4-view sheet (`K#.png`) while keeping
    a clean single-character hero image (`K#_clean.png`) for 3D sculpting and instant flexi conversion.
    """
    import numpy as np
    from PIL import ImageDraw

    with Image.open(io.BytesIO(raw_png_bytes)) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size

    # Legacy 3-view strip compatibility (e.g. unit-test mocks returning ["front", "side", "back"])
    if len(views) == 3 and w >= int(h * 1.6):
        hero_crop = rgb.crop((0, 0, w // 3, h))
        c_buf = io.BytesIO()
        hero_crop.save(c_buf, format="PNG")
        return raw_png_bytes, c_buf.getvalue(), views

    # 2x2 4-view sheet painted directly by FLUX: extract clean top-left hero for sculptor & label the 4 quadrants
    if len(views) >= 4 and w >= 400 and h >= 300:
        hw_q, hh_q = w // 2, h // 2
        hero_quad = rgb.crop((0, 0, hw_q, hh_q))
        arr_q = np.asarray(hero_quad, dtype=np.float32)
        border_q = np.concatenate([arr_q[0, :], arr_q[-1, :], arr_q[:, 0], arr_q[:, -1]], axis=0)
        bg_q = tuple(int(round(v)) for v in np.median(border_q, axis=0))
        side_len = max(hw_q, hh_q) + 48
        clean_canvas = Image.new("RGB", (side_len, side_len), bg_q)
        clean_canvas.paste(hero_quad, ((side_len - hw_q) // 2, (side_len - hh_q) // 2))
        c_buf = io.BytesIO()
        clean_canvas.save(c_buf, format="PNG")

        sheet = rgb.copy()
        dr = ImageDraw.Draw(sheet)
        dr.line([(hw_q, 0), (hw_q, h)], fill=(195, 202, 212), width=2)
        dr.line([(0, hh_q), (w, hh_q)], fill=(195, 202, 212), width=2)
        labels = [
            ("1 | 3/4 HERO", (0, 0)),
            ("2 | FRONT", (hw_q, 0)),
            ("3 | SIDE (Z=0 BED)", (0, hh_q)),
            ("4 | BACK", (hw_q, hh_q)),
        ]
        for lbl, (ox, oy) in labels:
            _draw_panel_badge(dr, ox, oy, lbl)
        p_buf = io.BytesIO()
        sheet.save(p_buf, format="PNG")
        return p_buf.getvalue(), c_buf.getvalue(), ["3/4 hero", "front", "side (Z=0)", "back"]

    # Single-view image fallback -> compose a 4-panel Character + Z=0 Print Reference Sheet automatically
    pw, ph = 560, 420
    sheet = Image.new("RGB", (pw * 2, ph * 2), (24, 28, 35))
    arr = np.asarray(rgb, dtype=np.float32)
    bg, fg, _, x0, x1 = _extract_foreground_profile(arr)
    bg_rgb = tuple(int(round(v)) for v in bg)

    # Panel 1 (Top-Left): 3/4 Studio Hero
    p1 = Image.new("RGB", (pw, ph), bg_rgb)
    fit = ImageOps.contain(rgb, (pw - 32, ph - 44), Image.Resampling.LANCZOS)
    p1.paste(fit, ((pw - fit.width) // 2, (ph - fit.height) // 2 + 12))
    sheet.paste(p1, (0, 0))

    # Measure silhouette heights for orthographic print-reference panels 2, 3, 4
    span = max(1, x1 - x0)
    n_s = 64
    xs_s = np.linspace(x0, x1, n_s)
    yts, ybs = [], []
    for xv in xs_s:
        yt, yb = _longest_run(fg[:, int(round(xv))])
        yts.append(float(yt))
        ybs.append(float(yb))
    yts_a = np.array(yts)
    ybs_a = np.array(ybs)
    hs_a = np.maximum(2.0, ybs_a - yts_a)
    y_bed_2d = float(np.percentile(ybs_a, 92))
    max_h = max(10.0, y_bed_2d - float(yts_a.min()))

    # Panel 2..4: Smooth 3D-shaded studio elevation, top envelope, and Z=0 bed contact panels
    k_s = np.array([1, 2, 4, 6, 4, 2, 1], dtype=np.float32)
    k_s /= k_s.sum()
    yts_s = np.convolve(np.pad(yts_a, 3, mode="edge"), k_s, mode="valid")
    ybs_s = np.convolve(np.pad(ybs_a, 3, mode="edge"), k_s, mode="valid")
    hs_s = np.maximum(2.0, ybs_s - yts_s)

    # Sample character dorsal/ventral colors from foreground pixels
    fg_pixels = arr[fg]
    if len(fg_pixels) > 40:
        dorsal_c = np.percentile(fg_pixels, 45, axis=0).astype(np.float32) / 255.0
        ventral_c = np.percentile(fg_pixels, 82, axis=0).astype(np.float32) / 255.0
    else:
        dorsal_c = np.array([0.42, 0.54, 0.68], dtype=np.float32)
        ventral_c = np.array([0.82, 0.84, 0.86], dtype=np.float32)

    bed_y2 = ph - 52
    sx_scale = (pw - 110) / max(1.0, float(span))
    sy_scale = min(sx_scale, (ph - 110) / max_h)
    ox2_local = int((pw - span * sx_scale) / 2)

    # Render Panel 2 (Side Z=0), Panel 3 (Top), and Panel 4 (Bottom Z=0) with smooth 3D volumetric shading
    yy_p, xx_p = np.mgrid[0:ph, 0:pw].astype(np.float32)
    bg_grad = (
        (1.0 - (yy_p / ph)[:, :, None]) * np.array([34.0, 39.0, 48.0], dtype=np.float32)
        + (yy_p / ph)[:, :, None] * np.array([22.0, 26.0, 33.0], dtype=np.float32)
    )

    # Panel 2: 3D-shaded Side Elevation on Z=0
    p2_arr = bg_grad.copy()
    x_src2 = x0 + (xx_p - ox2_local) / max(1e-5, sx_scale)
    in_x2 = (x_src2 >= x0) & (x_src2 <= x1)
    yt_2 = bed_y2 - (y_bed_2d - np.interp(x_src2, xs_s, yts_s)) * sy_scale
    yb_2 = np.minimum(float(bed_y2), bed_y2 - (y_bed_2d - np.interp(x_src2, xs_s, ybs_s)) * sy_scale)
    span_2 = np.maximum(3.0, yb_2 - yt_2)
    cov2 = _smoothstep(yt_2 - 1.2, yt_2 + 1.2, yy_p) * (1.0 - _smoothstep(yb_2 - 1.2, yb_2 + 1.2, yy_p))
    cov2 *= _smoothstep(x0 - 1.5, x0 + 1.5, x_src2) * (1.0 - _smoothstep(x1 - 1.5, x1 + 1.5, x_src2))
    v_norm2 = np.clip(((yy_p - yt_2) / span_2) * 2.0 - 1.0, -1.0, 1.0)  # -1 top, +1 bottom
    nz2 = np.sqrt(np.maximum(0.04, 1.0 - v_norm2 * v_norm2))
    sh2 = 0.30 + 0.54 * np.clip(-0.45 * v_norm2 + 0.72 * nz2, 0.0, 1.0) + 0.16 * nz2
    c_mix2 = _smoothstep(0.25, 0.65, v_norm2)[:, :, None]
    col2 = ((1.0 - c_mix2) * dorsal_c + c_mix2 * ventral_c) * sh2[:, :, None] * 255.0
    p2_arr = (1.0 - cov2[:, :, None]) * p2_arr + cov2[:, :, None] * col2
    sheet.paste(Image.fromarray(np.clip(p2_arr, 0, 255).astype(np.uint8)), (pw, 0))

    # Panel 3: 3D-shaded Top Envelope
    p3_arr = bg_grad.copy()
    cy3_l = ph * 0.52
    hw_3 = np.interp(x_src2, xs_s, hs_s * 0.36 * sy_scale)
    dy3 = yy_p - cy3_l
    cov3 = (1.0 - _smoothstep(np.maximum(1.5, hw_3 - 1.2), hw_3 + 1.2, np.abs(dy3))) * in_x2.astype(np.float32)
    u_lat3 = np.clip(dy3 / np.maximum(2.0, hw_3), -1.0, 1.0)
    nz3 = np.sqrt(np.maximum(0.04, 1.0 - u_lat3 * u_lat3))
    sh3 = 0.28 + 0.56 * np.clip(-0.35 * u_lat3 + 0.78 * nz3, 0.0, 1.0) + 0.16 * nz3
    p3_arr = (1.0 - cov3[:, :, None]) * p3_arr + cov3[:, :, None] * (dorsal_c * sh3[:, :, None] * 255.0)
    sheet.paste(Image.fromarray(np.clip(p3_arr, 0, 255).astype(np.uint8)), (0, ph))

    # Panel 4: 3D-shaded Bottom View + Z=0 Emerald Contact Pad
    p4_arr = bg_grad.copy()
    p4_arr = (1.0 - cov3[:, :, None]) * p4_arr + cov3[:, :, None] * (ventral_c * sh3[:, :, None] * 235.0)
    bed_contact = np.maximum(0.0, 1.0 - np.maximum(0.0, y_bed_2d - ybs_s) / (0.14 * max_h)) * (hs_s * 0.36 * sy_scale) * 0.62
    if float(bed_contact.max()) < 4.0:
        bed_contact = (hs_s * 0.36 * sy_scale) * 0.42
    bw_4 = np.interp(x_src2, xs_s, bed_contact)
    cov_pad4 = (1.0 - _smoothstep(np.maximum(0.8, bw_4 - 1.2), bw_4 + 1.2, np.abs(dy3))) * cov3 * (bw_4 > 1.5)
    em_col = np.array([16.0, 198.0, 136.0], dtype=np.float32) * (0.82 + 0.18 * nz3)[:, :, None]
    p4_arr = (1.0 - cov_pad4[:, :, None]) * p4_arr + cov_pad4[:, :, None] * em_col
    sheet.paste(Image.fromarray(np.clip(p4_arr, 0, 255).astype(np.uint8)), (pw, ph))

    dr = ImageDraw.Draw(sheet)
    dr.line([(pw + 24, bed_y2), (pw * 2 - 24, bed_y2)], fill=(52, 211, 153), width=3)
    dr.text((pw + 28, bed_y2 + 8), "Z = 0 BUILD PLATE", fill=(110, 231, 183))
    dr.text((pw + 24, ph * 2 - 28), "EMERALD = Z=0 FIRST-LAYER BED CONTACT", fill=(110, 231, 183))

    dr.line([(pw, 0), (pw, ph * 2)], fill=(64, 72, 86), width=2)
    dr.line([(0, ph), (pw * 2, ph)], fill=(64, 72, 86), width=2)
    for lbl, (ox, oy) in [
        ("1 | 3/4 STUDIO HERO", (0, 0)),
        ("2 | SIDE ELEVATION (Z=0 BED)", (pw, 0)),
        ("3 | TOP ENVELOPE", (0, ph)),
        ("4 | BOTTOM (Z=0 BED CONTACT)", (pw, ph)),
    ]:
        _draw_panel_badge(dr, ox, oy, lbl)

    out_buf = io.BytesIO()
    sheet.save(out_buf, format="PNG")
    return out_buf.getvalue(), raw_png_bytes, ["3/4 hero", "side (Z=0)", "top", "bottom (Z=0)"]


def render_flexi_concept_preview(
    png_bytes: bytes,
    flexi_segments: int = 6,
    flexi_cuts_pct: list[float] | None = None,
) -> tuple[bytes, list[float]]:
    """Build a 1280x900 4-View Bed-Seated Flexi Print-Engineering Design Sheet (`K#.png`):
    1. Top-Left:  3/4 Articulated Flexi Preview — warped/seated flat onto the Z=0 build plate (zero floating tail!)
    2. Top-Right: Side Orthographic on Z=0 Build Plate — showing all N segments flush on Z=0 + X-ray internal pins & loops
    3. Bot-Left:  Top Orthographic — showing intact lateral/dorsal fins + curved fin-safe S-wiggle cuts
    4. Bot-Right: Bottom Orthographic (Z=0 First-Layer Bed Contact) — verifying all N/N segments touch Z=0 support-free.
    """
    import numpy as np
    from PIL import ImageDraw

    with Image.open(io.BytesIO(png_bytes)) as im:
        rgb_im = im.convert("RGB")
    arr_orig = np.asarray(rgb_im, dtype=np.float32)
    H, W, _ = arr_orig.shape
    bg, fg_orig, _, x0, x1 = _extract_foreground_profile(arr_orig)
    span = max(1, x1 - x0)
    rb_diff_orig = arr_orig[:, :, 0] - arr_orig[:, :, 2]

    # Detect foreground pectoral fin hanging across/below a cream underbelly on the source image
    blue_mask_orig = (-rb_diff_orig > 10.0) & fg_orig
    cream_mask_orig = (arr_orig[:, :, 0] > 175.0) & (arr_orig[:, :, 1] > 170.0) & (rb_diff_orig > 7.0) & fg_orig
    pec_orig = np.zeros((H, W), dtype=bool)
    for x in range(x0, x1 + 1):
        ys_full = np.where(fg_orig[:, x])[0]
        if len(ys_full) < 20:
            continue
        yt, yb = int(ys_full[0]), int(ys_full[-1])
        mid_y = yt + int(0.32 * (yb - yt))
        cream_ys = np.where(cream_mask_orig[mid_y:yb + 1, x])[0]
        if len(cream_ys) >= 4:
            first_cream_y = mid_y + int(cream_ys[0])
            blue_below = np.where(blue_mask_orig[first_cream_y + 2:yb + 1, x])[0]
            if len(blue_below) >= 2:
                pec_orig[first_cream_y + 2 + int(blue_below[0]):, x] = True

    torso_orig = fg_orig & ~pec_orig
    yt_orig = np.full(W, np.nan, dtype=np.float32)
    yb_orig = np.full(W, np.nan, dtype=np.float32)
    for x in range(x0, x1 + 1):
        ys_t = np.where(torso_orig[:, x])[0]
        if len(ys_t) < 6:
            ys_t = np.where(fg_orig[:, x])[0]
        if len(ys_t) >= 2:
            # Use full outer envelope [ys_t[0], ys_t[-1]] so open mouths never split the head under the eye!
            yt_orig[x], yb_orig[x] = float(ys_t[0]), float(ys_t[-1])

    valid_x = ~np.isnan(yt_orig)
    x_all = np.arange(W)
    yt_orig = np.interp(x_all, x_all[valid_x], yt_orig[valid_x]).astype(np.float32)
    yb_orig = np.interp(x_all, x_all[valid_x], yb_orig[valid_x]).astype(np.float32)
    # Smooth yb_orig with a wide C1 kernel so 2D bed seating never produces vertical pixel streaks
    k_smooth = np.array([1, 3, 6, 9, 11, 9, 6, 3, 1], dtype=np.float32)
    k_smooth /= k_smooth.sum()
    yb_smooth = yb_orig.copy()
    yt_smooth = yt_orig.copy()
    for _ in range(4):
        yb_smooth = np.convolve(np.pad(yb_smooth, 4, mode="edge"), k_smooth, mode="valid")
        yt_smooth = np.convolve(np.pad(yt_smooth, 4, mode="edge"), k_smooth, mode="valid")
    h_orig = np.maximum(0.0, yb_smooth - yt_smooth)

    x_15_35 = [int(x0 + p * span) for p in np.linspace(0.15, 0.35, 9)]
    x_68_85 = [int(x0 + p * span) for p in np.linspace(0.68, 0.85, 9)]
    facing_left = float(np.mean(h_orig[x_15_35])) >= float(np.mean(h_orig[x_68_85]))

    def pct_to_x(pct: float) -> float:
        p = (pct / 100.0) if facing_left else (1.0 - pct / 100.0)
        return float(x0 + p * span)

    def x_to_pct(x_val: float) -> float:
        p = (x_val - x0) / max(1.0, float(span))
        return float(p * 100.0 if facing_left else (1.0 - p) * 100.0)

    # Detect eye/mouth end on source image BEFORE warping so head/eye/mouth are never stretched!
    head_feature_end_pct = 26.0
    for pct_probe in np.linspace(8.0, 37.5, 45):
        xp = int(np.clip(round(pct_to_x(float(pct_probe))), 0, W - 1))
        yt_p, yb_p = int(yt_smooth[xp]), int(yb_smooth[xp])
        hp = yb_p - yt_p
        if hp < 20:
            continue
        eye_rgb = arr_orig[yt_p + int(0.18 * hp):yt_p + int(0.48 * hp), xp]
        mouth_rgb = arr_orig[yt_p + int(0.35 * hp):yt_p + int(0.78 * hp), xp]
        has_eye = len(eye_rgb) > 0 and bool(np.any(eye_rgb.mean(axis=-1) < 42.0))
        has_mouth = len(mouth_rgb) > 0 and bool(np.any((mouth_rgb[:, 0] - mouth_rgb[:, 1]) > 24.0))
        if has_eye or has_mouth:
            head_feature_end_pct = max(head_feature_end_pct, float(pct_probe))

    # ------------------------------------------------------------------
    # STEP 1: Gently un-arch ONLY the upward-swept rear spine (behind the
    # pectoral fin, ~48%..84%) via pure rigid column translation — NEVER
    # stretching lower-belly pixels and NEVER creating sawtooth mask jaggies!
    # ------------------------------------------------------------------
    chest_xs = [int(np.clip(round(pct_to_x(p)), 0, W - 1)) for p in np.linspace(26.0, 42.0, 16)]
    chest_belly_y = float(np.median([yb_smooth[x] for x in chest_xs if h_orig[x] > 10.0] or [0.76 * H]))

    # Find where foreground pectoral fin ends so we never shift across the pectoral fin
    pec_cols = np.where(pec_orig.any(axis=0))[0]
    pec_end_pct = 46.0
    if len(pec_cols) >= 4:
        pec_pcts = [x_to_pct(float(c)) for c in pec_cols]
        pec_end_pct = min(56.0, max(44.0, float(np.percentile(pec_pcts, 92)) + 1.5))

    arr = arr_orig.copy()
    y_coords = np.arange(H, dtype=np.float32)
    rear_probe_xs = [int(np.clip(round(pct_to_x(p)), 0, W - 1)) for p in np.linspace(60.0, 76.0, 12)]
    rear_belly_y = float(np.median([yb_smooth[x] for x in rear_probe_xs if h_orig[x] > 6.0] or [chest_belly_y]))
    scalar_drop = min(0.11 * float(h_orig.max()), max(0.0, chest_belly_y - rear_belly_y) * 0.80)
    if scalar_drop > 0.5:
        for x in range(x0, x1 + 1):
            pct = x_to_pct(float(x))
            rf = float(_smoothstep(pec_end_pct, pec_end_pct + 15.0, np.array(pct)))
            rb = 1.0 - float(_smoothstep(74.0, 88.0, np.array(pct)))
            dy_down = scalar_drop * rf * rb
            if dy_down <= 0.20:
                continue
            src_y = y_coords - dy_down
            y0_i = np.clip(np.floor(src_y).astype(int), 0, H - 1)
            y1_i = np.clip(y0_i + 1, 0, H - 1)
            wy = np.clip(src_y - y0_i, 0.0, 1.0)[:, None]
            col_shifted = (1.0 - wy) * arr_orig[y0_i, x] + wy * arr_orig[y1_i, x]
            top_fill = src_y < 0.0
            if np.any(top_fill):
                col_shifted[top_fill] = bg
            arr[:, x] = col_shifted

    # Re-extract foreground & pectoral mask on the cleanly un-arched image
    Y_grid, X_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    _, fg, fg_alpha, x0, x1 = _extract_foreground_profile(arr)
    span = max(1, x1 - x0)
    rb_diff = arr[:, :, 0] - arr[:, :, 2]
    blue_mask = (-rb_diff > 10.0) & fg
    cream_mask = (arr[:, :, 0] > 175.0) & (arr[:, :, 1] > 170.0) & (rb_diff > 7.0) & fg
    pec_fin_mask = np.zeros((H, W), dtype=bool)
    for x in range(x0, x1 + 1):
        yt, yb = _longest_run(fg[:, x])
        if yb - yt < 20:
            continue
        mid_y = yt + int(0.32 * (yb - yt))
        cream_ys = np.where(cream_mask[mid_y:yb + 1, x])[0]
        if len(cream_ys) >= 4:
            first_cream_y = mid_y + int(cream_ys[0])
            blue_below = np.where(blue_mask[first_cream_y + 2:yb + 1, x])[0]
            if len(blue_below) >= 2:
                pec_fin_mask[first_cream_y + 2 + int(blue_below[0]):, x] = True

    pec_pil = (
        Image.fromarray((pec_fin_mask * 255).astype(np.uint8))
        .filter(ImageFilter.MaxFilter(7))
        .filter(ImageFilter.GaussianBlur(1.4))
    )
    pec_alpha = np.asarray(pec_pil, dtype=np.float32) / 255.0

    torso_fg = fg & (pec_alpha < 0.2)
    yt_col = np.zeros(W, dtype=np.float32)
    yb_col = np.zeros(W, dtype=np.float32)
    for x in range(W):
        yt, yb = _longest_run(torso_fg[:, x])
        if yb - yt < 5:
            yt, yb = _longest_run(fg[:, x])
        yt_col[x], yb_col[x] = float(yt), float(yb)
    k_env = np.array([1, 3, 6, 8, 6, 3, 1], dtype=np.float32)
    k_env /= k_env.sum()
    yt_col = np.convolve(np.pad(yt_col, 3, mode="edge"), k_env, mode="valid")
    yb_col = np.convolve(np.pad(yb_col, 3, mode="edge"), k_env, mode="valid")
    h_col = np.maximum(0.0, yb_col - yt_col)
    max_h = float(h_col.max()) if h_col.max() > 0 else 100.0

    # Detect dorsal fin zone & tail waist
    n_bins = 60
    pcts = np.linspace(5.0, 95.0, n_bins)
    b_yt = np.array([yt_col[int(np.clip(round(pct_to_x(float(p))), 0, W - 1))] for p in pcts])

    i_35 = int(np.argmin(np.abs(pcts - 35.0)))
    i_63 = int(np.argmin(np.abs(pcts - 63.0)))
    dorsal_pcts: list[float] = []
    for i in range(i_35, i_63 + 1):
        t = (pcts[i] - pcts[i_35]) / max(1e-5, pcts[i_63] - pcts[i_35])
        chord_y = b_yt[i_35] * (1.0 - t) + b_yt[i_63] * t
        if (chord_y - b_yt[i]) > 0.048 * max_h:
            dorsal_pcts.append(float(pcts[i]))

    fin_start = max(39.0, min(dorsal_pcts) - 1.8) if dorsal_pcts else 42.5
    fin_end = min(61.5, max(dorsal_pcts) + 1.8) if dorsal_pcts else 57.5

    waist_idx = [i for i, p in enumerate(pcts) if 72.0 <= p <= 89.0]
    waist_pct = float(pcts[min(waist_idx, key=lambda i: h_orig[int(np.clip(round(pct_to_x(float(pcts[i]))), 0, W - 1))])]) if waist_idx else 82.0
    max_cut_pct = min(78.5, waist_pct - 3.0)

    n_cuts = max(3, int(flexi_segments) - 1 if int(flexi_segments) >= 4 else 5)
    if flexi_cuts_pct and len(flexi_cuts_pct) >= 1:
        cuts_pct = sorted(round(float(c), 1) for c in flexi_cuts_pct if 18.0 <= float(c) <= 86.0)
    else:
        front_cut = max(head_feature_end_pct + 2.2, min(37.5, fin_start - 1.8))
        n_rear = max(2, n_cuts - 1)
        rear_start = max(fin_end + 1.4, 59.0)
        rear_end = max(rear_start + 6.0, max_cut_pct)
        rear_cuts = np.linspace(rear_start, rear_end, n_rear).tolist()
        cuts_pct = [round(front_cut, 1)] + [round(float(c), 1) for c in rear_cuts]

    # Render sleek, narrow (~0.9% span), 3D bevel-shaded gill-curved flexi seams on Panel 1
    out = arr.copy()
    slot_w = max(4.2, 0.0095 * span)
    half_w = slot_w * 0.5
    X_idx = np.clip(X_grid.astype(int), 0, W - 1)
    yt_grid = yt_col[X_idx]
    yb_grid = yb_col[X_idx]
    arch_sign = -1.0 if facing_left else 1.0

    for cp in cuts_pct:
        xc = pct_to_x(cp)
        x_int = int(np.clip(round(xc), 0, W - 1))
        yt_ref = float(yt_col[x_int])
        yb_ref = float(yb_col[x_int])
        h_ref = max(12.0, yb_ref - yt_ref)

        u_ref = np.clip((Y_grid - yt_ref) / h_ref, 0.0, 1.0)
        max_arch_pct = max(1.0, cp - (head_feature_end_pct + 2.0))
        arch = min(0.072 * h_ref, (max_arch_pct / 100.0) * span)
        x_center = xc + arch_sign * arch * (4.0 * u_ref * (1.0 - u_ref))
        dx = (X_grid - x_center) * (-arch_sign)

        in_y = (Y_grid >= yt_grid - 1.0) & (Y_grid <= yb_grid + 1.0) & (np.abs(dx) <= half_w + 3.5)
        mask_vis = fg_alpha * (1.0 - pec_alpha) * in_y.astype(np.float32)

        sample_x = np.clip(np.round(x_center - arch_sign * (half_w + 3.5)).astype(int), 0, W - 1)
        base_col = arr[Y_grid.astype(int), sample_x]
        bg_dist = np.linalg.norm(base_col - bg, axis=-1, keepdims=True)
        sample_x_opp = np.clip(np.round(x_center + arch_sign * (half_w + 3.5)).astype(int), 0, W - 1)
        base_col = np.where(bg_dist < 22.0, arr[Y_grid.astype(int), sample_x_opp], base_col)

        in_slot = _smoothstep(-half_w - 0.8, -half_w + 0.6, dx) * (1.0 - _smoothstep(half_w - 0.6, half_w + 0.8, dx))
        s = np.clip(dx / half_w, -1.0, 1.0)

        wall_blend = _smoothstep(-0.10, 0.25, s)
        cavity_shade = 0.20 + 0.08 * (s + 1.0)
        wall_shade = (0.48 + 0.22 * ((s + 0.10) / 1.10)) * (0.95 - 0.15 * u_ref)
        shade = (1.0 - wall_blend) * cavity_shade + wall_blend * wall_shade

        # Subtle internal horizontal chain-link bridge glimpse at mid-height
        link_y = _smoothstep(0.42, 0.46, u_ref) * (1.0 - _smoothstep(0.54, 0.58, u_ref))
        link_shade = 0.52 + 0.18 * np.sin(np.clip((u_ref - 0.42) / 0.16, 0.0, 1.0) * np.pi)
        shade = (1.0 - link_y * 0.8) * shade + (link_y * 0.8) * link_shade

        slot_rgb = base_col * shade[:, :, None]

        # Tiny sculpted bevel notch along the top dorsal contour so segments read as distinct 3D links
        top_edge_px = Y_grid - yt_grid
        notch_px = 2.4 * np.maximum(0.0, 1.0 - (s ** 2))
        top_notch_w = (1.0 - _smoothstep(np.maximum(0.0, notch_px - 0.8), notch_px + 0.9, top_edge_px)) * in_slot
        slot_rgb = (1.0 - top_notch_w[:, :, None]) * slot_rgb + top_notch_w[:, :, None] * bg[None, None, :]

        w_total = (in_slot * mask_vis)[:, :, None]
        out = (1.0 - w_total) * out + w_total * slot_rgb

        # Soft 3D bevel highlight on downstream lip + subtle bevel shadow on upstream lip
        rim_hi = _smoothstep(half_w - 0.4, half_w + 0.4, dx) * (1.0 - _smoothstep(half_w + 0.5, half_w + 2.0, dx))
        rim_hi = (rim_hi * mask_vis * (1.0 - top_notch_w))[:, :, None]
        rim_col = np.minimum(255.0, base_col * 1.14 + 12.0)
        out = (1.0 - rim_hi * 0.72) * out + (rim_hi * 0.72) * rim_col

    view1_im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # STEP 2: Compose the 1280x900 Studio 3D-Shaded 4-View Flexi Sheet!
    # ------------------------------------------------------------------
    pw, ph = 640, 450
    sheet = Image.new("RGB", (pw * 2, ph * 2), (22, 26, 33))
    bg_rgb = tuple(int(round(v)) for v in bg)

    # Panel 1 (Top-Left): 3/4 Articulated Hero Preview
    p1 = Image.new("RGB", (pw, ph), bg_rgb)
    fit1 = ImageOps.contain(view1_im, (pw - 24, ph - 44), Image.Resampling.LANCZOS)
    p1.paste(fit1, ((pw - fit1.width) // 2, (ph - fit1.height) // 2 + 12))
    sheet.paste(p1, (0, 0))

    # Sample actual character dorsal & ventral palette so Panels 2, 3, 4 match the character's colors!
    blue_px = arr_orig[blue_mask_orig]
    cream_px = arr_orig[cream_mask_orig]
    fg_px = arr_orig[fg_orig]
    if len(blue_px) > 50:
        dorsal_col = np.median(blue_px, axis=0).astype(np.float32) / 255.0
    elif len(fg_px) > 50:
        dorsal_col = np.percentile(fg_px, 45, axis=0).astype(np.float32) / 255.0
    else:
        dorsal_col = np.array([0.38, 0.50, 0.64], dtype=np.float32)

    if len(cream_px) > 50:
        ventral_col = np.median(cream_px, axis=0).astype(np.float32) / 255.0
    elif len(fg_px) > 50:
        ventral_col = np.percentile(fg_px, 82, axis=0).astype(np.float32) / 255.0
    else:
        ventral_col = np.minimum(1.0, dorsal_col * 1.25 + 0.12)

    # Build smooth C2 normalized spine profiles (p in 0..100%) seated flush on Z=0 for Panels 2, 3, 4
    p_dense = np.linspace(0.0, 100.0, 241)
    top_h_norm = np.zeros_like(p_dense)
    bot_z_norm = np.zeros_like(p_dense)
    y_bed_ref = chest_belly_y
    for idx_p, pv in enumerate(p_dense):
        xp = int(np.clip(round(pct_to_x(float(pv))), 0, W - 1))
        raw_th = max(0.02, (y_bed_ref - yt_col[xp]) / max(1.0, max_h))
        top_h_norm[idx_p] = raw_th
        if pv < 15.0:
            bot_z_norm[idx_p] = 0.24 * ((15.0 - pv) / 15.0) ** 1.65
        elif pv > 77.0:
            # Rise smoothly at caudal peduncle waist (~83%), then dip lower tail fin lobe back to Z=0 at ~92%
            waist_arch = 0.08 * math.exp(-(((pv - 82.5) / 4.0) ** 2))
            tip_taper = 0.22 * max(0.0, (pv - 92.0) / 8.0) ** 1.35
            bot_z_norm[idx_p] = waist_arch + tip_taper
        else:
            bot_z_norm[idx_p] = 0.0

    k_prof = np.array([1, 3, 6, 8, 6, 3, 1], dtype=np.float64)
    k_prof /= k_prof.sum()
    top_h_norm = np.convolve(np.pad(top_h_norm, 3, mode="edge"), k_prof, mode="valid")
    # Smoothly round the snout tip (0..8%) so top_h_norm meets bot_z_norm(0)=0.24 with zero upward nose spike!
    nose_ref = float(np.interp(8.0, p_dense, top_h_norm))
    for idx_p, pv in enumerate(p_dense):
        if pv <= 8.0:
            s_n = pv / 8.0
            top_h_norm[idx_p] = 0.24 + max(0.04, nose_ref - 0.24) * (s_n ** 0.65)

    # Sleek torpedo body width + tapered caudal fin blade (82..100%)
    torpedo_w = 0.29 * np.power(np.sin(np.pi * np.clip(p_dense / 90.0, 0.0, 1.0)), 0.72)
    torpedo_w = np.where(
        p_dense > 82.0,
        np.maximum(0.008, 0.038 * (1.0 - 0.78 * (np.maximum(0.0, p_dense - 82.0) / 18.0) ** 1.2)),
        torpedo_w,
    )
    torpedo_w = np.maximum(0.008, torpedo_w)

    yy_p, xx_p = np.mgrid[0:ph, 0:pw].astype(np.float32)
    bg_studio = (
        (1.0 - (yy_p / ph)[:, :, None]) * np.array([34.0, 39.0, 48.0], dtype=np.float32)
        + (yy_p / ph)[:, :, None] * np.array([22.0, 26.0, 33.0], dtype=np.float32)
    )

    ox_pad = 46.0
    draw_w = float(pw - 92)
    p_px = ((xx_p - ox_pad) / draw_w) * 100.0
    in_p = _smoothstep(0.0, 0.8, p_px) * (1.0 - _smoothstep(99.2, 100.0, p_px))
    gap_half_pct = 0.36  # sleek ~0.7% seam width matching real 0.85 mm flexi clearance

    # ---------- Panel 2 (Top-Right): 3D-Shaded Side Orthographic on Z=0 Build Plate ----------
    p2_arr = bg_studio.copy()
    bed_y2 = float(ph - 58)
    h_scale = min(ph - 126.0, draw_w * 0.43)
    th_grid = np.interp(p_px, p_dense, top_h_norm).astype(np.float32)
    bz_grid = np.interp(p_px, p_dense, bot_z_norm).astype(np.float32)
    yt_2 = bed_y2 - th_grid * h_scale
    yb_2 = bed_y2 - bz_grid * h_scale
    loc_h2 = np.maximum(4.0, yb_2 - yt_2)
    u2 = np.clip((yy_p - yt_2) / loc_h2, 0.0, 1.0)

    cov2 = _smoothstep(yt_2 - 1.1, yt_2 + 1.1, yy_p) * (1.0 - _smoothstep(yb_2 - 1.1, yb_2 + 1.1, yy_p)) * in_p
    # Carve crescent caudal fork notch on trailing edge of tail fin (p > 91%) so Side view has an iconic shark tail!
    fork_center_y = bed_y2 - 0.34 * h_scale
    fork_v = np.abs(yy_p - fork_center_y) / max(8.0, 0.34 * h_scale)
    tail_trail_p = np.where(
        yy_p <= fork_center_y,
        91.5 + 8.5 * np.clip(fork_v, 0.0, 1.0) ** 1.15,
        91.5 + 4.8 * np.clip(fork_v, 0.0, 1.0) ** 1.15,
    )
    cov2 *= 1.0 - _smoothstep(tail_trail_p - 0.8, tail_trail_p + 0.8, p_px)

    # Gill-curved flexi seams in Side view
    seam_mask2 = np.zeros_like(cov2)
    rim_hi2 = np.zeros_like(cov2)
    for cp in cuts_pct:
        c_curve = cp + 1.4 * (4.0 * u2 * (1.0 - u2))
        dp = p_px - c_curve
        s_cut = _smoothstep(-gap_half_pct - 0.16, -gap_half_pct + 0.12, dp) * (
            1.0 - _smoothstep(gap_half_pct - 0.12, gap_half_pct + 0.16, dp)
        )
        seam_mask2 = np.maximum(seam_mask2, s_cut)
        r_hi = _smoothstep(gap_half_pct - 0.1, gap_half_pct + 0.15, dp) * (
            1.0 - _smoothstep(gap_half_pct + 0.2, gap_half_pct + 0.75, dp)
        )
        rim_hi2 = np.maximum(rim_hi2, r_hi)

    v_n2 = u2 * 2.0 - 1.0
    nz2 = np.sqrt(np.maximum(0.05, 1.0 - v_n2 * v_n2))
    nx2 = np.gradient(th_grid, axis=1) * 2.5
    sh2 = 0.28 + 0.56 * np.clip(-0.25 * nx2 - 0.48 * v_n2 + 0.72 * nz2, 0.0, 1.0) + 0.16 * nz2
    # Two-tone countershade (dorsal slate -> cream belly on lower 32% of torso)
    belly_line = 0.66 + 0.08 * _smoothstep(55.0, 85.0, p_px)
    c_blend2 = _smoothstep(belly_line - 0.06, belly_line + 0.06, u2)[:, :, None]
    surf_c2 = (1.0 - c_blend2) * dorsal_col + c_blend2 * ventral_col
    rgb2 = surf_c2 * sh2[:, :, None] * 255.0

    # Sculpted swept-back lateral pectoral fin (36%..52%, sitting cleanly above Z=0!)
    pec_t = np.clip((p_px - 35.0) / 16.5, 0.0, 1.0)
    pec_y_top = bed_y2 - (0.28 - 0.16 * pec_t) * h_scale
    pec_y_bot = bed_y2 - (0.04 + 0.02 * (pec_t ** 2)) * h_scale
    in_pec2 = (
        _smoothstep(35.0, 36.5, p_px)
        * (1.0 - _smoothstep(49.5, 51.5, p_px))
        * _smoothstep(pec_y_top - 1.0, pec_y_top + 1.2, yy_p)
        * (1.0 - _smoothstep(pec_y_bot - 1.2, pec_y_bot + 1.0, yy_p))
    )
    pec_sh = 0.36 + 0.54 * (1.0 - np.clip((yy_p - pec_y_top) / np.maximum(3.0, pec_y_bot - pec_y_top), 0.0, 1.0))
    # Pectoral fin sits in front of seams and cream belly
    seam_mask2 *= (1.0 - in_pec2 * 0.95)
    rgb2 = (1.0 - (in_pec2 * 0.92)[:, :, None]) * rgb2 + (in_pec2 * 0.92)[:, :, None] * (dorsal_col * pec_sh[:, :, None] * 255.0)

    # Darken seam interior (leaving subtle internal link bridge) + brighten bevel lip
    link_band2 = _smoothstep(0.45, 0.50, u2) * (1.0 - _smoothstep(0.62, 0.67, u2))
    seam_dark2 = 0.22 + 0.30 * link_band2
    rgb2 = (1.0 - seam_mask2[:, :, None]) * rgb2 + seam_mask2[:, :, None] * (rgb2 * seam_dark2[:, :, None])
    rgb2 = np.minimum(255.0, rgb2 + (rim_hi2 * (1.0 - in_pec2))[:, :, None] * 26.0)
    p2_arr = (1.0 - cov2[:, :, None]) * p2_arr + cov2[:, :, None] * rgb2
    p2_im = Image.fromarray(np.clip(p2_arr, 0, 255).astype(np.uint8))
    d2 = ImageDraw.Draw(p2_im)
    # Crisp X-ray Z-axis captive post + horizontal chain-link callouts inside each joint
    for cp in cuts_pct:
        cx_i = ox_pad + ((cp + 1.1) / 100.0) * draw_w
        lh_i = float(np.interp(cp, p_dense, top_h_norm)) * h_scale
        if lh_i < 14.0:
            continue
        z_top_pin = bed_y2 - min(lh_i * 0.62, 26.0)
        z_bot_pin = bed_y2 - max(4.0, lh_i * 0.14)
        mid_pin = 0.5 * (z_top_pin + z_bot_pin)
        d2.rounded_rectangle([cx_i - 6, mid_pin - 4, cx_i + 6, mid_pin + 4], radius=3, outline=(56, 189, 248), width=1)
        d2.line([(cx_i - 3, z_top_pin), (cx_i - 3, z_bot_pin)], fill=(250, 204, 21), width=2)
    d2.line([(20, int(bed_y2) + 1), (pw - 20, int(bed_y2) + 1)], fill=(16, 185, 129), width=3)
    d2.text((26, int(bed_y2) + 10), f"Z = 0 BUILD PLATE  |  {len(cuts_pct) + 1}/{len(cuts_pct) + 1} SEGMENTS SEATED  |  CAPTIVE Z-PIN HINGES", fill=(110, 231, 183))
    sheet.paste(p2_im, (pw, 0))

    # ---------- Panel 3 (Bottom-Left) & Panel 4 (Bottom-Right): 3D-Shaded Top & Bottom Views ----------
    cy_tb = ph * 0.51
    w_scale = ph * 0.66
    dy_tb = yy_p - cy_tb
    abs_dy = np.abs(dy_tb)
    body_hw = np.interp(p_px, p_dense, torpedo_w).astype(np.float32) * w_scale

    # Swept-back hydrofoil pectoral fins (sweeping rearward at ~38° like real shark fins!)
    u_fin = np.maximum(0.0, abs_dy - body_hw * 0.82) / (0.24 * w_scale)
    fin_lead = 34.5 + 13.5 * (u_fin ** 1.15)
    fin_chord = np.maximum(0.0, 11.5 * (1.0 - 0.78 * (u_fin ** 1.3)))
    fin_trail = fin_lead + fin_chord
    in_pec_tb = (
        (u_fin <= 1.0)
        & (abs_dy >= body_hw * 0.75)
    ).astype(np.float32) * _smoothstep(fin_lead - 0.9, fin_lead + 0.9, p_px) * (
        1.0 - _smoothstep(fin_trail - 0.9, fin_trail + 0.9, p_px)
    ) * (1.0 - _smoothstep(0.90, 1.02, u_fin))

    cov_body_tb = (1.0 - _smoothstep(np.maximum(1.2, body_hw - 1.1), body_hw + 1.1, abs_dy)) * in_p
    cov_tb = np.clip(np.maximum(cov_body_tb, in_pec_tb), 0.0, 1.0)

    u_lat = np.clip(dy_tb / np.maximum(2.5, body_hw), -1.0, 1.0)
    nz_tb = np.sqrt(np.maximum(0.06, 1.0 - u_lat * u_lat))
    sh_tb = 0.28 + 0.56 * np.clip(-0.36 * u_lat + 0.76 * nz_tb, 0.0, 1.0) + 0.16 * nz_tb
    fin_blend = _smoothstep(0.0, 0.28, u_fin)
    sh_fin = (1.0 - fin_blend) * sh_tb + fin_blend * (0.38 + 0.42 * (1.0 - 0.35 * np.clip(u_fin, 0.0, 1.0)))

    # Gill-curved S-wiggle seams in Top & Bottom views (strictly outside pectoral fins!)
    seam_tb = np.zeros_like(cov_tb)
    rim_tb = np.zeros_like(cov_tb)
    for cp in cuts_pct:
        lat_frac = np.clip((abs_dy - 4.0) / np.maximum(6.0, body_hw - 4.0), 0.0, 1.0)
        c_top = cp + 1.5 * (lat_frac * lat_frac)
        dp_t = p_px - c_top
        sc_t = _smoothstep(-gap_half_pct - 0.16, -gap_half_pct + 0.12, dp_t) * (
            1.0 - _smoothstep(gap_half_pct - 0.12, gap_half_pct + 0.16, dp_t)
        )
        seam_tb = np.maximum(seam_tb, sc_t * (1.0 - in_pec_tb))
        rh_t = _smoothstep(gap_half_pct - 0.1, gap_half_pct + 0.15, dp_t) * (
            1.0 - _smoothstep(gap_half_pct + 0.2, gap_half_pct + 0.75, dp_t)
        )
        rim_tb = np.maximum(rim_tb, rh_t * (1.0 - in_pec_tb))

    # Render Panel 3 (Top View)
    p3_arr = bg_studio.copy()
    rgb3_body = dorsal_col * sh_tb[:, :, None] * 255.0
    rgb3_fin = dorsal_col * sh_fin[:, :, None] * 255.0
    fin_w3 = (in_pec_tb * (1.0 - cov_body_tb * 0.5))[:, :, None]
    rgb3 = (1.0 - fin_w3) * rgb3_body + fin_w3 * rgb3_fin
    # Sculpted dorsal fin ridge along centerline (fin_start .. fin_end)
    dor_hw = 4.8 * np.sin(np.pi * np.clip((p_px - fin_start) / max(3.0, fin_end - fin_start), 0.0, 1.0))
    in_dor = (
        _smoothstep(fin_start, fin_start + 1.5, p_px)
        * (1.0 - _smoothstep(fin_end - 1.5, fin_end, p_px))
        * (1.0 - _smoothstep(np.maximum(0.5, dor_hw - 1.0), dor_hw + 1.0, abs_dy))
    )
    dor_sh = 0.45 + 0.48 * np.clip(1.0 - dy_tb / np.maximum(1.5, dor_hw), 0.0, 1.2)
    rgb3 = (1.0 - in_dor[:, :, None]) * rgb3 + in_dor[:, :, None] * np.minimum(255.0, dorsal_col * dor_sh[:, :, None] * 255.0 + 10.0)
    rgb3 = (1.0 - seam_tb[:, :, None]) * rgb3 + seam_tb[:, :, None] * (rgb3 * 0.24)
    rgb3 = np.minimum(255.0, rgb3 + rim_tb[:, :, None] * 24.0)
    p3_arr = (1.0 - cov_tb[:, :, None]) * p3_arr + cov_tb[:, :, None] * rgb3
    p3_im = Image.fromarray(np.clip(p3_arr, 0, 255).astype(np.uint8))
    d3 = ImageDraw.Draw(p3_im)
    for cp in cuts_pct:
        cx3_i = ox_pad + (cp / 100.0) * draw_w
        d3.rounded_rectangle([cx3_i - 6, cy_tb - 3.5, cx3_i + 5, cy_tb + 3.5], radius=3, outline=(56, 189, 248), width=1)
        d3.ellipse([cx3_i - 4.5, cy_tb - 2.0, cx3_i - 0.5, cy_tb + 2.0], fill=(250, 204, 21))
    d3.text((24, ph - 28), f"FIN-SAFE S-SWIMMING CUTS  |  DORSAL & PECTORAL FINS 100% INTACT", fill=(191, 219, 254))
    sheet.paste(p3_im, (0, ph))

    # Render Panel 4 (Bottom View — Z=0 Emerald First-Layer Bed Contact Ribbon)
    p4_arr = bg_studio.copy()
    rgb4_body = ventral_col * sh_tb[:, :, None] * 242.0
    rgb4_fin = dorsal_col * sh_fin[:, :, None] * 235.0
    rgb4 = (1.0 - fin_w3) * rgb4_body + fin_w3 * rgb4_fin
    # Centered Z=0 ventral bed-contact ribbon across every segment (15%..77%) + caudal tail pad (89%..94%)
    pad_hw = body_hw * 0.38 * _smoothstep(13.0, 19.0, p_px) * (1.0 - _smoothstep(74.0, 78.5, p_px))
    tail_pad_hw = body_hw * 0.55 * _smoothstep(87.5, 89.5, p_px) * (1.0 - _smoothstep(93.5, 95.5, p_px))
    tot_pad_hw = np.maximum(pad_hw, tail_pad_hw)
    in_pad4 = (
        (1.0 - _smoothstep(np.maximum(0.8, tot_pad_hw - 1.1), tot_pad_hw + 1.1, abs_dy))
        * (tot_pad_hw > 1.2)
        * (1.0 - seam_tb)
    )
    em_rgb = np.array([16.0, 198.0, 136.0], dtype=np.float32) * (0.82 + 0.18 * nz_tb)[:, :, None]
    rgb4 = (1.0 - in_pad4[:, :, None]) * rgb4 + in_pad4[:, :, None] * em_rgb
    rgb4 = (1.0 - seam_tb[:, :, None]) * rgb4 + seam_tb[:, :, None] * (rgb4 * 0.22)
    p4_arr = (1.0 - cov_tb[:, :, None]) * p4_arr + cov_tb[:, :, None] * rgb4
    p4_im = Image.fromarray(np.clip(p4_arr, 0, 255).astype(np.uint8))
    d4 = ImageDraw.Draw(p4_im)
    d4.text(
        (24, ph - 28),
        f"EMERALD = Z=0 FIRST-LAYER BED PADS  |  {len(cuts_pct) + 1}/{len(cuts_pct) + 1} SEGMENTS SUPPORT-FREE",
        fill=(110, 231, 183),
    )
    sheet.paste(p4_im, (pw, ph))

    dr = ImageDraw.Draw(sheet)

    # Quadrant dividers & headers
    dr.line([(pw, 0), (pw, ph * 2)], fill=(64, 72, 86), width=2)
    dr.line([(0, ph), (pw * 2, ph)], fill=(64, 72, 86), width=2)
    for lbl, (ox, oy) in [
        ("1 | 3/4 ARTICULATED (SEATED ON Z=0)", (0, 0)),
        ("2 | SIDE ORTHOGRAPHIC (Z=0 BED & HINGES)", (pw, 0)),
        ("3 | TOP ORTHOGRAPHIC (FIN-SAFE CUTS)", (0, ph)),
        ("4 | BOTTOM VIEW (Z=0 BED CONTACT PADS)", (pw, ph)),
    ]:
        _draw_panel_badge(dr, ox, oy, lbl)

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue(), cuts_pct
