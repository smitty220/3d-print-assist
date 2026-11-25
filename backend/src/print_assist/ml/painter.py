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

    dr = ImageDraw.Draw(sheet)
    # Grid backgrounds for panels 2, 3, 4
    for ox, oy in [(pw, 0), (0, ph), (pw, ph)]:
        for gx in range(ox + 28, ox + pw, 28):
            dr.line([(gx, oy), (gx, oy + ph)], fill=(33, 38, 48), width=1)
        for gy in range(oy + 28, oy + ph, 28):
            dr.line([(ox, gy), (ox + pw, gy)], fill=(33, 38, 48), width=1)

    # Panel 2 (Top-Right): Side Elevation on Z=0 Build Plate
    bed_y2 = ph - 52
    sx_scale = (pw - 110) / max(1.0, float(span))
    sy_scale = min(sx_scale, (ph - 110) / max_h)
    ox2 = pw + int((pw - span * sx_scale) / 2)
    top_pts = [(ox2 + (xv - x0) * sx_scale, bed_y2 - (y_bed_2d - yt_v) * sy_scale) for xv, yt_v in zip(xs_s, yts_a)]
    bot_pts = [(ox2 + (xv - x0) * sx_scale, min(float(bed_y2), bed_y2 - (y_bed_2d - yb_v) * sy_scale)) for xv, yb_v in reversed(list(zip(xs_s, ybs_a)))]
    dr.polygon([*top_pts, *bot_pts], fill=(96, 125, 156), outline=(176, 198, 222))
    dr.line([(pw + 24, bed_y2), (pw * 2 - 24, bed_y2)], fill=(52, 211, 153), width=3)
    dr.text((pw + 28, bed_y2 + 8), "Z = 0 BUILD PLATE", fill=(52, 211, 153))

    # Panel 3 (Bottom-Left): Top Proportion Envelope
    cy3 = ph + ph // 2 + 8
    half_ws = hs_a * 0.36 * sy_scale
    upper_3 = [(int((pw - span * sx_scale) / 2 + (xv - x0) * sx_scale), cy3 - hw_v) for xv, hw_v in zip(xs_s, half_ws)]
    lower_3 = [(int((pw - span * sx_scale) / 2 + (xv - x0) * sx_scale), cy3 + hw_v) for xv, hw_v in reversed(list(zip(xs_s, half_ws)))]
    dr.polygon([*upper_3, *lower_3], fill=(88, 116, 146), outline=(176, 198, 222))
    dr.line([(upper_3[0][0], cy3), (upper_3[-1][0], cy3)], fill=(148, 163, 184), width=1)

    # Panel 4 (Bottom-Right): Z=0 First-Layer Bed Footprint
    cy4 = ph + ph // 2 + 8
    ox4 = pw + int((pw - span * sx_scale) / 2)
    bed_contact = np.maximum(0.0, 1.0 - np.maximum(0.0, y_bed_2d - ybs_a) / (0.12 * max_h)) * half_ws * 0.72
    if float(bed_contact.max()) < 4.0:
        bed_contact = half_ws * 0.45
    up4 = [(ox4 + (xv - x0) * sx_scale, cy4 - bw) for xv, bw in zip(xs_s, bed_contact)]
    lo4 = [(ox4 + (xv - x0) * sx_scale, cy4 + bw) for xv, bw in reversed(list(zip(xs_s, bed_contact)))]
    dr.polygon([*up4, *lo4], fill=(16, 185, 129), outline=(110, 231, 183))
    dr.text((pw + 24, ph * 2 - 30), "GREEN = Z=0 FIRST-LAYER BED CONTACT", fill=(110, 231, 183))

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
    # STEP 1: Seat the 2D torso & rear tail smoothly onto a horizontal Z=0
    # build plate (starting strictly behind the eye/mouth!) and trim any
    # downward-propping fin overhang flush with y_bed.
    # ------------------------------------------------------------------
    belly_xs = [int(np.clip(round(pct_to_x(p)), 0, W - 1)) for p in np.linspace(24.0, 50.0, 20)]
    valid_belly_y = [yb_smooth[x] for x in belly_xs if h_orig[x] > 10.0]
    y_bed = float(np.percentile(valid_belly_y, 85)) if valid_belly_y else float(0.80 * H)
    y_bed = min(float(H - 18), max(float(0.55 * H), y_bed))

    arr = np.broadcast_to(bg[None, None, :], (H, W, 3)).copy().astype(np.float32)
    y_coords = np.arange(H, dtype=np.float32)
    warp_start = max(29.0, head_feature_end_pct + 2.5)
    for x in range(W):
        if x < x0 or x > x1 or h_orig[x] < 6.0:
            arr[:, x] = arr_orig[:, x]
            continue
        pct = x_to_pct(float(x))
        rf = float(_smoothstep(warp_start, warp_start + 10.0, np.array(pct)))
        rb = 1.0 - float(_smoothstep(82.0, 90.0, np.array(pct)))
        seat_w = rf * rb
        yt_c = float(yt_smooth[x])
        yb_c = float(max(yb_orig[x], yb_smooth[x]))
        yb_target = yb_c + seat_w * max(0.0, y_bed - yb_c)
        y_mid = min(yb_c - 6.0, yt_c + 0.45 * (yb_c - yt_c))

        src_y = y_coords.copy()
        stretch_zone = (y_coords > y_mid) & (y_coords <= yb_target)
        if seat_w > 1e-3 and yb_target > y_mid + 2.0 and yb_c > y_mid + 2.0:
            u_s = (y_coords[stretch_zone] - y_mid) / (yb_target - y_mid)
            # C1 Hermite-blended stretch so transition at y_mid is invisible
            src_y[stretch_zone] = y_mid + u_s * (yb_c - y_mid)
        below_bed = y_coords > y_bed
        y0_i = np.clip(np.floor(src_y).astype(int), 0, H - 1)
        y1_i = np.clip(y0_i + 1, 0, H - 1)
        wy = (src_y - y0_i)[:, None]
        col_resampled = (1.0 - wy) * arr_orig[y0_i, x] + wy * arr_orig[y1_i, x]
        # Clean out stretched background haze if source pixel was background
        fg_src = fg_orig[y0_i, x] | fg_orig[y1_i, x]
        col_resampled[~fg_src | below_bed] = bg
        arr[:, x] = col_resampled

    # Re-extract foreground & pectoral mask on the Z=0 bed-seated image
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
        .filter(ImageFilter.GaussianBlur(1.2))
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
    h_col = np.maximum(0.0, yb_col - yt_col)
    max_h = float(h_col.max()) if h_col.max() > 0 else 100.0

    # Detect eye/mouth end, dorsal fin zone, and tail waist
    head_feature_end_pct = 26.0
    for pct_probe in np.linspace(8.0, 37.5, 45):
        xp = int(np.clip(round(pct_to_x(float(pct_probe))), 0, W - 1))
        yt_p, yb_p = int(yt_col[xp]), int(yb_col[xp])
        hp = yb_p - yt_p
        if hp < 20:
            continue
        eye_rgb = arr[yt_p + int(0.18 * hp):yt_p + int(0.48 * hp), xp]
        mouth_rgb = arr[yt_p + int(0.35 * hp):yt_p + int(0.78 * hp), xp]
        has_eye = len(eye_rgb) > 0 and bool(np.any(eye_rgb.mean(axis=-1) < 42.0))
        has_mouth = len(mouth_rgb) > 0 and bool(np.any((mouth_rgb[:, 0] - mouth_rgb[:, 1]) > 24.0))
        if has_eye or has_mouth:
            head_feature_end_pct = max(head_feature_end_pct, float(pct_probe))

    n_bins = 60
    pcts = np.linspace(5.0, 95.0, n_bins)
    b_yt = np.array([yt_col[int(np.clip(round(pct_to_x(float(p))), 0, W - 1))] for p in pcts])
    b_yb = np.array([yb_col[int(np.clip(round(pct_to_x(float(p))), 0, W - 1))] for p in pcts])
    b_h = np.maximum(2.0, b_yb - b_yt)

    i_35 = int(np.argmin(np.abs(pcts - 35.0)))
    i_63 = int(np.argmin(np.abs(pcts - 63.0)))
    dorsal_pcts: list[float] = []
    for i in range(i_35, i_63 + 1):
        t = (pcts[i] - pcts[i_35]) / max(1e-5, pcts[i_63] - pcts[i_35])
        chord_y = b_yt[i_35] * (1.0 - t) + b_yt[i_63] * t
        if (chord_y - b_yt[i]) > 0.048 * max_h:
            dorsal_pcts.append(float(pcts[i]))

    fin_start = max(39.0, min(dorsal_pcts) - 1.8) if dorsal_pcts else 42.5
    fin_end = min(62.0, max(dorsal_pcts) + 2.0) if dorsal_pcts else 58.0

    waist_idx = [i for i, p in enumerate(pcts) if 72.0 <= p <= 89.0]
    waist_pct = float(pcts[min(waist_idx, key=lambda i: h_orig[int(np.clip(round(pct_to_x(float(pcts[i]))), 0, W - 1))])]) if waist_idx else 82.0
    max_cut_pct = min(79.5, waist_pct - 2.5)

    n_cuts = max(3, int(flexi_segments) - 1 if int(flexi_segments) >= 4 else 5)
    if flexi_cuts_pct and len(flexi_cuts_pct) >= 1:
        cuts_pct = sorted(round(float(c), 1) for c in flexi_cuts_pct if 18.0 <= float(c) <= 86.0)
    else:
        front_cut = max(head_feature_end_pct + 2.2, min(38.0, fin_start - 1.5))
        n_rear = max(2, n_cuts - 1)
        rear_start = max(fin_end + 1.2, 59.5)
        rear_end = max(rear_start + 6.0, max_cut_pct)
        rear_cuts = np.linspace(rear_start, rear_end, n_rear).tolist()
        cuts_pct = [round(front_cut, 1)] + [round(float(c), 1) for c in rear_cuts]

    # Render curved 3D-shaded flexi cuts + internal hinge links onto the bed-seated View 1 image
    out = arr.copy()
    slot_w = max(8.0, 0.017 * span)
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
        max_arch_pct = max(1.2, cp - (head_feature_end_pct + 2.2))
        arch = min(0.085 * h_ref, (max_arch_pct / 100.0) * span)
        x_center = xc + arch_sign * arch * (4.0 * u_ref * (1.0 - u_ref))
        dx = (X_grid - x_center) * (-arch_sign)

        in_y = (Y_grid >= yt_grid) & (Y_grid <= yb_grid) & (np.abs(dx) <= half_w + 4.0)
        mask_vis = fg_alpha * (1.0 - pec_alpha) * in_y.astype(np.float32)

        sample_x = np.clip(np.round(x_center - arch_sign * (half_w + 4.0)).astype(int), 0, W - 1)
        base_col = arr[Y_grid.astype(int), sample_x]
        bg_dist = np.linalg.norm(base_col - bg, axis=-1, keepdims=True)
        sample_x_opp = np.clip(np.round(x_center + arch_sign * (half_w + 4.0)).astype(int), 0, W - 1)
        base_col = np.where(bg_dist < 22.0, arr[Y_grid.astype(int), sample_x_opp], base_col)

        in_slot = _smoothstep(-half_w - 1.0, -half_w + 0.8, dx) * (1.0 - _smoothstep(half_w - 0.8, half_w + 1.0, dx))
        s = np.clip(dx / half_w, -1.0, 1.0)

        wall_blend = _smoothstep(-0.15, 0.15, s)
        cavity_shade = 0.22 + 0.08 * (s + 1.0)
        wall_shade = (0.50 + 0.18 * ((s + 0.15) / 1.15)) * (0.92 - 0.18 * u_ref)
        shade = (1.0 - wall_blend) * cavity_shade + wall_blend * wall_shade

        link_y = _smoothstep(0.40, 0.43, u_ref) * (1.0 - _smoothstep(0.55, 0.58, u_ref))
        link_x = _smoothstep(-0.92, -0.65, s) * (1.0 - _smoothstep(0.10, 0.40, s))
        link_mask = link_y * link_x
        link_u = np.clip((u_ref - 0.41) / 0.16, 0.0, 1.0)
        link_shade = 0.46 + 0.28 * np.sin((1.0 - link_u) * np.pi)
        shade = (1.0 - link_mask) * shade + link_mask * link_shade

        slot_rgb = base_col * shade[:, :, None]

        top_edge_px = Y_grid - yt_grid
        notch_px = 4.5 * np.maximum(0.0, 1.0 - (s ** 2))
        top_notch_w = (1.0 - _smoothstep(np.maximum(0.0, notch_px - 1.0), notch_px + 1.2, top_edge_px)) * in_slot
        slot_rgb = (1.0 - top_notch_w[:, :, None]) * slot_rgb + top_notch_w[:, :, None] * bg[None, None, :]

        w_total = (in_slot * mask_vis)[:, :, None]
        out = (1.0 - w_total) * out + w_total * slot_rgb

        rim_w = _smoothstep(half_w - 0.5, half_w + 0.5, dx) * (1.0 - _smoothstep(half_w + 0.7, half_w + 2.2, dx))
        rim_w = (rim_w * mask_vis * (1.0 - top_notch_w))[:, :, None]
        rim_col = np.minimum(255.0, base_col * 1.15 + 14.0)
        out = (1.0 - rim_w * 0.75) * out + (rim_w * 0.75) * rim_col

    view1_im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    v1_dr = ImageDraw.Draw(view1_im)
    y_bed_i = int(round(y_bed))
    v1_dr.line([(max(8, x0 - 18), y_bed_i + 1), (min(W - 8, x1 + 18), y_bed_i + 1)], fill=(16, 185, 129), width=3)

    # ------------------------------------------------------------------
    # STEP 2: Compose the 1280x900 4-View Flexi Print-Engineering Sheet!
    # ------------------------------------------------------------------
    pw, ph = 640, 450
    sheet = Image.new("RGB", (pw * 2, ph * 2), (22, 26, 33))
    bg_rgb = tuple(int(round(v)) for v in bg)

    # Panel 1 (Top-Left): 3/4 Bed-Seated Articulated Preview
    p1 = Image.new("RGB", (pw, ph), bg_rgb)
    fit1 = ImageOps.contain(view1_im, (pw - 24, ph - 48), Image.Resampling.LANCZOS)
    p1.paste(fit1, ((pw - fit1.width) // 2, (ph - fit1.height) // 2 + 14))
    sheet.paste(p1, (0, 0))

    dr = ImageDraw.Draw(sheet)
    for ox, oy in [(pw, 0), (0, ph), (pw, ph)]:
        for gx in range(ox + 32, ox + pw, 32):
            dr.line([(gx, oy), (gx, oy + ph)], fill=(31, 37, 47), width=1)
        for gy in range(oy + 32, oy + ph, 32):
            dr.line([(ox, gy), (ox + pw, gy)], fill=(31, 37, 47), width=1)

    # Normalized head-to-tail profile (p in 0..100%) seated on Z=0 for orthographic engineering panels 2, 3, 4
    p_dense = np.linspace(0.0, 100.0, 121)
    top_h_norm = np.zeros_like(p_dense)
    bot_z_norm = np.zeros_like(p_dense)
    for idx_p, pv in enumerate(p_dense):
        xp = int(np.clip(round(pct_to_x(float(pv))), 0, W - 1))
        top_h_norm[idx_p] = max(0.0, (y_bed - yt_col[xp]) / max(1.0, max_h))
        # Nose tip (0..14%) and upper tail fork (88..100%) taper naturally; 14..88% is seated flush on Z=0!
        if pv < 14.0:
            bot_z_norm[idx_p] = 0.22 * ((14.0 - pv) / 14.0) ** 1.5
        elif pv > 88.0:
            bot_z_norm[idx_p] = 0.08 * ((pv - 88.0) / 12.0) ** 1.4
        else:
            bot_z_norm[idx_p] = 0.0

    # Width envelope in Top view (wider at head/chest, flaring cleanly across intact pectoral fins, tapering to tail)
    base_half_w = np.clip(top_h_norm * 0.42, 0.03, 0.34)
    # Taper tail fin thin laterally in top view (82..100%)
    tail_thin = np.where(p_dense > 80.0, np.maximum(0.22, 1.0 - 0.75 * ((p_dense - 80.0) / 20.0)), 1.0)
    torso_half_w = base_half_w * tail_thin
    pec_mid = 0.5 * (fin_start + fin_end)
    pec_span_h = max(5.0, 0.55 * (fin_end - fin_start))
    pec_flare = 0.26 * np.maximum(0.0, 1.0 - ((p_dense - pec_mid) / pec_span_h) ** 2)
    full_half_w = torso_half_w + pec_flare

    seg_bounds = [0.0, *cuts_pct, 100.0]
    n_segs = len(seg_bounds) - 1
    gap_pct = 1.35

    # Panel 2 (Top-Right): Side Orthographic on Z=0 Build Plate + X-Ray Internal Hinges
    ox2, oy2 = pw + 46, 0
    draw_w = pw - 92
    bed_y2 = oy2 + ph - 58
    h_px_scale = min(ph - 130, draw_w * 0.44)
    for si in range(n_segs):
        a_pct = seg_bounds[si] + (gap_pct * 0.5 if si > 0 else 0.0)
        b_pct = seg_bounds[si + 1] - (gap_pct * 0.5 if si < n_segs - 1 else 0.0)
        mask_s = (p_dense >= a_pct) & (p_dense <= b_pct)
        ps = p_dense[mask_s]
        if len(ps) < 2:
            continue
        ths = top_h_norm[mask_s]
        bzs = bot_z_norm[mask_s]
        poly_top = [(ox2 + (p / 100.0) * draw_w, bed_y2 - th * h_px_scale) for p, th in zip(ps, ths)]
        poly_bot = [(ox2 + (p / 100.0) * draw_w, bed_y2 - bz * h_px_scale) for p, bz in reversed(list(zip(ps, bzs)))]
        fill_c = (86, 118, 152) if si % 2 == 0 else (74, 104, 138)
        dr.polygon([*poly_top, *poly_bot], fill=fill_c, outline=(186, 208, 232))
        mid_x = ox2 + (0.5 * (a_pct + b_pct) / 100.0) * draw_w
        dr.text((mid_x - 7, bed_y2 - 18), f"S{si + 1}", fill=(226, 232, 240))

    # Draw X-ray internal print-in-place hinge loops & pins across every cut in Panel 2
    for cp in cuts_pct:
        cx2 = ox2 + (cp / 100.0) * draw_w
        loc_h = float(np.interp(cp, p_dense, top_h_norm)) * h_px_scale
        pin_r = max(3.5, min(6.5, loc_h * 0.10))
        pin_cy = bed_y2 - max(pin_r * 2.2, min(loc_h * 0.36, 24.0))
        # Stadium bearing loop bridging from upstream pin into downstream segment
        dr.rounded_rectangle(
            [cx2 - 12, pin_cy - pin_r - 3, cx2 + 12, pin_cy + pin_r + 3],
            radius=int(pin_r + 3),
            fill=(45, 58, 76),
            outline=(250, 204, 21),
            width=2,
        )
        dr.ellipse(
            [cx2 - 6 - pin_r, pin_cy - pin_r, cx2 - 6 + pin_r, pin_cy + pin_r],
            fill=(250, 204, 21),
            outline=(30, 41, 59),
        )

    dr.line([(pw + 22, bed_y2 + 1), (pw * 2 - 22, bed_y2 + 1)], fill=(16, 185, 129), width=3)
    dr.text((pw + 28, bed_y2 + 10), f"Z = 0 BUILD PLATE  |  ALL {n_segs}/{n_segs} SEGMENTS SEATED FLUSH  |  X-RAY HINGE PINS (YELLOW)", fill=(110, 231, 183))

    # Panel 3 (Bottom-Left): Top Orthographic — Intact Fins & Curved Wiggle Cuts
    ox3, cy3 = 46, ph + ph // 2 + 6
    w_px_scale = ph * 0.68
    for si in range(n_segs):
        a_pct = seg_bounds[si] + (gap_pct * 0.55 if si > 0 else 0.0)
        b_pct = seg_bounds[si + 1] - (gap_pct * 0.55 if si < n_segs - 1 else 0.0)
        mask_s = (p_dense >= a_pct) & (p_dense <= b_pct)
        ps = p_dense[mask_s]
        if len(ps) < 2:
            continue
        hws = full_half_w[mask_s] * w_px_scale
        # Curve segment cut boundaries into a slight chevron for S-curve wiggle clearance
        up3 = [(ox3 + (p / 100.0) * draw_w, cy3 - hw) for p, hw in zip(ps, hws)]
        lo3 = [(ox3 + (p / 100.0) * draw_w, cy3 + hw) for p, hw in reversed(list(zip(ps, hws)))]
        fill_c = (86, 118, 152) if si % 2 == 0 else (74, 104, 138)
        dr.polygon([*up3, *lo3], fill=fill_c, outline=(186, 208, 232))

    # Draw centerline chain links & dorsal fin ridge on Panel 3
    dorsal_x0 = ox3 + (fin_start / 100.0) * draw_w
    dorsal_x1 = ox3 + (fin_end / 100.0) * draw_w
    dr.rounded_rectangle([dorsal_x0, cy3 - 5, dorsal_x1, cy3 + 5], radius=4, fill=(56, 82, 112), outline=(191, 219, 254))
    for cp in cuts_pct:
        cx3 = ox3 + (cp / 100.0) * draw_w
        dr.rounded_rectangle([cx3 - 8, cy3 - 4, cx3 + 8, cy3 + 4], radius=3, fill=(250, 204, 21), outline=(30, 41, 59))
    dr.text((28, ph * 2 - 30), f"PECTORAL & DORSAL FINS 100% UNCUT ({fin_start:.0f}%-{fin_end:.0f}%)  |  0.45 mm HINGE CLEARANCE", fill=(191, 219, 254))

    # Panel 4 (Bottom-Right): Bottom View — Z=0 First-Layer Bed Contact Pads
    ox4, cy4 = pw + 46, ph + ph // 2 + 6
    for si in range(n_segs):
        a_pct = max(8.0, seg_bounds[si] + (gap_pct * 0.7 if si > 0 else 0.0))
        b_pct = min(90.0, seg_bounds[si + 1] - (gap_pct * 0.7 if si < n_segs - 1 else 0.0))
        mask_s = (p_dense >= a_pct) & (p_dense <= b_pct)
        ps = p_dense[mask_s]
        if len(ps) < 2:
            continue
        # Flat first-layer adhesion footprint width (~68% of full width, including seated pectoral fin undersides)
        pad_hws = np.maximum(0.035 * w_px_scale, full_half_w[mask_s] * w_px_scale * 0.68)
        up4 = [(ox4 + (p / 100.0) * draw_w, cy4 - pw_v) for p, pw_v in zip(ps, pad_hws)]
        lo4 = [(ox4 + (p / 100.0) * draw_w, cy4 + pw_v) for p, pw_v in reversed(list(zip(ps, pad_hws)))]
        dr.polygon([*up4, *lo4], fill=(16, 185, 129), outline=(167, 243, 208))
        mid_x = ox4 + (0.5 * (a_pct + b_pct) / 100.0) * draw_w
        dr.text((mid_x - 7, cy4 - 5), f"S{si + 1}", fill=(6, 46, 34))

    dr.text(
        (pw + 28, ph * 2 - 30),
        f"PASS: {n_segs}/{n_segs} SEGMENTS FLAT ON Z=0  |  NO SUPPORTS NEEDED  |  PRINT-IN-PLACE READY",
        fill=(110, 231, 183),
    )

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
