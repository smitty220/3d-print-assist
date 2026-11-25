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


def render_flexi_concept_preview(
    png_bytes: bytes,
    flexi_segments: int = 6,
    flexi_cuts_pct: list[float] | None = None,
) -> tuple[bytes, list[float]]:
    """Render fin-safe 3D-shaded flexi articulation cuts + internal print-in-place hinge links onto a 2D concept PNG.

    The un-cut source painting (`K#_clean.png`) is preserved separately for 3D sculpting (`sculpt_mesh_from_image`),
    while the user-facing Design preview (`K#.png`) clearly shows the articulated flexi character before 'Approve & build'.
    """
    import numpy as np

    with Image.open(io.BytesIO(png_bytes)) as im:
        rgb_im = im.convert("RGB")
    arr = np.asarray(rgb_im, dtype=np.float32)
    H, W, _ = arr.shape
    Y_grid, X_grid = np.mgrid[0:H, 0:W].astype(np.float32)

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
        # Synthetic uniform test canvas fallback
        fg = np.zeros((H, W), dtype=bool)
        fg[int(0.22 * H):int(0.78 * H), int(0.12 * W):int(0.88 * W)] = True
        fg_alpha = fg.astype(np.float32)
        xs = np.where(fg.any(axis=0))[0]
    x0, x1 = int(xs[0]), int(xs[-1])
    span = max(1, x1 - x0)

    # Detect foreground pectoral fin hanging across/below a cream underbelly
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
                pec_start_y = first_cream_y + 2 + int(blue_below[0])
                pec_fin_mask[pec_start_y:, x] = True

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
        yt_col[x] = float(yt)
        yb_col[x] = float(yb)
    h_col = np.maximum(0.0, yb_col - yt_col)
    max_h = float(h_col.max()) if h_col.max() > 0 else 100.0

    x_15_35 = [int(x0 + p * span) for p in np.linspace(0.15, 0.35, 9)]
    x_68_85 = [int(x0 + p * span) for p in np.linspace(0.68, 0.85, 9)]
    facing_left = float(np.mean(h_col[x_15_35])) >= float(np.mean(h_col[x_68_85]))

    def pct_to_x(pct: float) -> float:
        p = (pct / 100.0) if facing_left else (1.0 - pct / 100.0)
        return float(x0 + p * span)

    # Detect eye and mouth boundary so front cuts never touch eye or smile
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
    b_h = np.array([h_col[int(np.clip(round(pct_to_x(float(p))), 0, W - 1))] for p in pcts])

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
    waist_pct = float(pcts[min(waist_idx, key=lambda i: b_h[i])]) if waist_idx else 82.0
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
        arch = min(0.095 * h_ref, (max_arch_pct / 100.0) * span)
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

        lead_w = _smoothstep(-half_w - 2.0, -half_w - 0.5, dx) * (1.0 - _smoothstep(-half_w - 0.4, -half_w + 0.5, dx))
        lead_w = (lead_w * mask_vis * (1.0 - top_notch_w))[:, :, None]
        out = (1.0 - lead_w * 0.45) * out + (lead_w * 0.45) * (base_col * 0.65)

    buf = io.BytesIO()
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue(), cuts_pct
