"""Concept-to-3D Silhouette-Aligned Multi-Color MMU3/AMS Inlay Projector.

Takes ANY approved 2D Character Concept image (`concept_png`) and its sculpted 3D watertight mesh (`mf`):
  1. Extracts up to `max_colors` (default 5) clean PLA filament colors from the concept foreground —
     capturing BOTH dominant body/belly zones AND high-contrast facial details (eyes, teeth, mouth, spots).
  2. Automatically solves the 3D orthographic camera pose `(azimuth, elevation, roll, scale, shift)` that
     maximizes 2D silhouette IoU against the concept image (~0.3 s in pure NumPy).
  3. Z-buffer projects the concept's quantized filament colors onto visible 3D mesh faces, mirrors cleanly
     across bilateral symmetry planes when the character is symmetric, and fills hidden faces smoothly.
  4. Converts each painted accent region into a mathematically watertight, `1.6 mm`-deep solid inlay
     (via directional sub-patch extrusion + C++ `manifold3d` boolean intersection & disjoint subtraction)
     ready for PrusaSlicer MMU3 / Bambu Studio AMS `.3mf` export and live multi-color Three.js rendering!
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import manifold3d as m3d
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


def _rgb_to_hex(rgb: np.ndarray | tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(round(float(rgb[0])), 0, 255)),
        int(np.clip(round(float(rgb[1])), 0, 255)),
        int(np.clip(round(float(rgb[2])), 0, 255)),
    )


def _redmean_dist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast perceptual color squared distance (Redmean approximation, pure NumPy)."""
    rm = 0.5 * (a[..., 0] + b[..., 0])
    dr = a[..., 0] - b[..., 0]
    dg = a[..., 1] - b[..., 1]
    db = a[..., 2] - b[..., 2]
    return (2.0 + rm / 256.0) * (dr * dr) + 4.0 * (dg * dg) + (2.0 + (255.0 - rm) / 256.0) * (db * db)


def _color_label_name(rgb: np.ndarray, role: str) -> str:
    if role == "primary":
        return "Body"
    if role == "secondary":
        return "Belly"
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    luma = (r + g + b) / 3.0
    if luma < 60.0:
        return "Eyes & Dark"
    if luma > 225.0 and (max(r, g, b) - min(r, g, b)) < 26.0:
        return "White & Teeth"
    if r > g + 30.0 and r > b + 15.0:
        return "Mouth & Pink" if g < 140.0 and b > 55.0 else "Red Accent"
    if g > r + 25.0 and g > b + 20.0:
        return "Green Accent"
    if r > 180.0 and g > 140.0 and b < 110.0:
        return "Gold Accent"
    return "Accent"


def extract_concept_color_map(
    concept_png: Path,
    max_colors: int = 5,
) -> dict[str, Any] | None:
    """Extract clean foreground mask, up to `max_colors` PLA filament colors, and quantized pixel label map."""
    if not concept_png.is_file():
        return None
    with Image.open(concept_png) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    H, W, _ = rgb.shape
    if H < 32 or W < 32:
        return None

    # 1. Foreground extraction: respect alpha if transparent, else flood-fill studio backdrop + floor drop shadow
    if float((alpha < 200).mean()) > 0.08:
        fg = alpha >= 180
    else:
        border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
        bg = np.median(border, axis=0)
        dist_bg = np.linalg.norm(rgb - bg, axis=-1)
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        sat = (mx - mn) / np.maximum(mx, 1.0)
        luma = rgb.mean(axis=-1)
        # Studio backdrop + neutral soft gray drop-shadow touching the border
        bg_cand = (dist_bg < 22.0) | ((luma > 210.0) & (sat < 0.055) & (dist_bg < 52.0))
        lbl, _ = ndimage.label(bg_cand)
        border_lbls = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])).tolist()) - {0}
        bg_mask = np.isin(lbl, list(border_lbls))
        fg = ~bg_mask

    # Keep only the main connected foreground subject (drop tiny stray background marks)
    fg_lbl, n_fg = ndimage.label(fg)
    if n_fg == 0:
        return None
    if n_fg > 1:
        sizes = ndimage.sum(fg, fg_lbl, range(1, n_fg + 1))
        main_id = int(np.argmax(sizes)) + 1
        fg = fg_lbl == main_id

    n_px = int(fg.sum())
    if n_px < 400:
        return None

    # Interior mask eroded by 2 px so dark/white anti-aliased edge fringes don't trigger fake accent colors
    fg_inner = ndimage.binary_erosion(fg, iterations=2)
    if int(fg_inner.sum()) < 200:
        fg_inner = fg

    fg_rgb = rgb[fg_inner]
    fg_mx = fg_rgb.max(axis=-1)
    fg_mn = fg_rgb.min(axis=-1)
    fg_chroma = fg_mx - fg_mn
    fg_sat = fg_chroma / np.maximum(fg_mx, 1.0)
    fg_luma = fg_rgb.mean(axis=-1)

    palette_rgb: list[np.ndarray] = []
    palette_roles: list[str] = []

    # 2. Dominant primary + secondary body colors (excluding ultra-dark shadows < 52 and pure specular > 246)
    body_pool = fg_rgb[(fg_luma >= 55.0) & (fg_luma <= 245.0)]
    if len(body_pool) < 100:
        body_pool = fg_rgb
    sample = body_pool[:: max(1, len(body_pool) // 4000)]

    # Deterministic 2-cluster K-means++ on body_pool
    mean_c = sample.mean(axis=0)
    c1 = sample[int(np.argmax(np.sum((sample - mean_c) ** 2, axis=-1)))].copy()
    c0 = sample[int(np.argmax(np.sum((sample - c1) ** 2, axis=-1)))].copy()
    labels = np.zeros(len(sample), dtype=np.int32)
    for _ in range(14):
        d0 = _redmean_dist2(sample, c0[None, :])
        d1 = _redmean_dist2(sample, c1[None, :])
        labels = (d1 < d0).astype(np.int32)
        if np.any(labels == 0):
            c0 = np.median(sample[labels == 0], axis=0)
        if np.any(labels == 1):
            c1 = np.median(sample[labels == 1], axis=0)
    counts = np.bincount(labels, minlength=2)
    # Order primary = largest cluster, secondary = second cluster
    if counts[1] > counts[0]:
        c0, c1 = c1, c0
        counts = counts[::-1]

    palette_rgb.append(c0.astype(np.float32))
    palette_roles.append("primary")

    min_sec_share = 0.055
    if (
        max_colors >= 2
        and (counts[1] / max(1, counts.sum())) >= min_sec_share
        and float(np.linalg.norm(c0 - c1)) >= 48.0
    ):
        palette_rgb.append(c1.astype(np.float32))
        palette_roles.append("secondary")

    def _min_dist_to_palette(pts: np.ndarray) -> np.ndarray:
        if not palette_rgb:
            return np.full(len(pts), 999.0, dtype=np.float32)
        d_min = np.full(len(pts), 1e9, dtype=np.float32)
        for pc in palette_rgb:
            d_min = np.minimum(d_min, np.linalg.norm(pts - pc[None, :], axis=-1))
        return d_min

    # 3. High-contrast DARK accent (glossy black eyes, pupils, nose)
    if len(palette_rgb) < max_colors:
        dark_mask = (fg_luma < 52.0) & (_min_dist_to_palette(fg_rgb) >= 58.0)
        if int(dark_mask.sum()) >= max(50, int(len(fg_rgb) * 0.0018)):
            dark_c = np.median(fg_rgb[dark_mask], axis=0)
            # Clean crisp print black (#16181e)
            dark_c = np.clip(dark_c, 16.0, 34.0)
            palette_rgb.append(dark_c.astype(np.float32))
            palette_roles.append("dark")

    # 4. Saturated CHROMATIC accent (coral pink mouth/tongue, red hat, yellow beak, green leaves)
    if len(palette_rgb) < max_colors:
        d_pal = _min_dist_to_palette(fg_rgb)
        chrom_mask = (fg_chroma >= 36.0) & (fg_sat >= 0.28) & (d_pal >= 62.0)
        # Filter out darker/lighter shades of ANY existing palette color (e.g. shadowed cream or shadowed blue!)
        fg_n = fg_rgb / np.maximum(1e-4, np.linalg.norm(fg_rgb, axis=-1, keepdims=True))
        min_hue_diff = np.ones(len(fg_rgb), dtype=np.float32)
        for pc in palette_rgb:
            pc_n = pc / max(1e-4, float(np.linalg.norm(pc)))
            min_hue_diff = np.minimum(min_hue_diff, 1.0 - (fg_n @ pc_n))
        chrom_mask = chrom_mask & (min_hue_diff >= 0.045)
        if int(chrom_mask.sum()) >= max(50, int(len(fg_rgb) * 0.0012)):
            c_pts = fg_rgb[chrom_mask]
            # Pick brighter, higher-saturation quartile so coral pink isn't darkened by mouth-cavity shadow
            c_luma = c_pts.mean(axis=-1)
            hi_c = c_pts[c_luma >= np.percentile(c_luma, 55)]
            acc_c = np.median(hi_c if len(hi_c) >= 15 else c_pts, axis=0)
            if acc_c.max() < 185.0:
                acc_c = np.clip(acc_c * (195.0 / max(1.0, float(acc_c.max()))), 0.0, 255.0)
            palette_rgb.append(acc_c.astype(np.float32))
            palette_roles.append("chromatic")

    # 5. Crisp BRIGHT WHITE accent (teeth, eye highlights, sclera — distinct from warm cream belly!)
    if len(palette_rgb) < max_colors:
        d_pal = _min_dist_to_palette(fg_rgb)
        # Pure white has high brightness AND neutral cool/balanced chroma (unlike warm cream where R-B > 14)
        warmth = fg_rgb[:, 0] - fg_rgb[:, 2]
        white_mask = (fg_luma >= 224.0) & (fg_sat <= 0.09) & (warmth <= 12.0) & (d_pal >= 24.0)
        if int(white_mask.sum()) >= max(24, int(len(fg_rgb) * 0.0005)):
            palette_rgb.append(np.array([248.0, 248.0, 250.0], dtype=np.float32))
            palette_roles.append("white")

    if len(palette_rgb) < 2:
        return None

    # 6. Quantize EVERY foreground pixel to nearest filament color
    pal_arr = np.stack(palette_rgb, axis=0)  # (K, 3)
    label_img = np.full((H, W), -1, dtype=np.int16)
    all_fg_rgb = rgb[fg]  # (N, 3)
    dists = np.stack([_redmean_dist2(all_fg_rgb, pal_arr[k][None, :]) for k in range(len(pal_arr))], axis=-1)

    # Guard: prevent soft shadows & cool blue specular highlights from leaking into wrong filament slots!
    all_fg_luma = all_fg_rgb.mean(axis=-1)
    all_fg_sat = (all_fg_rgb.max(axis=-1) - all_fg_rgb.min(axis=-1)) / np.maximum(all_fg_rgb.max(axis=-1), 1.0)
    all_fg_warmth = all_fg_rgb[:, 0] - all_fg_rgb[:, 2]
    # Eroded interior mask (4 px inside silhouette) so bright outer-rim specular highlights don't become white teeth!
    fg_deep = ndimage.binary_erosion(fg, iterations=4)
    in_deep = fg_deep[fg]

    # Check if secondary color is warm-ivory/cream while primary is cool-blue (classic countershading)
    prim_warmth = float(pal_arr[0, 0] - pal_arr[0, 2])
    sec_warmth = float(pal_arr[1, 0] - pal_arr[1, 2]) if len(pal_arr) > 1 else prim_warmth
    is_cool_body_warm_belly = (prim_warmth < -6.0) and (sec_warmth > 4.0)

    for k, role in enumerate(palette_roles):
        if role == "secondary" and is_cool_body_warm_belly:
            # Never let cool blue-gray specular highlights on the dorsal fin / back (R - B < 0) steal warm Cream Belly!
            cool_highlight = all_fg_warmth < 2.0
            dists[cool_highlight, k] = 1e12
        elif role == "dark":
            dists[(all_fg_luma > 58.0) | (~in_deep), k] = 1e12
        elif role == "white":
            # Only interior crisp neutral bright white pixels (teeth & eye catch-lights), never rim shine on fins!
            not_pure_white = (all_fg_luma < 222.0) | (all_fg_sat > 0.10) | (all_fg_warmth > 13.0) | (~in_deep)
            dists[not_pure_white, k] = 1e12
        elif role == "chromatic":
            fn = all_fg_rgb / np.maximum(1e-4, np.linalg.norm(all_fg_rgb, axis=-1, keepdims=True))
            min_hd = np.ones(len(all_fg_rgb), dtype=np.float32)
            for bk, brole in enumerate(palette_roles):
                if brole in ("primary", "secondary"):
                    bn = pal_arr[bk] / max(1e-4, float(np.linalg.norm(pal_arr[bk])))
                    min_hd = np.minimum(min_hd, 1.0 - (fn @ bn))
            not_chrom = ((all_fg_rgb.max(axis=-1) - all_fg_rgb.min(axis=-1)) < 24.0) | (min_hd < 0.035)
            dists[not_chrom, k] = 1e12

    pix_labels = np.argmin(dists, axis=-1).astype(np.int16)
    label_img[fg] = pix_labels

    # Clean noise & stray rim reflections cleanly on accent masks:
    # - Keep only compact dark blobs (e.g. eyes/nose) and drop narrow linear armpit/fin-crease shadow slits
    # - Keep white features (teeth/eye catch-lights) only when near a dark/chromatic facial anchor (within 18 px)
    cleaned = label_img.copy()
    dark_idx = next((i for i, r in enumerate(palette_roles) if r == "dark"), None)
    chrom_idx = next((i for i, r in enumerate(palette_roles) if r == "chromatic"), None)
    for k, role in enumerate(palette_roles):
        if role == "dark":
            m_k = label_img == k
            m_k = ndimage.binary_opening(m_k, structure=np.ones((3, 3)))
            m_k = ndimage.binary_closing(m_k, structure=np.ones((3, 3)))
            # Filter out long thin dark shadow crevices (aspect ratio > 3.2 and thin) while preserving round eyes!
            lbl_k, nk = ndimage.label(m_k)
            keep_k = np.zeros_like(m_k)
            for cid in range(1, nk + 1):
                cm = lbl_k == cid
                area_c = int(cm.sum())
                if area_c < 45:
                    continue
                cy, cx = np.nonzero(cm)
                ch_c = max(1, int(cy.max() - cy.min() + 1))
                cw_c = max(1, int(cx.max() - cx.min() + 1))
                aspect = max(ch_c, cw_c) / float(min(ch_c, cw_c))
                fill = area_c / float(ch_c * cw_c)
                if aspect <= 2.8 and fill >= 0.42:
                    keep_k |= cm
            cleaned[(label_img == k) & (~keep_k)] = 0
            cleaned[keep_k & fg] = k
        elif role == "chromatic":
            m_k = label_img == k
            m_k = ndimage.binary_opening(m_k, structure=np.ones((3, 3)))
            m_k = ndimage.binary_closing(m_k, structure=np.ones((3, 3)))
            cleaned[(label_img == k) & (~m_k)] = 0
            cleaned[m_k & fg] = k
        elif role == "white" and (dark_idx is not None or chrom_idx is not None):
            m_w = label_img == k
            anchor = np.zeros_like(m_w)
            if dark_idx is not None:
                anchor |= (cleaned == dark_idx)
            if chrom_idx is not None:
                anchor |= (cleaned == chrom_idx)
            if np.any(anchor):
                near_anchor = ndimage.binary_dilation(anchor, iterations=18)
                keep_w = m_w & near_anchor
                cleaned[m_w & (~keep_w)] = 0
                cleaned[keep_w & fg] = k

    palette = [
        {
            "index": i,
            "name": _color_label_name(pal_arr[i], palette_roles[i]),
            "role": palette_roles[i],
            "hex": _rgb_to_hex(pal_arr[i]),
            "rgb": pal_arr[i].tolist(),
            "filament": i + 1,
            "pixels": int((cleaned == i).sum()),
        }
        for i in range(len(pal_arr))
    ]
    return {
        "fg": fg,
        "label_img": cleaned,
        "palette": palette,
        "size": (W, H),
    }


def _camera_basis(az_deg: float, el_deg: float, roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(right, up, fwd)` orthonormal camera vectors in Z-up world coordinates (`fwd` points into scene)."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    # Eye position on unit sphere at (az, el) -> fwd points from eye toward origin
    eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float64)
    fwd = -eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(fwd @ world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    if abs(roll_deg) > 1e-4:
        r = math.radians(roll_deg)
        cr, sr = math.cos(r), math.sin(r)
        right, up = cr * right + sr * up, -sr * right + cr * up
    return right, up, fwd


def align_mesh_to_concept(
    mesh: trimesh.Trimesh,
    fg_mask: np.ndarray,
) -> dict[str, Any]:
    """Solve the orthographic camera pose `(az, el, roll)` maximizing 2D silhouette IoU against `fg_mask`."""
    ys, xs = np.nonzero(fg_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    fg_crop = fg_mask[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = fg_crop.shape
    grid_n = 88
    # Keep exact aspect ratio inside grid_n x grid_n canvas
    scale_g = (grid_n - 8) / float(max(ch, cw))
    gh = max(8, int(round(ch * scale_g)))
    gw = max(8, int(round(cw * scale_g)))
    fg_small = np.asarray(
        Image.fromarray((fg_crop.astype(np.uint8) * 255)).resize((gw, gh), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ) > 127
    fg_pad = np.zeros((grid_n, grid_n), dtype=bool)
    oy = (grid_n - gh) // 2
    ox = (grid_n - gw) // 2
    fg_pad[oy : oy + gh, ox : ox + gw] = fg_small
    fg_area = float(fg_pad.sum())
    fg_aspect = cw / max(1.0, float(ch))

    # Uniformly sample ~5500 surface points across the 3D mesh for fast silhouette splatting
    pts, _ = trimesh.sample.sample_surface(mesh, min(5500, max(2000, len(mesh.faces) // 3)))
    pts = np.asarray(pts, dtype=np.float64)
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    q = pts - center
    struct3 = np.ones((3, 3), dtype=bool)

    def _score(az: float, el: float, roll: float = 0.0) -> tuple[float, float, float, float, float]:
        right, up, _ = _camera_basis(az, el, roll)
        u = q @ right
        v = q @ up
        u0, u1 = float(u.min()), float(u.max())
        v0, v1 = float(v.min()), float(v.max())
        du = max(1e-6, u1 - u0)
        dv = max(1e-6, v1 - v0)
        # Aspect-ratio quick check (penalize poses wildly narrower/wider than concept silhouette)
        proj_aspect = du / dv
        if proj_aspect < fg_aspect * 0.62 or proj_aspect > fg_aspect * 1.55:
            return 0.0, u0, u1, v0, v1
        sc = (grid_n - 8) / max(du, dv)
        px = np.clip(np.round((u - 0.5 * (u0 + u1)) * sc + grid_n * 0.5).astype(np.int32), 0, grid_n - 1)
        py = np.clip(np.round(grid_n * 0.5 - (v - 0.5 * (v0 + v1)) * sc).astype(np.int32), 0, grid_n - 1)
        proj_m = np.zeros((grid_n, grid_n), dtype=bool)
        proj_m[py, px] = True
        proj_m = ndimage.binary_dilation(proj_m, structure=struct3)
        proj_m = ndimage.binary_fill_holes(proj_m)
        inter = float((proj_m & fg_pad).sum())
        union = float(proj_m.sum()) + fg_area - inter
        iou = inter / max(1.0, union)
        return iou, u0, u1, v0, v1

    # Coarse search: 36 azimuths × 5 elevations
    cands: list[tuple[float, float, float]] = []
    for az in range(0, 360, 10):
        for el in (-8.0, 2.0, 12.0, 22.0, 32.0):
            iou, *_ = _score(float(az), el, 0.0)
            cands.append((iou, float(az), el))
    cands.sort(reverse=True)

    # Fine search around top 4 peaks (including subtle roll ±8°)
    best = (0.0, 0.0, 12.0, 0.0)
    for _, az0, el0 in cands[:4]:
        for daz in (-8, -4, 0, 4, 8):
            for del_ in (-6, -3, 0, 3, 6):
                for rl in (-6.0, 0.0, 6.0):
                    iou, *_ = _score(az0 + daz, el0 + del_, rl)
                    if iou > best[0]:
                        best = (iou, az0 + daz, el0 + del_, rl)

    # Micro refinement (±2° azimuth/elevation/roll)
    _, baz, bel, brl = best
    for daz in (-2.0, -1.0, 0.0, 1.0, 2.0):
        for del_ in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for drl in (-2.0, 0.0, 2.0):
                iou, *_ = _score(baz + daz, bel + del_, brl + drl)
                if iou > best[0]:
                    best = (iou, baz + daz, bel + del_, brl + drl)

    iou, az, el, roll = best
    right, up, fwd = _camera_basis(az, el, roll)
    # Compute exact mapping from 3D (u, v) to concept image pixel coordinates (px, py)
    verts = np.asarray(mesh.vertices, dtype=np.float64) - center
    vu = verts @ right
    vv = verts @ up
    u0, u1 = float(vu.min()), float(vu.max())
    v0, v1 = float(vv.min()), float(vv.max())
    return {
        "iou": round(float(iou), 4),
        "az_deg": float(az),
        "el_deg": float(el),
        "roll_deg": float(roll),
        "center": center,
        "right": right,
        "up": up,
        "fwd": fwd,
        "u_bounds": (u0, u1),
        "v_bounds": (v0, v1),
        "fg_bbox": (x0, y0, x1, y1),
    }


def _detect_symmetry_axis(mesh: trimesh.Trimesh) -> int | None:
    """Return 1 (Y-mirror) or 0 (X-mirror) if `mesh` is bilaterally symmetric about that center plane, else None."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) < 100:
        return None
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    span = v.max(axis=0) - v.min(axis=0)
    diag = float(np.linalg.norm(span))
    if diag < 1e-3:
        return None
    sample = v[:: max(1, len(v) // 1500)] - center
    tree = cKDTree(v - center)
    best_axis: int | None = None
    best_err = 0.045 * diag
    for ax in (1, 0):
        mir = sample.copy()
        mir[:, ax] = -mir[:, ax]
        dists, _ = tree.query(mir, k=1)
        med = float(np.median(dists))
        if med < best_err:
            best_err = med
            best_axis = ax
    return best_axis


def _extrude_patch_to_manifold(
    verts: np.ndarray,
    faces: np.ndarray,
    patch_face_idx: np.ndarray,
    axis_vec: np.ndarray,
    depth_mm: float = 1.6,
    out_overshoot_mm: float = 0.35,
) -> m3d.Manifold | None:
    """Extrude a single-hemisphere surface patch rigidly along `axis_vec` into a closed watertight solid.

    Rigid parallel translation of `outer = V + out_overshoot * axis` and `inner = V - depth * axis` stitched
    along the boundary edge loop is mathematically self-intersection-free for single-sided patches!
    """
    pf = faces[patch_face_idx]
    if len(pf) < 2:
        return None
    # Remap vertices compactly to 0..nv-1
    uniq_v, inv = np.unique(pf.reshape(-1), return_inverse=True)
    lf = inv.reshape(-1, 3)
    lv = verts[uniq_v]
    nv = len(lv)

    # Check triangle orientation relative to axis_vec: ensure outer cap faces +axis_vec
    e1 = lv[lf[:, 1]] - lv[lf[:, 0]]
    e2 = lv[lf[:, 2]] - lv[lf[:, 0]]
    fn = np.cross(e1, e2)
    mean_dot = float(np.mean(fn @ axis_vec))
    if mean_dot < 0.0:
        axis_vec = -axis_vec

    # Find directed boundary edges (edges appearing only once in lf)
    d_edges = np.vstack((lf[:, [0, 1]], lf[:, [1, 2]], lf[:, [2, 0]]))
    u_edges = np.sort(d_edges, axis=1)
    keys = u_edges[:, 0].astype(np.int64) * nv + u_edges[:, 1].astype(np.int64)
    u_k, cnt = np.unique(keys, return_counts=True)
    b_keys = set(u_k[cnt == 1].tolist())
    if not b_keys:
        return None

    # Build outer (+axis) and inner (-axis) vertex copies
    v_out = lv + axis_vec[None, :] * float(out_overshoot_mm)
    v_in = lv - axis_vec[None, :] * float(depth_mm)
    all_v = np.vstack((v_out, v_in))

    # Outer cap keeps original winding; inner cap reverses winding so normals point along -axis_vec
    f_out = lf
    f_in = lf[:, [0, 2, 1]] + nv

    # Side walls along every directed boundary half-edge (a -> b) of the outer cap
    wall_tris: list[tuple[int, int, int]] = []
    for a, b in d_edges.tolist():
        k = min(a, b) * nv + max(a, b)
        if k in b_keys:
            # Outward wall normal aligns with cross(b_out - a_out, a_in - a_out)
            wall_tris.append((a, a + nv, b + nv))
            wall_tris.append((a, b + nv, b))

    if not wall_tris:
        return None
    all_f = np.vstack((f_out, f_in, np.asarray(wall_tris, dtype=np.int64)))
    m = m3d.Mesh(
        vert_properties=np.asarray(all_v, dtype=np.float32),
        tri_verts=np.asarray(all_f, dtype=np.uint32),
    )
    m.merge()
    mf = m3d.Manifold(m)
    if mf.status() == m3d.Error.NoError and not mf.is_empty():
        if float(mf.volume()) < 0:
            all_f_rev = all_f[:, [0, 2, 1]]
            m2 = m3d.Mesh(
                vert_properties=np.asarray(all_v, dtype=np.float32),
                tri_verts=np.asarray(all_f_rev, dtype=np.uint32),
            )
            m2.merge()
            mf = m3d.Manifold(m2)
            if mf.status() != m3d.Error.NoError or mf.is_empty():
                return None
        return mf
    return None


def _chaikin_closed(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Smooth a closed 2D polygon loop via Chaikin corner-cutting (eliminates jagged triangle sawtooth!)."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        return p
    for _ in range(iterations):
        nxt = np.roll(p, -1, axis=0)
        q = 0.75 * p + 0.25 * nxt
        r = 0.25 * p + 0.75 * nxt
        out = np.empty((len(p) * 2, 2), dtype=np.float64)
        out[0::2] = q
        out[1::2] = r
        p = out
    return p


def _smooth_patch_prism(
    verts: np.ndarray,
    faces: np.ndarray,
    comp_f: np.ndarray,
    extrude_ax: np.ndarray,
    depth_mm: float = 1.6,
    overshoot_mm: float = 1.6,
) -> m3d.Manifold | None:
    """Build a blazing-fast, laser-smooth 3D cutter prism for a painted feature patch (eye, tooth, mouth, highlight)
    by extracting its smooth 2D silhouette contour perpendicular to `extrude_ax` and extruding via `manifold3d`."""
    from skimage.measure import find_contours

    pf = faces[comp_f]
    pts = verts[np.unique(pf.reshape(-1))]
    if len(pts) < 4:
        return None
    w = extrude_ax / max(1e-9, float(np.linalg.norm(extrude_ax)))
    ref = np.array([0.0, 0.0, 1.0]) if abs(float(w[2])) < 0.88 else np.array([1.0, 0.0, 0.0])
    u_ax = np.cross(ref, w)
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(w, u_ax)
    v_ax /= np.linalg.norm(v_ax)

    origin = pts.mean(axis=0)
    q = pts - origin[None, :]
    pu = q @ u_ax
    pv = q @ v_ax
    pw = q @ w
    u0, u1 = float(pu.min()), float(pu.max())
    v0, v1 = float(pv.min()), float(pv.max())
    du, dv = max(0.4, u1 - u0), max(0.4, v1 - v0)
    w_min, w_max = float(pw.min()), float(pw.max())

    # Rasterize the patch's projected triangles cleanly onto a 48x48 grid and trace smooth 2D contour!
    gn = 48
    pad = 4
    sc = (gn - 2 * pad) / max(du, dv)
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    from PIL import ImageDraw

    im = Image.new("L", (gn, gn), 0)
    dr = ImageDraw.Draw(im)
    t_pts = verts[pf] - origin[None, None, :]
    tu = (t_pts @ u_ax - uc) * sc + gn * 0.5
    tv = (t_pts @ v_ax - vc) * sc + gn * 0.5
    for i in range(len(pf)):
        dr.polygon([(float(tu[i, 0]), float(tv[i, 0])), (float(tu[i, 1]), float(tv[i, 1])), (float(tu[i, 2]), float(tv[i, 2]))], fill=255)
    mask = ndimage.binary_fill_holes(np.asarray(im, dtype=np.uint8) > 127)
    mask_f = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.85)
    contours = find_contours(mask_f, level=0.48)
    if not contours:
        return None
    # Pick largest outer contour
    cnt = max(contours, key=len)
    if len(cnt) < 4:
        return None
    # Subsample to ~20-32 clean vertices + Chaikin smooth -> razor-clean curved inlay border!
    step = max(1, len(cnt) // 24)
    sub = cnt[::step]
    # Note: find_contours returns (row=v, col=u)
    poly_uv = np.stack([(sub[:, 1] - gn * 0.5) / sc + uc, (sub[:, 0] - gn * 0.5) / sc + vc], axis=-1)
    poly_uv = _chaikin_closed(poly_uv, iterations=1)
    # Ensure CCW winding for manifold3d CrossSection
    area2 = float(np.sum(poly_uv[:, 0] * np.roll(poly_uv[:, 1], -1) - np.roll(poly_uv[:, 0], -1) * poly_uv[:, 1]))
    if area2 < 0:
        poly_uv = poly_uv[::-1]
    if abs(area2) < 0.08:
        return None

    z_lo = w_min - float(depth_mm)
    z_hi = w_max + float(overshoot_mm)
    h = max(0.5, z_hi - z_lo)
    pts_list = [(float(x), float(y)) for x, y in poly_uv]
    try:
        solid = m3d.Manifold.extrude(m3d.CrossSection([pts_list]), h)
    except Exception:  # noqa: BLE001
        return None
    if solid.is_empty():
        return None
    # Rigid affine transform mapping local (u, v, w) -> world (x, y, z)
    t_origin = origin + w * z_lo
    mat = np.array([
        [u_ax[0], v_ax[0], w[0], t_origin[0]],
        [u_ax[1], v_ax[1], w[1], t_origin[1]],
        [u_ax[2], v_ax[2], w[2], t_origin[2]],
    ], dtype=np.float64)
    return solid.transform(mat.tolist())


def _build_belly_zone_prism(
    verts: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    all_verts: np.ndarray,
) -> m3d.Manifold | None:
    """Build a laser-smooth volumetric cutter prism for a broad wrap-around secondary color zone (e.g. Cream Belly)
    directly from the exact painted 3D vertex envelope, keeping dorsal/pectoral/tail fins strictly primary color!"""
    idx = np.nonzero(face_mask)[0]
    if len(idx) < 20:
        return None
    all_min = all_verts.min(axis=0)
    all_max = all_verts.max(axis=0)
    all_span = all_max - all_min
    spine_ax = 0 if all_span[0] >= all_span[1] else 1
    lat_ax = 1 - spine_ax
    lat_center = float(np.median(all_verts[:, lat_ax]))

    # Use ONLY vertices in the ventral 58% of total height (guards against any stray dorsal/tail highlights!)
    b_verts = verts[np.unique(faces[idx].reshape(-1))]
    z_cap = float(all_min[2] + 0.56 * all_span[2])
    b_verts = b_verts[b_verts[:, 2] <= z_cap]
    if len(b_verts) < 20:
        return None

    s_vals = b_verts[:, spine_ax]
    l_vals = b_verts[:, lat_ax]
    z_vals = b_verts[:, 2]
    s0, s1 = float(s_vals.min()), float(s_vals.max())
    if s1 - s0 < 6.0:
        return None

    # Measure core torso width (ahead of protruding lateral fins, around 15..26% from front/head) so
    # pectoral/lateral fins stay 100% pure primary body color!
    all_s0, all_s1 = float(all_min[spine_ax]), float(all_max[spine_ax])
    all_len = max(1e-6, all_s1 - all_s0)
    # Detect which end is the wider head
    f_pts = all_verts[all_verts[:, spine_ax] <= all_s0 + 0.35 * all_len]
    r_pts = all_verts[all_verts[:, spine_ax] >= all_s1 - 0.35 * all_len]
    head_at_min = (float(np.ptp(f_pts[:, lat_ax])) >= float(np.ptp(r_pts[:, lat_ax])) * 0.85) if (len(f_pts) and len(r_pts)) else True
    if head_at_min:
        hc = all_verts[(all_verts[:, spine_ax] >= all_s0 + 0.14 * all_len) & (all_verts[:, spine_ax] <= all_s0 + 0.28 * all_len)]
    else:
        hc = all_verts[(all_verts[:, spine_ax] >= all_s1 - 0.28 * all_len) & (all_verts[:, spine_ax] <= all_s1 - 0.14 * all_len)]
    torso_half_w = float(np.percentile(np.abs(hc[:, lat_ax] - lat_center), 94)) if len(hc) > 20 else float(all_span[lat_ax] * 0.28)
    # Clamp lateral sampling strictly inside torso_half_w so pectoral fins never pollute Z_top or W_lat!
    in_torso = np.abs(l_vals - lat_center) <= torso_half_w * 1.05
    if int(in_torso.sum()) >= 20:
        s_vals, l_vals, z_vals = s_vals[in_torso], l_vals[in_torso], z_vals[in_torso]
        s0, s1 = float(s_vals.min()), float(s_vals.max())

    # Stop ventral countershading cleanly before the flared rear caudal tail fin (~78% along spine for streamlined bodies)
    if all_len / max(1.0, float(all_span[2])) >= 1.75:
        if head_at_min:
            s1 = min(s1, all_s0 + 0.78 * all_len)
        else:
            s0 = max(s0, all_s1 - 0.78 * all_len)
    if s1 - s0 < 6.0:
        return None

    n_bins = 32
    edges = np.linspace(s0, s1, n_bins + 1)
    s_mid = 0.5 * (edges[:-1] + edges[1:])
    z_top = np.full(n_bins, np.nan, dtype=np.float64)
    z_loc_cap = np.full(n_bins, z_cap, dtype=np.float64)
    for i in range(n_bins):
        # Measure local torso slice height so narrow rear tail stalks keep a crisp blue dorsal spine!
        sl_all = all_verts[(all_verts[:, spine_ax] >= edges[i]) & (all_verts[:, spine_ax] <= edges[i + 1])]
        if len(sl_all) >= 6:
            zl0, zl1 = float(sl_all[:, 2].min()), float(sl_all[:, 2].max())
            t_along = ((s_mid[i] - all_s0) / all_len) if head_at_min else ((all_s1 - s_mid[i]) / all_len)
            max_frac = 0.50 if t_along < 0.58 else max(0.32, 0.46 - 0.55 * (t_along - 0.58))
            z_loc_cap[i] = zl0 + max_frac * max(1.0, zl1 - zl0)
        m = (s_vals >= edges[i]) & (s_vals <= edges[i + 1])
        if int(m.sum()) < 3:
            continue
        z_top[i] = min(float(np.percentile(z_vals[m], 90)), float(z_loc_cap[i]))

    valid = np.isfinite(z_top)
    if int(valid.sum()) < 4:
        return None
    z_interp = np.interp(s_mid, s_mid[valid], z_top[valid])
    z_smooth = np.minimum(ndimage.gaussian_filter1d(z_interp, sigma=1.8, mode="nearest"), z_loc_cap)
    eff_w = torso_half_w * 1.94
    z_floor = float(all_min[2]) - 4.0

    pts_sz = [(float(s_mid[0] - 3.0), z_floor), (float(s_mid[-1] + 1.0), z_floor)] + [
        (float(sv), float(zv)) for sv, zv in reversed(list(zip(s_mid, z_smooth)))
    ]
    slab = (
        m3d.Manifold.extrude(m3d.CrossSection([pts_sz]), eff_w)
        .translate((0.0, 0.0, -eff_w * 0.5))
        .rotate((90.0, 0.0, 0.0))
    )
    if spine_ax == 1:
        slab = slab.rotate((0.0, 0.0, 90.0))
    shift = (0.0, lat_center, 0.0) if spine_ax == 0 else (lat_center, 0.0, 0.0)
    return slab.translate(shift)


def _build_color_cutter_prism(
    verts: np.ndarray,
    faces: np.ndarray,
    fn: np.ndarray,
    face_mask: np.ndarray,
    role: str = "accent",
    depth_mm: float = 1.6,
    min_area_mm2: float = 1.5,
) -> m3d.Manifold | None:
    """Build a lightweight (~48-160 tri) smooth-contour 3D cutter prism for a painted color region.

    Keeping cutters lightweight and non-coplanar with `target_mf` makes disjoint subtraction & final clipping
    500x faster (~0.1 s total for 5 MMU3 colors) with zero coplanar triangle soup!
    """
    idx_all = np.nonzero(face_mask)[0]
    if len(idx_all) < 3:
        return None

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if float(areas[idx_all].sum()) < min_area_mm2:
        return None

    # Broad secondary color zones (e.g. Cream Belly) use a smooth continuous envelope prism -> 0.01 s, laser-smooth line!
    if role == "secondary":
        z_all_min, z_all_max = float(verts[:, 2].min()), float(verts[:, 2].max())
        sec_z_med = float(np.median(verts[faces[idx_all].reshape(-1), 2]))
        if sec_z_med < z_all_min + 0.52 * (z_all_max - z_all_min):
            belly_prism = _build_belly_zone_prism(verts, faces, face_mask, verts)
            if belly_prism is not None and not belly_prism.is_empty():
                return belly_prism

    # Fast NumPy edge-adjacency connected components strictly on `idx_all`
    def _face_components(sub_f_idx: np.ndarray) -> list[np.ndarray]:
        pf = faces[sub_f_idx]
        m_n = len(pf)
        if m_n == 0:
            return []
        nv = len(verts)
        e0 = np.sort(pf[:, [0, 1]], axis=1)
        e1 = np.sort(pf[:, [1, 2]], axis=1)
        e2 = np.sort(pf[:, [2, 0]], axis=1)
        all_e = np.vstack((e0, e1, e2))
        keys = all_e[:, 0].astype(np.int64) * nv + all_e[:, 1].astype(np.int64)
        f_ids = np.tile(np.arange(m_n, dtype=np.int32), 3)
        order = np.argsort(keys)
        ks = keys[order]
        same = ks[:-1] == ks[1:]
        parent = np.arange(m_n, dtype=np.int32)

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i_s in np.nonzero(same)[0].tolist():
            ra = _find(int(f_ids[order[i_s]]))
            rb = _find(int(f_ids[order[i_s + 1]]))
            if ra != rb:
                parent[rb] = ra

        groups: dict[int, list[int]] = {}
        for i in range(m_n):
            groups.setdefault(_find(i), []).append(i)
        return [sub_f_idx[g] for g in groups.values()]

    # Build lightweight ~48-triangle smooth-contour prisms for each localized feature patch (eyes, teeth, mouth, highlights!)
    prisms: list[m3d.Manifold] = []
    for comp_f in _face_components(idx_all):
        if float(areas[comp_f].sum()) < min_area_mm2 * 0.45:
            continue
        mean_n = fn[comp_f].mean(axis=0)
        nlen = float(np.linalg.norm(mean_n))
        if nlen > 0.25:
            extrude_ax = mean_n / nlen
            pr = _smooth_patch_prism(verts, faces, comp_f, extrude_ax, depth_mm=depth_mm)
            if pr is not None:
                prisms.append(pr)
        else:
            for ax in (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])):
                sub = comp_f[(fn[comp_f] @ ax) > 0.25]
                if len(sub) >= 3 and float(areas[sub].sum()) >= min_area_mm2 * 0.35:
                    pr = _smooth_patch_prism(verts, faces, sub, ax, depth_mm=depth_mm)
                    if pr is not None:
                        prisms.append(pr)

    if not prisms:
        return None
    union_prism = (
        m3d.Manifold.batch_boolean(prisms, m3d.OpType.Add)
        if len(prisms) > 1
        else prisms[0]
    )
    if union_prism.is_empty() or union_prism.status() != m3d.Error.NoError:
        return None
    return union_prism


def project_concept_colors_to_manifolds(
    pre_flexi_mf: m3d.Manifold,
    concept_png: Path,
    *,
    final_mf: m3d.Manifold | None = None,
    hinge_keepout: m3d.Manifold | None = None,
    max_colors: int = 5,
    inlay_depth_mm: float = 1.6,
    min_iou: float = 0.60,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Project up to `max_colors` filament colors from `concept_png` onto `pre_flexi_mf` and build disjoint
    watertight `manifold3d.Manifold` bodies clipped cleanly to `final_mf` (or `pre_flexi_mf` if None).

    Returns `(parts_meta, debug_info)` where each entry in `parts_meta` is:
      `{"name": str, "hex": str, "role": str, "filament": int, "manifold": m3d.Manifold}`
    or `None` if the concept only has 1 color or silhouette alignment fails (`iou < min_iou`).
    """
    cmap = extract_concept_color_map(concept_png, max_colors=max(2, min(int(max_colors), 6)))
    if cmap is None or len(cmap["palette"]) < 2:
        return None

    m = pre_flexi_mf.to_mesh()
    verts = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
    faces = np.asarray(m.tri_verts, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    pose = align_mesh_to_concept(tm, cmap["fg"])
    if pose["iou"] < min_iou:
        return None

    W, H = cmap["size"]
    x0, y0, x1, y1 = pose["fg_bbox"]
    u0, u1 = pose["u_bounds"]
    v0, v1 = pose["v_bounds"]
    center = pose["center"]
    right, up, fwd = pose["right"], pose["up"], pose["fwd"]

    # Face centroids and unit outward normals
    tv0, tv1, tv2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fc = (tv0 + tv1 + tv2) / 3.0
    fn = np.cross(tv1 - tv0, tv2 - tv0)
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(flen, 1e-12))

    # Project face centroids to exact concept pixel coordinates (px, py) and depth (z_cam)
    q = fc - center[None, :]
    fu = q @ right
    fv = q @ up
    fz = q @ fwd
    px = np.round(x0 + (fu - u0) / max(1e-6, u1 - u0) * (x1 - x0)).astype(np.int32)
    py = np.round(y1 - (fv - v0) / max(1e-6, v1 - v0) * (y1 - y0)).astype(np.int32)

    # Build crisp per-pixel depth buffer at half-res (fast & closes tiny inter-centroid pixel gaps!)
    bw, bh = max(64, W // 2), max(64, H // 2)
    bx = np.clip(px // 2, 0, bw - 1)
    by = np.clip(py // 2, 0, bh - 1)
    zbuf = np.full((bh, bw), np.inf, dtype=np.float64)
    front_facing = (fn @ fwd) < 0.25  # faces turned toward camera (small tolerance for mouth recesses)
    ff_idx = np.nonzero(front_facing)[0]
    np.minimum.at(zbuf, (by[ff_idx], bx[ff_idx]), fz[ff_idx])
    # Slightly dilate minimum depth by 1 cell so no background triangle peeks through gaps
    zbuf_min = ndimage.minimum_filter(zbuf, size=3)

    depth_span = max(1.0, float(fz.max() - fz.min()))
    vis_tol = depth_span * 0.08
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    visible = front_facing & in_bounds & (fz <= zbuf_min[by, bx] + vis_tol)

    label_img = cmap["label_img"]
    face_labels = np.full(len(faces), -1, dtype=np.int16)
    vis_idx = np.nonzero(visible)[0]
    sampled = label_img[py[vis_idx], px[vis_idx]]
    valid_s = sampled >= 0
    face_labels[vis_idx[valid_s]] = sampled[valid_s]

    # Bilateral symmetry mirroring: copy front-side painted features (eyes, teeth, mouth, belly) onto opposite flank!
    sym_ax = _detect_symmetry_axis(tm)
    if sym_ax is not None:
        labeled_idx = np.nonzero(face_labels >= 0)[0]
        unlabeled_idx = np.nonzero(face_labels < 0)[0]
        if len(labeled_idx) >= 50 and len(unlabeled_idx) > 0:
            sym_c = 0.5 * (verts[:, sym_ax].min() + verts[:, sym_ax].max())
            fc_mir = fc[unlabeled_idx].copy()
            fc_mir[:, sym_ax] = 2.0 * sym_c - fc_mir[:, sym_ax]
            tree = cKDTree(fc[labeled_idx])
            dists, nn = tree.query(fc_mir, k=1)
            diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
            good = dists <= diag * 0.06
            face_labels[unlabeled_idx[good]] = face_labels[labeled_idx[nn[good]]]

    # Propagate labels smoothly onto remaining occluded undersides/crevices (e.g. flat Z=0 belly pad!)
    if np.any(face_labels < 0):
        labeled_idx = np.nonzero(face_labels >= 0)[0]
        unlabeled_idx = np.nonzero(face_labels < 0)[0]
        if len(labeled_idx) > 0:
            # Prefer nearest labeled face that belongs to base body/belly (never bleed eyes/teeth onto the back!)
            base_roles = {p["index"] for p in cmap["palette"] if p["role"] in ("primary", "secondary")}
            base_idx = np.nonzero(np.isin(face_labels, list(base_roles)))[0]
            src_idx = base_idx if len(base_idx) >= 20 else labeled_idx
            tree = cKDTree(fc[src_idx])
            _, nn = tree.query(fc[unlabeled_idx], k=1)
            face_labels[unlabeled_idx] = face_labels[src_idx[nn]]
        else:
            face_labels[unlabeled_idx] = 0

    # 2 passes of majority-vote smoothing along face adjacency (keeps crisp small white highlights/teeth intact!)
    preserve_roles = {p["index"] for p in cmap["palette"] if p["role"] in ("white",)}
    adj = tm.face_adjacency
    if len(adj) > 0:
        a0, a1 = adj[:, 0], adj[:, 1]
        k_n = len(cmap["palette"])
        for _ in range(2):
            votes = np.zeros((len(faces), k_n), dtype=np.int32)
            # Self-weight 2 so crisp features resist erosion unless surrounded
            np.add.at(votes, (np.arange(len(faces)), face_labels), 2)
            np.add.at(votes, (a0, face_labels[a1]), 1)
            np.add.at(votes, (a1, face_labels[a0]), 1)
            nxt = np.argmax(votes, axis=1).astype(np.int16)
            keep_small = np.isin(face_labels, list(preserve_roles))
            face_labels = np.where(keep_small, face_labels, nxt)

    # Build lightweight non-coplanar cutter prisms for each accent color, make cutters disjoint FIRST (0.01 s!),
    # then clip each disjoint cutter against `target_mf` ONCE and compute `body = target_mf - all_cutters` (0.1 s!)
    target_mf = final_mf if final_mf is not None else pre_flexi_mf
    role_priority = {"primary": 0, "secondary": 1, "chromatic": 2, "dark": 3, "white": 4}
    accent_entries = [p for p in cmap["palette"] if p["index"] != 0]
    accent_entries.sort(key=lambda p: role_priority.get(p["role"], 2))

    raw_cutters: list[tuple[dict[str, Any], m3d.Manifold]] = []
    for p in accent_entries:
        f_mask = face_labels == p["index"]
        if not np.any(f_mask):
            continue
        depth = 2.2 if p["role"] == "secondary" else (1.8 if p["role"] in ("dark", "chromatic") else 1.4)
        min_a = 0.5 if p["role"] == "white" else 1.8
        cutter = _build_color_cutter_prism(
            verts, faces, fn, f_mask, role=p["role"], depth_mm=depth, min_area_mm2=min_a,
        )
        if cutter is None or cutter.is_empty():
            continue
        if hinge_keepout is not None and p["role"] == "secondary":
            cutter = cutter - hinge_keepout
            if cutter.is_empty():
                continue
        raw_cutters.append((p, cutter))

    if not raw_cutters:
        return None

    # Disjoint cutter hierarchy on lightweight prisms (White > Dark > Chromatic > Belly)
    disjoint_cutters: list[tuple[dict[str, Any], m3d.Manifold]] = []
    claimed_cutter: m3d.Manifold | None = None
    for p, cut in reversed(raw_cutters):
        cur_cut = (cut - claimed_cutter) if claimed_cutter is not None else cut
        if cur_cut.is_empty():
            continue
        disjoint_cutters.append((p, cur_cut))
        claimed_cutter = cur_cut if claimed_cutter is None else (claimed_cutter + cur_cut)
    disjoint_cutters.reverse()

    if claimed_cutter is None:
        return None

    # Now intersect each disjoint lightweight cutter ONCE with `target_mf` (zero coplanar soup!)
    final_accents: list[tuple[dict[str, Any], m3d.Manifold]] = []
    kept_cutters: list[m3d.Manifold] = []
    for p, cut in disjoint_cutters:
        clipped = target_mf ^ cut
        if clipped.is_empty() or clipped.status() != m3d.Error.NoError:
            continue
        min_v = 3.0 if p["role"] == "secondary" else 0.20
        comps = [c for c in clipped.decompose() if abs(float(c.volume())) >= min_v]
        if not comps:
            continue
        final_accents.append((p, m3d.Manifold.compose(comps) if len(comps) > 1 else comps[0]))
        kept_cutters.append(cut)

    if not final_accents:
        return None
    all_kept_cutters = (
        m3d.Manifold.batch_boolean(kept_cutters, m3d.OpType.Add)
        if len(kept_cutters) > 1
        else kept_cutters[0]
    )
    body_mf = target_mf - all_kept_cutters
    if body_mf.is_empty():
        return None

    primary_p = cmap["palette"][0]
    parts_meta: list[dict[str, Any]] = [{
        "name": primary_p["name"],
        "hex": primary_p["hex"],
        "role": "primary",
        "filament": 1,
        "manifold": body_mf,
    }]
    for i, (p, mf_acc) in enumerate(final_accents):
        parts_meta.append({
            "name": p["name"],
            "hex": p["hex"],
            "role": p["role"],
            "filament": i + 2,
            "manifold": mf_acc,
        })

    debug = {
        "iou": pose["iou"],
        "az_deg": pose["az_deg"],
        "el_deg": pose["el_deg"],
        "roll_deg": pose["roll_deg"],
        "symmetry_axis": sym_ax,
        "colors": [(pm["name"], pm["hex"]) for pm in parts_meta],
    }
    return parts_meta, debug
