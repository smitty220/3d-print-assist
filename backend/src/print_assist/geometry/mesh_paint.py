"""Per-triangle MMU3 / AMS painting, UV-to-filament quantization, smooth countershading,
and interactive 3D crease-guided Smart-Fill & Brush painting for Character Mode sculpts.

Unlike destructive boolean color prisms that slice delicate 1–2 mm sculpted teeth and bleed through
thin lips, per-triangle MMU3 segmentation preserves ONE 100% watertight articulated solid and exports
native PrusaSlicer / OrcaSlicer / Bambu Studio `slic3rpe:mmu_segmentation` attributes in `.3mf`.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def encode_mmu_segmentation(slot: int) -> str:
    """Encode a 1-based extruder/filament slot for an unsplit triangle in PrusaSlicer / Bambu Studio 3MF.

    Verified directly against `TriangleSelector::serialize()` and `FacetsAnnotation::get_triangle_as_string()`:
      - Slot 1 (default object extruder): "" (omit attribute)
      - Slot 2: "8"
      - Slot 3..17: f"{slot - 3:X}C"  (3 -> "0C", 4 -> "1C", 5 -> "2C", 6 -> "3C", ...)
    """
    s = int(slot)
    if s <= 1:
        return ""
    if s == 2:
        return "8"
    n = s - 3
    nibbles = ["C"]
    while n >= 15:
        nibbles.append("F")
        n -= 15
    nibbles.append(f"{n:X}")
    return "".join(reversed(nibbles))


def normalize_hex(col: str | None, fallback: str = "#527292") -> str:
    if not col:
        return fallback
    s = col.strip()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 7:
        try:
            int(s[1:], 16)
            return s.lower()
        except ValueError:
            pass
    return fallback


def hex_to_rgb(hx: str) -> np.ndarray:
    s = normalize_hex(hx).lstrip("#")
    return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float32)


def rgb_to_hex(rgb: Iterable[float]) -> str:
    r, g, b = [int(np.clip(round(float(v)), 0, 255)) for v in rgb][:3]
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB `(..., 3)` in `[0, 255]` to CIE Lab (D65) in pure NumPy."""
    c = np.clip(np.asarray(rgb, dtype=np.float32) / 255.0, 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
    X = (lin[..., 0] * 0.4124564 + lin[..., 1] * 0.3575761 + lin[..., 2] * 0.1804375) / 0.95047
    Y = (lin[..., 0] * 0.2126729 + lin[..., 1] * 0.7151522 + lin[..., 2] * 0.0721750) / 1.00000
    Z = (lin[..., 0] * 0.0193339 + lin[..., 1] * 0.1191920 + lin[..., 2] * 0.9503041) / 1.08883
    eps = 0.008856
    kappa = 903.3
    fx = np.where(X > eps, np.cbrt(np.maximum(X, 0.0)), (kappa * X + 16.0) / 116.0)
    fy = np.where(Y > eps, np.cbrt(np.maximum(Y, 0.0)), (kappa * Y + 16.0) / 116.0)
    fz = np.where(Z > eps, np.cbrt(np.maximum(Z, 0.0)), (kappa * Z + 16.0) / 116.0)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def sample_mesh_face_rgb(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Extract per-triangle sRGB `(F, 3) float32` in `[0, 255]` from a UV-textured or vertex-colored mesh.

    Returns `None` if the mesh has no texture/color variation (e.g. untextured monochrome sculpt).
    """
    vis = getattr(mesh, "visual", None)
    if vis is None or len(mesh.faces) == 0:
        return None

    # 1. UV texture map (e.g. baked by Hunyuan3D-Paint)
    uv = getattr(vis, "uv", None)
    mat = getattr(vis, "material", None)
    tex_img = None
    if mat is not None:
        tex_img = getattr(mat, "image", None)
        if tex_img is None:
            tex_img = getattr(mat, "baseColorTexture", None)
    if uv is not None and len(uv) == len(mesh.vertices) and tex_img is not None:
        arr = np.asarray(tex_img.convert("RGB"), dtype=np.float32)
        H, W, _ = arr.shape
        if H >= 8 and W >= 8:
            uv_arr = np.asarray(uv, dtype=np.float32)
            tri_uv = uv_arr[mesh.faces]  # (F, 3, 2)
            # Sample centroid + 3 vertices (4 samples per triangle) to avoid edge aliasing while staying crisp
            c_uv = tri_uv.mean(axis=1, keepdims=True)
            pts_uv = np.concatenate([c_uv, 0.65 * tri_uv + 0.35 * c_uv], axis=1)  # (F, 4, 2)
            u = np.mod(pts_uv[..., 0], 1.0)
            v = np.mod(pts_uv[..., 1], 1.0)
            px = np.clip(np.rint(u * (W - 1)).astype(np.int32), 0, W - 1)
            py = np.clip(np.rint((1.0 - v) * (H - 1)).astype(np.int32), 0, H - 1)
            samples = arr[py, px]  # (F, 4, 3)
            # Weight centroid higher for clean crisp features
            face_rgb = (0.46 * samples[:, 0] + 0.18 * (samples[:, 1] + samples[:, 2] + samples[:, 3])).astype(np.float32)
            if float(face_rgb.std()) > 6.5:
                return face_rgb

    # 2. Per-face or per-vertex colors
    fcols = getattr(vis, "face_colors", None)
    if fcols is not None and len(fcols) == len(mesh.faces):
        fc = np.asarray(fcols, dtype=np.float32)[:, :3]
        if float(fc.std()) > 6.5:
            return fc
    vcols = getattr(vis, "vertex_colors", None)
    if vcols is not None and len(vcols) == len(mesh.vertices):
        vc = np.asarray(vcols, dtype=np.float32)[:, :3]
        if float(vc.std()) > 6.5:
            return vc[mesh.faces].mean(axis=1)
    return None


def build_face_adjacency(points: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute shared-edge face adjacency `(f1, f2, edge_cos, face_normals)` in `O(F log F)` vectorized NumPy."""
    n_verts = len(points)
    n_faces = len(tris)
    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(flen, 1e-12))

    if n_faces < 2:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, fn.astype(np.float32)

    all_e = np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    ea = np.minimum(all_e[:, 0], all_e[:, 1]).astype(np.int64)
    eb = np.maximum(all_e[:, 0], all_e[:, 1]).astype(np.int64)
    f_idx = np.tile(np.arange(n_faces, dtype=np.int32), 3)
    keys = ea * max(1, n_verts) + eb
    order = np.argsort(keys)
    ks = keys[order]
    same = ks[:-1] == ks[1:]
    if not np.any(same):
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, fn.astype(np.float32)

    o1 = order[:-1][same]
    o2 = order[1:][same]
    f1 = f_idx[o1]
    f2 = f_idx[o2]
    dots = np.clip(np.einsum("ij,ij->i", fn[f1], fn[f2]), -1.0, 1.0).astype(np.float32)
    return f1, f2, dots, fn.astype(np.float32)


def transfer_face_rgb_to_target(
    src_points: np.ndarray,
    src_tris: np.ndarray,
    src_face_rgb: np.ndarray,
    dst_points: np.ndarray,
    dst_tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer per-triangle RGB from pre-flexi textured mesh to post-flexi target mesh via nearest surface centroid.

    Returns `(dst_rgb (F_dst, 3), dist_mm (F_dst,))`. Triangles far from the original outer skin
    (e.g. newly cut internal flexi walls & hinge channels) have larger `dist_mm` so they can be forced to Slot 1.
    """
    src_c = src_points[src_tris].mean(axis=1)
    dst_c = dst_points[dst_tris].mean(axis=1)
    tree = cKDTree(src_c)
    dists, idxs = tree.query(dst_c, k=1)
    return src_face_rgb[idxs].astype(np.float32), dists.astype(np.float32)


def _shading_invariant_feat(lab: np.ndarray) -> np.ndarray:
    """Map CIE Lab `(..., 3)` into a shading-invariant filament feature space.

    Ambient-occlusion shading on 3D painted textures shifts luminance `L` smoothly across a single
    filament (e.g. lit slate blue L=50 vs shadowed flank slate blue L=30, or lit cream L=90 vs
    shadowed belly tan L=60) WITHOUT changing hue `(a, b)`. Compressing mid-tone `L` while
    stretching true Jet Black (`L < 24`) and Pure White (`L > 86, low chroma`) ensures:
      - Lit + shadowed regions of the same plastic stay ONE clean uniform filament slot (zero stripes!)
      - Small crisp Jet-Black pupils & Pure-White teeth/sclera ALWAYS get dedicated filament slots!
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.hypot(a, b)
    # Gently compress mid-tone L (0.36x) so lit & shadowed plastic of the same hue merge cleanly,
    # while expanding extreme dark (L < 26) and pure crisp white (L > 86 & chroma < 14) so pupils & teeth pop!
    L_feat = 50.0 + (L - 50.0) * 0.38
    dark_boost = np.clip((19.0 - L) / 11.0, 0.0, 1.8) * 28.0
    white_boost = np.clip((L - 87.0) / 9.0, 0.0, 1.5) * np.clip((13.0 - chroma) / 9.0, 0.0, 1.0) * 22.0
    L_feat = L_feat - dark_boost + white_boost
    return np.stack([L_feat, a * 1.35, b * 1.35], axis=-1).astype(np.float32)


def quantize_face_colors_to_filaments(
    points: np.ndarray,
    tris: np.ndarray,
    face_rgb: np.ndarray,
    *,
    max_colors: int = 5,
    protected_mask: np.ndarray | None = None,
    crease_deg: float = 24.0,
    min_delta_e: float = 22.0,
    min_cluster_frac: float = 0.0008,
) -> tuple[np.ndarray, dict[int, str], list[str]]:
    """Quantize continuous 3D surface RGB into up to `max_colors` crisp PLA filament slots with
    shading-invariant chromaticity clustering + 3D crease-respecting Potts smoothing (color boundaries
    snap cleanly to real 3D eye/tooth/lip creases with ZERO shading stripes!).

    Returns `(face_filaments [F] int32 1..K, palette {slot: '#rrggbb'}, summary_notes)`.
    """
    n_faces = len(tris)
    k_max = max(1, min(8, int(max_colors)))
    if n_faces == 0:
        return np.ones(0, dtype=np.int32), {1: "#527292"}, []

    v0, v1, v2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    areas = (0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)).astype(np.float32)
    areas = np.maximum(areas, 1e-6)

    prot = np.zeros(n_faces, dtype=bool) if protected_mask is None else np.asarray(protected_mask, dtype=bool)
    outer_idx = np.nonzero(~prot)[0]
    if len(outer_idx) < 20:
        outer_idx = np.arange(n_faces, dtype=np.int32)

    lab_all = rgb_to_lab(face_rgb)
    feat_all = _shading_invariant_feat(lab_all)
    rgb_out = face_rgb[outer_idx]
    lab_out = lab_all[outer_idx]
    feat_out = feat_all[outer_idx]
    area_out = areas[outer_idx]
    tot_area = float(area_out.sum())

    if k_max == 1:
        mean_rgb = np.average(rgb_out, axis=0, weights=area_out)
        hx = rgb_to_hex(mean_rgb)
        return np.ones(n_faces, dtype=np.int32), {1: hx}, [f"single-color ({hx})"]

    # Deterministic subsample for fast clustering
    rng = np.random.default_rng(42)
    n_sub = min(16000, len(outer_idx))
    sub_sel = rng.choice(len(outer_idx), size=n_sub, replace=False) if len(outer_idx) > n_sub else np.arange(len(outer_idx))
    sub_feat = feat_out[sub_sel]
    sub_lab = lab_out[sub_sel]
    sub_w = area_out[sub_sel]
    sub_tot_w = float(sub_w.sum())

    # 1. Seed 0: Dominant body filament (largest shading-invariant mode in chromaticity space)
    bins = np.rint(sub_feat / 11.0).astype(np.int32)
    uniq_b, inv_b = np.unique(bins, axis=0, return_inverse=True)
    bin_w = np.bincount(inv_b, weights=sub_w)
    dom_b = int(np.argmax(bin_w))
    c0 = np.average(sub_feat[inv_b == dom_b], axis=0, weights=sub_w[inv_b == dom_b]).astype(np.float32)
    centers: list[np.ndarray] = [c0]

    # 2. Dedicated high-contrast feature detectors (never let huge lit/shadow body regions crowd out tiny
    #    Jet-Black eyes/pupils, Pure-White teeth/sclera, or vivid Red/Coral tongues!)
    if k_max >= 3:
        # Jet Black (pupils / dark accents): L < 17, low chroma (excludes soft ambient-occlusion shadows!)
        is_black = (sub_lab[:, 0] < 17.5) & (np.hypot(sub_lab[:, 1], sub_lab[:, 2]) < 14.0)
        if float(sub_w[is_black].sum()) >= sub_tot_w * 0.0012:
            cb = np.average(sub_feat[is_black], axis=0, weights=sub_w[is_black]).astype(np.float32)
            if float(np.linalg.norm(cb - c0)) >= 20.0:
                centers.append(cb)
        # Pure Bright White (teeth / eye sclera): L > 88, low chroma (< 10 — distinct from warm cream belly!)
        is_white = (sub_lab[:, 0] > 88.0) & (np.hypot(sub_lab[:, 1], sub_lab[:, 2]) < 10.5)
        if len(centers) < k_max and float(sub_w[is_white].sum()) >= sub_tot_w * 0.0020:
            cw = np.average(sub_feat[is_white], axis=0, weights=sub_w[is_white]).astype(np.float32)
            if all(float(np.linalg.norm(cw - c)) >= 18.0 for c in centers):
                centers.append(cw)
        # Vivid Chromatic Accent (e.g. red/pink mouth/tongue, gold badge, bright green trim): chroma > 26
        is_vivid = np.hypot(sub_lab[:, 1], sub_lab[:, 2]) > 26.0
        if len(centers) < k_max and float(sub_w[is_vivid].sum()) >= sub_tot_w * 0.0015:
            cv = np.average(sub_feat[is_vivid], axis=0, weights=sub_w[is_vivid]).astype(np.float32)
            if all(float(np.linalg.norm(cv - c)) >= 24.0 for c in centers):
                centers.append(cv)

    # 3. Fill remaining slots (e.g. Warm Cream Belly, secondary color zones) with farthest distinct hue modes
    #    Require strong shading-invariant separation (`>= min_delta_e`) so light/dark shading of the same plastic
    #    NEVER splits into two redundant slots!
    min_seed_w = sub_tot_w * max(0.015, min_cluster_frac * 4.0)
    while len(centers) < k_max:
        c_arr = np.stack(centers, axis=0)
        d2 = np.sum((sub_feat[:, None, :] - c_arr[None, :, :]) ** 2, axis=-1)
        min_d = np.sqrt(d2.min(axis=1))
        best_score = 0.0
        best_center = None
        for bi in range(len(uniq_b)):
            if bin_w[bi] < min_seed_w:
                continue
            m_bi = inv_b == bi
            mean_d = float(np.average(min_d[m_bi], weights=sub_w[m_bi]))
            if mean_d < min_delta_e:
                continue
            cand = np.average(sub_feat[m_bi], axis=0, weights=sub_w[m_bi]).astype(np.float32)
            # Reject pure luminance-only shadows of an existing center (nearly identical chromaticity a*, b*!)
            same_hue_shadow = any(
                math.hypot(float(cand[1] - ex[1]), float(cand[2] - ex[2])) < 11.0
                and abs(float(cand[0] - ex[0])) < 24.0
                for ex in centers
            )
            if same_hue_shadow:
                continue
            score = mean_d * (float(bin_w[bi]) ** 0.35)
            if score > best_score:
                best_score = score
                best_center = cand
        if best_center is None:
            break
        centers.append(best_center)

    # Gentle K-means refinement (only 4 steps so locked black/white eye/teeth centers don't drift into grey!)
    c_arr = np.stack(centers, axis=0)
    for _ in range(4):
        d2 = np.sum((sub_feat[:, None, :] - c_arr[None, :, :]) ** 2, axis=-1)
        lbl = np.argmin(d2, axis=1)
        for ci in range(len(c_arr)):
            m = lbl == ci
            if np.any(m):
                nc = np.average(sub_feat[m], axis=0, weights=sub_w[m]).astype(np.float32)
                # Dampen drift on tiny extreme anchors
                alpha = 0.55 if float(sub_w[m].sum()) < sub_tot_w * 0.04 else 0.85
                c_arr[ci] = (1.0 - alpha) * c_arr[ci] + alpha * nc

    # Drop vanished or nearly-duplicate clusters
    d2_all = np.sum((feat_out[:, None, :] - c_arr[None, :, :]) ** 2, axis=-1)
    lbl_out = np.argmin(d2_all, axis=1)
    keep_c: list[int] = []
    for ci in range(len(c_arr)):
        m = lbl_out == ci
        frac = float(area_out[m].sum()) / max(1e-6, tot_area)
        if frac < min_cluster_frac and ci > 0:
            continue
        if any(float(np.linalg.norm(c_arr[ci] - c_arr[kj])) < 16.0 for kj in keep_c):
            continue
        keep_c.append(ci)
    if not keep_c:
        keep_c = [0]
    c_arr = c_arr[keep_c]

    # Sort slots by descending outer surface area: Slot 1 = Body, Slot 2 = Belly, Slots 3..K = Teeth/Eyes/Mouth!
    d2_full = np.sum((feat_all[:, None, :] - c_arr[None, :, :]) ** 2, axis=-1)
    lbl_full = np.argmin(d2_full, axis=1)
    slot_areas = [float(areas[(lbl_full == ci) & ~prot].sum()) for ci in range(len(c_arr))]
    order = np.argsort(slot_areas)[::-1]
    c_arr = c_arr[order]
    K = len(c_arr)

    # 3D Crease-Respecting Potts Graph Smoothing:
    # Smooth skin edges (< crease_deg) eliminate single-triangle noise; sharp 3D creases (>= crease_deg)
    # act as crisp paint-stop barriers so eyes, teeth, and mouth align cleanly to the 3D sculpt!
    f1, f2, dots, _ = build_face_adjacency(points, tris)
    cos_crease = math.cos(math.radians(crease_deg))
    e_weight = np.where(dots >= cos_crease, 1.0, 0.04).astype(np.float32)

    data_dist = np.sqrt(np.sum((feat_all[:, None, :] - c_arr[None, :, :]) ** 2, axis=-1)).astype(np.float32)
    labels = np.argmin(data_dist, axis=1).astype(np.int32)
    labels[prot] = 0

    if len(f1) > 0 and K > 1:
        smooth_lambda = 12.0
        for _ in range(5):
            votes = np.zeros((n_faces, K), dtype=np.float32)
            valid_e = (~prot[f1]) & (~prot[f2])
            if not np.any(valid_e):
                break
            ef1, ef2, ew = f1[valid_e], f2[valid_e], e_weight[valid_e]
            np.add.at(votes, (ef1, labels[ef2]), ew)
            np.add.at(votes, (ef2, labels[ef1]), ew)
            v_sum = np.maximum(votes.sum(axis=1, keepdims=True), 1e-6)
            nbr_frac = votes / v_sum
            energy = data_dist - smooth_lambda * nbr_frac
            new_labels = np.argmin(energy, axis=1).astype(np.int32)
            new_labels[prot] = 0
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        # Absorb tiny isolated (< 8 triangles) disconnected specks into their dominant surrounding color
        adj_lists: list[list[int]] = [[] for _ in range(n_faces)]
        for a, b in zip(f1.tolist(), f2.tolist()):
            if not prot[a] and not prot[b]:
                adj_lists[a].append(b)
                adj_lists[b].append(a)
        visited = prot.copy()
        for start in range(n_faces):
            if visited[start]:
                continue
            lbl_s = int(labels[start])
            comp = [start]
            visited[start] = True
            q = deque([start])
            border_counts: dict[int, int] = {}
            while q:
                cur = q.popleft()
                for nb in adj_lists[cur]:
                    if int(labels[nb]) == lbl_s:
                        if not visited[nb]:
                            visited[nb] = True
                            comp.append(nb)
                            q.append(nb)
                    else:
                        nl = int(labels[nb])
                        border_counts[nl] = border_counts.get(nl, 0) + 1
            if len(comp) < 8 and border_counts:
                best_nb = max(border_counts.items(), key=lambda kv: kv[1])[0]
                for fi in comp:
                    labels[fi] = best_nb

    # Compute crisp representative sRGB hex for each surviving slot (using lit albedo percentile so
    # shadow pixels never muddy clean filament colors; dark slots stay true dark!)
    used_slots = sorted(set(labels.tolist()))
    remap = {old: new_idx + 1 for new_idx, old in enumerate(used_slots)}
    face_filaments = np.array([remap[int(l)] for l in labels], dtype=np.int32)
    face_filaments[prot] = 1

    palette: dict[int, str] = {}
    for old, slot in remap.items():
        m = (face_filaments == slot) & (~prot)
        if not np.any(m):
            m = face_filaments == slot
        px = face_rgb[m]
        luma = px.mean(axis=1)
        med_luma = float(np.median(luma))
        if len(px) >= 16:
            if med_luma < 60.0:
                # True black/dark filament -> use crisp dark core (15..45th percentile)
                lo, hi = np.percentile(luma, [10, 45])
            else:
                # Lit matte filament -> use well-lit albedo core (55..88th percentile, excluding AO shadows!)
                lo, hi = np.percentile(luma, [55, 88])
            clean_px = px[(luma >= lo) & (luma <= hi)]
            if len(clean_px) >= 4:
                px = clean_px
        med_rgb = np.median(px, axis=0)
        palette[slot] = rgb_to_hex(med_rgb)

    summary = [f"Slot {s} {palette[s]}" for s in sorted(palette)]
    return face_filaments, palette, summary


def compute_clean_sweeping_belly_paint(
    points: np.ndarray,
    tris: np.ndarray,
    *,
    body_hex: str = "#527292",
    belly_hex: str = "#ece6d6",
    flexi_axis: str = "auto",
    protected_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Compute a clean, organic 2-color countershaded Belly on per-triangle `face_filaments`.

    - Smooth sweeping contour under the chin and lower flanks
    - Smoothly tapers ALL THE WAY down to `Z = 0` over the rear torso (`60%..78%`) — ZERO vertical cliff wall!
    - Keeps lateral pectoral fins and all internal flexi joints / hinges 100% Slot 1 (Primary Body).
    """
    n_faces = len(tris)
    palette = {1: normalize_hex(body_hex, "#527292"), 2: normalize_hex(belly_hex, "#ece6d6")}
    if n_faces == 0:
        return np.ones(0, dtype=np.int32), palette

    pts = np.asarray(points, dtype=np.float64)
    span = pts.max(axis=0) - pts.min(axis=0)
    rot_y = (flexi_axis == "y") or (flexi_axis == "auto" and float(span[1]) > float(span[0]))
    if rot_y:
        # Rotate -90 deg about Z so spine is along X
        work = np.stack([pts[:, 1], -pts[:, 0], pts[:, 2]], axis=-1)
    else:
        work = pts.copy()

    # Detect head vs tail orientation along X (head half has larger mid-cross-section volume than tail half)
    x0, x1 = float(work[:, 0].min()), float(work[:, 0].max())
    length = max(1e-6, x1 - x0)
    front_v = work[(work[:, 0] >= x0 + 0.08 * length) & (work[:, 0] <= x0 + 0.35 * length)]
    back_v = work[(work[:, 0] >= x0 + 0.65 * length) & (work[:, 0] <= x0 + 0.92 * length)]
    f_span = (float(np.ptp(front_v[:, 1])) * float(np.ptp(front_v[:, 2]))) if len(front_v) > 8 else 0.0
    b_span = (float(np.ptp(back_v[:, 1])) * float(np.ptp(back_v[:, 2]))) if len(back_v) > 8 else 0.0
    if b_span > f_span * 1.12:
        work[:, 0] = -work[:, 0]
        work[:, 1] = -work[:, 1]
        x0, x1 = float(work[:, 0].min()), float(work[:, 0].max())

    y0, y1 = float(work[:, 1].min()), float(work[:, 1].max())
    z0_all, z1_all = float(work[:, 2].min()), float(work[:, 2].max())
    width_all = max(1e-6, y1 - y0)
    height_all = max(1e-6, z1_all - z0_all)
    yc = 0.5 * (y0 + y1)

    # Sample local torso height & core width along X spine
    nx = 48
    x_e = np.linspace(x0, x1, nx + 1)
    x_m = 0.5 * (x_e[:-1] + x_e[1:])
    t_m = (x_m - x0) / length
    z_bot_bin = np.full(nx, z0_all, dtype=np.float64)
    z_top_bin = np.full(nx, z1_all, dtype=np.float64)
    torso_hw_bin = np.full(nx, width_all * 0.35, dtype=np.float64)
    for i in range(nx):
        sl = work[(work[:, 0] >= x_e[i]) & (work[:, 0] <= x_e[i + 1])]
        if len(sl) < 4:
            continue
        sl_core = sl[np.abs(sl[:, 1] - yc) <= width_all * 0.28]
        use = sl_core if len(sl_core) >= 4 else sl
        z_bot_bin[i] = float(use[:, 2].min())
        z_top_bin[i] = float(use[:, 2].max())
        # Measure upper torso half-width (above pectoral fins: upper 45..80% of local height)
        h_loc = max(1e-3, z_top_bin[i] - z_bot_bin[i])
        upper = sl[sl[:, 2] >= z_bot_bin[i] + 0.45 * h_loc]
        if len(upper) >= 6:
            torso_hw_bin[i] = float(np.percentile(np.abs(upper[:, 1] - yc), 88)) * 0.90
        else:
            torso_hw_bin[i] = float(np.percentile(np.abs(sl[:, 1] - yc), 80)) * 0.90

    # Smooth profile curves so boundary never jitters
    kern = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    for _ in range(2):
        z_bot_bin = np.convolve(np.pad(z_bot_bin, 2, mode="edge"), kern, mode="valid")
        z_top_bin = np.convolve(np.pad(z_top_bin, 2, mode="edge"), kern, mode="valid")
        torso_hw_bin = np.convolve(np.pad(torso_hw_bin, 2, mode="edge"), kern, mode="valid")

    # Organic countershading belly contour:
    # - Gently sweeps up to ~32% under jaw/chin (0..22%), eases to ~25% along chest/midbody (22..56%),
    # - Smooth C2 cubic ease-out ALL THE WAY to 0.0 over 56%..78% so it merges cleanly into Z=0 with ZERO vertical cliff!
    elongated = (length / height_all) >= 1.65
    t_taper_start = 0.52 if elongated else 0.72
    t_taper_end = 0.78 if elongated else 0.96
    frac_bin = np.zeros(nx, dtype=np.float64)
    for i, t in enumerate(t_m):
        if t <= 0.22:
            # Smooth chin rise
            u = t / 0.22
            frac_bin[i] = 0.20 + 0.13 * (u * u * (3.0 - 2.0 * u))
        elif t <= t_taper_start:
            u = (t - 0.22) / max(1e-6, t_taper_start - 0.22)
            frac_bin[i] = 0.33 - 0.08 * (u * u * (3.0 - 2.0 * u))
        elif t <= t_taper_end:
            u = (t - t_taper_start) / max(1e-6, t_taper_end - t_taper_start)
            # Smooth ease-out to 0.0 — zero vertical cliff!
            frac_bin[i] = 0.25 * (1.0 - (u * u * (3.0 - 2.0 * u)))
        else:
            frac_bin[i] = 0.0
    z_cut_bin = z_bot_bin + (z_top_bin - z_bot_bin) * frac_bin
    z_cut_bin = np.where(frac_bin <= 0.005, z_bot_bin - 1.0, z_cut_bin)

    tri_c = work[tris].mean(axis=1)  # (F, 3)
    cx, cy, cz = tri_c[:, 0], tri_c[:, 1], tri_c[:, 2]
    z_cut = np.interp(cx, x_m, z_cut_bin)
    hw_cut = np.interp(cx, x_m, torso_hw_bin)
    t_tri = (cx - x0) / length

    # Keep internal flexi cut caps (flat planar walls in the mid/rear spine `0.18 < t < 0.90`)
    # and hinge_keepout pure Slot 1 Body — NEVER mask the front chin/snout (`t <= 0.18`)!
    v0, v1, v2 = work[tris[:, 0]], work[tris[:, 1]], work[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = np.divide(fn, np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12))
    is_cap = (np.abs(fn[:, 0]) > 0.90) & (t_tri > 0.18) & (t_tri < 0.90)

    prot = np.zeros(n_faces, dtype=bool) if protected_mask is None else np.asarray(protected_mask, dtype=bool)
    prot = prot | is_cap

    is_belly = (cz <= z_cut) & (np.abs(cy - yc) <= hw_cut) & (t_tri <= t_taper_end) & (~prot)
    filaments = np.where(is_belly, 2, 1).astype(np.int32)

    # 2 passes of smooth-edge majority cleanup so the border line is razor-smooth
    f1, f2, dots, _ = build_face_adjacency(points, tris)
    if len(f1) > 0:
        valid_e = (~prot[f1]) & (~prot[f2]) & (dots >= 0.85)
        ef1, ef2 = f1[valid_e], f2[valid_e]
        for _ in range(2):
            v1_c = np.zeros(n_faces, dtype=np.int32)
            v2_c = np.zeros(n_faces, dtype=np.int32)
            np.add.at(v1_c, ef1, (filaments[ef2] == 1).astype(np.int32))
            np.add.at(v1_c, ef2, (filaments[ef1] == 1).astype(np.int32))
            np.add.at(v2_c, ef1, (filaments[ef2] == 2).astype(np.int32))
            np.add.at(v2_c, ef2, (filaments[ef1] == 2).astype(np.int32))
            flip_to_2 = (~prot) & (filaments == 1) & (v2_c >= 2) & (v2_c > v1_c)
            flip_to_1 = (~prot) & (filaments == 2) & (v1_c >= 2) & (v1_c > v2_c)
            filaments[flip_to_2] = 2
            filaments[flip_to_1] = 1

    filaments[prot] = 1
    return filaments, palette


def find_nearest_triangle(points: np.ndarray, tris: np.ndarray, hit_point: Iterable[float]) -> int:
    """Locate the triangle closest to a 3D viewport raycast hit point `[x, y, z]` (mm)."""
    if len(tris) == 0:
        return -1
    hp = np.asarray(list(hit_point), dtype=np.float64).reshape(1, 3)
    centroids = points[tris].mean(axis=1)
    d2 = np.sum((centroids - hp) ** 2, axis=1)
    return int(np.argmin(d2))


def smart_fill_triangles(
    points: np.ndarray,
    tris: np.ndarray,
    face_filaments: np.ndarray,
    hit_point: Iterable[float],
    target_filament: int,
    *,
    crease_deg: float = 26.0,
    max_tris: int = 0,
) -> tuple[np.ndarray, int]:
    """1-click 3D Crease-Guided Smart-Fill flood-fill starting from `hit_point`.

    Crosses shared mesh edges ONLY when their dihedral angle is shallower than `crease_deg`,
    so clicking an eyeball fills only the eyeball, clicking a tooth fills only that tooth, and
    clicking a tongue fills only the tongue — stopping dead at real 3D sculpted creases!
    """
    out = np.asarray(face_filaments, dtype=np.int32).copy()
    n_faces = len(tris)
    if n_faces == 0:
        return out, 0
    seed = find_nearest_triangle(points, tris, hit_point)
    if seed < 0:
        return out, 0

    f1, f2, dots, _ = build_face_adjacency(points, tris)
    cos_thresh = math.cos(math.radians(max(4.0, min(85.0, float(crease_deg)))))
    smooth = dots >= cos_thresh
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in zip(f1[smooth].tolist(), f2[smooth].tolist()):
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(n_faces, dtype=bool)
    visited[seed] = True
    q = deque([seed])
    filled = []
    limit = int(max_tris) if max_tris > 0 else n_faces
    while q and len(filled) < limit:
        cur = q.popleft()
        filled.append(cur)
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                q.append(nb)

    slot = max(1, min(8, int(target_filament)))
    out[filled] = slot
    return out, len(filled)


def brush_paint_triangles(
    points: np.ndarray,
    tris: np.ndarray,
    face_filaments: np.ndarray,
    hit_point: Iterable[float],
    target_filament: int,
    *,
    radius_mm: float = 4.0,
    crease_deg: float = 55.0,
) -> tuple[np.ndarray, int]:
    """Geodesic/sphere-bounded 3D surface brush centered at `hit_point` that respects steep creases (`crease_deg`)."""
    out = np.asarray(face_filaments, dtype=np.int32).copy()
    n_faces = len(tris)
    if n_faces == 0:
        return out, 0
    seed = find_nearest_triangle(points, tris, hit_point)
    if seed < 0:
        return out, 0

    hp = np.asarray(list(hit_point), dtype=np.float64).reshape(1, 3)
    centroids = points[tris].mean(axis=1)
    r2 = max(0.5, float(radius_mm)) ** 2
    in_sphere = np.sum((centroids - hp) ** 2, axis=1) <= r2
    in_sphere[seed] = True

    f1, f2, dots, _ = build_face_adjacency(points, tris)
    cos_thresh = math.cos(math.radians(max(10.0, min(179.0, float(crease_deg)))))
    ok_edge = (dots >= cos_thresh) & in_sphere[f1] & in_sphere[f2]
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in zip(f1[ok_edge].tolist(), f2[ok_edge].tolist()):
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(n_faces, dtype=bool)
    visited[seed] = True
    q = deque([seed])
    filled = []
    while q:
        cur = q.popleft()
        filled.append(cur)
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                q.append(nb)

    slot = max(1, min(8, int(target_filament)))
    out[filled] = slot
    return out, len(filled)


def bucket_fill_triangles(
    points: np.ndarray,
    tris: np.ndarray,
    face_filaments: np.ndarray,
    hit_point: Iterable[float],
    target_filament: int,
) -> tuple[np.ndarray, int]:
    """Flood-fill the connected region currently sharing the clicked triangle's filament slot."""
    out = np.asarray(face_filaments, dtype=np.int32).copy()
    n_faces = len(tris)
    if n_faces == 0:
        return out, 0
    seed = find_nearest_triangle(points, tris, hit_point)
    if seed < 0:
        return out, 0
    src_slot = int(out[seed])
    slot = max(1, min(8, int(target_filament)))
    if src_slot == slot:
        return out, 0

    f1, f2, _, _ = build_face_adjacency(points, tris)
    same = (out[f1] == src_slot) & (out[f2] == src_slot)
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in zip(f1[same].tolist(), f2[same].tolist()):
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(n_faces, dtype=bool)
    visited[seed] = True
    q = deque([seed])
    filled = []
    while q:
        cur = q.popleft()
        filled.append(cur)
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                q.append(nb)

    out[filled] = slot
    return out, len(filled)
