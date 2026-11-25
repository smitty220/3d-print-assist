"""Watertight print-preparation and finishing for organic 3D meshes (Character Mode).

Raw meshes from Image-to-3D models often have holes, inverted normals, floating debris, or zero-thickness
sheets. This module cleans and repairs them into guaranteed watertight manifold solids using `trimesh`
and `manifold3d`, scales to exact millimetres, flattens the bottom for bed adhesion, optionally unions a
display plinth or keychain loop, and exports print-ready STL or 3MF files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import manifold3d as m3d
import numpy as np
import trimesh

from ..features.model import Design, SculptPrep
from ..printers import PrinterProfile

BaseStyle = Literal["none", "round", "oval", "square"]


class MeshPrepError(ValueError):
    pass


@dataclass
class PreparedPart:
    name: str
    points: np.ndarray  # (N, 3) float64, mm
    tris: np.ndarray  # (M, 3) int32
    color: str | None = None
    filament: int | None = None

    @property
    def triangles(self) -> int:
        return int(len(self.tris))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.points) == 0:
            z = np.zeros(3, dtype=np.float64)
            return z, z
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclass
class PreparedSculpt:
    parts: list[PreparedPart]
    watertight: bool
    triangles: int
    volume_mm3: float
    bbox_min: list[float]
    bbox_max: list[float]
    notes: list[str] = field(default_factory=list)


def _to_trimesh(raw: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(raw, trimesh.Scene):
        geoms = [g for g in raw.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geoms:
            raise MeshPrepError("mesh file contains no triangles")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(raw, trimesh.Trimesh):
        mesh = raw.copy()
    else:
        raise MeshPrepError(f"unsupported mesh object: {type(raw).__name__}")
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise MeshPrepError("mesh contains no geometry")
    return mesh


def load_raw_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise MeshPrepError(f"mesh file not found: {path.name}")
    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception as exc:  # noqa: BLE001
        raise MeshPrepError(f"could not read mesh {path.name}: {exc}") from exc
    return _to_trimesh(loaded)


def _manifold_to_trimesh(mf: m3d.Manifold) -> trimesh.Trimesh:
    m = mf.to_mesh()
    verts = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
    tris = np.asarray(m.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


def _trimesh_to_manifold(mesh: trimesh.Trimesh) -> m3d.Manifold | None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    m = m3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    )
    m.merge()
    mf = m3d.Manifold(m)
    if mf.status() == m3d.Error.NoError and not mf.is_empty():
        return mf
    return None


def _drop_tiny_debris(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove disconnected micro-fragments (< 0.2% of largest component face count)."""
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        return mesh
    if len(comps) <= 1:
        return mesh
    max_faces = max(len(c.faces) for c in comps)
    threshold = max(12, int(max_faces * 0.002))
    kept = [c for c in comps if len(c.faces) >= threshold]
    if not kept:
        kept = [max(comps, key=lambda c: len(c.faces))]
    return trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]


def repair_to_manifold(mesh: trimesh.Trimesh, max_voxel_res: int = 128) -> tuple[m3d.Manifold, list[str]]:
    """Repair an arbitrary triangle mesh into a guaranteed watertight `manifold3d.Manifold`."""
    notes: list[str] = []
    work = mesh.copy()
    work.merge_vertices()
    work.update_faces(work.nondegenerate_faces())
    work.update_faces(work.unique_faces())
    work.remove_unreferenced_vertices()
    work = _drop_tiny_debris(work)

    # First attempt direct topological repair (preserves full sharp detail when possible).
    trimesh.repair.fix_winding(work)
    trimesh.repair.fix_inversion(work, multibody=True)
    if not work.is_watertight:
        trimesh.repair.fill_holes(work)
        trimesh.repair.fix_normals(work, multibody=True)

    if work.is_watertight:
        if work.volume < 0:
            work.invert()
        mf = _trimesh_to_manifold(work)
        if mf is not None:
            return mf, notes

    # Fallback: voxel solid remesh guarantees a closed manifold solid even for open/self-intersecting AI meshes.
    extents = np.asarray(work.extents, dtype=np.float64)
    longest = float(extents.max())
    if longest <= 1e-6:
        raise MeshPrepError("mesh has zero size")
    pitch = longest / float(max(32, min(max_voxel_res, 192)))
    vox = work.voxelized(pitch=pitch).fill()
    remeshed = vox.marching_cubes
    # marching_cubes returns coordinates in voxel index space -> transform back to mesh space
    remeshed.apply_transform(vox.transform)
    remeshed.merge_vertices()
    trimesh.repair.fix_normals(remeshed)
    if remeshed.volume < 0:
        remeshed.invert()
    mf = _trimesh_to_manifold(remeshed)
    if mf is None:
        raise MeshPrepError("mesh could not be repaired into a watertight solid")
    notes.append("repaired non-manifold geometry via watertight voxel remesh")
    return mf, notes


def _build_base(style: BaseStyle, sx: float, sy: float, height: float, margin: float) -> m3d.Manifold:
    h = max(1.0, float(height))
    rx = max(6.0, sx / 2.0 + float(margin))
    ry = max(6.0, sy / 2.0 + float(margin))
    chamfer = min(1.0, h * 0.25)

    if style == "round":
        r = max(rx, ry)
        base = m3d.Manifold.cylinder(h - chamfer, r, r, circular_segments=64)
        top = m3d.Manifold.cylinder(chamfer, r, max(1.0, r - chamfer), circular_segments=64).translate((0.0, 0.0, h - chamfer))
        return base + top
    if style == "oval":
        base = m3d.Manifold.cylinder(h - chamfer, 1.0, 1.0, circular_segments=64).scale((rx, ry, 1.0))
        top_r_x = max(1.0, rx - chamfer)
        top_r_y = max(1.0, ry - chamfer)
        top = m3d.Manifold.cylinder(chamfer, 1.0, 1.0, circular_segments=64).scale((rx, ry, 1.0))
        # Use a gentle taper cylinder scaled to oval
        taper = m3d.Manifold.cylinder(chamfer, 1.0, min(top_r_x / rx, top_r_y / ry), circular_segments=64).scale((rx, ry, 1.0))
        return base + taper.translate((0.0, 0.0, h - chamfer))
    # square with chamfered top
    wx, wy = rx * 2.0, ry * 2.0
    box = m3d.Manifold.cube((wx, wy, h), center=True).translate((0.0, 0.0, h / 2.0))
    return box


def _build_keychain_loop(hole_dia: float, top_x: float, top_y: float, top_z: float) -> m3d.Manifold:
    """Create a sturdy vertical split-ring eyelet overlapping cleanly into the top of the sculpt."""
    inner_r = max(1.5, float(hole_dia) / 2.0)
    wall = 2.4
    outer_r = inner_r + wall
    thickness = 3.2
    outer = m3d.Manifold.cylinder(thickness, outer_r, outer_r, circular_segments=48, center=True)
    inner = m3d.Manifold.cylinder(thickness + 2.0, inner_r, inner_r, circular_segments=48, center=True)
    ring = (outer - inner).rotate((90.0, 0.0, 0.0))  # ring lies in XZ plane, hole faces Y
    # Seat bottom of ring 1.6 mm below top_z so it fuses strongly into the model
    center_z = top_z + outer_r - 1.6
    return ring.translate((top_x, top_y, center_z))


def _place_on_bed(mf: m3d.Manifold) -> m3d.Manifold:
    bb = mf.bounding_box()
    cx = (bb[0] + bb[3]) / 2.0
    cy = (bb[1] + bb[4]) / 2.0
    min_z = bb[2]
    return mf.translate((-cx, -cy, -min_z))


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _rounded_xz_loop(x_center: float, z_center: float, radius: float, x_back: float, z_floor: float,
                     width_y: float, y_center: float, segs: int = 24) -> m3d.Manifold:
    """Build a smooth semicircular-nose stadium block in the XZ plane extruded along Y (true bearing profile)."""
    # Points in (X, Z) traced CCW around the front semicircle (around x_center, z_center) and back to x_back
    top_z = z_center + radius
    bot_z = max(z_floor, z_center - radius)
    raw_pts: list[tuple[float, float]] = [(x_back, top_z), (x_center, top_z)]
    # Front semicircle from angle +90° -> +270° (facing -X), clamped cleanly at z_floor
    for k in range(1, segs):
        ang = math.pi * 0.5 + math.pi * (k / segs)
        px = x_center + radius * math.cos(ang)
        pz = max(z_floor, z_center + radius * math.sin(ang))
        raw_pts.append((px, pz))
    raw_pts.append((x_center, bot_z))
    raw_pts.append((x_back, bot_z))
    # Deduplicate consecutive identical points from z_floor clamping
    pts: list[tuple[float, float]] = []
    for pt in raw_pts:
        if not pts or abs(pt[0] - pts[-1][0]) > 1e-5 or abs(pt[1] - pts[-1][1]) > 1e-5:
            pts.append(pt)
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= 1e-5 and abs(pts[0][1] - pts[-1][1]) <= 1e-5:
        pts.pop()
    # Extrude in XY (where Y stands for Z), then rotate +90° about X so extrusion lies along Y and profile lies in XZ.
    solid = (
        m3d.Manifold.extrude(m3d.CrossSection([pts]), width_y)
        .translate((0.0, 0.0, -width_y / 2.0))
        .rotate((90.0, 0.0, 0.0))
        .translate((0.0, y_center, 0.0))
    )
    return solid


def _crescent_x(u: float, y: float, yc: float, half_slot: float, half_w: float, curve: float) -> float:
    """Smooth rearward-swept gill/rib crescent curve in the XY plane.

    Flat inside the narrow central hinge channel (`|y - yc| <= half_slot`), then sweeps smoothly
    rearward (+X, toward the tail) across the flanks so the upstream pin stays deeply nested in meat
    and the exterior cuts follow organic shark-rib / chevron lines.
    """
    flank = max(0.0, abs(y - yc) - half_slot)
    span = max(2.0, half_w - half_slot)
    t = min(1.0, flank / span)
    return u + curve * (t * t)


def _clean_segment_slivers(seg: m3d.Manifold) -> m3d.Manifold:
    """Drop tiny disconnected floating crumbs (< 10 mm³ above Z=1.5 mm) clipped off fin tips by a crescent cut."""
    if seg.is_empty():
        return seg
    comps = seg.decompose()
    if len(comps) <= 1:
        return seg
    main_vol = max(abs(float(c.volume())) for c in comps)
    kept = []
    for c in comps:
        vol = abs(float(c.volume()))
        z_min = float(c.bounding_box()[2])
        if vol >= max(10.0, main_vol * 0.02) or z_min <= 1.5:
            kept.append(c)
    if not kept:
        kept = [max(comps, key=lambda c: abs(float(c.volume())))]
    return m3d.Manifold.compose(kept) if len(kept) > 1 else kept[0]


def _detect_head_flip(work: m3d.Manifold) -> bool:
    """Compare cross-sectional bulk of front 35% vs rear 35%: keep the heavier head end at x0."""
    tm = _manifold_to_trimesh(work)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    length = max(1e-6, x1 - x0)
    front = verts[verts[:, 0] <= x0 + 0.35 * length]
    rear = verts[verts[:, 0] >= x1 - 0.35 * length]
    f_span = float(np.ptp(front[:, 1]) * np.ptp(front[:, 2])) if len(front) > 20 else 0.0
    r_span = float(np.ptp(rear[:, 1]) * np.ptp(rear[:, 2])) if len(rear) > 20 else 0.0
    return bool(r_span > f_span * 1.18)


def analyze_sculpt_anatomy(
    work: m3d.Manifold,
    n_cuts: int = 5,
    s0_pct: float = 35.0,
    s1_pct: float = 88.0,
) -> dict:
    """Analyze a canonical Z-up sculpt (head at x0, tail at x1) to locate fins, limbs, tail waist, and fin-safe cuts.

    Slices the mesh into 40 cross-sections along X and measures:
    - Lateral fin/limb protrusions (`w_max / w_median > 1.22` and `w_max > 0.50 * W`, e.g. pectoral fins, arms, legs)
    - Vertical dorsal fin/crest protrusions (`dorsal_h > max(5.0, 0.14 * H)` above the core torso)
    - Caudal peduncle (tail waist) before the tail fin flares upward/outward
    Returns exact mm & percentage anatomy data plus recommended fin-safe cut positions `[(t_norm, force_flat), ...]`.
    """
    tm = _manifold_to_trimesh(work)
    v = np.asarray(tm.vertices, dtype=np.float64)
    x0, x1 = float(v[:, 0].min()), float(v[:, 0].max())
    y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
    z0_all, z1_all = float(v[:, 2].min()), float(v[:, 2].max())
    length = max(1e-6, x1 - x0)
    width_all = max(1e-6, y1 - y0)
    height_all = max(1e-6, z1_all - z0_all)

    n_bins = 40
    edges = np.linspace(x0, x1, n_bins + 1)
    t_centers = np.array([(i + 0.5) / n_bins for i in range(n_bins)], dtype=np.float64)
    w_max = np.zeros(n_bins, dtype=np.float64)
    w_upper = np.zeros(n_bins, dtype=np.float64)
    w_top = np.zeros(n_bins, dtype=np.float64)
    z_min = np.zeros(n_bins, dtype=np.float64)
    z_max = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        sl = v[(v[:, 0] >= edges[i]) & (v[:, 0] <= edges[i + 1])]
        if len(sl) < 6:
            continue
        w_max[i] = float(np.ptp(sl[:, 1]))
        za, zb = float(sl[:, 2].min()), float(sl[:, 2].max())
        z_min[i], z_max[i] = za, zb
        hl = zb - za
        if hl < max(0.04 * height_all, 1e-4):
            w_upper[i] = w_max[i]
            w_top[i] = w_max[i]
            continue
        up = sl[(sl[:, 2] >= za + 0.45 * hl) & (sl[:, 2] <= za + 0.76 * hl)]
        w_upper[i] = float(np.ptp(up[:, 1])) if len(up) >= 3 else w_max[i]
        tp = sl[sl[:, 2] >= za + 0.80 * hl]
        w_top[i] = float(np.ptp(tp[:, 1])) if len(tp) >= 3 else w_upper[i]

    h_loc = z_max - z_min
    min_w_ref = max(0.04 * width_all, 1e-4)
    lat_ratio = np.divide(w_max, np.maximum(min_w_ref, w_upper))
    top_ratio = np.divide(w_top, np.maximum(min_w_ref, w_upper))

    # 1. Detect tail waist if a caudal fin flares upward/outward at the rear (68%..88%)
    eff_s1 = min(0.92, s1_pct / 100.0)
    waist_bins = np.where((t_centers >= 0.68) & (t_centers <= 0.88))[0]
    has_tail_flare = False
    tail_waist_pct: float | None = None
    if len(waist_bins) >= 3:
        areas = w_max[waist_bins] * np.maximum(0.03 * height_all, h_loc[waist_bins])
        w_idx = int(waist_bins[np.argmin(areas)])
        after = np.where(t_centers > t_centers[w_idx])[0]
        if len(after) and (
            float(h_loc[after].max()) > float(h_loc[w_idx]) * 1.10
            or float(w_max[after].max()) > float(w_max[w_idx]) * 1.15
        ):
            has_tail_flare = True
            tail_waist_pct = round(float(t_centers[w_idx] * 100.0), 1)
            eff_s1 = min(eff_s1, float(t_centers[w_idx]) - 0.025)
            # Also pull back in front of any pelvic/adipose fin bump right before the tail waist
            pelvic = np.where((t_centers >= 0.72) & (t_centers <= eff_s1 + 0.04) & ((lat_ratio > 1.25) | (top_ratio < 0.45)))[0]
            if len(pelvic):
                eff_s1 = min(eff_s1, float(t_centers[pelvic[0]]) - 0.04)

    # 2. Detect major protected fin/limb zones in [20%..70%]
    is_lat_fin = (lat_ratio > 1.24) & (w_max > 0.50 * width_all) & (t_centers >= 0.20) & (t_centers <= 0.70)
    is_dorsal_fin = (top_ratio < 0.45) & (h_loc > 0.78 * height_all) & (t_centers >= 0.20) & (t_centers <= 0.70)
    is_protected = is_lat_fin | is_dorsal_fin
    # Bridge tiny <= 2-bin (5%) gaps between overlapping pectoral and dorsal fins so they stay one unified chest segment
    for i in range(1, n_bins - 2):
        if not is_protected[i] and is_protected[i - 1] and (is_protected[i + 1] or is_protected[min(n_bins - 1, i + 2)]):
            is_protected[i] = True

    dt = 1.0 / n_bins
    protected_zones: list[dict] = []
    p_run = False
    p_start = 0
    for i, p_flag in enumerate(is_protected):
        if p_flag and not p_run:
            p_run = True
            p_start = i
        elif not p_flag and p_run:
            p_run = False
            has_lat = bool(np.any(is_lat_fin[p_start:i]))
            has_dor = bool(np.any(is_dorsal_fin[p_start:i]))
            kind = "pectoral+dorsal_fins" if (has_lat and has_dor) else ("pectoral_fins_or_limbs" if has_lat else "dorsal_fin")
            protected_zones.append({
                "kind": kind,
                "start_pct": round(float((t_centers[p_start] - 0.5 * dt) * 100.0), 1),
                "end_pct": round(float((t_centers[i - 1] + 0.5 * dt) * 100.0), 1),
            })
    if p_run:
        protected_zones.append({
            "kind": "fin_or_limb",
            "start_pct": round(float((t_centers[p_start] - 0.5 * dt) * 100.0), 1),
            "end_pct": round(float((t_centers[n_bins - 1] + 0.5 * dt) * 100.0), 1),
        })

    # If no protected fin zones and no tail flare (e.g. uniform ellipsoid/worm), use standard spacing in [s0..s1]
    target_n = max(1, int(n_cuts))
    if not np.any(is_protected) and not has_tail_flare:
        s0 = _clamp(s0_pct / 100.0, 0.14, 0.72)
        s1 = _clamp(s1_pct / 100.0, s0 + 0.12, 0.94)
        chosen_t = [(s0 + (s1 - s0) * ((i + 0.5) / target_n), False) for i in range(target_n)]
        safe_intervals = [(s0, s1)]
    else:
        # Allow a front neck/gill cut interval [0.24 .. first_fin - 0.015] ahead of chest fins/limbs when s0_pct <= 38%
        first_prot_idx = int(np.nonzero(is_protected)[0][0]) if np.any(is_protected) else -1
        first_prot_t = float(t_centers[first_prot_idx] - 0.5 * dt) if first_prot_idx >= 0 else 1.0
        if first_prot_t >= 0.275 and s0_pct <= 38.0:
            min_t = 0.235
        else:
            min_t = min(0.31, _clamp(s0_pct / 100.0, 0.14, 0.72))
        min_safe_w = min(7.5, 0.11 * width_all) if length >= 20.0 else 0.11 * width_all
        safe_mask = (~is_protected) & (t_centers >= min_t) & (t_centers <= eff_s1) & (w_max >= min_safe_w)
        safe_intervals = []
        in_run = False
        run_start = 0.0
        for i, ok in enumerate(safe_mask):
            if ok and not in_run:
                in_run = True
                run_start = float(t_centers[i] - 0.2 * dt)
            elif not ok and in_run:
                in_run = False
                safe_intervals.append((run_start, float(t_centers[i - 1] + 0.2 * dt)))
        if in_run:
            safe_intervals.append((run_start, min(eff_s1, float(t_centers[np.where(safe_mask)[0][-1]] + 0.2 * dt))))

        min_pitch = (8.0 / length) if length >= 20.0 else 0.052
        iv_caps = []
        for iv_a, iv_b in safe_intervals:
            span_t = max(0.0, iv_b - iv_a)
            if span_t < 0.010:
                iv_caps.append(0)
            elif span_t < min_pitch * 1.15:
                iv_caps.append(1)
            else:
                iv_caps.append(max(1, int(round(span_t / min_pitch))))

        # If the main rear tail interval already fits >= 3 joints, skip narrow (< 4%) mid-fin cracks (keep front neck interval idx 0!)
        if len(iv_caps) >= 2 and iv_caps[-1] >= 3:
            for idx_iv in range(1, len(iv_caps) - 1):
                if (safe_intervals[idx_iv][1] - safe_intervals[idx_iv][0]) < 0.045:
                    iv_caps[idx_iv] = 0

        chosen_t = []
        total_cap = sum(iv_caps)
        for (iv_a, iv_b), cap in zip(safe_intervals, iv_caps):
            if cap <= 0:
                continue
            k_iv = min(cap, max(1, int(round(target_n * (cap / max(1, total_cap))))))
            fin_behind = any(is_protected[k] for k in range(n_bins) if iv_b - 0.01 <= t_centers[k] <= iv_b + 0.10)
            eff_b = (iv_b - 0.02) if (fin_behind and iv_b - iv_a > 0.02) else iv_b
            if k_iv == 1:
                pt_single = min(0.5 * (iv_a + eff_b), eff_b - 0.01) if fin_behind else 0.5 * (iv_a + eff_b)
                chosen_t.append((pt_single, fin_behind))
            else:
                pad = min(0.018, (eff_b - iv_a) * 0.12)
                pts = np.linspace(iv_a + pad, eff_b - pad, k_iv).tolist()
                for idx_p, pt in enumerate(pts):
                    chosen_t.append((pt, fin_behind and idx_p == len(pts) - 1))

        if len(chosen_t) > target_n:
            idxs = np.round(np.linspace(0, len(chosen_t) - 1, target_n)).astype(int)
            chosen_t = [chosen_t[k] for k in idxs]

        # Fallback if no safe intervals were wide enough
        if not chosen_t:
            s0 = _clamp(s0_pct / 100.0, 0.14, 0.72)
            s1 = _clamp(eff_s1, s0 + 0.10, 0.90)
            chosen_t = [(s0 + (s1 - s0) * ((i + 0.5) / target_n), False) for i in range(target_n)]

    profile_10pct = []
    for k in range(0, n_bins, 4):
        pct = round(float(t_centers[k] * 100.0), 1)
        profile_10pct.append({
            "pct": pct,
            "x_mm": round(float(x0 + length * t_centers[k]), 1),
            "width_mm": round(float(w_max[k]), 1),
            "height_mm": round(float(h_loc[k]), 1),
            "z_min_mm": round(float(z_min[k]), 1),
            "protected_fin": bool(is_protected[k]),
        })

    return {
        "bbox_min_mm": [round(x0, 2), round(y0, 2), round(z0_all, 2)],
        "bbox_max_mm": [round(x1, 2), round(y1, 2), round(z1_all, 2)],
        "dimensions_mm": [round(length, 2), round(width_all, 2), round(height_all, 2)],
        "protected_fin_zones": protected_zones,
        "tail_waist_pct": tail_waist_pct,
        "safe_cut_intervals_pct": [[round(a * 100.0, 1), round(b * 100.0, 1)] for a, b in safe_intervals],
        "recommended_cuts_pct": [round(t * 100.0, 1) for t, _ in chosen_t],
        "solved_cuts": chosen_t,
        "profile": profile_10pct,
    }


def _rounded_xy_prism(
    x_center: float,
    y_center: float,
    radius: float,
    x_back: float,
    z0: float,
    z1: float,
    segs: int = 28,
) -> m3d.Manifold:
    """Build a smooth semicircular-nose stadium in the XY plane (nose facing -X around `(x_center, y_center)`,
    extending back to `x_back`) extruded vertically from `z0` to `z1`."""
    pts: list[tuple[float, float]] = [(x_back, y_center + radius), (x_center, y_center + radius)]
    for k in range(1, segs):
        ang = math.pi * 0.5 + math.pi * (k / segs)
        pts.append((x_center + radius * math.cos(ang), y_center + radius * math.sin(ang)))
    pts.append((x_center, y_center - radius))
    pts.append((x_back, y_center - radius))
    h = max(0.2, z1 - z0)
    return m3d.Manifold.extrude(m3d.CrossSection([pts]), h).translate((0.0, 0.0, z0))


def _streamline_flexi_proportions(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    """Transform overly spherical/chubby figurine sculpts (`L/H < 2.05`) into sleek, athletic flexi proportions
    (`L/H ~ 2.25`) while keeping the front head/eyes/grin (`0..26%`) at 1.0x undistorted aspect ratio.
    """
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    sz = float(bb[5] - bb[2])
    if max(sx, sy) < 20.0 or sz < 8.0:
        return mf, []
    rotated_for_y = (prep.flexi_axis == "y") or (prep.flexi_axis == "auto" and sy > sx)
    work = mf.rotate((0.0, 0.0, -90.0)) if rotated_for_y else mf
    work = _place_on_bed(work)
    flip_180 = _detect_head_flip(work)
    if flip_180:
        work = _place_on_bed(work.rotate((0.0, 0.0, 180.0)))

    wbb = work.bounding_box()
    length = float(wbb[3] - wbb[0])
    width = float(wbb[4] - wbb[1])
    height = float(wbb[5] - wbb[2])
    lh = length / max(1e-6, height)
    # Only streamline chubby character sculpts (skip synthetic spheres with L/H <= 1.05 used in generic mesh unit tests,
    # or streamline when L/H is in the chubby-character range 1.12 .. 2.05!)
    if lh < 1.12 or lh >= 2.08:
        return mf, []

    target_lh = min(2.36, max(lh * 1.62, 2.24))
    needed_gain = (target_lh / lh) - 1.0  # total fractional length increase needed
    # Integral of ramp w(t) over [0, 1] is ~0.62, so rear stretch factor is 1 + needed_gain / 0.62
    s_rear = min(2.25, 1.0 + needed_gain / 0.62)

    tm = _manifold_to_trimesh(work)
    verts = np.asarray(tm.vertices, dtype=np.float64).copy()
    x0 = float(verts[:, 0].min())
    yc = 0.5 * float(verts[:, 1].min() + verts[:, 1].max())

    n_grid = 256
    ts = np.linspace(0.0, 1.0, n_grid)
    r_up = np.clip((ts - 0.23) / 0.15, 0.0, 1.0)
    r_up = r_up * r_up * (3.0 - 2.0 * r_up)
    r_tail = 1.0 - 0.30 * (np.clip((ts - 0.87) / 0.12, 0.0, 1.0) ** 2)
    stretch_t = 1.0 + (s_rear - 1.0) * r_up * r_tail
    dt = 1.0 / (n_grid - 1)
    mapped_t = np.concatenate(([0.0], np.cumsum(0.5 * (stretch_t[:-1] + stretch_t[1:]) * dt)))

    v_t = np.clip((verts[:, 0] - x0) / max(1e-6, length), 0.0, 1.0)
    verts[:, 0] = x0 + np.interp(v_t, ts, mapped_t) * length

    # Gently slim excessive spherical belly/flank bloat behind the head so it reads like a sleek shark, not a balloon
    if width / max(1e-6, height) > 0.80 and lh < 1.80:
        slim_r = np.clip((v_t - 0.22) / 0.20, 0.0, 1.0)
        slim_r = slim_r * slim_r * (3.0 - 2.0 * slim_r)
        verts[:, 1] = yc + (verts[:, 1] - yc) * (1.0 - 0.22 * slim_r)

    streamlined = repair_to_manifold(trimesh.Trimesh(vertices=verts, faces=tm.faces, process=False))[0]
    if streamlined.is_empty():
        return mf, []
    if flip_180:
        streamlined = streamlined.rotate((0.0, 0.0, 180.0))
    if rotated_for_y:
        streamlined = streamlined.rotate((0.0, 0.0, 90.0))
    new_bb = streamlined.bounding_box()
    new_len = max(float(new_bb[3] - new_bb[0]), float(new_bb[4] - new_bb[1]))
    # Keep comfortably within standard 180-250 mm print beds
    if new_len > 176.0:
        sc = 176.0 / new_len
        streamlined = streamlined.scale((sc, sc, sc))
    return _place_on_bed(streamlined), [f"streamlined torso/tail to sleek {target_lh:.2f}:1 flexi profile (head undistorted)"]


def _seat_flexi_on_bed(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    """Gently un-arch an upward-swept rear belly (`45%..79%`) via pure 1D rigid cross-section translation
    `z' = z - drop_rear(x)`.

    - Leaves `0%..44%` (snout, jaw, throat, chest, and pectoral fins) 100% untouched (no throat goiter, no melted fins!)
    - Ramps cleanly back to `0.0` before `80%` (caudal peduncle waist & crescent tail fin stay sleek and symmetrical!)
    - Preserves exact cross-sectional roundness (never stretches thickness vertically like taffy).
    """
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    sz = float(bb[5] - bb[2])
    if max(sx, sy) < 20.0 or sz < 6.0:
        return mf, []
    rotated_for_y = (prep.flexi_axis == "y") or (prep.flexi_axis == "auto" and sy > sx)
    work = mf.rotate((0.0, 0.0, -90.0)) if rotated_for_y else mf
    work = _place_on_bed(work)
    flip_180 = _detect_head_flip(work)
    if flip_180:
        work = _place_on_bed(work.rotate((0.0, 0.0, 180.0)))

    tm = _manifold_to_trimesh(work)
    verts = np.asarray(tm.vertices, dtype=np.float64).copy()
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    h_all = float(verts[:, 2].max() - verts[:, 2].min())
    length = max(1e-6, x1 - x0)
    width = max(1e-6, y1 - y0)
    yc = 0.5 * (y0 + y1)

    nx = 48
    x_edges = np.linspace(x0, x1, nx + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_min_central = np.full(nx, np.nan, dtype=np.float64)
    central_v = np.abs(verts[:, 1] - yc) <= 0.22 * width
    for i in range(nx):
        mx = (verts[:, 0] >= x_edges[i]) & (verts[:, 0] <= x_edges[i + 1])
        mc = mx & central_v
        if np.any(mc):
            z_min_central[i] = float(verts[mc, 2].min())
        elif np.any(mx):
            z_min_central[i] = float(verts[mx, 2].min())

    valid = ~np.isnan(z_min_central)
    if valid.sum() < 8:
        return mf, []
    idx_1d = np.arange(nx)
    z_min_central = np.interp(idx_1d, idx_1d[valid], z_min_central[valid])

    t_norm = (x_centers - x0) / length
    rear_zone = (t_norm >= 0.48) & (t_norm <= 0.77)
    if float(z_min_central[rear_zone].max()) < 1.8:
        return mf, []

    # Protect tail fin past its lowest point
    tail_region = np.where((t_norm >= 0.74) & (t_norm <= 0.96))[0]
    tail_low_idx = int(tail_region[np.argmin(z_min_central[tail_region])]) if len(tail_region) else (nx - 1)
    z_for_drop = z_min_central.copy()
    for i in range(tail_low_idx, nx):
        z_for_drop[i] = min(z_for_drop[i], z_min_central[tail_low_idx])

    # Strictly zero on head/throat/pectoral fins (t < 0.42), full smooth seating across rear flexi zone (0.52..0.75),
    # and a wide gentle C2 ease-out (0.75..0.91) so there is never an abrupt step before the tail fin!
    rf = np.clip((t_norm - 0.41) / 0.12, 0.0, 1.0)
    rf = rf * rf * (3.0 - 2.0 * rf)
    rb = np.clip((0.90 - t_norm) / 0.15, 0.0, 1.0)
    rb = rb * rb * (3.0 - 2.0 * rb)
    raw_drop = np.minimum(z_for_drop, h_all * 0.28) * rf * rb

    kernel = np.array([0.08, 0.24, 0.36, 0.24, 0.08], dtype=np.float64)
    drop_smooth = raw_drop.copy()
    for _ in range(3):
        drop_smooth = np.convolve(np.pad(drop_smooth, 2, mode="edge"), kernel, mode="valid")
    drop_safe = np.minimum(drop_smooth, z_for_drop) * (t_norm >= 0.40) * (t_norm <= 0.91)
    drop_safe = np.convolve(np.pad(drop_safe, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid")
    if float(drop_safe.max()) < 1.0:
        return mf, []

    verts[:, 2] -= np.interp(verts[:, 0], x_centers, drop_safe)
    verts[:, 2] -= float(verts[:, 2].min())
    seated = _trimesh_to_manifold(trimesh.Trimesh(vertices=verts, faces=tm.faces, process=False))
    if seated is None or seated.is_empty():
        return mf, []
    if flip_180:
        seated = seated.rotate((0.0, 0.0, 180.0))
    if rotated_for_y:
        seated = seated.rotate((0.0, 0.0, 90.0))
    return _place_on_bed(seated), ["gently un-arched rear torso onto Z=0 bed (head, fins, and tail preserved)"]


def _bevel_segment_rims(seg: m3d.Manifold, bevel_mm: float = 0.45) -> m3d.Manifold:
    """Soften sharp 90° saw-cut rims into smooth pillowed designer links by tucking ONLY outer-rim vertices
    (shared between cut-cap faces `|nx| > 0.78` and outer-skin faces `|nx| <= 0.78`, above `Z = 0.40 mm`)
    inward along the local YZ skin normal. Never creates stair-step rabbets and keeps `Z = 0` bed pads flat."""
    if seg.is_empty():
        return seg
    tm = _manifold_to_trimesh(seg)
    verts = np.asarray(tm.vertices, dtype=np.float64).copy()
    faces = np.asarray(tm.faces, dtype=np.int64)
    if len(verts) < 12 or len(faces) < 12:
        return seg
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(flen, 1e-12))

    is_cap = np.abs(fn[:, 0]) > 0.78
    is_skin = ~is_cap
    if not np.any(is_cap) or not np.any(is_skin):
        return seg

    cap_v = np.zeros(len(verts), dtype=bool)
    cap_v[faces[is_cap].reshape(-1)] = True
    skin_faces = faces[is_skin]
    skin_fn = fn[is_skin]
    skin_v = np.zeros(len(verts), dtype=bool)
    skin_v[skin_faces.reshape(-1)] = True
    rim_mask = cap_v & skin_v & (verts[:, 2] > 0.40)
    if not np.any(rim_mask):
        return seg

    # Accumulate outward YZ skin normal and shortest skin edge length at each rim vertex
    vn_yz = np.zeros((len(verts), 2), dtype=np.float64)
    for slot in (0, 1, 2):
        np.add.at(vn_yz, skin_faces[:, slot], skin_fn[:, 1:3])
    nlen = np.linalg.norm(vn_yz, axis=1, keepdims=True)
    vn_yz = np.divide(vn_yz, np.maximum(nlen, 1e-12))

    min_edge = np.full(len(verts), np.inf, dtype=np.float64)
    for a_s, b_s in ((0, 1), (1, 2), (2, 0)):
        va = skin_faces[:, a_s]
        vb = skin_faces[:, b_s]
        elen = np.linalg.norm(verts[va] - verts[vb], axis=1)
        np.minimum.at(min_edge, va, elen)
        np.minimum.at(min_edge, vb, elen)

    rim_idx = np.nonzero(rim_mask)[0]
    disp = np.minimum(bevel_mm, 0.36 * min_edge[rim_idx])
    disp = np.where(np.isfinite(disp) & (disp > 0.04), disp, 0.0)
    verts[rim_idx, 1] -= vn_yz[rim_idx, 0] * disp
    verts[rim_idx, 2] = np.maximum(0.40, verts[rim_idx, 1 * 0 + 2] - vn_yz[rim_idx, 1] * disp)
    cand = _trimesh_to_manifold(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
    if cand is not None and not cand.is_empty() and cand.status() == m3d.Error.NoError:
        return cand
    return seg


def _seat_single_segment_organic(seg: m3d.Manifold, ceiling_z: float) -> m3d.Manifold:
    """If an individual segment hovers slightly above Z=0 (`0.15..8 mm`) OR touches Z=0 at only a tiny
    pinpoint (`< 6.5 mm²` flat bed pad), gently lower ONLY its ventral belly vertices (`z < ceiling_z`)
    and shave cleanly at Z=0 so every segment gets `>= 6.5 mm²` support-free bed adhesion."""
    sbb = seg.bounding_box()
    z_min = float(sbb[2])
    if z_min > 12.0:
        return seg
    tm = _manifold_to_trimesh(seg)
    v = np.asarray(tm.vertices, dtype=np.float64).copy()
    f = np.asarray(tm.faces, dtype=np.int64)
    if len(f) == 0:
        return seg
    # Measure current Z=0 bed contact area on this segment
    tv0, tv1, tv2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(tv1 - tv0, tv2 - tv0)
    dbl_a = np.linalg.norm(cross, axis=1)
    nz = np.divide(cross[:, 2], np.maximum(dbl_a, 1e-12))
    tz_max = np.maximum(np.maximum(tv0[:, 2], tv1[:, 2]), tv2[:, 2])
    cur_bed_area = float((0.5 * dbl_a[(tz_max <= 0.28) & (nz < -0.65)]).sum())
    if z_min <= 0.14 and cur_bed_area >= 6.5:
        return seg

    eff_ceil = max(ceiling_z, z_min + 2.4)
    span = max(1.6, eff_ceil - z_min)
    extra_pad = 0.58 if cur_bed_area < 6.5 else 0.35
    u = np.clip((v[:, 2] - z_min) / span, 0.0, 1.0)
    below = v[:, 2] < eff_ceil
    v[:, 2] = np.where(below, v[:, 2] - ((1.0 - u) ** 2) * (max(0.0, z_min) + extra_pad), v[:, 2])
    repaired = repair_to_manifold(trimesh.Trimesh(vertices=v, faces=tm.faces, process=False))[0]
    if repaired.is_empty():
        return seg
    # Shave the sub-zero pad flush at Z=0 so it forms a crisp flat first-layer contact patch
    rbb = repaired.bounding_box()
    if float(rbb[2]) < -0.005:
        sx = (rbb[3] - rbb[0]) + 20.0
        sy = (rbb[4] - rbb[1]) + 20.0
        sz = (rbb[5] - rbb[2]) + 20.0
        keep = m3d.Manifold.cube((sx, sy, sz), center=True).translate(
            ((rbb[0] + rbb[3]) / 2.0, (rbb[1] + rbb[4]) / 2.0, sz / 2.0)
        )
        trimmed = repaired ^ keep
        if not trimmed.is_empty() and trimmed.status() == m3d.Error.NoError:
            return trimmed
    return repaired


def _apply_flexi_articulation(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    """Slice a sculpted character into N+1 fin-safe, side-to-side S-swimming flexi segments.

    - Uses captive vertical Z-axis posts anchored into Segment i's Z=0 floor & upper roof, mated with
      rounded horizontal chain-link tongues from Segment i+1 (true side-to-side swimming S-wiggle + support-free 0° pin!)
    - Keeps a snug ~0.85 mm gill-curved exterior seam with a 360° sculpted outer rim bevel (no wide bread-loaf canyons!)
    """
    n_cuts = int(prep.flexi_segments)
    explicit_pcts = [float(p) for p in (prep.flexi_cuts_pct or []) if 10.0 <= float(p) <= 92.0]
    if n_cuts <= 0 and not explicit_pcts:
        return mf, []
    notes: list[str] = []
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    rotated_for_y = (prep.flexi_axis == "y") or (prep.flexi_axis == "auto" and sy > sx)
    work = mf.rotate((0.0, 0.0, -90.0)) if rotated_for_y else mf
    work = _place_on_bed(work)

    flip_180 = _detect_head_flip(work)
    if flip_180:
        work = _place_on_bed(work.rotate((0.0, 0.0, 180.0)))

    bb = work.bounding_box()
    x0, y0, z0, x1, y1, z1 = [float(v) for v in bb]
    length = x1 - x0
    width_all = y1 - y0
    height_all = z1 - z0
    if length < 20.0 or width_all < 8.0 or height_all < 6.0:
        notes.append("model is too small for print-in-place flexi joints (skipped)")
        return mf, notes

    c = _clamp(float(prep.flexi_clearance_mm), 0.25, 0.80)
    gap = max(0.82, 1.85 * c)
    if explicit_pcts:
        sorted_pcts = sorted(explicit_pcts)
        solved_cuts = [(p / 100.0, False) for p in sorted_pcts]
    else:
        # Target 5 cuts (6 segments: head + chest + 3 rear links + tail) when n_cuts >= 4 on a streamlined >= 1.95:1 body
        req_cuts = max(5, n_cuts) if (n_cuts >= 4 and length / max(1.0, height_all) >= 1.95) else max(1, n_cuts)
        anat = analyze_sculpt_anatomy(work, n_cuts=req_cuts, s0_pct=float(prep.flexi_start_pct), s1_pct=float(prep.flexi_end_pct))
        solved_cuts = anat["solved_cuts"]

    cuts = [x0 + length * t_val for t_val, _ in solved_cuts]
    force_flat = [bool(fl) for _, fl in solved_cuts]
    bounds = [x0 - 5.0, *cuts, x1 + 5.0]
    pad_y = width_all + 40.0
    pad_z = height_all + 40.0
    # Tight, refined flank flare (~3.5° half-angle, max 1.35 mm half-gap) so exterior seams look sleek, never gaping!
    eff_deg = _clamp(float(prep.flexi_angle_deg), 6.0, 14.0)
    tan_half = math.tan(math.radians(eff_deg / 2.0)) * 0.48

    joint_specs: list[dict[str, float]] = []
    active_cuts: list[float] = []
    for i, u in enumerate(cuts):
        slab_box = m3d.Manifold.cube((2.0, pad_y, pad_z), center=True).translate((u, (y0 + y1) / 2.0, height_all / 2.0))
        slab = work ^ slab_box
        if slab.is_empty():
            continue
        sbb = slab.bounding_box()
        w_loc = float(sbb[4] - sbb[1])
        z_bot = max(0.0, float(sbb[2]))
        z_top = float(sbb[5])
        h_loc = z_top - z_bot
        yc = float((sbb[1] + sbb[4]) / 2.0)
        prev_cut = active_cuts[-1] if active_cuts else bounds[i]
        seg_pitch = min(u - prev_cut, bounds[i + 2] - u)
        # Skip elevated thin tail stems so tail waist + crescent tail fin remain one graceful solid tail piece!
        if z_bot > 5.5 or w_loc < 7.2 or h_loc < 7.0 or seg_pitch < (gap + 4.0 * c + 2.8):
            continue

        floor_t = _clamp(h_loc * 0.14, 1.40, 2.2)
        roof_t = _clamp(h_loc * 0.15, 1.50, 2.5)
        avail_h = h_loc - floor_t - roof_t - 2.0 * c
        if avail_h < 2.1:
            continue
        loop_h = _clamp(avail_h, 2.1, 6.2)
        cav_z0 = z_bot + floor_t
        cav_z1 = cav_z0 + loop_h + 2.0 * c

        pin_r = _clamp(min(w_loc * 0.085, (seg_pitch - gap) * 0.15), 1.05, 1.75)
        wall_xy = _clamp(pin_r * 0.92, 1.15, 1.65)
        r_loop_xy = pin_r + c + wall_xy
        slot_w = 2.0 * (r_loop_xy + c + 0.25)
        if slot_w > w_loc - 3.0:
            # Scale down pin_r & wall_xy proportionally on narrower rear tail joints
            max_r_loop = (w_loc - 3.0) / 2.0 - c - 0.25
            if max_r_loop < 1.95 + c:
                continue
            pin_r = max(0.95, (max_r_loop - c) * 0.48)
            wall_xy = max(1.00, (max_r_loop - c) * 0.52)
            r_loop_xy = pin_r + c + wall_xy
            slot_w = 2.0 * (r_loop_xy + c + 0.25)

        u_pin = u - (gap / 2.0 + c + pin_r + 0.35)
        cavity_front_x = u_pin - (r_loop_xy + c)
        min_front = (active_cuts[-1] + gap / 2.0 + 1.2) if active_cuts else (x0 + 1.5)
        if cavity_front_x <= min_front:
            continue

        curve = 0.0 if force_flat[i] else min(w_loc * 0.12, seg_pitch * 0.20, 3.8)
        max_half_gap = min(1.45, max(gap / 2.0 + 0.25, seg_pitch * 0.16))
        active_cuts.append(u)
        joint_specs.append({
            "u": u, "yc": yc, "w_loc": w_loc, "z_bot": z_bot, "z_top": z_top, "h_loc": h_loc,
            "pin_r": pin_r, "r_loop_xy": r_loop_xy, "slot_w": slot_w,
            "cav_z0": cav_z0, "cav_z1": cav_z1, "u_pin": u_pin, "cavity_front_x": cavity_front_x,
            "curve": curve, "max_half_gap": max_half_gap,
        })

    if not active_cuts:
        notes.append("no cross-sections were wide enough for flexi joints")
        return mf, notes

    # Partition into N+1 sleek gill-curved segments + subtle 360° outer skin bevels
    n_samples = 29
    ys = np.linspace(y0 - 8.0, y1 + 8.0, n_samples).tolist()

    def _cut_curve_x(j: int, y_val: float, side: float, extra_bevel: float = 0.0) -> float:
        sp = joint_specs[j]
        u = sp["u"]
        yc = sp["yc"]
        half_w = max(4.0, sp["w_loc"] / 2.0)
        half_slot = sp["slot_w"] / 2.0 + 0.35
        base_x = _crescent_x(u, y_val, yc, half_slot, half_w, sp["curve"])
        flank = min(half_w + 2.0, max(0.0, abs(y_val - yc) - half_slot))
        half_gap = min(sp["max_half_gap"], gap / 2.0 + flank * tan_half) + extra_bevel
        return base_x + side * half_gap

    segments: list[m3d.Manifold] = []
    slab_prisms: list[m3d.Manifold] = []
    for seg_i in range(len(active_cuts) + 1):
        right_pts = [
            (x1 + 8.0, yv) if seg_i == len(active_cuts) else (_cut_curve_x(seg_i, yv, -1.0), yv)
            for yv in ys
        ]
        left_pts = [
            (x0 - 8.0, yv) if seg_i == 0 else (_cut_curve_x(seg_i - 1, yv, +1.0), yv)
            for yv in reversed(ys)
        ]
        poly = [*right_pts, *left_pts]
        slab_prism = m3d.Manifold.extrude(m3d.CrossSection([poly]), pad_z).translate((0.0, 0.0, -5.0))
        slab_prisms.append(slab_prism)
        seg_raw = _clean_segment_slivers(work ^ slab_prism)
        seg_raw = _bevel_segment_rims(seg_raw, bevel_mm=0.45)

        # Ensure middle segments touch Z=0 organically BEFORE hinge pockets are cut
        ceil_z = joint_specs[min(seg_i, len(joint_specs) - 1)]["cav_z0"] - 0.2
        seg_raw = _seat_single_segment_organic(seg_raw, ceiling_z=ceil_z)
        segments.append(seg_raw)

    # Carve captive horizontal pockets + vertical Z-pins in Segment i, and union horizontal chain-link tongues on Segment i+1
    for j_idx, sp in enumerate(joint_specs):
        u = sp["u"]
        yc = sp["yc"]
        pin_r = sp["pin_r"]
        r_loop_xy = sp["r_loop_xy"]
        cav_z0 = sp["cav_z0"]
        cav_z1 = sp["cav_z1"]
        u_pin = sp["u_pin"]
        cavity_back_x = u + gap / 2.0 + 1.2

        # 1. Segment i (upstream): subtract enclosed horizontal pocket (preserving Z=0 floor & top roof!), add vertical Z-post
        cavity = _rounded_xy_prism(
            x_center=u_pin, y_center=yc, radius=r_loop_xy + c,
            x_back=cavity_back_x, z0=cav_z0, z1=cav_z1,
        )
        pin_h = (cav_z1 - cav_z0) + 1.8
        pin = m3d.Manifold.cylinder(pin_h, pin_r, pin_r, circular_segments=32).translate((u_pin, yc, cav_z0 - 0.9))
        pin = pin ^ work ^ slab_prisms[j_idx]
        segments[j_idx] = (segments[j_idx] - cavity) + pin

        # 2. Segment i+1 (downstream): horizontal rounded chain-link tongue wrapping 360° around vertical Z-post
        next_limit = (
            joint_specs[j_idx + 1]["cavity_front_x"] - 0.6
            if j_idx + 1 < len(joint_specs)
            else x1 - 1.2
        )
        anchor_x = min(u + gap / 2.0 + 3.2, next_limit)
        if anchor_x <= u + gap / 2.0 + 0.8:
            anchor_x = u + gap / 2.0 + 1.0
        z_l0 = cav_z0 + c
        z_l1 = cav_z1 - c
        tongue_outer = _rounded_xy_prism(
            x_center=u_pin, y_center=yc, radius=r_loop_xy,
            x_back=anchor_x, z0=z_l0, z1=z_l1,
        )
        pin_hole = m3d.Manifold.cylinder(
            (z_l1 - z_l0) + 4.0, pin_r + c, pin_r + c, circular_segments=32
        ).translate((u_pin, yc, z_l0 - 2.0))
        tongue = (tongue_outer - pin_hole) ^ work
        segments[j_idx + 1] = segments[j_idx + 1] + tongue

    non_empty = [s for s in segments if not s.is_empty() and s.status() == m3d.Error.NoError]
    if len(non_empty) < 2:
        return mf, notes
    composed = m3d.Manifold.compose(non_empty)
    if flip_180:
        composed = composed.rotate((0.0, 0.0, 180.0))
    if rotated_for_y:
        composed = composed.rotate((0.0, 0.0, 90.0))
    composed = _place_on_bed(composed)
    cut_pct_strs = [f"{((u - x0) / length) * 100.0:.1f}%" for u in active_cuts]
    notes.append(
        f"sliced {len(active_cuts)} fin-safe Z-pin swimming flexi joints at [{', '.join(cut_pct_strs)}] "
        f"({len(non_empty)} beveled segments, {c:.2f} mm clearance)"
    )
    return composed, notes


def inspect_sculpt_anatomy_file(raw_path: Path, prep: SculptPrep) -> dict:
    """Load, orient, and scale a raw sculpt according to `prep` and return its 3D anatomical profile."""
    raw = load_raw_mesh(raw_path)
    mf, _ = repair_to_manifold(raw)
    if prep.rotate_x or prep.rotate_y or prep.rotate_z:
        mf = mf.rotate((float(prep.rotate_x), float(prep.rotate_y), float(prep.rotate_z)))
    bb = mf.bounding_box()
    cur_h = float(bb[5] - bb[2])
    if prep.target_height_mm > 0 and cur_h > 1e-6:
        s = float(prep.target_height_mm) / cur_h
        mf = mf.scale((s, s, s))
    mf = _place_on_bed(mf)
    if int(prep.flexi_segments) > 0 or bool(prep.flexi_cuts_pct):
        mf, _ = _streamline_flexi_proportions(mf, prep)
        mf, _ = _seat_flexi_on_bed(mf, prep)
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    rotated_for_y = (prep.flexi_axis == "y") or (prep.flexi_axis == "auto" and sy > sx)
    work = _place_on_bed(mf.rotate((0.0, 0.0, -90.0)) if rotated_for_y else mf)
    flip_180 = _detect_head_flip(work)
    if flip_180:
        work = _place_on_bed(work.rotate((0.0, 0.0, 180.0)))
    anat = analyze_sculpt_anatomy(
        work,
        n_cuts=max(1, int(prep.flexi_segments) or 5),
        s0_pct=float(prep.flexi_start_pct),
        s1_pct=float(prep.flexi_end_pct),
    )
    anat.pop("solved_cuts", None)
    anat["spine_axis"] = "Y" if rotated_for_y else "X"
    anat["head_flipped_180"] = flip_180
    return anat


def _apply_magnet_hole(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    if prep.magnet_hole == "none":
        return mf, []
    bb = mf.bounding_box()
    cx = float((bb[0] + bb[3]) / 2.0)
    cy = float((bb[1] + bb[4]) / 2.0)
    cz = float((bb[2] + bb[5]) / 2.0)
    r = float(prep.magnet_dia_mm) / 2.0
    depth = float(prep.magnet_depth_mm)
    if prep.magnet_hole == "bottom":
        cyl = m3d.Manifold.cylinder(depth + 1.0, r, r, circular_segments=48).translate((cx, cy, float(bb[2]) - 1.0))
    else:  # "back" (-Y face)
        cyl = (
            m3d.Manifold.cylinder(depth + 1.0, r, r, circular_segments=48)
            .rotate((-90.0, 0.0, 0.0))
            .translate((cx, float(bb[1]) - 1.0, cz))
        )
    cut = mf - cyl
    if cut.is_empty() or cut.status() != m3d.Error.NoError:
        return mf, []
    return cut, [f"cut {prep.magnet_hole} magnet pocket (Ø{prep.magnet_dia_mm:.1f} × {depth:.1f} mm)"]


def _apply_hollow_cavity(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    if not prep.hollow_cavity:
        return mf, []
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    sz = float(bb[5] - bb[2])
    wall = max(1.5, float(prep.hollow_wall_mm))
    rx = sx / 2.0 - wall
    ry = sy / 2.0 - wall
    if rx < 4.0 or ry < 4.0 or sz < wall * 2.5:
        return mf, []
    cx = float((bb[0] + bb[3]) / 2.0)
    cy = float((bb[1] + bb[4]) / 2.0)
    cav = (
        m3d.Manifold.cylinder(sz + 5.0, 1.0, 1.0, circular_segments=48)
        .scale((rx, ry, 1.0))
        .translate((cx, cy, float(bb[2]) + wall))
    )
    cut = mf - cav
    if cut.is_empty() or cut.status() != m3d.Error.NoError:
        return mf, []
    return cut, [f"hollowed top cavity ({wall:.1f} mm wall)"]


def prepare_print_mesh(
    raw_path: Path,
    prep: SculptPrep,
    name: str = "Sculpt",
    color: str | None = "#b8c2cc",
    filament: int | None = None,
    cad_cuts: list[tuple[np.ndarray, np.ndarray]] | None = None,
    cad_unions: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> PreparedSculpt:
    """Load, repair, orient, scale, flatten bottom, apply hybrid CAD booleans, and cut flexi/hardware finishing."""
    raw = load_raw_mesh(raw_path)
    mf, notes = repair_to_manifold(raw)

    if prep.rotate_x or prep.rotate_y or prep.rotate_z:
        mf = mf.rotate((float(prep.rotate_x), float(prep.rotate_y), float(prep.rotate_z)))

    bb = mf.bounding_box()
    cur_h = float(bb[5] - bb[2])
    if prep.target_height_mm > 0 and cur_h > 1e-6:
        s = float(prep.target_height_mm) / cur_h
        mf = mf.scale((s, s, s))

    mf = _place_on_bed(mf)
    has_flexi = int(prep.flexi_segments) > 0 or bool(prep.flexi_cuts_pct)

    # For print-in-place flexi models, streamline chubby figurine proportions behind the head and gently un-arch
    # any floating rear belly onto Z=0 BEFORE subtle belly bed-ribbon trimming.
    if has_flexi:
        mf, stream_notes = _streamline_flexi_proportions(mf, prep)
        notes.extend(stream_notes)
        mf, seat_notes = _seat_flexi_on_bed(mf, prep)
        notes.extend(seat_notes)

    # Flatten bottom by trimming off Z=0 so bed adhesion is rock-solid.
    # For print-in-place flexi models, shave only a subtle ~1.8% ventral ribbon (capped at 1.05 mm) so the shark
    # keeps its sleek rounded belly silhouette while still establishing solid Z=0 first-layer pads.
    eff_flat = float(prep.flatten_bottom_mm)
    if has_flexi and eff_flat < 0.9:
        bb_cur = mf.bounding_box()
        h_cur = float(bb_cur[5] - bb_cur[2])
        eff_flat = max(eff_flat, min(1.05, h_cur * 0.018))
    if eff_flat > 0:
        bb = mf.bounding_box()
        cut = min(eff_flat, max(0.0, (bb[5] - bb[2]) * 0.15))
        if cut > 0.01:
            span_x = (bb[3] - bb[0]) + 40.0
            span_y = (bb[4] - bb[1]) + 40.0
            span_z = (bb[5] - bb[2]) + 40.0
            keep_box = m3d.Manifold.cube((span_x, span_y, span_z), center=True).translate(
                ((bb[0] + bb[3]) / 2.0, (bb[1] + bb[4]) / 2.0, cut + span_z / 2.0)
            )
            trimmed = mf ^ keep_box
            if not trimmed.is_empty() and trimmed.status() == m3d.Error.NoError:
                mf = _place_on_bed(trimmed)
                notes.append(f"flattened bottom by {cut:.2f} mm")

    # Optional display base / plinth (skip if flexi articulation is enabled so joints remain free on bed)
    if prep.base_style != "none" and not has_flexi:
        bb = mf.bounding_box()
        sx = float(bb[3] - bb[0])
        sy = float(bb[4] - bb[1])
        bh = max(1.5, float(prep.base_height_mm))
        overlap = min(0.6, bh * 0.3)
        base = _build_base(prep.base_style, sx, sy, bh, prep.base_margin_mm)
        mf = _place_on_bed(base + mf.translate((0.0, 0.0, bh - overlap)))
        notes.append(f"added {prep.base_style} display base ({bh:.1f} mm)")

    # Optional keychain loop at top peak (applied before flexi cuts so it stays anchored to head/body)
    if prep.keychain_loop:
        tm = _manifold_to_trimesh(mf)
        top_z = float(tm.vertices[:, 2].max())
        near_top = tm.vertices[tm.vertices[:, 2] >= top_z - 2.0]
        tx = float(near_top[:, 0].mean()) if len(near_top) else 0.0
        ty = float(near_top[:, 1].mean()) if len(near_top) else 0.0
        loop = _build_keychain_loop(prep.keychain_hole_mm, tx, ty, top_z)
        mf = _place_on_bed(mf + loop)
        notes.append(f"added keychain loop (Ø{prep.keychain_hole_mm:.1f} mm hole)")

    # Hybrid CAD + Clay boolean operations (union or subtract parametric OpenCascade CAD bodies into the sculpt)
    for u_pts, u_tris in (cad_unions or ()):
        if len(u_pts) and len(u_tris):
            u_mf, _ = repair_to_manifold(trimesh.Trimesh(vertices=u_pts, faces=u_tris, process=False))
            unioned = mf + u_mf
            if not unioned.is_empty() and unioned.status() == m3d.Error.NoError:
                mf = unioned
                notes.append("unioned parametric CAD body")
    for c_pts, c_tris in (cad_cuts or ()):
        if len(c_pts) and len(c_tris):
            c_mf, _ = repair_to_manifold(trimesh.Trimesh(vertices=c_pts, faces=c_tris, process=False))
            carved = mf - c_mf
            if not carved.is_empty() and carved.status() == m3d.Error.NoError:
                mf = carved
                notes.append("subtracted parametric CAD body")

    # Optional functional hardware cutouts (magnet pocket, hollowed planter/cup cavity)
    if prep.magnet_hole != "none":
        mf, m_notes = _apply_magnet_hole(mf, prep)
        notes.extend(m_notes)
    if prep.hollow_cavity:
        mf, h_notes = _apply_hollow_cavity(mf, prep)
        notes.extend(h_notes)

    # Optional print-in-place flexi articulation (precision CAD hinge cuts through sculpted character)
    if has_flexi:
        mf, f_notes = _apply_flexi_articulation(mf, prep)
        notes.extend(f_notes)

    final_tm = _manifold_to_trimesh(mf)
    bb = mf.bounding_box()
    vol = abs(float(mf.volume()))
    part = PreparedPart(
        name=name,
        points=np.asarray(final_tm.vertices, dtype=np.float64),
        tris=np.asarray(final_tm.faces, dtype=np.int32),
        color=color,
        filament=filament,
    )
    return PreparedSculpt(
        parts=[part],
        watertight=bool(mf.status() == m3d.Error.NoError and final_tm.is_watertight),
        triangles=part.triangles,
        volume_mm3=vol,
        bbox_min=[round(float(bb[0]), 4), round(float(bb[1]), 4), round(float(bb[2]), 4)],
        bbox_max=[round(float(bb[3]), 4), round(float(bb[4]), 4), round(float(bb[5]), 4)],
        notes=notes,
    )


def _split_sharp_edges(points: np.ndarray, tris: np.ndarray, angle_deg: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Auto-smooth vertex splitting across sharp dihedral creases (e.g., 90° flexi cuts & flat bellies).

    Keeps smooth organic skin vertices welded so Three.js `computeVertexNormals()` renders silky shading,
    while splitting vertices across creases steeper than `angle_deg` so CAD cut rims render razor-crisp
    with zero dark normal-averaging bruises. Never alters exported STL/3MF geometry.
    """
    n_verts = len(points)
    n_faces = len(tris)
    if n_verts == 0 or n_faces == 0:
        return points, tris

    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(flen, 1e-12))

    # Build undirected edge keys for all 3*F half-edges and match shared edges in O(F log F)
    e0 = tris[:, [0, 1]]
    e1 = tris[:, [1, 2]]
    e2 = tris[:, [2, 0]]
    all_e = np.vstack((e0, e1, e2))
    ea = np.minimum(all_e[:, 0], all_e[:, 1]).astype(np.int64)
    eb = np.maximum(all_e[:, 0], all_e[:, 1]).astype(np.int64)
    f_idx = np.tile(np.arange(n_faces, dtype=np.int32), 3)

    keys = ea * n_verts + eb
    order = np.argsort(keys)
    k_sorted = keys[order]
    same = k_sorted[:-1] == k_sorted[1:]
    if not np.any(same):
        return points, tris

    o1 = order[:-1][same]
    o2 = order[1:][same]
    f1 = f_idx[o1]
    f2 = f_idx[o2]
    dots = np.einsum("ij,ij->i", fn[f1], fn[f2])
    cos_thresh = math.cos(math.radians(angle_deg))
    sharp_mask = dots < cos_thresh
    if not np.any(sharp_mask):
        return points, tris

    sharp_verts = np.unique(np.concatenate((ea[o1[sharp_mask]], eb[o1[sharp_mask]])))
    is_sharp_v = np.zeros(n_verts, dtype=bool)
    is_sharp_v[sharp_verts] = True

    # Collect incident (face, corner_slot) only for sharp vertices
    v_corners: dict[int, list[tuple[int, int]]] = {int(v): [] for v in sharp_verts}
    t_flat = tris.reshape(-1)
    sharp_corner_idx = np.nonzero(is_sharp_v[t_flat])[0]
    for ci in sharp_corner_idx.tolist():
        v_corners[int(t_flat[ci])].append((ci // 3, ci % 3))

    # Collect smooth edge face-adjacency only at sharp vertices
    smooth_mask = ~sharp_mask
    sm_a = ea[o1[smooth_mask]]
    sm_b = eb[o1[smooth_mask]]
    sm_f1 = f1[smooth_mask]
    sm_f2 = f2[smooth_mask]
    touch_sharp = is_sharp_v[sm_a] | is_sharp_v[sm_b]
    smooth_adj: dict[int, dict[int, list[int]]] = {}
    for a_v, b_v, fa, fb in zip(
        sm_a[touch_sharp].tolist(), sm_b[touch_sharp].tolist(),
        sm_f1[touch_sharp].tolist(), sm_f2[touch_sharp].tolist(),
    ):
        if is_sharp_v[a_v]:
            d = smooth_adj.setdefault(a_v, {})
            d.setdefault(fa, []).append(fb)
            d.setdefault(fb, []).append(fa)
        if is_sharp_v[b_v]:
            d = smooth_adj.setdefault(b_v, {})
            d.setdefault(fa, []).append(fb)
            d.setdefault(fb, []).append(fa)

    new_tris = tris.copy()
    extra_pts: list[np.ndarray] = []
    next_vid = n_verts

    for v in sharp_verts.tolist():
        corners = v_corners[v]
        if len(corners) <= 1:
            continue
        adj = smooth_adj.get(v)
        # Map face -> corner slot(s) at vertex v
        f_to_slots: dict[int, list[int]] = {}
        for f, slot in corners:
            f_to_slots.setdefault(f, []).append(slot)
        visited: set[int] = set()
        group_idx = 0
        for start_f in f_to_slots:
            if start_f in visited:
                continue
            # BFS over faces incident to v connected by smooth edges
            comp = [start_f]
            visited.add(start_f)
            if adj:
                head = 0
                while head < len(comp):
                    cur = comp[head]
                    head += 1
                    for nxt in adj.get(cur, ()):
                        if nxt in f_to_slots and nxt not in visited:
                            visited.add(nxt)
                            comp.append(nxt)
            if group_idx > 0:
                vid = next_vid
                next_vid += 1
                extra_pts.append(points[v])
                for f in comp:
                    for slot in f_to_slots[f]:
                        new_tris[f, slot] = vid
            group_idx += 1

    if not extra_pts:
        return points, tris
    new_points = np.vstack((points, np.asarray(extra_pts, dtype=points.dtype)))
    return new_points, new_tris


def to_viewer_mesh(part: PreparedPart, bid: str) -> dict:
    """Convert a `PreparedPart` into the `ShapeMesh`-compatible dict that `ViewportPane` renders in Three.js."""
    bmin, bmax = part.bounds()
    centroid = ((bmin + bmax) / 2.0).tolist()
    pts, tris = _split_sharp_edges(part.points, part.tris, angle_deg=40.0)
    pos = np.round(pts.astype(np.float64), 4).reshape(-1).tolist()
    idx = tris.astype(np.int64).reshape(-1).tolist()
    return {
        "name": bid,
        "units": "mm",
        "bbox_min": [round(float(v), 4) for v in bmin],
        "bbox_max": [round(float(v), 4) for v in bmax],
        "faces": [{
            "id": 0,
            "type": "mesh",
            "area": 0.0,
            "centroid": [round(float(v), 4) for v in centroid],
            "positions": pos,
            "indices": idx,
        }],
        "edges": [],
        "face_names": ["sculpt.surface"],
    }


def export_prepared_meshes(parts: list[PreparedPart], printer: PrinterProfile, fmt: str, path: Path,
                           title: str = "", design: Design | None = None) -> Path:
    """Export prepared organic meshes to STL or 3MF centred on the printer's bed."""
    if fmt == "step":
        raise MeshPrepError("Character Mode sculpted meshes export to 3MF or STL (STEP is for Engineer CAD solids)")
    if fmt not in ("stl", "3mf"):
        raise MeshPrepError(f"unsupported export format {fmt!r}")
    if not parts:
        raise MeshPrepError("nothing is built to export")

    all_pts = np.vstack([p.points for p in parts])
    bmin, bmax = all_pts.min(axis=0), all_pts.max(axis=0)
    dx = printer.bed[0] / 2.0 - float(bmin[0] + bmax[0]) / 2.0
    dy = printer.bed[1] / 2.0 - float(bmin[1] + bmax[1]) / 2.0
    dz = -float(bmin[2])
    shift = np.array([dx, dy, dz], dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "stl":
        meshes = [trimesh.Trimesh(vertices=p.points + shift, faces=p.tris, process=False) for p in parts]
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        combined.export(str(path), file_type="stl")
        return path

    # 3MF export: write clean multi-component 3MF with units=millimeter and optional color/filament info
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    mat_ns = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    ET.register_namespace("", ns)
    ET.register_namespace("m", mat_ns)
    model = ET.Element(f"{{{ns}}}model", {"unit": "millimeter", "xml:lang": "en-US"})
    meta = ET.SubElement(model, f"{{{ns}}}metadata", {"name": "Title"})
    meta.text = title or "3D Print Assist Character"
    res = ET.SubElement(model, f"{{{ns}}}resources")
    basemats = ET.SubElement(res, f"{{{mat_ns}}}basematerials", {"id": "1"})
    for i, p in enumerate(parts):
        col = (p.color or "#b8c2cc").upper()
        if len(col) == 7:
            col = col + "FF"
        ET.SubElement(basemats, f"{{{mat_ns}}}base", {"name": p.name or f"Part {i + 1}", "displaycolor": col})

    part_ids: list[int] = []
    for i, p in enumerate(parts):
        oid = i + 2
        part_ids.append(oid)
        obj = ET.SubElement(res, f"{{{ns}}}object", {
            "id": str(oid), "type": "model", "name": p.name or f"Sculpt {i + 1}",
            "pid": "1", "pindex": str(i),
        })
        mesh_el = ET.SubElement(obj, f"{{{ns}}}mesh")
        verts_el = ET.SubElement(mesh_el, f"{{{ns}}}vertices")
        pts = p.points + shift
        for v in pts:
            ET.SubElement(verts_el, f"{{{ns}}}vertex", {
                "x": f"{v[0]:.5f}", "y": f"{v[1]:.5f}", "z": f"{v[2]:.5f}",
            })
        tris_el = ET.SubElement(mesh_el, f"{{{ns}}}triangles")
        for t in p.tris:
            ET.SubElement(tris_el, f"{{{ns}}}triangle", {
                "v1": str(int(t[0])), "v2": str(int(t[1])), "v3": str(int(t[2])),
            })

    build = ET.SubElement(model, f"{{{ns}}}build")
    if len(part_ids) > 1:
        asm_id = max(part_ids) + 1
        asm = ET.SubElement(res, f"{{{ns}}}object", {"id": str(asm_id), "type": "model", "name": title or "Character"})
        comps = ET.SubElement(asm, f"{{{ns}}}components")
        for pid in part_ids:
            ET.SubElement(comps, f"{{{ns}}}component", {"objectid": str(pid)})
        ET.SubElement(build, f"{{{ns}}}item", {"objectid": str(asm_id)})
    else:
        ET.SubElement(build, f"{{{ns}}}item", {"objectid": str(part_ids[0])})

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )
    xml_bytes = ET.tostring(model, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", xml_bytes)
    return path


def analyze_sculpt_printability(parts: list[PreparedPart], max_overhang_deg: float = 48.0) -> dict:
    """Measure real FDM printability across all connected components (segments) of prepared sculpt meshes:
    - Per-component `z_min` and flat `Z=0` first-layer bed contact area (`mm²`)
    - Floating component detection (`z_min > 0.25 mm` or `bed_area < 4.0 mm²`)
    - Unsupported steep downward overhang area (`z > 1.6 mm`, slope steeper than `max_overhang_deg`)
    """
    if not parts:
        return {
            "component_count": 0,
            "seated_count": 0,
            "floating_count": 0,
            "total_bed_area_mm2": 0.0,
            "min_component_bed_area_mm2": 0.0,
            "overhang_area_mm2": 0.0,
            "components": [],
        }

    all_pts = np.vstack([p.points for p in parts])
    global_z0 = float(all_pts[:, 2].min())
    cos_over = math.cos(math.radians(max(25.0, min(75.0, float(max_overhang_deg)))))

    comp_rows: list[dict] = []
    overhang_area = 0.0
    for p in parts:
        tm = trimesh.Trimesh(vertices=p.points, faces=p.tris, process=False)
        try:
            comps = tm.split(only_watertight=False)
        except Exception:  # noqa: BLE001
            comps = [tm]
        # Ignore microscopic dust (< 5 triangles)
        comps = [c for c in comps if len(c.faces) >= 6] or [tm]
        # Sort along X so segments read head-to-tail
        comps.sort(key=lambda c: float(c.vertices[:, 0].mean()))
        for idx, comp in enumerate(comps):
            v = np.asarray(comp.vertices, dtype=np.float64)
            f = np.asarray(comp.faces, dtype=np.int64)
            z_min = float(v[:, 2].min()) - global_z0
            v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            dbl_area = np.linalg.norm(cross, axis=1)
            areas = 0.5 * dbl_area
            nz = np.divide(cross[:, 2], np.maximum(dbl_area, 1e-12))
            tri_z_max = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2]) - global_z0
            tri_z_min = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2]) - global_z0

            on_bed = (tri_z_max <= 0.30) & (nz < -0.65)
            bed_area = float(areas[on_bed].sum())
            # Count vertices near Z=0 if bottom wasn't planar-trimmed yet
            if bed_area < 1.0 and z_min <= 0.25:
                near_bed = (tri_z_min <= 0.35) & (nz < -0.35)
                bed_area = float(areas[near_bed].sum()) * 0.5

            # Steep downward faces above first 2 mm (excluding short internal self-bridging hinge pin ceilings)
            steep_down = (tri_z_min > 2.2) & (nz < -cos_over)
            overhang_area += float(areas[steep_down].sum())

            seated = (z_min <= 0.28) and (bed_area >= 4.0)
            comp_rows.append({
                "index": idx + 1,
                "z_min_mm": round(z_min, 2),
                "bed_area_mm2": round(bed_area, 1),
                "seated": seated,
            })

    seated_n = sum(1 for r in comp_rows if r["seated"])
    floating_n = len(comp_rows) - seated_n
    total_bed = round(sum(r["bed_area_mm2"] for r in comp_rows), 1)
    min_bed = round(min((r["bed_area_mm2"] for r in comp_rows), default=0.0), 1)
    return {
        "component_count": len(comp_rows),
        "seated_count": seated_n,
        "floating_count": floating_n,
        "total_bed_area_mm2": total_bed,
        "min_component_bed_area_mm2": min_bed,
        "overhang_area_mm2": round(overhang_area, 1),
        "components": comp_rows,
    }


def _render_single_mesh_panel(
    parts: list[PreparedPart],
    view: str,
    w: int,
    h: int,
    highlight_bed: bool = False,
    draw_bed_line: bool = False,
) -> "Image.Image":
    """True per-pixel Z-buffered, smooth-vertex-normal, 2x SSAA 3-point studio renderer for sculpted meshes."""
    from PIL import Image, ImageDraw

    bg_top = np.array([34.0, 39.0, 48.0], dtype=np.float32)
    bg_bot = np.array([22.0, 26.0, 33.0], dtype=np.float32)
    if not parts:
        return Image.new("RGB", (w, h), (28, 32, 40))

    all_pts = np.vstack([p.points for p in parts])
    span_xyz = all_pts.max(axis=0) - all_pts.min(axis=0)
    y_dominant = float(span_xyz[1]) > float(span_xyz[0]) * 1.08

    # Camera basis: `fwd` points FROM eye INTO scene (larger depth `sz = q @ fwd` = farther from camera).
    # For X-dominant models (head at -X, tail at +X), looking along +Y (`d = (0, 1, 0)`) puts head on LEFT, tail on RIGHT!
    if y_dominant:
        view_defs = {
            "iso": ((-0.90, -0.44, -0.36), (0.0, 0.0, 1.0)),
            "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "back": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "top": ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
            "bottom": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        }
    else:
        view_defs = {
            "iso": ((0.44, 0.90, -0.36), (0.0, 0.0, 1.0)),
            "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "back": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            "bottom": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        }
    d_vec, up_hint = view_defs.get(view, view_defs["iso"])
    fwd = np.asarray(d_vec, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_v = np.asarray(up_hint, dtype=np.float64)
    right = np.cross(fwd, up_v)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)

    ss = 2 if max(w, h) <= 900 else 1
    rw, rh = w * ss, h * ss
    gy = np.linspace(0.0, 1.0, rh, dtype=np.float32)[:, None, None]
    fb = (1.0 - gy) * bg_top[None, None, :] + gy * bg_bot[None, None, :]
    fb = np.broadcast_to(fb, (rh, rw, 3)).copy()
    zbuf = np.full((rh, rw), np.inf, dtype=np.float64)

    global_z0 = float(all_pts[:, 2].min())
    global_h = max(1e-3, float(all_pts[:, 2].max() - global_z0))
    center = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
    q_all = all_pts - center
    proj_x = q_all @ right
    proj_y = q_all @ up
    span_x = max(float(proj_x.max() - proj_x.min()), 1e-3)
    span_y = max(float(proj_y.max() - proj_y.min()), 1e-3)
    scale = min((rw * 0.84) / span_x, (rh * 0.76) / span_y)
    cx_mid = 0.5 * float(proj_x.min() + proj_x.max())
    cy_mid = 0.5 * float(proj_y.min() + proj_y.max())

    # 3-point studio lighting in view space (x=right, y=up, z=into-screen -> outward normals toward camera have nv_z < 0)
    key_l = np.array([-0.38, 0.58, -0.72], dtype=np.float64)
    key_l /= np.linalg.norm(key_l)
    fill_l = np.array([0.62, 0.12, -0.77], dtype=np.float64)
    fill_l /= np.linalg.norm(fill_l)
    back_l = np.array([0.15, 0.78, 0.60], dtype=np.float64)
    back_l /= np.linalg.norm(back_l)

    for p in parts:
        if len(p.tris) == 0:
            continue
        hex_c = (p.color or "#cbd5e1").lstrip("#")
        base_c = (
            np.array([int(hex_c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
            if len(hex_c) == 6
            else np.array([0.80, 0.84, 0.88], dtype=np.float32)
        )
        # Split sharp CAD creases (42°) while computing silky area-weighted smooth normals on organic clay skin
        s_pts, s_tris = _split_sharp_edges(p.points, p.tris, angle_deg=42.0)
        tv0, tv1, tv2 = s_pts[s_tris[:, 0]], s_pts[s_tris[:, 1]], s_pts[s_tris[:, 2]]
        fn_raw = np.cross(tv1 - tv0, tv2 - tv0)
        vn = np.zeros_like(s_pts)
        np.add.at(vn, s_tris[:, 0], fn_raw)
        np.add.at(vn, s_tris[:, 1], fn_raw)
        np.add.at(vn, s_tris[:, 2], fn_raw)
        vn_len = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = np.divide(vn, np.maximum(vn_len, 1e-12))

        flen = np.linalg.norm(fn_raw, axis=1, keepdims=True)
        fn_unit = np.divide(fn_raw, np.maximum(flen, 1e-12))
        # Backface cull closed solids slightly leniently (-0.08) so silhouette grazing triangles never drop out
        keep = (fn_unit @ fwd) < 0.08
        if not np.any(keep):
            keep = np.ones(len(s_tris), dtype=bool)
        f_keep = s_tris[keep]
        fn_keep = fn_unit[keep]

        T = s_pts[f_keep]  # (F, 3, 3)
        N = vn[f_keep]     # (F, 3, 3)
        q = T - center
        vx = q @ right - cx_mid
        vy = q @ up - cy_mid
        sz = q @ fwd
        sx = vx * scale + rw * 0.5
        sy = rh * 0.5 - vy * scale

        nv = np.stack([N @ right, N @ up, N @ fwd], axis=-1)
        nv = np.where(nv[..., 2:3] > 0.0, -nv, nv)
        kd = np.clip(nv @ key_l, 0.0, 1.0)
        fd = np.clip(nv @ fill_l, 0.0, 1.0)
        bd = np.clip(nv @ back_l, 0.0, 1.0)
        rim = np.power(np.clip(1.0 - np.abs(nv[..., 2]), 0.0, 1.0), 3.0) * 0.14
        # Subtle vertical height ambient-occlusion gradient so flexi valleys have crisp depth
        h_ao = 0.90 + 0.10 * np.clip((T[..., 2] - global_z0) / global_h, 0.0, 1.0)
        shade = ((0.24 + 0.58 * kd + 0.22 * fd + 0.08 * bd) * h_ao + rim).astype(np.float32)
        spec = (np.power(kd, 22.0) * 0.12).astype(np.float32)

        tri_cols = np.tile(base_c, (len(f_keep), 1))
        if highlight_bed:
            z_max_t = T[..., 2].max(axis=1) - global_z0
            on_bed = (z_max_t <= 0.30) & (fn_keep[:, 2] < -0.65)
            tri_cols[on_bed] = np.array([0.08, 0.82, 0.56], dtype=np.float32)

        xmin = np.clip(np.floor(sx.min(axis=1)), 0, rw - 1).astype(int)
        xmax = np.clip(np.ceil(sx.max(axis=1)), 0, rw - 1).astype(int)
        ymin = np.clip(np.floor(sy.min(axis=1)), 0, rh - 1).astype(int)
        ymax = np.clip(np.ceil(sy.max(axis=1)), 0, rh - 1).astype(int)

        W_box = 6
        small = ((xmax - xmin) < W_box) & ((ymax - ymin) < W_box)
        s_idx = np.nonzero(small)[0]
        off = np.array([(dx, dy) for dy in range(W_box) for dx in range(W_box)], dtype=np.int32)
        for c0 in range(0, len(s_idx), 35000):
            k = s_idx[c0:c0 + 35000]
            px = xmin[k][:, None] + off[None, :, 0]
            py = ymin[k][:, None] + off[None, :, 1]
            cx_p, cy_p = px + 0.5, py + 0.5
            ax, ay = sx[k, 0:1], sy[k, 0:1]
            bx, by = sx[k, 1:2], sy[k, 1:2]
            qx, qy = sx[k, 2:3], sy[k, 2:3]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            den = np.where(np.abs(den) < 1e-12, np.nan, den)
            w0 = ((by - qy) * (cx_p - qx) + (qx - bx) * (cy_p - qy)) / den
            w1 = ((qy - ay) * (cx_p - qx) + (ax - qx) * (cy_p - qy)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5) & (px < rw) & (py < rh)
            t_i, c_i = np.nonzero(inside)
            if len(t_i) == 0:
                continue
            fx, fy = px[t_i, c_i], py[t_i, c_i]
            W0, W1, W2 = w0[t_i, c_i], w1[t_i, c_i], w2[t_i, c_i]
            tri = k[t_i]
            fz = W0 * sz[tri, 0] + W1 * sz[tri, 1] + W2 * sz[tri, 2]
            flat = fy * rw + fx
            order = np.lexsort((fz, flat))
            first = np.ones(len(order), dtype=bool)
            first[1:] = flat[order][1:] != flat[order][:-1]
            o = order[first]
            fx, fy, fz, tri = fx[o], fy[o], fz[o], tri[o]
            nearer = fz < zbuf[fy, fx]
            if not np.any(nearer):
                continue
            fx, fy, fz, tri, o = fx[nearer], fy[nearer], fz[nearer], tri[nearer], o[nearer]
            zbuf[fy, fx] = fz
            sh = (W0[o] * shade[tri, 0] + W1[o] * shade[tri, 1] + W2[o] * shade[tri, 2]).astype(np.float32)
            sp = (W0[o] * spec[tri, 0] + W1[o] * spec[tri, 1] + W2[o] * spec[tri, 2]).astype(np.float32)
            fb[fy, fx] = np.clip(tri_cols[tri] * sh[:, None] + sp[:, None], 0.0, 1.0) * 255.0

        for i in np.nonzero(~small)[0]:
            x0_i, x1_i, y0_i, y1_i = xmin[i], xmax[i], ymin[i], ymax[i]
            if x1_i < x0_i or y1_i < y0_i:
                continue
            px, py = np.meshgrid(np.arange(x0_i, x1_i + 1) + 0.5, np.arange(y0_i, y1_i + 1) + 0.5)
            ax, ay, bx, by, qx, qy = sx[i, 0], sy[i, 0], sx[i, 1], sy[i, 1], sx[i, 2], sy[i, 2]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            if abs(den) < 1e-12:
                continue
            w0 = ((by - qy) * (px - qx) + (qx - bx) * (py - qy)) / den
            w1 = ((qy - ay) * (px - qx) + (ax - qx) * (py - qy)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if not np.any(inside):
                continue
            z = w0 * sz[i, 0] + w1 * sz[i, 1] + w2 * sz[i, 2]
            win = zbuf[y0_i:y1_i + 1, x0_i:x1_i + 1]
            hit = inside & (z < win)
            if not np.any(hit):
                continue
            win[hit] = z[hit]
            sh = (w0 * shade[i, 0] + w1 * shade[i, 1] + w2 * shade[i, 2])[hit].astype(np.float32)
            sp = (w0 * spec[i, 0] + w1 * spec[i, 1] + w2 * spec[i, 2])[hit].astype(np.float32)
            fb[y0_i:y1_i + 1, x0_i:x1_i + 1][hit] = np.clip(tri_cols[i][None, :] * sh[:, None] + sp[:, None], 0.0, 1.0) * 255.0

    img = Image.fromarray(np.clip(fb, 0.0, 255.0).astype(np.uint8))
    if ss > 1:
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    if draw_bed_line:
        draw = ImageDraw.Draw(img)
        bed_pt = np.array([center[0], center[1], global_z0]) - center
        bed_vy = float(bed_pt @ up) - cy_mid
        bed_y = int(round(h * 0.5 - bed_vy * (scale / ss)))
        if 4 <= bed_y <= h - 4:
            draw.line([(16, bed_y), (w - 16, bed_y)], fill=(52, 211, 153), width=2)
    return img


def render_prepared_meshes(parts: list[PreparedPart], view: str = "iso", size: int = 640) -> bytes:
    """Fast shaded PNG render of prepared organic meshes using PIL/numpy (supports `view="sheet"` 4-view grid!)."""
    import io
    from PIL import Image, ImageDraw

    if view == "sheet":
        pw = max(320, min(int(size), 1400) // 2)
        ph = int(round(pw * 0.75))
        sheet = Image.new("RGB", (pw * 2, ph * 2), (24, 27, 33))
        all_pts = np.vstack([p.points for p in parts]) if parts else np.zeros((1, 3))
        span_xy = all_pts.max(axis=0) - all_pts.min(axis=0)
        # Pick true broadside view ("front" looks along Y showing X-Z profile; "right" looks along X showing Y-Z profile)
        side_view = "front" if float(span_xy[0]) >= float(span_xy[1]) else "right"
        panels = [
            ("1 | 3/4 ISO", _render_single_mesh_panel(parts, "iso", pw, ph), (0, 0)),
            ("2 | SIDE (Z=0 BUILD PLATE)", _render_single_mesh_panel(parts, side_view, pw, ph, draw_bed_line=True), (pw, 0)),
            ("3 | TOP VIEW", _render_single_mesh_panel(parts, "top", pw, ph), (0, ph)),
            ("4 | BOTTOM (Z=0 BED CONTACT)", _render_single_mesh_panel(parts, "bottom", pw, ph, highlight_bed=True), (pw, ph)),
        ]
        for label, pim, (ox, oy) in panels:
            sheet.paste(pim, (ox, oy))
        dr = ImageDraw.Draw(sheet)
        dr.line([(pw, 0), (pw, ph * 2)], fill=(58, 64, 76), width=2)
        dr.line([(0, ph), (pw * 2, ph)], fill=(58, 64, 76), width=2)
        for label, _, (ox, oy) in panels:
            dr.rectangle([ox + 8, oy + 8, ox + 12 + len(label) * 6 + 8, oy + 26], fill=(18, 20, 26))
            dr.text((ox + 14, oy + 12), label, fill=(215, 222, 235))
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        return buf.getvalue()

    w = h = max(128, min(int(size), 1600))
    side_flag = view in ("front", "back", "left", "right")
    img = _render_single_mesh_panel(parts, view, w, h, highlight_bed=(view == "bottom"), draw_bed_line=side_flag)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
