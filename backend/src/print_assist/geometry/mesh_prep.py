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


def _auto_flexi_orientation_and_start(work: m3d.Manifold, start_pct: float) -> tuple[bool, float]:
    """Detect head vs tail end and avoid chopping lateral pectoral fins when start_pct is near default."""
    tm = _manifold_to_trimesh(work)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    length = max(1e-6, x1 - x0)
    # Compare cross-sectional bulk of front 35% vs rear 35%: keep the heavier head end at x0!
    front = verts[verts[:, 0] <= x0 + 0.35 * length]
    rear = verts[verts[:, 0] >= x1 - 0.35 * length]
    f_span = float(np.ptp(front[:, 1]) * np.ptp(front[:, 2])) if len(front) > 20 else 0.0
    r_span = float(np.ptp(rear[:, 1]) * np.ptp(rear[:, 2])) if len(rear) > 20 else 0.0
    flip_180 = r_span > f_span * 1.18

    xs = (x1 - verts[:, 0]) / length if flip_180 else (verts[:, 0] - x0) / length
    # Sample width profile in 30 bins from head (0) to tail (1) to detect lateral pectoral fins
    widths = []
    for b in range(30):
        t0, t1 = b / 30.0, (b + 1) / 30.0
        sl = verts[(xs >= t0) & (xs < t1)]
        widths.append(float(np.ptp(sl[:, 1])) if len(sl) > 5 else 0.0)
    adj_start = start_pct
    if abs(start_pct - 35.0) < 3.0 and max(widths) > 1.0:
        # Look for widest lateral fin peak in [20%..52%] and nudge first cut cleanly behind it
        fin_zone = widths[6:16]
        peak_val = max(fin_zone) if fin_zone else 0.0
        torso_med = float(np.median(widths[6:22])) or peak_val
        if peak_val > torso_med * 1.18:
            peak_idx = 6 + fin_zone.index(peak_val)
            # Advance past the fin peak until width drops back near torso width (capped at 52%)
            end_idx = peak_idx + 1
            while end_idx < 16 and widths[end_idx] > torso_med * 1.12:
                end_idx += 1
            fin_end_pct = (end_idx / 30.0) * 100.0
            adj_start = max(start_pct, min(52.0, fin_end_pct + 2.0))
    return flip_180, adj_start


def _seat_flexi_on_bed(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    """Smoothly lower an arched floating torso/tail onto Z=0 via a volume-preserving vertical shear.

    Real swimming sharks, whales, dolphins, and arched-tail creatures taper upward behind the chest so their
    rear segments float 5-15 mm off the build plate. A pure vertical shear `z' = z - drop(x)` has Jacobian
    determinant +1 everywhere (100% watertight & volume-preserving) and seats every flexing segment flat on
    Z=0 for support-free print-in-place printing while leaving the head, fins, and tail tip untouched.
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
    flip_180, _ = _auto_flexi_orientation_and_start(work, float(prep.flexi_start_pct))
    if flip_180:
        work = _place_on_bed(work.rotate((0.0, 0.0, 180.0)))

    tm = _manifold_to_trimesh(work)
    verts = np.asarray(tm.vertices, dtype=np.float64).copy()
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    h_all = float(verts[:, 2].max() - verts[:, 2].min())
    length = max(1e-6, x1 - x0)

    n_bins = 48
    edges = np.linspace(x0, x1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    z_mins = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        m = (verts[:, 0] >= edges[i]) & (verts[:, 0] <= edges[i + 1])
        if np.any(m):
            z_mins[i] = float(verts[m, 2].min())
    valid = ~np.isnan(z_mins)
    if valid.sum() < 8:
        return mf, []
    z_mins = np.interp(np.arange(n_bins), np.nonzero(valid)[0], z_mins[valid])

    # Only seat if the rear torso/tail zone (30%..85%) actually floats above the build plate
    t_norm = (centers - x0) / length
    rear_zone = (t_norm >= 0.30) & (t_norm <= 0.85)
    if float(z_mins[rear_zone].max()) < 2.2:
        return mf, []

    # Smoothstep ramp from 0 at snout (0..22%) to 1 across torso/tail (>= 34%)
    ramp = np.clip((t_norm - 0.22) / 0.12, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    raw_drop = np.minimum(z_mins, h_all * 0.35) * ramp
    kernel = np.array([0.06, 0.24, 0.40, 0.24, 0.06], dtype=np.float64)
    drop_smooth = raw_drop.copy()
    for _ in range(3):
        drop_smooth = np.convolve(np.pad(drop_smooth, 2, mode="edge"), kernel, mode="valid")
    # Never shear below actual local z_min (preserves lower tail lobe contact with Z=0)
    drop_safe = np.minimum(drop_smooth, z_mins)
    drop_safe = np.convolve(np.pad(drop_safe, 1, mode="edge"), np.array([0.2, 0.6, 0.2]), mode="valid")
    if float(drop_safe.max()) < 1.2:
        return mf, []

    verts[:, 2] -= np.interp(verts[:, 0], centers, drop_safe)
    verts[:, 2] -= float(verts[:, 2].min())
    seated = _trimesh_to_manifold(trimesh.Trimesh(vertices=verts, faces=tm.faces, process=False))
    if seated is None or seated.is_empty():
        return mf, []
    if flip_180:
        seated = seated.rotate((0.0, 0.0, 180.0))
    if rotated_for_y:
        seated = seated.rotate((0.0, 0.0, 90.0))
    return _place_on_bed(seated), ["seated arched tail smoothly onto print bed for support-free flexi printing"]


def _apply_flexi_articulation(mf: m3d.Manifold, prep: SculptPrep) -> tuple[m3d.Manifold, list[str]]:
    """Slice a solid sculpted character into N+1 sleek, curved-crescent print-in-place flexi segments.

    Uses smooth rib-curved crescent cuts (instead of flat bread-knife slabs) and compact rounded-stadium
    internal chain links rotating around horizontal cylindrical pins with guaranteed `prep.flexi_clearance_mm`
    air gaps in X, Y, and Z so segments print flat on Z=0, never fuse, preserve lateral fins, and wiggle smoothly.
    """
    n_cuts = int(prep.flexi_segments)
    if n_cuts <= 0:
        return mf, []
    notes: list[str] = []
    bb = mf.bounding_box()
    sx = float(bb[3] - bb[0])
    sy = float(bb[4] - bb[1])
    rotated_for_y = (prep.flexi_axis == "y") or (prep.flexi_axis == "auto" and sy > sx)
    work = mf.rotate((0.0, 0.0, -90.0)) if rotated_for_y else mf
    work = _place_on_bed(work)

    flip_180, eff_start_pct = _auto_flexi_orientation_and_start(work, float(prep.flexi_start_pct))
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
    gap = max(0.95, 2.1 * c)
    s0 = _clamp(eff_start_pct / 100.0, 0.14, 0.72)
    s1 = _clamp(float(prep.flexi_end_pct) / 100.0, s0 + 0.12, 0.94)
    cuts = [x0 + length * (s0 + (s1 - s0) * ((i + 0.5) / n_cuts)) for i in range(n_cuts)]
    bounds = [x0 - 5.0, *cuts, x1 + 5.0]
    pad_y = width_all + 40.0
    pad_z = height_all + 40.0
    # Subtle side taper for natural S-curve swimming wiggle without chewing big Pac-Man chunks out of the character
    tan_half = math.tan(math.radians(_clamp(float(prep.flexi_angle_deg), 6.0, 35.0) / 2.0)) * 0.80

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
        h_loc = float(sbb[5]) - z_bot
        yc = float((sbb[1] + sbb[4]) / 2.0)
        prev_cut = active_cuts[-1] if active_cuts else bounds[i]
        seg_pitch = min(u - prev_cut, bounds[i + 2] - u)
        # Skip elevated thin tail stems so tail fins stay solidly attached to the last flexing segment
        if z_bot > 3.5 or w_loc < 8.0 or h_loc < 6.0 or seg_pitch < (gap + 4.0 * c + 4.2):
            continue
        max_pin_h = (h_loc - 1.4 - 3.0 * c) / 4.2
        max_pin_x = (seg_pitch - gap - 3.5 * c) / 4.2
        unit = _clamp(min(max_pin_h, max_pin_x, 1.85), 1.0, 1.85)
        pin_r = unit
        arm_t = max(1.15, unit * 1.05)
        # Compact narrow central hinge channel with room for both vertical drape AND side-to-side S-curve wiggle
        loop_w = _clamp(w_loc * 0.13, 2.5, 3.4)
        slot_w = _clamp(max(w_loc * 0.22, loop_w + 2.0 * c + 1.6), 4.6, 6.8)
        if w_loc - slot_w < 3.6:
            slot_w = max(3.6, w_loc - 3.6)
            loop_w = min(loop_w, slot_w - 2.0 * c)
        if loop_w < 2.2:
            continue
        pin_z = z_bot + arm_t + c + pin_r
        r_loop = pin_r + c + arm_t
        if pin_z + r_loop + c + 1.2 > float(sbb[5]):
            continue
        u_pin = u - (gap / 2.0 + c + pin_r + 0.45)
        cavity_front_x = u_pin - (r_loop + c)
        min_front = (active_cuts[-1] + gap / 2.0 + 1.2) if active_cuts else (x0 + 1.0)
        if cavity_front_x <= min_front:
            continue
        # Gentle organic rearward crescent curve guaranteed never to cross adjacent cuts
        curve = min(w_loc * 0.14, seg_pitch * 0.20, 4.8)
        max_half_gap = max(gap / 2.0, seg_pitch * 0.26)
        active_cuts.append(u)
        joint_specs.append({
            "u": u, "yc": yc, "w_loc": w_loc, "z_bot": z_bot, "h_loc": h_loc,
            "pin_r": pin_r, "arm_t": arm_t, "r_loop": r_loop, "slot_w": slot_w, "loop_w": loop_w,
            "pin_z": pin_z, "u_pin": u_pin, "cavity_front_x": cavity_front_x,
            "curve": curve, "max_half_gap": max_half_gap,
        })

    if not active_cuts:
        notes.append("no cross-sections were wide enough for flexi joints")
        return mf, notes

    # Partition into N+1 sleek curved-crescent segments via smooth extruded CCW 2D rib polygons
    n_samples = 25
    ys = np.linspace(y0 - 8.0, y1 + 8.0, n_samples).tolist()

    def _cut_curve_x(j: int, y_val: float, side: float) -> float:
        sp = joint_specs[j]
        u = sp["u"]
        yc = sp["yc"]
        half_w = max(4.0, sp["w_loc"] / 2.0)
        half_slot = sp["slot_w"] / 2.0 + 0.45
        base_x = _crescent_x(u, y_val, yc, half_slot, half_w, sp["curve"])
        flank = min(half_w + 2.0, max(0.0, abs(y_val - yc) - half_slot))
        half_gap = min(sp["max_half_gap"], gap / 2.0 + flank * tan_half)
        return base_x + side * half_gap

    segments: list[m3d.Manifold] = []
    slab_prisms: list[m3d.Manifold] = []
    for seg_i in range(len(active_cuts) + 1):
        # Strictly CCW polygon: right boundary (increasing Y) -> left boundary (decreasing Y)
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
        segments.append(_clean_segment_slivers(work ^ slab_prism))

    # Carve compact rounded cavities, anchor cylindrical crossbar pins, and union smooth rounded bearing loops
    for j_idx, sp in enumerate(joint_specs):
        u = sp["u"]
        yc = sp["yc"]
        w_loc = sp["w_loc"]
        z_bot = sp["z_bot"]
        pin_r = sp["pin_r"]
        r_loop = sp["r_loop"]
        slot_w = sp["slot_w"]
        loop_w = sp["loop_w"]
        pin_z = sp["pin_z"]
        u_pin = sp["u_pin"]
        cavity_back_x = u + gap / 2.0 + 0.6

        # 1. Segment A (upstream): subtract sleek rounded stadium cavity, then union horizontal crossbar pin
        cavity = _rounded_xz_loop(
            x_center=u_pin, z_center=pin_z, radius=r_loop + c,
            x_back=cavity_back_x, z_floor=-1.0, width_y=slot_w, y_center=yc,
        )
        pin_span = min(w_loc - 1.2, slot_w + 4.8)
        pin = (
            m3d.Manifold.cylinder(pin_span, pin_r, pin_r, circular_segments=32, center=True)
            .rotate((90.0, 0.0, 0.0))
            .translate((u_pin, yc, pin_z))
        )
        pin = pin ^ work ^ slab_prisms[j_idx]
        segments[j_idx] = (segments[j_idx] - cavity) + pin

        # 2. Segment B (downstream): smooth rounded bearing loop wrapping 360° around pin with exact `c` clearance
        next_limit = (
            joint_specs[j_idx + 1]["cavity_front_x"] - 0.5
            if j_idx + 1 < len(joint_specs)
            else x1 - 1.0
        )
        anchor_x = min(u + gap / 2.0 + 3.0, next_limit)
        if anchor_x <= u + gap / 2.0 + 0.7:
            anchor_x = u + gap / 2.0 + 0.9
        loop_outer = _rounded_xz_loop(
            x_center=u_pin, z_center=pin_z, radius=r_loop,
            x_back=anchor_x, z_floor=z_bot, width_y=loop_w, y_center=yc,
        )
        eye_hole = (
            m3d.Manifold.cylinder(loop_w + 4.0, pin_r + c, pin_r + c, circular_segments=32, center=True)
            .rotate((90.0, 0.0, 0.0))
            .translate((u_pin, yc, pin_z))
        )
        loop = (loop_outer - eye_hole) ^ work
        segments[j_idx + 1] = segments[j_idx + 1] + loop

    non_empty = [s for s in segments if not s.is_empty() and s.status() == m3d.Error.NoError]
    if len(non_empty) < 2:
        return mf, notes
    composed = m3d.Manifold.compose(non_empty)
    if flip_180:
        composed = composed.rotate((0.0, 0.0, 180.0))
    if rotated_for_y:
        composed = composed.rotate((0.0, 0.0, 90.0))
    composed = _place_on_bed(composed)
    notes.append(f"sliced {len(active_cuts)} curved print-in-place flexi joints ({len(non_empty)} segments, {c:.2f} mm clearance)")
    return composed, notes


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


def prepare_print_mesh(raw_path: Path, prep: SculptPrep, name: str = "Sculpt",
                       color: str | None = "#b8c2cc", filament: int | None = None) -> PreparedSculpt:
    """Load, repair, orient, scale, flatten bottom, and apply optional base/keychain finishing."""
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

    # For print-in-place flexi models, smoothly seat any floating arched tail onto Z=0 BEFORE belly flattening
    # so every flexing segment gets uniform flat bed adhesion with zero supports.
    if prep.flexi_segments > 0:
        mf, seat_notes = _seat_flexi_on_bed(mf, prep)
        notes.extend(seat_notes)

    # Flatten bottom by trimming off Z=0 so bed adhesion is rock-solid.
    # For print-in-place flexi models, automatically trim gently (~5% of height, capped at 3.5 mm) so every
    # flexing segment rests flat on the print bed without lopping off lower tail fin tips.
    eff_flat = float(prep.flatten_bottom_mm)
    if prep.flexi_segments > 0 and eff_flat < 1.5:
        bb_cur = mf.bounding_box()
        h_cur = float(bb_cur[5] - bb_cur[2])
        eff_flat = max(eff_flat, min(3.5, h_cur * 0.05))
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
    if prep.base_style != "none" and prep.flexi_segments == 0:
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

    # Optional functional hardware cutouts (magnet pocket, hollowed planter/cup cavity)
    if prep.magnet_hole != "none":
        mf, m_notes = _apply_magnet_hole(mf, prep)
        notes.extend(m_notes)
    if prep.hollow_cavity:
        mf, h_notes = _apply_hollow_cavity(mf, prep)
        notes.extend(h_notes)

    # Optional print-in-place flexi articulation (precision CAD hinge cuts through sculpted character)
    if prep.flexi_segments > 0:
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


def render_prepared_meshes(parts: list[PreparedPart], view: str = "iso", size: int = 640) -> bytes:
    """Fast shaded PNG render of prepared organic meshes using PIL/numpy (for AI `render_preview`)."""
    import io
    from PIL import Image, ImageDraw

    w = h = max(128, min(int(size), 1600))
    img = Image.new("RGB", (w, h), (30, 34, 41))
    if not parts:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    views = {
        "iso": (math.radians(35), math.radians(35)),
        "front": (0.0, 0.0),
        "back": (math.pi, 0.0),
        "right": (math.pi / 2, 0.0),
        "left": (-math.pi / 2, 0.0),
        "top": (0.0, math.pi / 2 - 1e-3),
        "bottom": (0.0, -math.pi / 2 + 1e-3),
    }
    az, el = views.get(view, views["iso"])
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    # Camera basis: view from azimuth/elevation around Z-up
    right = np.array([ca, -sa, 0.0])
    up = np.array([-sa * se, -ca * se, ce])
    fwd = np.array([sa * ce, ca * ce, se])

    all_pts = np.vstack([p.points for p in parts])
    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
    proj_all_x = (all_pts - center) @ right
    proj_all_y = (all_pts - center) @ up
    span = max(float(proj_all_x.max() - proj_all_x.min()), float(proj_all_y.max() - proj_all_y.min()), 1e-3)
    scale = (w * 0.78) / span

    draw = ImageDraw.Draw(img)
    light = np.array([0.4, 0.5, 0.76])
    light /= np.linalg.norm(light)

    # Painter's algorithm (downsample if huge so software preview stays < 0.2s)
    polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    for p in parts:
        hex_c = (p.color or "#b8c2cc").lstrip("#")
        base_rgb = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4)) if len(hex_c) == 6 else (184, 194, 204)
        pts = p.points - center
        tris = p.tris
        if len(tris) > 120000:
            step = max(1, len(tris) // 120000)
            tris = tris[::step]
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.maximum(nlen, 1e-9))
        facing = normals @ fwd
        keep = facing > -0.05
        v0, v1, v2, normals = v0[keep], v1[keep], v2[keep], normals[keep]
        depths = ((v0 + v1 + v2) / 3.0) @ fwd
        shades = np.clip(0.28 + 0.72 * np.abs(normals @ light), 0.2, 1.0)
        for j in range(len(depths)):
            tri_pts = [v0[j], v1[j], v2[j]]
            xy = [(w / 2.0 + float(pt @ right) * scale, h / 2.0 - float(pt @ up) * scale) for pt in tri_pts]
            sh = float(shades[j])
            rgb = (int(base_rgb[0] * sh), int(base_rgb[1] * sh), int(base_rgb[2] * sh))
            polys.append((float(depths[j]), xy, rgb))

    polys.sort(key=lambda item: item[0])
    for _, xy, rgb in polys:
        draw.polygon(xy, fill=rgb)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
