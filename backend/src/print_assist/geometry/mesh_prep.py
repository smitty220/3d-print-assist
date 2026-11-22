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

    # Flatten bottom by trimming `flatten_bottom_mm` off Z=0 so bed adhesion is rock-solid
    if prep.flatten_bottom_mm > 0:
        bb = mf.bounding_box()
        cut = min(float(prep.flatten_bottom_mm), max(0.0, (bb[5] - bb[2]) * 0.15))
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

    # Optional display base / plinth
    if prep.base_style != "none":
        bb = mf.bounding_box()
        sx = float(bb[3] - bb[0])
        sy = float(bb[4] - bb[1])
        bh = max(1.5, float(prep.base_height_mm))
        overlap = min(0.6, bh * 0.3)
        base = _build_base(prep.base_style, sx, sy, bh, prep.base_margin_mm)
        mf = _place_on_bed(base + mf.translate((0.0, 0.0, bh - overlap)))
        notes.append(f"added {prep.base_style} display base ({bh:.1f} mm)")

    # Optional keychain loop at top peak
    if prep.keychain_loop:
        tm = _manifold_to_trimesh(mf)
        top_z = float(tm.vertices[:, 2].max())
        near_top = tm.vertices[tm.vertices[:, 2] >= top_z - 2.0]
        tx = float(near_top[:, 0].mean()) if len(near_top) else 0.0
        ty = float(near_top[:, 1].mean()) if len(near_top) else 0.0
        loop = _build_keychain_loop(prep.keychain_hole_mm, tx, ty, top_z)
        mf = _place_on_bed(mf + loop)
        notes.append(f"added keychain loop (Ø{prep.keychain_hole_mm:.1f} mm hole)")

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


def to_viewer_mesh(part: PreparedPart, bid: str) -> dict:
    """Convert a `PreparedPart` into the `ShapeMesh`-compatible dict that `ViewportPane` renders in Three.js."""
    bmin, bmax = part.bounds()
    centroid = ((bmin + bmax) / 2.0).tolist()
    pos = np.round(part.points.astype(np.float64), 4).reshape(-1).tolist()
    idx = part.tris.astype(np.int64).reshape(-1).tolist()
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
        if len(tris) > 12000:
            step = max(1, len(tris) // 12000)
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
