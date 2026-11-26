"""In-Blender Python helper library for 3D Print Assist Character Mode.

Executed INSIDE standalone headless Blender 4.5 LTS (`bpy` + `bmesh` + `mathutils`).
Provides glass-smooth Catmull-Clark Subdivision-Surface organic sculpting primitives,
multi-color MMU3/AMS material assignment, and true Flexi-Factory non-destructive
articulation (wide ±24° V-clearance cuts, `bmesh.ops.bevel` rounded cut rims with
ZERO voxelization, captive vertical Z-axis pins + horizontal chain-link tongues, and
flat chamfered `Z = 0` build-plate pads).

Coordinate convention:
- 1 Blender unit = 1 mm, Z is up, print bed is Z = 0.
- Exported via `pa_export_glb(..., export_yup=False)` so Z-up millimetres load 1:1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector


# ---------------------------------------------------------------- scene & materials


def pa_clear_scene() -> None:
    """Reset Blender to a clean empty millimetre scene (Z-up, 1 unit = 1 mm)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def pa_hex_to_rgba_linear(hex_color: str) -> tuple[float, float, float, float]:
    """Convert `#rrggbb` sRGB hex string to Blender linear RGBA so glTF exports exact filament hex colors."""
    s = (hex_color or "#b8c2cc").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        s = "b8c2cc"
    r = _srgb_to_linear(int(s[0:2], 16) / 255.0)
    g = _srgb_to_linear(int(s[2:4], 16) / 255.0)
    b = _srgb_to_linear(int(s[4:6], 16) / 255.0)
    return (r, g, b, 1.0)


def pa_material(
    name: str,
    hex_color: str,
    roughness: float = 0.42,
    metallic: float = 0.03,
) -> bpy.types.Material:
    """Get or create a Principled BSDF filament material with exact glTF sRGB color."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        rgba = pa_hex_to_rgba_linear(hex_color)
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
    mat["pa_color"] = hex_color if hex_color.startswith("#") else f"#{hex_color}"
    return mat


def pa_assign_color(
    obj: bpy.types.Object,
    part_name: str,
    hex_color: str,
    filament: int | None = None,
    roughness: float = 0.42,
) -> bpy.types.Object:
    """Assign a clean PLA filament color material & metadata to a mesh object."""
    norm_hex = hex_color if hex_color.startswith("#") else f"#{hex_color}"
    mat = pa_material(f"MAT_{part_name}", norm_hex, roughness=roughness)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj["pa_color"] = norm_hex
    obj["pa_part_name"] = part_name
    if filament is not None:
        obj["pa_filament"] = int(filament)
    return obj


def _link_object(name: str, mesh: bpy.types.Mesh) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def pa_shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    if obj.type == "MESH":
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def pa_add_subsurf(obj: bpy.types.Object, levels: int = 3, apply: bool = True) -> bpy.types.Object:
    """Apply smooth Catmull-Clark Subdivision Surface (Pixar / Flexi-Factory finish)."""
    if levels <= 0:
        return obj
    mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
    mod.subdivision_type = "CATMULL_CLARK"
    mod.levels = int(levels)
    mod.render_levels = int(levels)
    pa_shade_smooth(obj)
    if apply:
        pa_apply_all_modifiers(obj)
    return obj


def pa_apply_all_modifiers(obj: bpy.types.Object) -> bpy.types.Object:
    """Bake object transforms and all active modifiers cleanly into mesh vertices."""
    if obj.type != "MESH":
        return obj
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(eval_obj)
    # Bake world matrix directly into vertices so all booleans & exports share world mm space
    new_mesh.transform(obj.matrix_world)
    old_mesh = obj.data
    old_mats = list(old_mesh.materials)
    obj.modifiers.clear()
    obj.data = new_mesh
    obj.matrix_world = Matrix.Identity(4)
    if not obj.data.materials and old_mats:
        for m in old_mats:
            obj.data.materials.append(m)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return obj


def pa_delete(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    data = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.meshes.remove(data)


def pa_duplicate(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    pa_apply_all_modifiers(obj)
    dup_mesh = obj.data.copy()
    dup = bpy.data.objects.new(name, dup_mesh)
    bpy.context.scene.collection.objects.link(dup)
    for k in obj.keys():
        if k != "_RNA_UI":
            dup[k] = obj[k]
    return dup


def pa_bounds(objs: bpy.types.Object | Sequence[bpy.types.Object]) -> tuple[Vector, Vector]:
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in objs:
        if obj.type != "MESH" or not obj.data.vertices:
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            mins.x = min(mins.x, p.x)
            mins.y = min(mins.y, p.y)
            mins.z = min(mins.z, p.z)
            maxs.x = max(maxs.x, p.x)
            maxs.y = max(maxs.y, p.y)
            maxs.z = max(maxs.z, p.z)
    return mins, maxs


def _ensure_outward_winding(bm: bmesh.types.BMesh) -> None:
    """Ensure every connected closed shell in `bm` has strictly positive signed volume (outward normals).
    Unlike `bmesh.ops.recalc_face_normals`, this NEVER flips thin concave flexi-pocket shells inside-out!"""
    bm.faces.ensure_lookup_table()
    visited: set[int] = set()
    for start_f in bm.faces:
        if start_f.index in visited:
            continue
        comp: list[bmesh.types.BMFace] = []
        stack = [start_f]
        visited.add(start_f.index)
        while stack:
            f = stack.pop()
            comp.append(f)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf.index not in visited:
                        visited.add(nf.index)
                        stack.append(nf)
        # Signed volume of this connected shell: (1/6) * sum(v0 · (v1 × v2)) over its triangles
        vol6 = 0.0
        for f in comp:
            vs = f.verts
            if len(vs) < 3:
                continue
            v0 = vs[0].co
            for k in range(1, len(vs) - 1):
                vol6 += v0.dot(vs[k].co.cross(vs[k + 1].co))
        if vol6 < 0.0:
            bmesh.ops.reverse_faces(bm, faces=comp)


def pa_repair_mesh(obj: bpy.types.Object, merge_dist: float = 0.0) -> bpy.types.Object:
    """Triangulate cleanly and enforce strictly positive outward winding per shell without flipping concave pockets."""
    pa_apply_all_modifiers(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if merge_dist > 0.0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    _ensure_outward_winding(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


# ---------------------------------------------------------------- organic SubSurf sculpting helpers


def pa_loft_profile(
    name: str,
    sections: Sequence[dict],
    *,
    ring_verts: int = 24,
    subsurf: int = 2,
    cap_ends: bool = True,
) -> bpy.types.Object:
    """Build a glass-smooth quad-topology organic body loft along the X spine with Catmull-Clark SubSurf.

    Each section dict in `sections` (ordered front-to-back along X) supports:
      - `x`: float (mm along spine)
      - `ry`: float (half-width along ±Y)
      - `z_bot`: float (ventral bottom Z, e.g. 0.0 for bed-seated belly)
      - `z_top`: float (dorsal crown Z)
      - `yc`: float (optional lateral Y offset, default 0.0)
      - `belly_flat`: float in [0..1] (0 = rounded ellipse, 0.7 = flat-bottomed D-profile for bed adhesion!)
      - `crown_sharp`: float in [0..0.5] (optional subtle dorsal ridge taper)
    """
    if len(sections) < 2:
        raise ValueError("pa_loft_profile requires at least 2 cross-sections")
    n = max(12, int(ring_verts))
    if n % 2 == 1:
        n += 1

    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []

    for sec in sections:
        x = float(sec["x"])
        ry = max(0.2, float(sec["ry"]))
        z_bot = float(sec["z_bot"])
        z_top = float(sec["z_top"])
        yc = float(sec.get("yc", 0.0))
        belly_flat = max(0.0, min(0.95, float(sec.get("belly_flat", 0.45))))
        crown_sharp = max(0.0, min(0.45, float(sec.get("crown_sharp", 0.0))))

        zc = 0.5 * (z_bot + z_top)
        rz = max(0.2, 0.5 * (z_top - z_bot))
        ring: list[bmesh.types.BMVert] = []
        for k in range(n):
            # theta = 0 at +Y flank, pi/2 at +Z crown, pi at -Y flank, 3pi/2 at -Z belly
            theta = (2.0 * math.pi * k) / n
            ct = math.cos(theta)
            st = math.sin(theta)
            if st < 0.0 and belly_flat > 0.0:
                # Superellipse flattening on the ventral half so the underside meets Z=0 cleanly
                exp = 2.0 + 3.2 * belly_flat
                ay = abs(ct) ** (2.0 / exp)
                az = abs(st) ** (2.0 / exp)
                y = yc + math.copysign(ry * ay, ct)
                z = zc - rz * az
            else:
                y_scale = 1.0 - crown_sharp * (st ** 2)
                y = yc + ry * ct * y_scale
                z = zc + rz * st
            ring.append(bm.verts.new((x, y, z)))
        rings.append(ring)

    # Connect adjacent rings with pure quads
    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for k in range(n):
            kn = (k + 1) % n
            bm.faces.new((r0[k], r1[k], r1[kn], r0[kn]))

    if cap_ends:
        # Smooth quad-friendly domed end-caps (two concentric inset rings + tip vertex)
        for end_idx, direction in ((0, -1.0), (len(rings) - 1, +1.0)):
            sec = sections[end_idx]
            nb = sections[1 if end_idx == 0 else -2]
            dx_span = abs(float(sec["x"]) - float(nb["x"]))
            cap_ext = max(0.5, min(dx_span * 0.45, float(sec["ry"]) * 0.45))
            yc = float(sec.get("yc", 0.0))
            zc = 0.5 * (float(sec["z_bot"]) + float(sec["z_top"]))
            x_base = float(sec["x"])
            outer_r = rings[end_idx]

            mid_r: list[bmesh.types.BMVert] = []
            for v in outer_r:
                mx = x_base + direction * cap_ext * 0.65
                my = yc + (v.co.y - yc) * 0.52
                mz = zc + (v.co.z - zc) * 0.52
                mid_r.append(bm.verts.new((mx, my, mz)))
            tip = bm.verts.new((x_base + direction * cap_ext, yc, zc))

            for k in range(n):
                kn = (k + 1) % n
                if direction < 0:
                    bm.faces.new((mid_r[k], outer_r[k], outer_r[kn], mid_r[kn]))
                    bm.faces.new((tip, mid_r[k], mid_r[kn]))
                else:
                    bm.faces.new((outer_r[k], mid_r[k], mid_r[kn], outer_r[kn]))
                    bm.faces.new((mid_r[k], tip, mid_r[kn]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = _link_object(name, mesh)
    return pa_add_subsurf(obj, levels=subsurf, apply=True)


def pa_sculpt_fin_strip(
    name: str,
    stations: Sequence[dict],
    *,
    ring_verts: int = 16,
    subsurf: int = 2,
    mirror_y: bool = False,
) -> bpy.types.Object:
    """Build a smooth airfoil-profile fin, tail, wing, ear, or limb from ordered cross-section stations.

    Each station dict defines a local 3D frame:
      - `center`: `(x, y, z)` center of cross-section
      - `chord_vec`: `(dx, dy, dz)` vector from trailing/back to leading/front edge (length = half-chord)
      - `thick_vec`: `(tx, ty, tz)` normal thickness vector (length = half-thickness)
    Creates a clean quad loft with rounded airfoil tip + SubSurf, optionally mirrored across Y=0.
    """
    if len(stations) < 2:
        raise ValueError("pa_sculpt_fin_strip requires at least 2 stations")
    n = max(10, int(ring_verts))
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []

    for st in stations:
        c = Vector(st["center"])
        u = Vector(st["chord_vec"])
        w = Vector(st["thick_vec"])
        ring: list[bmesh.types.BMVert] = []
        for k in range(n):
            theta = (2.0 * math.pi * k) / n
            ct = math.cos(theta)
            st_s = math.sin(theta)
            # Sleek teardrop airfoil: slightly plumper toward leading edge (+u), tapered toward trailing edge (-u)
            taper = 0.85 + 0.25 * ct
            pt = c + u * ct + w * (st_s * taper)
            ring.append(bm.verts.new(pt))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for k in range(n):
            kn = (k + 1) % n
            bm.faces.new((r0[k], r1[k], r1[kn], r0[kn]))

    # Cap root (index 0) and rounded tip (index -1)
    c0 = Vector(stations[0]["center"])
    v0 = bm.verts.new(c0)
    for k in range(n):
        kn = (k + 1) % n
        bm.faces.new((v0, rings[0][kn], rings[0][k]))

    c_last = Vector(stations[-1]["center"])
    c_prev = Vector(stations[-2]["center"])
    tip_dir = (c_last - c_prev)
    if tip_dir.length > 1e-4:
        tip_dir.normalize()
    else:
        tip_dir = Vector((0.0, 0.0, 1.0))
    tip_ext = max(0.3, Vector(stations[-1]["thick_vec"]).length * 0.9)
    v_tip = bm.verts.new(c_last + tip_dir * tip_ext)
    for k in range(n):
        kn = (k + 1) % n
        bm.faces.new((rings[-1][k], v_tip, rings[-1][kn]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = _link_object(name, mesh)
    pa_add_subsurf(obj, levels=subsurf, apply=True)

    if mirror_y:
        mod = obj.modifiers.new(name="MirrorY", type="MIRROR")
        mod.use_axis = (False, True, False)
        mod.use_bisect_axis = (False, False, False)
        mod.use_clip = False
        pa_apply_all_modifiers(obj)
    return obj


def pa_quad_sphere(
    name: str,
    center: Sequence[float],
    radius_xyz: Sequence[float] | float,
    *,
    rot_deg: Sequence[float] = (0.0, 0.0, 0.0),
    subsurf: int = 3,
    mirror_y: bool = False,
) -> bpy.types.Object:
    """Create a smooth quad-topology sphere/ellipsoid (no pinched poles) — ideal for Pixar eyes, brows, cheeks."""
    rx, ry, rz = (
        (float(radius_xyz), float(radius_xyz), float(radius_xyz))
        if isinstance(radius_xyz, (int, float))
        else (float(radius_xyz[0]), float(radius_xyz[1]), float(radius_xyz[2]))
    )
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True)
    for v in bm.verts:
        if v.co.length > 1e-6:
            v.co.normalize()
        v.co.x *= rx
        v.co.y *= ry
        v.co.z *= rz
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = _link_object(name, mesh)
    obj.rotation_euler = Euler((math.radians(rot_deg[0]), math.radians(rot_deg[1]), math.radians(rot_deg[2])), "XYZ")
    obj.location = Vector((float(center[0]), float(center[1]), float(center[2])))
    pa_apply_all_modifiers(obj)
    if subsurf > 0:
        pa_add_subsurf(obj, levels=subsurf, apply=True)
    if mirror_y:
        mod = obj.modifiers.new(name="MirrorY", type="MIRROR")
        mod.use_axis = (False, True, False)
        pa_apply_all_modifiers(obj)
    return obj


def pa_cone(
    name: str,
    base_center: Sequence[float],
    tip_point: Sequence[float],
    radius_base: float,
    radius_tip: float = 0.25,
    *,
    segments: int = 16,
    subsurf: int = 1,
    mirror_y: bool = False,
) -> bpy.types.Object:
    """Create a smooth rounded cone/tooth/horn oriented from `base_center` to `tip_point`."""
    p0 = Vector(base_center)
    p1 = Vector(tip_point)
    axis = p1 - p0
    length = max(0.2, axis.length)
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=max(8, int(segments)),
        radius1=float(radius_base),
        radius2=float(radius_tip),
        depth=length,
    )
    # Default cone is centered on Z in [-length/2, +length/2]; shift base to 0 and align +Z to axis
    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(axis.normalized()).to_matrix().to_4x4()
    mat = Matrix.Translation(p0) @ rot @ Matrix.Translation((0.0, 0.0, length * 0.5))
    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = _link_object(name, mesh)
    if subsurf > 0:
        pa_add_subsurf(obj, levels=subsurf, apply=True)
    if mirror_y:
        mod = obj.modifiers.new(name="MirrorY", type="MIRROR")
        mod.use_axis = (False, True, False)
        pa_apply_all_modifiers(obj)
    return obj


def pa_box(
    name: str,
    center: Sequence[float],
    size_xyz: Sequence[float],
    *,
    rot_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= float(size_xyz[0])
        v.co.y *= float(size_xyz[1])
        v.co.z *= float(size_xyz[2])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = _link_object(name, mesh)
    obj.rotation_euler = Euler((math.radians(rot_deg[0]), math.radians(rot_deg[1]), math.radians(rot_deg[2])), "XYZ")
    obj.location = Vector((float(center[0]), float(center[1]), float(center[2])))
    pa_apply_all_modifiers(obj)
    return obj


def pa_cylinder(
    name: str,
    center: Sequence[float],
    radius: float,
    height: float,
    *,
    segments: int = 32,
) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=max(12, int(segments)),
        radius1=float(radius),
        radius2=float(radius),
        depth=float(height),
    )
    bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(center)), verts=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _link_object(name, mesh)


def pa_extrude_polygon_z(
    name: str,
    pts_xy: Sequence[tuple[float, float]],
    z0: float,
    z1: float,
) -> bpy.types.Object:
    """Extrude a closed 2D XY polygon cleanly from `z0` to `z1` into a watertight manifold solid."""
    bm = bmesh.new()
    bot = [bm.verts.new((float(x), float(y), float(z0))) for x, y in pts_xy]
    top = [bm.verts.new((float(x), float(y), float(z1))) for x, y in pts_xy]
    n = len(bot)
    bm.faces.new(list(reversed(bot)))
    bm.faces.new(top)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _link_object(name, mesh)


def pa_stadium_xy(
    name: str,
    x_center: float,
    y_center: float,
    radius: float,
    x_back: float,
    z0: float,
    z1: float,
    segs: int = 24,
) -> bpy.types.Object:
    """Horizontal rounded-nose stadium prism in XY extruded from `z0` to `z1` (for flexi chain-link tongues & pockets)."""
    pts: list[tuple[float, float]] = [(x_back, y_center + radius), (x_center, y_center + radius)]
    for k in range(1, segs):
        ang = math.pi * 0.5 + math.pi * (k / segs)
        pts.append((x_center + radius * math.cos(ang), y_center + radius * math.sin(ang)))
    pts.append((x_center, y_center - radius))
    pts.append((x_back, y_center - radius))
    return pa_extrude_polygon_z(name, pts, z0, z1)


def _collapse_to_single_material(obj: bpy.types.Object) -> None:
    """Ensure every polygon on `obj` uses material slot 0 so glTF NEVER splits a solid into open multi-primitive shells!"""
    if obj.type != "MESH" or obj.data is None:
        return
    mats = list(obj.data.materials)
    primary = mats[0] if mats else None
    for poly in obj.data.polygons:
        poly.material_index = 0
    if len(mats) > 1:
        obj.data.materials.clear()
        if primary is not None:
            obj.data.materials.append(primary)


def pa_bool(
    target: bpy.types.Object,
    tool: bpy.types.Object,
    op: str = "DIFFERENCE",
    *,
    solver: str = "MANIFOLD",
    delete_tool: bool = True,
) -> bpy.types.Object:
    """Run a fast, watertight boolean (`UNION`, `DIFFERENCE`, `INTERSECT`) via Blender 4.5's `MANIFOLD` solver
    (with automatic fallback to `EXACT` if an input isn't strictly closed)."""
    pa_apply_all_modifiers(target)
    pa_apply_all_modifiers(tool)
    before_verts = len(target.data.vertices)
    backup_mesh = target.data.copy() if solver == "MANIFOLD" else None
    try:
        mod = target.modifiers.new(name=f"Bool_{op}", type="BOOLEAN")
        mod.operation = op.upper()
        mod.solver = solver
        mod.object = tool
        if solver == "EXACT":
            mod.use_self = True
            mod.use_hole_tolerant = True
        pa_apply_all_modifiers(target)
        # If MANIFOLD failed silently (produced 0 verts on a UNION), retry once with EXACT
        if solver == "MANIFOLD" and op.upper() == "UNION" and len(target.data.vertices) == 0 and before_verts > 0 and backup_mesh is not None:
            old = target.data
            target.data = backup_mesh
            backup_mesh = None
            if old.users == 0:
                bpy.data.meshes.remove(old)
            mod2 = target.modifiers.new(name=f"Bool_{op}_exact", type="BOOLEAN")
            mod2.operation = op.upper()
            mod2.solver = "EXACT"
            mod2.object = tool
            mod2.use_hole_tolerant = True
            pa_apply_all_modifiers(target)
        _collapse_to_single_material(target)
    finally:
        if backup_mesh is not None and backup_mesh.users == 0:
            bpy.data.meshes.remove(backup_mesh)
        if delete_tool:
            pa_delete(tool)
    return target


def pa_join(objs: Sequence[bpy.types.Object], name: str | None = None) -> bpy.types.Object:
    """Combine multiple mesh objects into one single object (without changing their geometry)."""
    objs = [o for o in objs if o is not None and o.type == "MESH"]
    if not objs:
        raise ValueError("pa_join received no objects")
    for o in objs:
        pa_apply_all_modifiers(o)
    if len(objs) == 1:
        if name:
            objs[0].name = name
        return objs[0]
    bm_all = bmesh.new()
    for o in objs:
        bm_all.from_mesh(o.data)
    _ensure_outward_winding(bm_all)
    target = objs[0]
    if name:
        target.name = name
    bm_all.to_mesh(target.data)
    bm_all.free()
    target.data.update()
    for extra in objs[1:]:
        pa_delete(extra)
    return target


def pa_seat_on_bed(objs: Sequence[bpy.types.Object], flatten_mm: float = 0.0) -> None:
    """Center `objs` on XY=0 and seat their lowest point on Z=0, optionally shaving `flatten_mm` off Z=0."""
    valid = [o for o in objs if o is not None and o.type == "MESH" and len(o.data.vertices) > 0]
    if not valid:
        return
    for o in valid:
        pa_apply_all_modifiers(o)
    bmin, bmax = pa_bounds(valid)
    shift = Vector((-(bmin.x + bmax.x) * 0.5, -(bmin.y + bmax.y) * 0.5, -bmin.z - max(0.0, float(flatten_mm))))
    for o in valid:
        o.data.transform(Matrix.Translation(shift))
        o.data.update()
    if flatten_mm > 0.01:
        span_x = (bmax.x - bmin.x) + 40.0
        span_y = (bmax.y - bmin.y) + 40.0
        span_z = (bmax.z - bmin.z) + 40.0
        for o in valid:
            keep = pa_box("_bed_keep", (0.0, 0.0, span_z * 0.5), (span_x, span_y, span_z))
            pa_bool(o, keep, "INTERSECT", delete_tool=True)


# ---------------------------------------------------------------- TRUE FLEXI-FACTORY ARTICULATION IN BLENDER


def _is_manifold_closed(obj: bpy.types.Object) -> bool:
    """Return True iff every edge in `obj` is a closed 2-manifold edge (exactly 2 adjacent faces)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    ok = len(bm.faces) > 0 and all(len(e.link_faces) == 2 for e in bm.edges)
    bm.free()
    return ok


def _bevel_cut_rim_edges(
    obj: bpy.types.Object,
    bevel_mm: float = 1.15,
    segments: int = 4,
    z_floor_lock: float = 0.35,
) -> bpy.types.Object:
    """Smoothly round ONLY the sharp boolean cut-rim edge loops while strictly guaranteeing closed 2-manifold topology!

    Uses Blender's native `bmesh.ops.bevel` on clean cut-rim edges (`|nx| > 0.62` cap meeting outer skin)
    with a zero-risk manifold guard: if beveling would ever create an open/non-manifold edge, falls back cleanly
    to an un-pinched subtle rim tuck or leaves the silky SubSurf segment intact (NEVER breaks watertightness!).
    """
    if bevel_mm <= 0.08 or len(obj.data.polygons) < 8:
        return obj
    pa_apply_all_modifiers(obj)
    backup = obj.data.copy()
    try:
        # Try clean angle-limited BEVEL modifier first (handles miters & overlaps smoothly in C++)
        # Mark only cut-rim edges with bevel weight = 1.0 so face details, eyes, teeth & fins are NEVER touched!
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        bw_layer = bm.edges.layers.float.get("bevel_weight_edge") or bm.edges.layers.float.new("bevel_weight_edge")
        marked = 0
        for e in bm.edges:
            e[bw_layer] = 0.0
            if len(e.link_faces) != 2:
                continue
            f0, f1 = e.link_faces[0], e.link_faces[1]
            c0 = abs(f0.normal.x) > 0.60 and abs(f0.normal.z) < 0.72
            c1 = abs(f1.normal.x) > 0.60 and abs(f1.normal.z) < 0.72
            if c0 == c1:
                continue
            dot = max(-1.0, min(1.0, f0.normal.dot(f1.normal)))
            if dot > 0.86:
                continue
            # Keep Z=0 bed perimeter clean & crisp (avoid pinching corners right at Z=0!)
            if min(e.verts[0].co.z, e.verts[1].co.z) <= z_floor_lock:
                continue
            e[bw_layer] = 1.0
            marked += 1
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()

        if marked > 0:
            for try_w in (float(bevel_mm), float(bevel_mm) * 0.55, 0.35):
                # Restore clean mesh before each attempt
                if try_w != float(bevel_mm):
                    old_d = obj.data
                    obj.data = backup.copy()
                    if old_d.users == 0:
                        bpy.data.meshes.remove(old_d)
                    # Re-mark weights on restored backup
                    bm2 = bmesh.new()
                    bm2.from_mesh(obj.data)
                    bm2.normal_update()
                    bw2 = bm2.edges.layers.float.get("bevel_weight_edge") or bm2.edges.layers.float.new("bevel_weight_edge")
                    for e in bm2.edges:
                        e[bw2] = 0.0
                        if len(e.link_faces) != 2:
                            continue
                        f0, f1 = e.link_faces[0], e.link_faces[1]
                        c0 = abs(f0.normal.x) > 0.60 and abs(f0.normal.z) < 0.72
                        c1 = abs(f1.normal.x) > 0.60 and abs(f1.normal.z) < 0.72
                        if c0 != c1 and f0.normal.dot(f1.normal) <= 0.86 and min(e.verts[0].co.z, e.verts[1].co.z) > z_floor_lock:
                            e[bw2] = 1.0
                    bm2.to_mesh(obj.data)
                    bm2.free()
                    obj.data.update()

                mod = obj.modifiers.new(name="RimBevel", type="BEVEL")
                mod.affect = "EDGES"
                mod.limit_method = "WEIGHT"
                mod.edge_weight = "bevel_weight_edge"
                mod.offset_type = "OFFSET"
                mod.width = try_w
                mod.segments = max(2, int(segments))
                mod.profile = 0.5
                mod.use_clamp_overlap = True
                mod.miter_outer = "MITER_ARC"
                pa_apply_all_modifiers(obj)
                pa_repair_mesh(obj, merge_dist=0.0)
                if _is_manifold_closed(obj):
                    pa_shade_smooth(obj)
                    return obj

        # If every bevel attempt broke manifoldness, restore exact clean manifold segment
        old_d = obj.data
        obj.data = backup
        backup = None
        if old_d.users == 0:
            bpy.data.meshes.remove(old_d)
        pa_shade_smooth(obj)
        return obj
    finally:
        if backup is not None and backup.users == 0:
            bpy.data.meshes.remove(backup)


def _make_flexi_slab_prism(
    name: str,
    x_left_fn,
    x_right_fn,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n_samples: int = 25,
) -> bpy.types.Object:
    ys = [y_min + (y_max - y_min) * (i / (n_samples - 1)) for i in range(n_samples)]
    right_pts = [(float(x_right_fn(y)), float(y)) for y in ys]
    left_pts = [(float(x_left_fn(y)), float(y)) for y in reversed(ys)]
    return pa_extrude_polygon_z(name, [*right_pts, *left_pts], z_min, z_max)


def pa_flexi_articulate(
    primary_body: bpy.types.Object,
    detail_parts: Sequence[bpy.types.Object] = (),
    color_zones: Sequence[bpy.types.Object] = (),
    *,
    cuts_x: Sequence[float] | None = None,
    cuts_pct: Sequence[float] | None = None,
    clearance_mm: float = 0.45,
    angle_deg: float = 24.0,
    bevel_mm: float = 1.25,
    bevel_segments: int = 4,
    curve_mm: float = 2.4,
    flatten_bottom_mm: float = 0.5,
) -> list[bpy.types.Object]:
    """Slice a smooth multi-color Blender character into Flexi-Factory articulated links with:
      - Wide ±24° curved V-flank swing gaps (so segments genuinely wiggle freely!)
      - Native Blender `BEVEL` circular fillets STRICTLY on cut rims (ZERO voxelization, 100% manifold!)
      - Captive vertical Z-axis pins + horizontal chain-link tongues (0° overhang, support-free print-in-place!)
      - Exact non-overlapping multi-color MMU3/AMS partition:
          * `detail_parts` (e.g. Eyes, Teeth, Mouth, Horns): additive 3D features unioned onto the body silhouette.
          * `color_zones` (e.g. Belly, Stripes, Spots, Patches): volumetric paint masks that ONLY intersect the
            existing sculpted silhouette (never inflate or bulge out past the body!).

    Assumes the character faces -X (head at min X, tail at max X), Z-up, resting near Z=0.
    """
    valid_details = [p for p in detail_parts if p is not None and p.type == "MESH"]
    valid_zones = [p for p in color_zones if p is not None and p.type == "MESH"]

    # Seat ONLY the real physical character silhouette (`primary_body` + `detail_parts`) on Z=0,
    # and apply the exact same translation shift to `color_zones` so paint masks stay registered!
    phys_parts = [primary_body, *valid_details]
    for o in (*phys_parts, *valid_zones):
        pa_apply_all_modifiers(o)
    bmin0, bmax0 = pa_bounds(phys_parts)
    shift0 = Vector((
        -(bmin0.x + bmax0.x) * 0.5,
        -(bmin0.y + bmax0.y) * 0.5,
        -bmin0.z - max(0.0, float(flatten_bottom_mm)),
    ))
    for o in (*phys_parts, *valid_zones):
        o.data.transform(Matrix.Translation(shift0))
        o.data.update()
    if flatten_bottom_mm > 0.01:
        span_x = (bmax0.x - bmin0.x) + 40.0
        span_y = (bmax0.y - bmin0.y) + 40.0
        span_z = (bmax0.z - bmin0.z) + 40.0
        for o in phys_parts:
            keep = pa_box("_bed_keep", (0.0, 0.0, span_z * 0.5), (span_x, span_y, span_z))
            pa_bool(o, keep, "INTERSECT", delete_tool=True)

    # Unified physical master hull = primary_body ONLY (detail_parts like Eyes/Teeth/Mouth stay clean closed solids!)
    master = pa_duplicate(primary_body, "_flexi_master")

    bmin, bmax = pa_bounds(master)
    x0, y0, z0 = bmin.x, bmin.y, max(0.0, bmin.z)
    x1, y1, z1 = bmax.x, bmax.y, bmax.z
    length = max(1.0, x1 - x0)
    width_all = max(1.0, y1 - y0)
    height_all = max(1.0, z1 - z0)
    yc_global = 0.5 * (y0 + y1)

    if cuts_x:
        cut_xs = sorted(float(x) for x in cuts_x if x0 + 6.0 < float(x) < x1 - 6.0)
    elif cuts_pct:
        cut_xs = sorted(x0 + length * (float(p) / 100.0) for p in cuts_pct if 10.0 <= float(p) <= 92.0)
    else:
        cut_xs = [x0 + length * p for p in (0.31, 0.54, 0.64, 0.73, 0.81)]

    c = max(0.28, min(0.75, float(clearance_mm)))
    gap = max(1.15, 2.5 * c)
    eff_deg = max(12.0, min(32.0, float(angle_deg)))
    tan_half = math.tan(math.radians(eff_deg * 0.5)) * 0.68
    pad_y = width_all + 30.0
    pad_z = height_all + 30.0

    # Measure local cross-section at each cut plane to size Z-axis pin & chain-link tongue safely
    bounds_x = [x0 - 5.0, *cut_xs, x1 + 5.0]
    joint_specs: list[dict[str, float]] = []
    active_cuts: list[float] = []
    for i, u in enumerate(cut_xs):
        probe = pa_duplicate(master, f"_probe_{i}")
        slab = pa_box(f"_slab_{i}", (u, yc_global, z0 + height_all * 0.5), (2.2, pad_y, pad_z))
        pa_bool(probe, slab, "INTERSECT", delete_tool=True)
        if not probe.data.vertices:
            pa_delete(probe)
            continue
        pmin, pmax = pa_bounds(probe)
        pa_delete(probe)
        w_loc = pmax.y - pmin.y
        z_bot = max(0.0, pmin.z)
        z_top = pmax.z
        h_loc = z_top - z_bot
        yc = 0.5 * (pmin.y + pmax.y)
        prev_cut = active_cuts[-1] if active_cuts else bounds_x[i]
        seg_pitch = min(u - prev_cut, bounds_x[i + 2] - u)
        if z_bot > 4.5 or w_loc < 7.5 or h_loc < 7.0 or seg_pitch < (gap + 4.0 * c + 2.8):
            continue

        floor_t = max(1.45, min(2.2, h_loc * 0.14))
        roof_t = max(1.50, min(2.4, h_loc * 0.15))
        avail_h = h_loc - floor_t - roof_t - 2.0 * c
        if avail_h < 2.2:
            continue
        loop_h = max(2.2, min(6.0, avail_h))
        cav_z0 = z_bot + floor_t
        cav_z1 = cav_z0 + loop_h + 2.0 * c

        pin_r = max(1.05, min(1.75, min(w_loc * 0.088, (seg_pitch - gap) * 0.155)))
        wall_xy = max(1.15, min(1.65, pin_r * 0.92))
        r_loop_xy = pin_r + c + wall_xy
        slot_w = 2.0 * (r_loop_xy + c + 0.25)
        if slot_w > w_loc - 3.0:
            max_r = (w_loc - 3.0) * 0.5 - c - 0.25
            if max_r < 1.95 + c:
                continue
            pin_r = max(0.95, (max_r - c) * 0.48)
            wall_xy = max(1.00, (max_r - c) * 0.52)
            r_loop_xy = pin_r + c + wall_xy
            slot_w = 2.0 * (r_loop_xy + c + 0.25)

        u_pin = u - (gap * 0.5 + c + pin_r + 0.45)
        cav_front = u_pin - (r_loop_xy + c)
        min_front = (active_cuts[-1] + gap * 0.5 + 1.2) if active_cuts else (x0 + 1.5)
        if cav_front <= min_front:
            continue

        local_curve = min(float(curve_mm), w_loc * 0.10, seg_pitch * 0.16)
        max_half_gap = min(2.4, max(gap * 0.5 + 0.65, seg_pitch * 0.20))
        active_cuts.append(u)
        joint_specs.append({
            "u": u, "yc": yc, "w_loc": w_loc, "z_bot": z_bot, "z_top": z_top, "h_loc": h_loc,
            "pin_r": pin_r, "r_loop_xy": r_loop_xy, "slot_w": slot_w,
            "cav_z0": cav_z0, "cav_z1": cav_z1, "u_pin": u_pin, "cav_front": cav_front,
            "curve": local_curve, "max_half_gap": max_half_gap,
        })

    if not active_cuts:
        pa_delete(master)
        return [primary_body, *valid_zones, *valid_details]

    def _cut_x(j: int, y_val: float, side: float) -> float:
        sp = joint_specs[j]
        half_w = max(4.0, sp["w_loc"] * 0.5)
        half_slot = sp["slot_w"] * 0.5 + 0.35
        flank = max(0.0, abs(y_val - sp["yc"]) - half_slot)
        span = max(2.0, half_w - half_slot)
        t = min(1.0, flank / span)
        base_x = sp["u"] + sp["curve"] * (t * t)
        half_gap = min(sp["max_half_gap"], gap * 0.5 + min(half_w + 2.0, flank) * tan_half)
        return base_x + side * half_gap

    # Slice master solid into N+1 sleek beveled segments
    n_segs = len(active_cuts) + 1
    seg_objs: list[bpy.types.Object] = []
    slab_prisms: list[bpy.types.Object] = []
    for seg_i in range(n_segs):
        left_fn = (lambda _y: x0 - 10.0) if seg_i == 0 else (lambda yv, idx=seg_i - 1: _cut_x(idx, yv, +1.0))
        right_fn = (lambda _y: x1 + 10.0) if seg_i == len(active_cuts) else (lambda yv, idx=seg_i: _cut_x(idx, yv, -1.0))
        prism = _make_flexi_slab_prism(
            f"_seg_prism_{seg_i}", left_fn, right_fn, y0 - 12.0, y1 + 12.0, -5.0, z1 + 12.0, n_samples=25,
        )
        slab_prisms.append(prism)
        seg = pa_duplicate(master, f"_flexi_seg_{seg_i}")
        pa_bool(seg, pa_duplicate(prism, f"_seg_prism_cp_{seg_i}"), "INTERSECT", delete_tool=True)
        # Bevel ONLY the freshly created cut-rim edge loops (ZERO voxelization!)
        eff_bevel = min(float(bevel_mm), max(0.45, (bounds_x[seg_i + 1] - bounds_x[seg_i]) * 0.14))
        _bevel_cut_rim_edges(seg, bevel_mm=eff_bevel, segments=bevel_segments)
        seg_objs.append(seg)

    # Install captive vertical Z-axis pins (Segment i) & horizontal chain-link tongues (Segment i+1)
    hinge_keepouts: list[bpy.types.Object] = []
    for j_idx, sp in enumerate(joint_specs):
        u = sp["u"]
        yc = sp["yc"]
        pin_r = sp["pin_r"]
        r_loop = sp["r_loop_xy"]
        cav_z0 = sp["cav_z0"]
        cav_z1 = sp["cav_z1"]
        u_pin = sp["u_pin"]
        cav_back_x = u + gap * 0.5 + 2.4

        # 1. Carve enclosed horizontal pocket in Segment j_idx & add vertical Z-pin
        cavity = pa_stadium_xy(f"_cav_{j_idx}", u_pin, yc, r_loop + c, cav_back_x, cav_z0, cav_z1)
        pa_bool(seg_objs[j_idx], cavity, "DIFFERENCE", delete_tool=True)

        pin = pa_cylinder(f"_pin_{j_idx}", (u_pin, yc, 0.5 * (cav_z0 + cav_z1)), pin_r, (cav_z1 - cav_z0) + 1.8)
        pa_bool(pin, pa_duplicate(master, f"_pin_clip_m_{j_idx}"), "INTERSECT", delete_tool=True)
        pa_bool(pin, pa_duplicate(slab_prisms[j_idx], f"_pin_clip_s_{j_idx}"), "INTERSECT", delete_tool=True)
        pa_bool(seg_objs[j_idx], pin, "UNION", delete_tool=True)

        # 2. Horizontal chain-link tongue wrapping 360° around pin, fused into Segment j_idx + 1
        next_lim = joint_specs[j_idx + 1]["cav_front"] - 0.6 if (j_idx + 1) < len(joint_specs) else x1 - 1.2
        anchor_x = min(u + gap * 0.5 + 3.2, next_lim)
        if anchor_x <= u + gap * 0.5 + 0.8:
            anchor_x = u + gap * 0.5 + 1.0
        z_l0 = cav_z0 + c
        z_l1 = cav_z1 - c
        tongue = pa_stadium_xy(f"_tongue_{j_idx}", u_pin, yc, r_loop, anchor_x, z_l0, z_l1)
        pin_hole = pa_cylinder(f"_pinhole_{j_idx}", (u_pin, yc, 0.5 * (z_l0 + z_l1)), pin_r + c, (z_l1 - z_l0) + 4.0)
        pa_bool(tongue, pin_hole, "DIFFERENCE", delete_tool=True)
        pa_bool(tongue, pa_duplicate(master, f"_t_clip_{j_idx}"), "INTERSECT", delete_tool=True)
        pa_bool(seg_objs[j_idx + 1], tongue, "UNION", delete_tool=True)

        # Keepout volume around hinge so secondary colors (e.g. Belly) never split across internal pin/tongue
        keepout = pa_stadium_xy(
            f"_hkeep_{j_idx}", u_pin, yc, r_loop + c + 0.4, anchor_x + 0.4, cav_z0 - 0.3, cav_z1 + 0.3
        )
        hinge_keepouts.append(keepout)

    for sp_obj in slab_prisms:
        pa_delete(sp_obj)
    pa_delete(master)

    # Combine all articulated segments into the final flexi body solid (ZERO coplanar detail booleans!)
    flexi_body = pa_join(seg_objs, name="_flexi_body")
    pa_repair_mesh(flexi_body, merge_dist=0.0)
    _collapse_to_single_material(flexi_body)

    # Subtract internal hinge keepouts from color_zones (e.g. Belly) so Z-pins & chain-link tongues stay
    # 100% solid primary Body filament; exact coplanar multi-color clipping & disjoint subtraction are
    # handled cleanly in C++ `manifold3d` by `mesh_prep.py` (guaranteeing zero T-junctions!).
    keepout_all = pa_join(hinge_keepouts, name="_hinge_keepout") if hinge_keepouts else None
    for idx, z_obj in enumerate(valid_zones):
        if keepout_all is not None:
            pa_bool(z_obj, pa_duplicate(keepout_all, f"_zhk_{idx}"), "DIFFERENCE", delete_tool=True)
        pa_repair_mesh(z_obj, merge_dist=0.0)
        _collapse_to_single_material(z_obj)
        z_obj["pa_color_zone"] = True
        z_obj["pa_flexi_baked"] = True

    if keepout_all is not None:
        pa_delete(keepout_all)

    for dp in valid_details:
        pa_repair_mesh(dp, merge_dist=0.0)
        _collapse_to_single_material(dp)
        dp["pa_color_zone"] = False
        dp["pa_flexi_baked"] = True

    body_mats = list(primary_body.data.materials)
    body_props = {k: primary_body[k] for k in primary_body.keys() if k != "_RNA_UI"}
    body_name = primary_body.name
    pa_delete(primary_body)

    body_out = flexi_body
    body_out.name = body_name
    body_out.data.materials.clear()
    if body_mats:
        body_out.data.materials.append(body_mats[0])
    _collapse_to_single_material(body_out)
    for k, v in body_props.items():
        body_out[k] = v
    body_out["pa_color_zone"] = False
    body_out["pa_flexi_baked"] = True

    return [body_out, *valid_zones, *valid_details]


def pa_partition_colors(
    primary_body: bpy.types.Object,
    detail_parts: Sequence[bpy.types.Object] = (),
    color_zones: Sequence[bpy.types.Object] = (),
    *,
    flatten_bottom_mm: float = 0.4,
) -> list[bpy.types.Object]:
    """Seat a non-flexi multi-color Blender character on Z=0 and tag color_zones for exact C++ manifold3d splitting."""
    valid_details = [p for p in detail_parts if p is not None and p.type == "MESH"]
    valid_zones = [p for p in color_zones if p is not None and p.type == "MESH"]
    phys_parts = [primary_body, *valid_details]
    for o in (*phys_parts, *valid_zones):
        pa_apply_all_modifiers(o)
    bmin0, bmax0 = pa_bounds(phys_parts)
    shift0 = Vector((
        -(bmin0.x + bmax0.x) * 0.5,
        -(bmin0.y + bmax0.y) * 0.5,
        -bmin0.z - max(0.0, float(flatten_bottom_mm)),
    ))
    for o in (*phys_parts, *valid_zones):
        o.data.transform(Matrix.Translation(shift0))
        o.data.update()
    if flatten_bottom_mm > 0.01:
        span_x = (bmax0.x - bmin0.x) + 40.0
        span_y = (bmax0.y - bmin0.y) + 40.0
        span_z = (bmax0.z - bmin0.z) + 40.0
        for o in phys_parts:
            keep = pa_box("_bed_keep", (0.0, 0.0, span_z * 0.5), (span_x, span_y, span_z))
            pa_bool(o, keep, "INTERSECT", delete_tool=True)
    for o in phys_parts:
        pa_repair_mesh(o, merge_dist=0.0)
        _collapse_to_single_material(o)
        o["pa_color_zone"] = False
    for z in valid_zones:
        pa_repair_mesh(z, merge_dist=0.0)
        _collapse_to_single_material(z)
        z["pa_color_zone"] = True
    return [primary_body, *valid_zones, *valid_details]


def pa_export_glb(out_path: str | Path, *, baked_flexi: bool | None = None) -> Path:
    """Export all visible scene meshes to a Z-up millimetre `.glb` preserving multi-color materials and extras."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and not obj.hide_get():
            pa_apply_all_modifiers(obj)
            _collapse_to_single_material(obj)
            if baked_flexi is not None:
                obj["pa_flexi_baked"] = bool(baked_flexi)
    bpy.ops.export_scene.gltf(
        filepath=str(out),
        export_format="GLB",
        use_selection=False,
        export_yup=False,
        export_apply=True,
        export_materials="EXPORT",
        export_extras=True,
        export_normals=False,
    )
    return out
