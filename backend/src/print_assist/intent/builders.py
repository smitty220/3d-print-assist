"""Build intent objects from faces picked on imported mating parts."""

from __future__ import annotations

import math

from build123d import CenterOf, Face, GeomType, Shape, Vector

from ..catalog import suggest_fastener
from ..geometry.signature import face_signature
from .models import Bore, BoltPattern, DesignSpace, FaceRef, Frame, Hole, Intent, LoadCase, PlanarMount


class BuildError(ValueError):
    pass


def _t(v: Vector) -> tuple[float, float, float]:
    return (round(v.X, 6), round(v.Y, 6), round(v.Z, 6))


def _perpendicular(z: Vector) -> Vector:
    helper = Vector(0, 0, 1) if abs(z.Z) < 0.9 else Vector(1, 0, 0)
    return helper.cross(z).normalized()


def _plane_frame(face: Face) -> tuple[Vector, Vector, Vector]:
    """Origin at the face centroid, Z along the outward normal, X along the longest straight outer edge."""
    z = face.normal_at().normalized()
    origin = face.center(CenterOf.MASS)
    lines = [e for e in face.outer_wire().edges() if e.geom_type == GeomType.LINE]
    if lines:
        e = max(lines, key=lambda e: e.length)
        d = (e @ 1) - (e @ 0)
        x = (d - z * d.dot(z)).normalized()
    else:
        x = _perpendicular(z)
    return origin, z, x


def _to_local(p: Vector, origin: Vector, z: Vector, x: Vector) -> tuple[float, float]:
    y = z.cross(x)
    v = p - origin
    return (round(v.dot(x), 6), round(v.dot(y), 6))


def _face_ref(shape: Shape, part: str, face_id: int) -> tuple[Face, FaceRef]:
    faces = shape.faces()
    if not 0 <= face_id < len(faces):
        raise BuildError(f"{part} has no face {face_id}")
    face = faces[face_id]
    return face, FaceRef(part=part, face_id=face_id, signature=face_signature(face))


def _circular_holes(face: Face) -> list[tuple[Vector, float]]:
    """(centre, diameter) for every inner loop of a planar face made only of circular arcs on one centre."""
    holes = []
    for wire in face.inner_wires():
        edges = wire.edges()
        if not edges or any(e.geom_type != GeomType.CIRCLE for e in edges):
            continue
        centres = [e.arc_center for e in edges]
        if any((c - centres[0]).length > 1e-3 for c in centres):
            continue
        holes.append((centres[0], 2 * edges[0].radius))
    return holes


def planar_mount(intent: Intent, shape: Shape, part: str, face_id: int) -> PlanarMount:
    face, ref = _face_ref(shape, part, face_id)
    if face.geom_type != GeomType.PLANE:
        raise BuildError("A planar mount needs a planar face")
    origin, z, x = _plane_frame(face)
    pts: list[tuple[float, float]] = []
    for e in face.outer_wire().edges():
        n = 1 if e.geom_type == GeomType.LINE else 24
        pts += [_to_local(e @ (i / n), origin, z, x) for i in range(n)]
    return PlanarMount(
        id=intent.next_id("planar_mount"),
        label=f"Mount on {part}",
        frame=Frame(origin=_t(origin), z=_t(z), x=_t(x)),
        footprint=pts,
        face=ref,
    )


def _mount_face_for_hole(shape: Shape, cyl: Face) -> int:
    """The largest planar face of the part with a circular inner loop on this cylinder's axis."""
    sig = face_signature(cyl)
    a_o, a_d = Vector(*sig.axis_origin), Vector(*sig.axis_dir)
    best: tuple[float, int] | None = None
    for i, f in enumerate(shape.faces()):
        if f.geom_type != GeomType.PLANE:
            continue
        for c, d in _circular_holes(f):
            v = c - a_o
            off_axis = (v - a_d * v.dot(a_d)).length
            if off_axis < 1e-3 and math.isclose(d, 2 * sig.radius, rel_tol=1e-3):
                if best is None or f.area > best[0]:
                    best = (f.area, i)
    if best is None:
        raise BuildError("Could not find the planar face this hole passes through")
    return best[1]


def bolt_pattern(intent: Intent, shape: Shape, part: str, face_id: int) -> BoltPattern:
    face = shape.faces()[face_id] if 0 <= face_id < len(shape.faces()) else None
    if face is None:
        raise BuildError(f"{part} has no face {face_id}")
    if face.geom_type == GeomType.CYLINDER:
        face_id = _mount_face_for_hole(shape, face)
    face, ref = _face_ref(shape, part, face_id)
    if face.geom_type != GeomType.PLANE:
        raise BuildError("Pick the face the holes pass through, or one of the holes")
    holes = _circular_holes(face)
    if not holes:
        raise BuildError("That face has no round holes")
    origin, z, x = _plane_frame(face)
    diameters = sorted({round(d, 3) for _, d in holes})
    parent = next(
        (o.id for o in intent.objects if o.kind == "planar_mount" and o.face.part == part and o.face.face_id == face_id),
        None,
    )
    return BoltPattern(
        id=intent.next_id("bolt_pattern"),
        label=f"{len(holes)}x hole pattern on {part}",
        frame=Frame(origin=_t(origin), z=_t(z), x=_t(x)),
        holes=[
            Hole(center=_to_local(c, origin, z, x), diameter=round(d, 6), depth=_hole_depth(shape, c, z, d))
            for c, d in holes
        ],
        fastener=suggest_fastener(diameters[0]) if len(diameters) == 1 else None,
        parent=parent,
        face=ref,
    )


def bore(intent: Intent, shape: Shape, part: str, face_id: int) -> Bore:
    face, ref = _face_ref(shape, part, face_id)
    if face.geom_type != GeomType.CYLINDER:
        raise BuildError("A bore needs a cylindrical face")
    sig = ref.signature
    a_o, a_d = Vector(*sig.axis_origin), Vector(*sig.axis_dir).normalized()
    r = sig.radius
    seg_len = _axial_length(face, a_d)
    if face.area / (2 * math.pi * r * max(seg_len, 1e-9)) < 0.95:
        raise BuildError("That cylindrical face is only a partial arc; pick a full bore")

    # Every full cylinder of the same radius on the same axis is part of this bore (a forked end has two).
    spans = []
    for f in shape.faces():
        if f.geom_type != GeomType.CYLINDER:
            continue
        s = face_signature(f)
        if abs(s.radius - r) > 1e-3 or abs(Vector(*s.axis_dir).dot(a_d)) < 0.999:
            continue
        v = Vector(*s.axis_origin) - a_o
        if (v - a_d * v.dot(a_d)).length > 1e-3:
            continue
        proj = [(e @ t - a_o).dot(a_d) for e in f.edges() for t in (0.0, 0.5, 1.0)]
        spans.append((min(proj), max(proj)))
    spans.sort()
    gap = None
    if len(spans) == 2 and spans[1][0] > spans[0][1] + 1e-3:
        gap = spans[1][0] - spans[0][1]
        mid_t = (spans[0][1] + spans[1][0]) / 2
    else:
        c = Vector(*sig.centroid)
        mid_t = (c - a_o).dot(a_d)
    mid = a_o + a_d * mid_t
    label = f"Ø{2 * r / 25.4:.3f} bore on {part}" + (f", {gap / 25.4:.3f} gap" if gap else "")
    return Bore(
        id=intent.next_id("bore"),
        label=label,
        frame=Frame(origin=_t(mid), z=_t(a_d), x=_t(_perpendicular(a_d))),
        diameter=round(2 * r, 6),
        length=round(seg_len, 6),
        gap=round(gap, 6) if gap else None,
        face=ref,
    )


def _hole_depth(shape: Shape, centre: Vector, axis: Vector, diameter: float) -> float | None:
    """Axial length of the cylindrical wall of the hole at `centre`, if the part has one."""
    for f in shape.faces():
        if f.geom_type != GeomType.CYLINDER:
            continue
        sig = face_signature(f)
        if not math.isclose(2 * sig.radius, diameter, rel_tol=1e-3):
            continue
        a = Vector(*sig.axis_dir)
        v = centre - Vector(*sig.axis_origin)
        if abs(a.dot(axis)) > 0.999 and (v - a * v.dot(a)).length < 1e-3:
            return round(_axial_length(f, a), 6)
    return None


def _axial_length(face: Face, axis: Vector) -> float:
    """Extent of the face along the axis, from points sampled on its boundary edges."""
    proj = [(e @ t).dot(axis) for e in face.edges() for t in (0.0, 0.5, 1.0)]
    return max(proj) - min(proj)


DESIGN_SPACE_MARGIN_MM = 12.7  # 0.5 in around the interfaces


def _world(frame: Frame, local: tuple[float, float, float]) -> Vector:
    o, z, x = Vector(*frame.origin), Vector(*frame.z), Vector(*frame.x)
    y = z.cross(x)
    return o + x * local[0] + y * local[1] + z * local[2]


def _local3(frame: Frame, p: Vector) -> tuple[float, float, float]:
    o, z, x = Vector(*frame.origin), Vector(*frame.z), Vector(*frame.x)
    v = p - o
    return (v.dot(x), v.dot(z.cross(x)), v.dot(z))


def default_design_space(intent: Intent) -> DesignSpace:
    """A box in the primary mount's frame that encloses every interface plus a margin, starting at the mount face."""
    mount = next((o for o in intent.objects if o.kind == "planar_mount"), None)
    if mount is None:
        return _pin_mounted_design_space(intent)
    f = mount.frame
    pts: list[tuple[float, float, float]] = []
    for o in intent.objects:
        if o.kind == "bolt_pattern":
            for h in o.holes:
                pts.append(_local3(f, _world(o.frame, (h.center[0], h.center[1], 0.0))))
        elif o.kind == "bore":
            r = o.diameter / 2
            for t in (-o.length / 2, o.length / 2):
                c = _local3(f, _world(o.frame, (0.0, 0.0, t)))
                pts += [(c[0] + dx, c[1] + dy, c[2] + dz) for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)]
    if not pts:
        pts = [(x, y, 0.0) for x, y in mount.footprint]
    m = DESIGN_SPACE_MARGIN_MM
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return DesignSpace(
        id=intent.next_id("design_space"),
        label="Design space",
        source="typed",
        frame=f,
        min=(round(lo[0] - m, 4), round(lo[1] - m, 4), 0.0),
        max=(round(hi[0] + m, 4), round(hi[1] + m, 4), round(max(hi[2], 0.0) + m, 4)),
    )


def _pin_mounted_design_space(intent: Intent) -> DesignSpace:
    """No mount face (a pin-mounted part such as a bell crank): a box round every bore plus the margin, in a frame
    at the first fixed bore (else the first bore) with Z along its pin."""
    bores = [o for o in intent.objects if o.kind == "bore"]
    if not bores:
        raise BuildError("Create a planar mount or a bore first; the design space is placed in its frame")
    ref = next((b for b in bores if b.support in ("fixed", "pin")), bores[0])
    z = Vector(*ref.frame.z).normalized()
    wx = Vector(1, 0, 0) if abs(z.X) < 0.9 else Vector(0, 1, 0)  # keep the box square to world where possible
    f = Frame(origin=ref.frame.origin, z=_t(z), x=_t((wx - z * wx.dot(z)).normalized()))
    pts = []
    for b in bores:
        r = b.diameter / 2
        half = b.gap / 2 if b.gap else b.length / 2  # a forked mating end: the part lives in its gap
        for t in (-half, half):
            c = _local3(f, _world(b.frame, (0.0, 0.0, t)))
            pts += [(c[0] + dx, c[1] + dy, c[2]) for dx in (-r, r) for dy in (-r, r)]
    m = DESIGN_SPACE_MARGIN_MM
    lo = [round(min(p[i] for p in pts) - m, 4) for i in range(3)]
    hi = [round(max(p[i] for p in pts) + m, 4) for i in range(3)]
    return DesignSpace(id=intent.next_id("design_space"), label="Design space", source="typed", frame=f, min=tuple(lo), max=tuple(hi))


def new_load_case(intent: Intent, target: str | None) -> LoadCase:
    if target is None:
        target = next((o.id for o in intent.objects if o.kind == "bore"), None) or next(
            (o.id for o in intent.objects if o.kind in ("planar_mount", "bolt_pattern")), None
        )
    elif intent.get(target) is None or intent.get(target).kind not in ("planar_mount", "bolt_pattern", "bore"):
        raise BuildError(f"{target} is not an interface")
    n = intent.next_id("load_case")
    return LoadCase(id=n, label=f"Load case {n[3:]}", source="typed", target=target)
