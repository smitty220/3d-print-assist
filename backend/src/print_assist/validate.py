"""Validator: pre-flight checks on the intent, and checks of the built part against it.

Pre-flight runs on the intent alone and answers "can any part satisfy this?" before anything is built
(or before the AI loop spends a generation on it). Part checks answer "does this part satisfy the intent?":
keep-out clearance, design-space containment, interference with mating parts, every interface actually
matched, minimum wall, and mass. Lengths in mm, points in world coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from build123d import Align, Cylinder, Edge, Face, GeomType, Plane, Shape, Vector, Vertex
from OCP.BRep import BRep_Tool
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.ShapeAnalysis import ShapeAnalysis_Surface

from .catalog import FASTENERS
from .features.context import frame_plane
from .features.design import BuildResult
from .intent.models import BoltPattern, Bore, DesignSpace, Intent, KeepOut, LoadCase, PlanarMount, resolved_support

IN = 25.4
Status = Literal["ok", "warn", "fail"]
Vec3 = tuple[float, float, float]

AXIS_TOL = 0.05  # mm: a hole is "on" an interface axis within this
FIT_TOL = 0.05  # mm on radius: a part bore matches a pin bore within this
VOL_TOL = 0.5  # mm³: overlaps smaller than this are boolean noise
MIN_Z = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Finding:
    stage: Literal["preflight", "part"]
    code: str
    title: str
    status: Status
    detail: str
    value: float | None = None
    quantity: Literal["length", "area", "ratio", "percent", "mass", "count"] | None = None
    refs: list[str] = field(default_factory=list)
    points: list[Vec3] = field(default_factory=list)  # 1 = marker, 2 = dimension between them

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    preflight: list[Finding]
    part: list[Finding]

    def to_dict(self) -> dict:
        everything = self.preflight + self.part
        counts = {s: sum(f.status == s for f in everything) for s in ("ok", "warn", "fail")}
        return {
            "preflight": [f.to_dict() for f in self.preflight],
            "part": [f.to_dict() for f in self.part],
            "counts": counts,
            "preflight_ready": not any(f.status == "fail" for f in self.preflight),
        }


def _in(v: float) -> str:
    return f"{v / IN:.3f} in"


def _t(v: Vector) -> Vec3:
    return (round(v.X, 4), round(v.Y, 4), round(v.Z, 4))


def _vol(shape: Shape | None) -> float:
    if shape is None:
        return 0.0
    try:
        return max(0.0, shape.volume)
    except Exception:
        return 0.0


def _distance(a: Shape, b: Shape) -> tuple[float, Vector, Vector]:
    d = BRepExtrema_DistShapeShape()
    d.LoadS1(a.wrapped)
    d.LoadS2(b.wrapped)
    d.SetMultiThread(True)
    d.Perform()
    if not d.IsDone() or d.NbSolution() == 0:
        raise ValueError("distance computation failed")
    p1, p2 = d.PointOnShape1(1), d.PointOnShape2(1)
    return d.Value(), Vector(p1.X(), p1.Y(), p1.Z()), Vector(p2.X(), p2.Y(), p2.Z())


def _plane(frame) -> Plane:
    return frame_plane(frame)


def _of(intent: Intent, kind: str) -> list:
    return [o for o in intent.objects if o.kind == kind]


def _hole_points(bp: BoltPattern) -> list[Vector]:
    pl = _plane(bp.frame)
    return [pl.from_local_coords(Vector(h.center[0], h.center[1], 0)) for h in bp.holes]


def _pin_segment(b: Bore, extra: float) -> tuple[Vector, Vector]:
    o, z = Vector(*b.frame.origin), Vector(*b.frame.z).normalized()
    half = (b.gap if b.gap is not None else b.length) / 2 + extra
    return o - z * half, o + z * half


# ---------------------------------------------------------------- pre-flight


def preflight(intent: Intent, shape_of: Callable[[str], Shape]) -> list[Finding]:
    out: list[Finding] = []
    F = lambda *a, **k: out.append(Finding("preflight", *a, **k))  # noqa: E731

    mounts: list[PlanarMount] = _of(intent, "planar_mount")
    if mounts:
        F("mount", "Primary mount", "ok", f"{mounts[0].id} sets the build frame: Z = 0 on its face", refs=[mounts[0].id])
    elif any(o.kind == "bore" and resolved_support(intent, o) != "free" for o in intent.objects):
        F("mount", "Primary mount", "warn", "Pin-mounted, no planar mount: topology optimisation can run, but the feature library builds parts on a mount face")
    else:
        F("mount", "Primary mount", "fail", "Pick the face the part bolts to and create a planar mount; parts are built on it")

    spaces: list[DesignSpace] = _of(intent, "design_space")
    ds = spaces[0] if spaces else None
    if ds:
        size = [ds.max[i] - ds.min[i] for i in range(3)]
        F("design_space", "Design space", "ok", f"{ds.id} is {' × '.join(f'{s / IN:.2f}' for s in size)} in", refs=[ds.id])
    else:
        F("design_space", "Design space", "warn", "No design space: only the keep-outs bound the part's size")

    for bp in _of(intent, "bolt_pattern"):
        hole_d = min(h.diameter for h in bp.holes)
        f = FASTENERS.get(bp.fastener or "")
        if f is None:
            F("fastener", f"{bp.id} fastener", "warn", "No fastener assigned: holes are sized from the mating holes and head clearance is not checked", refs=[bp.id])
        elif f.shank_d * IN > hole_d + 1e-3:
            F("fastener", f"{bp.id} fastener", "fail", f"{f.part_number} shank Ø{f.shank_d:.3f} in will not pass the mating Ø{hole_d / IN:.3f} in holes", refs=[bp.id])
        else:
            F("fastener", f"{bp.id} fastener", "ok", f"{len(bp.holes)}× {f.part_number} through Ø{hole_d / IN:.3f} in mating holes", refs=[bp.id])

    mount_pl = _plane(mounts[0].frame) if mounts else None
    for b in _of(intent, "bore"):
        if mount_pl is None:
            break
        c = mount_pl.to_local_coords(Vector(*b.frame.origin))
        tilt = abs(Vector(*b.frame.z).dot(mount_pl.z_dir))
        if c.Z <= b.diameter / 2:
            F("bore", f"{b.id} pin position", "fail", f"The pin axis is {_in(c.Z)} above the mount face; a lug cannot reach under it", value=c.Z, quantity="length", refs=[b.id])
        elif tilt > 0.02:
            F("bore", f"{b.id} pin position", "warn", f"The pin axis is {math.degrees(math.asin(min(1, tilt))):.1f}° out of parallel with the mount face; the library lugs assume parallel", refs=[b.id])
        else:
            F("bore", f"{b.id} pin position", "ok", f"Pin axis {_in(c.Z)} above the mount face, parallel to it", value=c.Z, quantity="length", refs=[b.id])

    if ds:
        pl = _plane(ds.frame)
        for o in intent.objects:
            if o.kind == "bolt_pattern":
                pts = _hole_points(o)
            elif o.kind == "bore":
                pts = list(_pin_segment(o, 0))
            else:
                continue
            outside = [p for p in pts if not _inside_box(pl.to_local_coords(p), ds, 1e-3)]
            if outside:
                F("in_space", f"{o.id} in design space", "fail", f"{len(outside)} point(s) of {o.id} lie outside {ds.id}; grow the design space to include it", refs=[o.id, ds.id], points=[_t(outside[0])])

    for ko in _of(intent, "keep_out"):
        _preflight_keepout(intent, ko, shape_of, ds, F)

    loads: list[LoadCase] = _of(intent, "load_case")
    if not loads:
        F("loads", "Load cases", "warn", "No load case: the part can be built but not sized or checked for strength")
    for lc in loads:
        mag = math.dist((0, 0, 0), lc.force)
        if lc.target is None:
            F("loads", f"{lc.id} load", "fail", "Not applied at an interface; pick one in its properties", refs=[lc.id])
        elif mag < 1e-9 and math.dist((0, 0, 0), lc.moment) < 1e-9:
            F("loads", f"{lc.id} load", "warn", f"Zero force and moment at {lc.target}", refs=[lc.id, lc.target])
        else:
            F("loads", f"{lc.id} load", "ok", f"{mag / 4.4482216:.0f} lbf {lc.category} at {lc.target}", refs=[lc.id, lc.target])

    ifaces = [o for o in intent.objects if o.kind in ("planar_mount", "bolt_pattern", "bore")]
    held = [(o, resolved_support(intent, o)) for o in ifaces if resolved_support(intent, o) != "free"]
    if ifaces and not held:
        F("supports", "Supports", "warn", "Nothing holds the part for analysis: set a bolt pattern or bore to Fixed, or the mount face to Normal")
    elif held:
        loaded = {lc.target for lc in loads}
        both = [o.id for o, _s in held if o.id in loaded]
        text = ", ".join(f"{o.id} {s}" for o, s in held)
        if both:
            F("supports", "Supports", "warn", f"{text}; {', '.join(both)} is also loaded, so that load goes straight into the support", refs=[o.id for o, _ in held])
        else:
            F("supports", "Supports", "ok", text, refs=[o.id for o, _ in held])

    if intent.rules.mass_target is None:
        F("mass_target", "Mass target", "warn", "No mass target: the AI will size for strength and rules only")
    else:
        F("mass_target", "Mass target", "ok", f"{intent.rules.mass_target / 0.45359237:.2f} lb ({intent.rules.material})")
    return out


def _inside_box(p: Vector, ds: DesignSpace, tol: float) -> bool:
    return all(ds.min[i] - tol <= (p.X, p.Y, p.Z)[i] <= ds.max[i] + tol for i in range(3))


def _preflight_keepout(intent: Intent, ko: KeepOut, shape_of, ds: DesignSpace | None, F) -> None:
    try:
        vol = shape_of(ko.part)
    except KeyError:
        F("keepout", f"{ko.id} volume", "fail", f"The keep-out part {ko.part} is missing; re-import it", refs=[ko.id])
        return
    # The connection points every candidate part must reach: hole heads and the pin between its lugs.
    worst = None
    for o in intent.objects:
        if o.kind == "bolt_pattern":
            f = FASTENERS.get(o.fastener or "")
            r = (max(f.head_size, f.washer_od) * IN / 2) if f else max(h.diameter for h in o.holes)
            for p in _hole_points(o):
                d, a, b = _distance(vol, _vertex(p))
                if worst is None or d - r < worst[0]:
                    worst = (d - r, o.id, b, a)
        elif o.kind == "bore":
            s0, s1 = _pin_segment(o, 0.5 * IN)
            d, a, b = _distance(vol, Edge.make_line(s0, s1))
            r = intent.rules.lug_edge_full_bearing * o.diameter  # crown radius of a full-bearing lug
            if worst is None or d - r < worst[0]:
                worst = (d - r, o.id, b, a)
    if worst is not None:
        gap, oid, p_if, p_ko = worst
        if gap < ko.clearance:
            F("keepout", f"{ko.id} vs interfaces", "fail" if ko.strength == "hard" else "warn",
              f"Only {_in(max(gap, 0))} between {ko.id} and the material {oid} needs; clearance is {_in(ko.clearance)}",
              value=gap, quantity="length", refs=[ko.id, oid], points=[_t(p_if), _t(p_ko)])
        else:
            F("keepout", f"{ko.id} vs interfaces", "ok", f"{_in(gap)} from the nearest interface ({oid}); needs {_in(ko.clearance)}",
              value=gap, quantity="length", refs=[ko.id, oid], points=[_t(p_if), _t(p_ko)])
    if ds:
        box = _plane(ds.frame).location * _box(ds)
        share = _vol(box & vol) / _vol(box) * 100 if _vol(box) else 0
        F("keepout", f"{ko.id} share of space", "warn" if share > 40 else "ok",
          f"Takes {share:.0f}% of the design space", value=share, quantity="percent", refs=[ko.id, ds.id])


def _vertex(p: Vector) -> Vertex:
    return Vertex(p.X, p.Y, p.Z)


def _box(ds: DesignSpace) -> Shape:
    from build123d import Box, Pos

    size = [ds.max[i] - ds.min[i] for i in range(3)]
    mid = [(ds.max[i] + ds.min[i]) / 2 for i in range(3)]
    return Pos(*mid) * Box(*size)


# ---------------------------------------------------------------- part checks


def check_part(intent: Intent, build: BuildResult, shape_of: Callable[[str], Shape], roles: dict[str, str]) -> list[Finding]:
    out: list[Finding] = []
    F = lambda *a, **k: out.append(Finding("part", *a, **k))  # noqa: E731

    errors = [f for f in build.features if f.error]
    for f in errors:
        F("build", f"{f.id} failed", "fail", f.error or "", refs=[f.id])
    part = build.world
    if part is None:
        F("build", "Part", "fail", "Nothing was built")
        return out
    built = [f for f in build.features if not f.error and not f.suppressed]
    if not errors:
        F("build", "Build", "ok", f"{len(built)} features built, every face named")
    solids = len(part.solids())
    if solids > 1:
        F("build", "One body", "fail", f"The part is {solids} separate bodies; join them")

    bad = [c for c in build.checks if c.status != "ok"]
    for c in bad:
        F("rules", f"{c.feature} {c.name}", c.status, c.rule, value=c.value, refs=[c.feature])
    if not bad:
        F("rules", "Feature rule checks", "ok", f"All {len(build.checks)} pass")

    for ko in _of(intent, "keep_out"):
        try:
            vol = shape_of(ko.part)
        except KeyError:
            continue
        over = part & vol
        if _vol(over) > VOL_TOL:
            c = over.center()
            F("keepout", f"Clearance to {ko.id}", "fail" if ko.strength == "hard" else "warn",
              f"The part enters {ko.id} ({_vol(over) / IN**3:.4f} in³)", value=0.0, quantity="length", refs=[ko.id], points=[_t(c)])
            continue
        d, a, b = _distance(part, vol)
        status = "ok" if d >= ko.clearance - 1e-6 else ("fail" if ko.strength == "hard" else "warn")
        F("keepout", f"Clearance to {ko.id}", status, f"{_in(d)} at the closest point; needs {_in(ko.clearance)}",
          value=d, quantity="length", refs=[ko.id], points=[_t(a), _t(b)])

    for ds in _of(intent, "design_space"):
        pl = _plane(ds.frame)
        local = pl.to_local_coords(part)
        bb = local.bounding_box(optimal=True)
        lo, hi = (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)
        excess = []
        for i, ax in enumerate("XYZ"):
            if lo[i] < ds.min[i] - 1e-3:
                excess.append((ds.min[i] - lo[i], f"−{ax}"))
            if hi[i] > ds.max[i] + 1e-3:
                excess.append((hi[i] - ds.max[i], f"+{ax}"))
        if excess:
            worst = max(excess)
            F("in_space", f"Inside {ds.id}", "fail",
              "Past the design space: " + ", ".join(f"{_in(e)} on {ax}" for e, ax in sorted(excess, reverse=True)),
              value=worst[0], quantity="length", refs=[ds.id])
        else:
            # The part sits on the mount face (the space's floor), so that wall does not count.
            gaps = [lo[0] - ds.min[0], ds.max[0] - hi[0], lo[1] - ds.min[1], ds.max[1] - hi[1], ds.max[2] - hi[2]]
            if lo[2] - ds.min[2] > 1e-3:
                gaps.append(lo[2] - ds.min[2])
            margin = min(gaps)
            F("in_space", f"Inside {ds.id}", "ok", f"Fits with {_in(margin)} to the nearest wall", value=margin, quantity="length", refs=[ds.id])

    for name, role in roles.items():
        if role != "mating":
            continue
        try:
            mate = shape_of(name)
        except KeyError:
            continue
        over = part & mate
        if _vol(over) > VOL_TOL:
            F("interference", f"Clear of {name}", "fail", f"Overlaps {name} by {_vol(over) / IN**3:.4f} in³", refs=[], points=[_t(over.center())])
        else:
            F("interference", f"Clear of {name}", "ok", "No interference")

    for o in intent.objects:
        if o.kind == "planar_mount":
            out += _check_mount(o, part, shape_of)
        elif o.kind == "bolt_pattern":
            out += _check_bolts(o, part)
        elif o.kind == "bore":
            out += _check_bore(o, part)

    out += _check_wall(intent, part)

    target = intent.rules.mass_target
    if target is not None:
        lb = 0.45359237
        F("mass", "Mass", "ok" if build.mass <= target else "warn",
          f"{build.mass / lb:.3f} lb against a {target / lb:.2f} lb target" + ("" if build.mass <= target else f" ({(build.mass / target - 1) * 100:.0f}% over)"),
          value=build.mass, quantity="mass")
    return out


def _check_mount(m: PlanarMount, part: Shape, shape_of) -> list[Finding]:
    pl = _plane(m.frame)
    base = [f for f in part.faces() if f.geom_type == GeomType.PLANE and f.normal_at().dot(pl.z_dir) < -0.9999
            and abs(pl.to_local_coords(f.center()).Z) < 1e-3]
    title = f"{m.id} seated"
    if not base:
        return [Finding("part", "mount", title, "fail", f"No part face lies on {m.id}; the part does not sit on its mount", refs=[m.id])]
    try:
        mate = shape_of(m.face.part).faces()[m.face.face_id]
    except (KeyError, IndexError):
        return [Finding("part", "mount", title, "fail", f"Cannot read the mating face of {m.id}", refs=[m.id])]
    area = sum(f.area for f in base)
    contact = sum(f.area for f in (_common_faces(b, mate) for b in base) for f in f)
    over = max(0.0, area - contact)
    pts = [_t(base[0].center())]
    if contact <= 1e-6:
        return [Finding("part", "mount", title, "fail", "The base face does not touch the mating face", refs=[m.id], points=pts)]
    if over > 0.005 * area:
        return [Finding("part", "mount", title, "warn",
                        f"{contact / IN**2:.2f} in² in contact; {over / area * 100:.0f}% of the base overhangs the mating face",
                        value=contact, quantity="area", refs=[m.id], points=pts)]
    return [Finding("part", "mount", title, "ok", f"{contact / IN**2:.2f} in² in contact, fully supported", value=contact, quantity="area", refs=[m.id], points=pts)]


def _common_faces(a: Face, b: Face) -> list[Face]:
    try:
        c = a & b
    except Exception:
        return []
    return list(c.faces()) if c is not None else []


def _coaxial_cylinders(part: Shape, origin: Vector, axis: Vector) -> list[tuple[Face, float, float]]:
    """(face, radius, axial position of its centre) for cylinder faces on the given axis."""
    found = []
    for f in part.faces():
        if f.geom_type != GeomType.CYLINDER:
            continue
        ax = f.axis_of_rotation
        if abs(abs(ax.direction.normalized().dot(axis)) - 1) > 1e-4:
            continue
        rel = Vector(ax.position) - origin
        off = rel - axis * rel.dot(axis)
        if off.length > AXIS_TOL:
            continue
        found.append((f, f.radius, (f.center() - origin).dot(axis)))
    return found


def _check_bolts(bp: BoltPattern, part: Shape) -> list[Finding]:
    axis = Vector(*bp.frame.z).normalized()
    fast = FASTENERS.get(bp.fastener or "")
    shank_r = (fast.shank_d * IN if fast else min(h.diameter for h in bp.holes)) / 2
    missing, small, blocked = [], [], []
    for i, p in enumerate(_hole_points(bp)):
        cyl = _coaxial_cylinders(part, p, axis)
        if not cyl:
            missing.append((i, p))
            continue
        if min(r for _f, r, _s in cyl) < shank_r - 1e-3:
            small.append((i, p))
            continue
        if fast is not None:
            top = max(_axial_extent(f, p, axis)[1] for f, _r, _s in cyl)
            seat = Plane(origin=p + axis * (top + 0.05), z_dir=axis).location * Cylinder(
                max(fast.head_size, fast.washer_od) * IN / 2, fast.head_height * IN + 0.1 * IN, align=MIN_Z)
            if _vol(part & seat) > VOL_TOL:
                blocked.append((i, p + axis * top))
    title = f"{bp.id} bolt holes"
    n = len(bp.holes)
    if missing:
        return [Finding("part", "bolts", title, "fail", f"{len(missing)} of {n} holes have no matching hole in the part", refs=[bp.id], points=[_t(missing[0][1])])]
    if small:
        return [Finding("part", "bolts", title, "fail", f"{len(small)} of {n} holes are smaller than the {bp.fastener} shank", refs=[bp.id], points=[_t(small[0][1])])]
    if blocked:
        return [Finding("part", "bolts", title, "fail",
                        f"No room for the {bp.fastener} head and washer at {len(blocked)} of {n} holes (part material within Ø{max(fast.head_size, fast.washer_od):.3f} in above the seat)",
                        refs=[bp.id], points=[_t(blocked[0][1])])]
    extra = f", head and washer clearance at all {n}" if fast else ""
    return [Finding("part", "bolts", title, "ok", f"{n} of {n} holes on the pattern{extra}", value=n, quantity="count", refs=[bp.id])]


def _axial_extent(face: Face, origin: Vector, axis: Vector) -> tuple[float, float]:
    vals = [(v.center() - origin).dot(axis) for v in face.vertices()]
    if not vals:
        vals = [(face.center() - origin).dot(axis)]
    # Seam-only cylinders can have few vertices; include edge midpoints for a fuller extent.
    vals += [(e.center() - origin).dot(axis) for e in face.edges()]
    return min(vals), max(vals)


def _check_bore(b: Bore, part: Shape) -> list[Finding]:
    o, axis = Vector(*b.frame.origin), Vector(*b.frame.z).normalized()
    r = b.diameter / 2
    cyl = [(f, rr, s) for f, rr, s in _coaxial_cylinders(part, o, axis) if abs(rr - r) <= FIT_TOL]
    title = f"{b.id} pin bore"
    if b.gap is None:
        half = b.length / 2
        sides = {s < 0 for _f, _r, s in cyl if abs(s) > half}
        ok = len(sides) == 2
        need = "a bore on each side of the rod end"
    else:
        ok = any(abs(s) < b.gap / 2 for _f, _r, s in cyl)
        need = "a lug bore inside the fork gap"
    if not ok:
        return [Finding("part", "bore", title, "fail", f"Needs {need}, coaxial with Ø{b.diameter / IN:.3f} in", refs=[b.id], points=[_t(o)])]
    lo = min(_axial_extent(f, o, axis)[0] for f, _r, _s in cyl)
    hi = max(_axial_extent(f, o, axis)[1] for f, _r, _s in cyl)
    pin = Plane(origin=o + axis * (lo - 1.0), z_dir=axis).location * Cylinder(r - 0.02, hi - lo + 2.0, align=MIN_Z)
    if _vol(part & pin) > VOL_TOL:
        return [Finding("part", "bore", title, "fail", "Part material blocks the pin path between the bores", refs=[b.id], points=[_t(o)])]
    grip = sum(_axial_extent(f, o, axis)[1] - _axial_extent(f, o, axis)[0] for f, _r, _s in cyl)
    return [Finding("part", "bore", title, "ok", f"{len(cyl)} coaxial bore(s), Ø{b.diameter / IN:.3f} in, {_in(grip)} total bearing length, pin path clear",
                    value=grip, quantity="length", refs=[b.id], points=[_t(o)])]


def _check_wall(intent: Intent, part: Shape) -> list[Finding]:
    """Sampled ray thickness: from points across every face, the distance through the material along the inward normal
    (corner readings excluded: the tip of an acute corner reads thin but is not a thin wall)."""
    rules = intent.rules
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(part.wrapped, 1e-4)
    best: tuple[float, Vector, Vector] | None = None
    for face in part.faces():
        surf = BRep_Tool.Surface_s(face.wrapped)
        proj = ShapeAnalysis_Surface(surf)
        try:
            verts, tris = face.tessellate(0.3, 0.5)
        except AttributeError:  # a sliver face the mesher can't triangulate; it has no thickness to sample
            continue
        for a, b, c in tris:
            m = (verts[a] + verts[b] + verts[c]) * (1 / 3)
            uv = proj.ValueOfUV(gp_Pnt(m.X, m.Y, m.Z), 1e-7)
            gp = surf.Value(uv.X(), uv.Y())
            pt = Vector(gp.X(), gp.Y(), gp.Z())
            n = face.normal_at(pt)
            inter.Perform(gp_Lin(gp, gp_Dir(-n.X, -n.Y, -n.Z)), 1e-3, 1e9)
            hits = [inter.WParameter(i) for i in range(1, inter.NbPnt() + 1) if inter.WParameter(i) > 1e-2]
            if hits:
                t = min(hits)
                # Near a corner the ray meets the neighbouring face at once (an acute corner is not a thin wall): a
                # reading counts only if the point is farther from its face's edges than twice the thickness found.
                if t < rules.min_wall_preferred and min(e.distance_to(pt) for e in face.edges()) < 2 * t:
                    continue
                if best is None or t < best[0]:
                    best = (t, pt, pt - n * t)
    if best is None:
        return []
    t, p0, p1 = best
    status = "ok" if t >= rules.min_wall_preferred - 1e-6 else "warn" if t >= rules.min_wall - 1e-6 else "fail"
    return [Finding("part", "wall", "Minimum wall", status,
                    f"{_in(t)} thinnest (sampled); {_in(rules.min_wall_preferred)} preferred, {_in(rules.min_wall)} min",
                    value=t, quantity="length", points=[_t(p0), _t(p1)])]


def validate(intent: Intent, build: BuildResult | None, shape_of, roles: dict[str, str]) -> Report:
    pre = preflight(intent, shape_of)
    part = check_part(intent, build, shape_of, roles) if build is not None else []
    return Report(pre, part)
