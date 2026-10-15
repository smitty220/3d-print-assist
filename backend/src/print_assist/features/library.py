"""The parametric feature library.

Each feature declares its parameters and references, builds a tool solid in world millimetres (Z up, the print
bed at Z = 0) and reports print-rule checks. The AI composes these same features; one definition drives the
edit dialog, the AI tool description and the validator.
"""

from __future__ import annotations

from typing import ClassVar

import math

from build123d import (Align, Axis, Box, Face, GeomType, Keep, Kind, Location, Plane, Pos, Shape, Solid, Text,
                       TextAlign, Vector, Wire, extrude, offset, revolve)

from .context import BASE_PLANES, BuildContext, FeatureError
from .model import Check, FeatureOutput, ParamSpec, RefSpec

P = ParamSpec
MIN_Z = (Align.CENTER, Align.CENTER, Align.MIN)


def _status(value: float, good: float, limit: float, lower_is_better: bool = False):
    if lower_is_better:
        return "ok" if value <= good else "warn" if value <= limit else "fail"
    return "ok" if value >= good else "warn" if value >= limit else "fail"


def _wall_check(ctx, fid: str, name: str, t: float) -> Check:
    r = ctx.rules
    return Check(fid, name, t, _status(t, r.min_wall_preferred, r.min_wall),
                 f"≥ {r.min_wall:g} mm ({r.min_wall_preferred:g} mm preferred)")


def _planar(faces, normal: Vector, tol: float = 1e-6) -> list[Face]:
    return [f for f in faces if f.geom_type == GeomType.PLANE and f.normal_at().dot(normal) > 1 - tol]


class Feature:
    kind: ClassVar[str]
    label: ClassVar[str]
    group: ClassVar[str]
    summary: ClassVar[str]
    params: ClassVar[tuple[ParamSpec, ...]] = ()
    refs: ClassVar[tuple[RefSpec, ...]] = ()

    def build(self, ctx: BuildContext, p: dict, refs: dict[str, str]) -> FeatureOutput:
        raise NotImplementedError

    def checks(self, ctx: BuildContext, fid: str, p: dict, out: FeatureOutput) -> list[Check]:
        return []

    @classmethod
    def describe(cls) -> dict:
        return {
            "kind": cls.kind,
            "label": cls.label,
            "group": cls.group,
            "summary": cls.summary,
            "params": [s.to_dict() for s in cls.params],
            "refs": [r.__dict__ | {"kinds": list(r.kinds)} for r in cls.refs],
        }


# ---------------------------------------------------------------- base


class Plate(Feature):
    kind = "plate"
    label = "Plate"
    group = "Base"
    summary = "Rectangular plate standing on the bed, rounded vertical corners."
    params = (
        P("length", "Length X", "length", 60.0, 1.0, 1000.0),
        P("width", "Width Y", "length", 40.0, 1.0, 1000.0),
        P("thickness", "Thickness", "length", 5.0, 0.2, 500.0),
        P("corner_radius", "Corner radius", "length", 3.0, 0.0, 500.0),
        P("center_x", "Center X", "length", 0.0, -1000.0, 1000.0),
        P("center_y", "Center Y", "length", 0.0, -1000.0, 1000.0),
    )

    def build(self, ctx, p, refs):
        L, W, t, r = p["length"], p["width"], p["thickness"], p["corner_radius"]
        if r >= min(L, W) / 2:
            raise FeatureError("corner radius must be less than half the smaller side")
        plate = Pos(p["center_x"], p["center_y"], 0) * Box(L, W, t, align=MIN_Z)
        if r > 0:
            plate = plate.solid().fillet(r, plate.edges().filter_by(Axis.Z))
        faces = plate.faces()
        bottom = _planar(faces, Vector(0, 0, -1))
        top = _planar(faces, Vector(0, 0, 1))
        sides = [f for f in faces if f not in bottom and f not in top]
        return FeatureOutput(
            "add", plate, named={"bottom": bottom, "top": top, "sides": sides},
            info={"top_z": t, "thickness": t},
        )

    def checks(self, ctx, fid, p, out):
        return [_wall_check(ctx, fid, "Thickness", p["thickness"])]


# ---------------------------------------------------------------- 2.5D profiles


def _offset(faces: list[Face], d: float) -> list[Face]:
    """2-D offset of planar faces (no holes) with round joins; an inward offset may split a face or remove it."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
    from OCP.GeomAbs import GeomAbs_Arc

    out: list[Face] = []
    for f in faces:
        mo = BRepOffsetAPI_MakeOffset(f.wrapped, GeomAbs_Arc)
        mo.Perform(d)
        if not mo.IsDone():
            raise FeatureError("could not round the outline's corners")
        wires = [w for w in Shape.cast(mo.Shape()).wires() if w.is_closed]
        loops = sorted((Face(w) for w in wires), key=lambda g: -g.area)
        for g in loops:  # a loop inside a bigger one is a hole in it
            host = next((k for k, o in enumerate(out) if o.area > g.area and o.is_inside(g.center())), None)
            if host is None:
                out.append(g)
            else:
                out[host] = (out[host] - g).faces()[0]
    return [f for f in out if f.area > 1e-6]


def _profile_face(points: list, r: float, round_convex: bool = True, round_concave: bool = True) -> list[Face]:
    """Faces of a closed outline in the XY plane, corners rounded to at least `r` (the cutter radius): growing then
    shrinking by r rounds the inside corners, shrinking then growing rounds the outside ones (and drops anything
    narrower than the cutter)."""
    if len(points) < 3:
        raise FeatureError("an outline needs at least 3 points")
    faces = [Face(Wire.make_polygon([Vector(float(x), float(y), 0) for x, y in points], close=True))]
    if r > 0:
        if round_concave:
            faces = _offset(_offset(faces, r), -r)
        if round_convex:
            faces = _offset(_offset(faces, -r), r)
    if not faces:
        raise FeatureError("the outline is narrower than the cutter everywhere")
    return faces


def _union(faces: list[Face]):
    out = faces[0]
    for f in faces[1:]:
        out = out + f
    return out


def _prism(faces: list[Face], z0: float, z1: float) -> Shape:
    out = None
    for f in faces:
        solid = extrude(Pos(0, 0, z0) * f, amount=z1 - z0, dir=(0, 0, 1))  # a clockwise outline faces -Z
        out = solid if out is None else out + solid
    return out


class ProfilePlate(Feature):
    kind = "profile_plate"
    label = "Profile plate"
    group = "Base"
    summary = "Plate cut to a closed outline between two Z levels, inside corners rounded to a radius."
    params = (
        P("outline", "Outline", "profile", [], help="closed outline, [x, y] points in mm"),
        P("z_min", "Bottom Z", "length", 0.0, -1000.0, 1000.0),
        P("z_max", "Top Z", "length", 5.0, -1000.0, 1000.0),
        P("corner_radius", "Corner radius", "length", 3.0, 0.0, 100.0),
    )

    def build(self, ctx, p, refs):
        z0, z1 = p["z_min"], p["z_max"]
        if z1 - z0 < 0.1:
            raise FeatureError("top Z must be above bottom Z")
        plate = _prism(_profile_face(p["outline"], p["corner_radius"], round_convex=False), z0, z1)
        faces = plate.faces()
        bottom, top = _planar(faces, Vector(0, 0, -1)), _planar(faces, Vector(0, 0, 1))
        sides = [f for f in faces if f not in bottom and f not in top]
        return FeatureOutput("add", plate, named={"bottom": bottom, "top": top, "sides": sides},
                             info={"top_z": z1, "bottom_z": z0, "thickness": z1 - z0})

    def checks(self, ctx, fid, p, out):
        return [_wall_check(ctx, fid, "Thickness", out.info["thickness"])]


class ProfileCut(Feature):
    kind = "profile_cut"
    label = "Profile cut"
    group = "Material removal"
    summary = ("Pocket or through cut-out of any outline, from the top (+z) or bottom (-z) of a profile plate to a "
               "depth, walls straight, outside corners rounded to a radius.")
    params = (
        P("outline", "Outline", "profile", [], help="closed outline, [x, y] points in mm"),
        P("islands", "Islands", "profiles", [], help="outlines of material left standing inside the pocket"),
        P("side", "Cut from", "choice", "+z", choices=("+z", "-z")),
        P("through", "Through", "choice", "no", choices=("no", "yes")),
        P("depth", "Depth", "length", 2.0, 0.1, 1000.0),
        P("corner_radius", "Corner radius", "length", 1.0, 0.0, 100.0),
    )
    refs = (RefSpec("base", "Profile plate", ("profile_plate",), "feature"),)

    def build(self, ctx, p, refs):
        base, _ = ctx.feature_ref(refs, "base")
        top, bot = base.info["top_z"], base.info["bottom_z"]
        through = p["through"] == "yes"
        depth = top - bot if through else p["depth"]
        if not through and depth >= top - bot - 0.1:
            raise FeatureError("depth reaches the far face; make it a through cut")
        faces = _profile_face(p["outline"], p["corner_radius"], round_concave=False)
        if p["islands"]:
            islands = _union([f for poly in p["islands"] for f in _profile_face(poly, p["corner_radius"], round_convex=False)])
            faces = [g for f in faces for g in (f - islands).faces()]
        if p["side"] == "+z":
            tool, floor_z = _prism(faces, top - depth, top + 1.0), top - depth
        else:
            tool, floor_z = _prism(faces, bot - 1.0, bot + depth), bot + depth
        if through:
            tool = _prism(faces, bot - 1.0, top + 1.0)
        tf = tool.faces()
        floors = [] if through else _planar(tf, Vector(0, 0, -1 if p["side"] == "+z" else 1))
        caps = _planar(tf, Vector(0, 0, 1)) + _planar(tf, Vector(0, 0, -1))
        walls = [f for f in tf if f not in caps]
        return FeatureOutput("subtract", tool, named={"floor": floors, "walls": walls},
                             info={"depth": depth, "floor_z": floor_z, "through": float(through)})


# ---------------------------------------------------------------- sketch-based (Pad / Pocket, as in CATIA)


def _sketch_faces(ctx, refs) -> tuple[list[Face], "object"]:
    from ..sketch.geometry import SketchGeometryError, regions

    sk = ctx.sketch_ref(refs)
    try:
        return regions(sk), sk
    except SketchGeometryError as exc:
        raise FeatureError(str(exc)) from exc


class Pad(Feature):
    kind = "pad"
    label = "Pad"
    group = "Base"
    summary = "Extrude a sketch's closed profiles between two Z levels (nested profiles are holes)."
    params = (
        P("z_min", "Bottom Z", "length", 0.0, -1000.0, 1000.0),
        P("z_max", "Top Z", "length", 5.0, -1000.0, 1000.0),
    )
    refs = (RefSpec("sketch", "Sketch", (), "sketch"),)

    def build(self, ctx, p, refs):
        z0, z1 = p["z_min"], p["z_max"]
        if z1 - z0 < 0.1:
            raise FeatureError("top Z must be above bottom Z")
        faces, sk = _sketch_faces(ctx, refs)
        pad = _prism(faces, z0, z1)
        tf = pad.faces()
        bottom, top = _planar(tf, Vector(0, 0, -1)), _planar(tf, Vector(0, 0, 1))
        sides = [f for f in tf if f not in bottom and f not in top]
        return FeatureOutput("add", pad, named={"bottom": bottom, "top": top, "sides": sides},
                             driven={"sketch": sk.id}, info={"top_z": z1, "bottom_z": z0, "thickness": z1 - z0})

    def checks(self, ctx, fid, p, out):
        return [_wall_check(ctx, fid, "Thickness", out.info["thickness"])]


class SketchPocket(Feature):
    kind = "sketch_pocket"
    label = "Pocket"
    group = "Material removal"
    summary = ("Cut a sketch's closed profiles into a pad from the top (+z) or bottom (-z), to a depth or through; "
               "nested profiles inside are islands left standing.")
    params = (
        P("side", "Cut from", "choice", "+z", choices=("+z", "-z")),
        P("through", "Through", "choice", "no", choices=("no", "yes")),
        P("depth", "Depth", "length", 2.0, 0.1, 1000.0),
    )
    refs = (RefSpec("base", "Pad", ("pad",), "feature"), RefSpec("sketch", "Sketch", (), "sketch"))

    def build(self, ctx, p, refs):
        base, _ = ctx.feature_ref(refs, "base")
        top, bot = base.info["top_z"], base.info["bottom_z"]
        through = p["through"] == "yes"
        depth = top - bot if through else p["depth"]
        if not through and depth >= top - bot - 0.1:
            raise FeatureError("depth reaches the far face; make it a through cut")
        faces, sk = _sketch_faces(ctx, refs)
        if through:
            tool = _prism(faces, bot - 1.0, top + 1.0)
        elif p["side"] == "+z":
            tool = _prism(faces, top - depth, top + 1.0)
        else:
            tool = _prism(faces, bot - 1.0, bot + depth)
        tf = tool.faces()
        caps = _planar(tf, Vector(0, 0, 1)) + _planar(tf, Vector(0, 0, -1))
        floors = [] if through else _planar(tf, Vector(0, 0, -1 if p["side"] == "+z" else 1))
        return FeatureOutput("subtract", tool, named={"floor": floors, "walls": [f for f in tf if f not in caps]},
                             driven={"sketch": sk.id}, info={"depth": depth})


# ---------------------------------------------------------------- sketch-based solids


OPS = ("add", "cut", "intersect")
THROUGH = 5000.0  # mm, "through all"


def _op_output(op: str, tool: Shape, ctx, named: dict, **kw) -> FeatureOutput:
    if op == "add":
        return FeatureOutput("add", tool, named=named, **kw)
    if op == "cut":
        return FeatureOutput("subtract", tool, named=named, **kw)
    if ctx.part is None:
        raise FeatureError("nothing to intersect with yet")
    return FeatureOutput("replace", ctx.part & tool, named=named, **kw)


def _sections(face: Face, z0: float, z1: float, twist: float, end_scale: float) -> list[Face]:
    """Copies of a sketch face (local XY) from z0 to z1, turned and scaled progressively about the sketch origin."""
    n = max(2, math.ceil(abs(twist) / 10) + 1)
    out = []
    for k in range(n):
        u = k / (n - 1)
        f = face.scale(1 + (end_scale - 1) * u) if end_scale != 1 else face
        if twist:
            f = f.rotate(Axis.Z, twist * u)
        out.append(f.translate((0, 0, z0 + (z1 - z0) * u)))
    return out


def _loft(face: Face, z0: float, z1: float, twist: float, end_scale: float) -> Shape:
    secs = _sections(face, z0, z1, twist, end_scale)
    solid = Solid.make_loft([f.outer_wire() for f in secs])
    for i in range(len(face.inner_wires())):
        solid = solid - Solid.make_loft([f.inner_wires()[i] for f in secs])
    return solid


def _extrude_face(face: Face, z0: float, z1: float, twist: float, end_scale: float, draft: float) -> Shape:
    """One sketch face as a solid between z0 and z1 (local), twisted, scaled or drafted."""
    if draft and (twist or end_scale != 1):
        raise FeatureError("use either draft or twist / end scale, not both")
    if draft:
        return Solid.extrude_taper(face.translate((0, 0, z0)), Vector(0, 0, z1 - z0), draft)
    if end_scale != 1:
        return _loft(face, z0, z1, twist, end_scale)
    if twist:
        base = face.translate((0, 0, z0))
        return Solid.extrude_linear_with_rotation(base.outer_wire(), Vector(0, 0, z0), Vector(0, 0, z1 - z0), twist,
                                                  inner_wires=base.inner_wires())
    return extrude(face.translate((0, 0, z0)), amount=z1 - z0)


def _local_named(tool: Shape, z0: float, z1: float, plane) -> dict[str, list[Face]]:
    """start / end / side faces of a solid built along local Z from z0 to z1 (tool already in world)."""
    start, end, side = [], [], []
    for f in tool.faces():
        z = plane.to_local_coords(f.center()).Z
        if f.geom_type == GeomType.PLANE and abs(abs(f.normal_at().dot(plane.z_dir)) - 1) < 1e-6 and abs(z - z0) < 1e-6:
            start.append(f)
        elif f.geom_type == GeomType.PLANE and abs(abs(f.normal_at().dot(plane.z_dir)) - 1) < 1e-6 and abs(z - z1) < 1e-6:
            end.append(f)
        else:
            side.append(f)
    return {"start": start, "end": end, "side": side}


class Extrude(Feature):
    kind = "extrude"
    label = "Extrude"
    group = "Create"
    summary = ("Extrude a sketch's closed profiles along the sketch's normal: add, cut or intersect. Optional twist "
               "(degrees over the length), end scale (taper the end to this fraction, about the sketch origin) or draft "
               "angle. Faces are named start, end and side.")
    params = (
        P("op", "Operation", "choice", "add", choices=OPS),
        P("extent", "Extent", "choice", "distance", choices=("distance", "symmetric", "through_all")),
        P("distance", "Distance", "length", 10.0, 0.01, 2000.0),
        P("reverse", "Reverse", "bool", False),
        P("twist", "Twist", "angle", 0.0, -3600.0, 3600.0, help="degrees turned over the length"),
        P("end_scale", "End scale", "ratio", 1.0, 0.01, 100.0, help="size of the end relative to the sketch"),
        P("draft", "Draft", "angle", 0.0, -60.0, 60.0, help="degrees; positive narrows toward the end"),
    )
    refs = (RefSpec("sketch", "Sketch", (), "sketch"),)

    def build(self, ctx, p, refs):
        faces, sk = _sketch_faces(ctx, refs)
        plane = ctx.sketch_plane(sk)
        d = p["distance"]
        if p["extent"] == "symmetric":
            z0, z1 = -d / 2, d / 2
        elif p["extent"] == "through_all":
            z0, z1 = (-THROUGH, THROUGH) if p["op"] == "cut" else (0.0, THROUGH)
        else:
            z0, z1 = (-d, 0.0) if p["reverse"] else (0.0, d)
        solids = [_extrude_face(f, z0, z1, p["twist"], p["end_scale"], p["draft"]) for f in faces]
        local = solids[0]
        for s_ in solids[1:]:
            local = local + s_
        tool = plane.location * local
        named = _local_named(tool, z0, z1, plane)
        return _op_output(p["op"], tool, ctx, named, driven={"sketch": sk.id}, info={"length": z1 - z0})

    def checks(self, ctx, fid, p, out):
        if p["op"] != "add" or p["extent"] == "through_all":
            return []
        return [_wall_check(ctx, fid, "Length", out.info["length"])]


class Revolve(Feature):
    kind = "revolve"
    label = "Revolve"
    group = "Create"
    summary = ("Spin a sketch's closed profiles about the sketch's X or Y axis (the profile must not cross the axis): "
               "add, cut or intersect.")
    params = (
        P("op", "Operation", "choice", "add", choices=OPS),
        P("axis", "Axis", "choice", "Y", choices=("X", "Y")),
        P("angle", "Angle", "angle", 360.0, 0.1, 360.0),
    )
    refs = (RefSpec("sketch", "Sketch", (), "sketch"),)

    def build(self, ctx, p, refs):
        faces, sk = _sketch_faces(ctx, refs)
        plane = ctx.sketch_plane(sk)
        axis = Axis.X if p["axis"] == "X" else Axis.Y
        k = 1 if p["axis"] == "X" else 0  # the other coordinate must not change sign
        for f in faces:
            bb = f.bounding_box()
            lo, hi = (bb.min.Y, bb.max.Y) if k else (bb.min.X, bb.max.X)
            if lo < -1e-6 and hi > 1e-6:
                raise FeatureError(f"the profile crosses the sketch's {p['axis']} axis; keep it on one side")
        local = revolve(faces, axis=axis, revolution_arc=p["angle"])
        tool = plane.location * local
        named = {"side": list(tool.faces())}
        return _op_output(p["op"], tool, ctx, named, driven={"sketch": sk.id})


class Import(Feature):
    kind = "import"
    label = "Import"
    group = "Create"
    summary = ("An exact solid from a STEP file kept in the design's references folder (File > Import STEP puts it "
               "there): add, cut or intersect. Placed on the bed, centred, or where the file puts it; then scaled, "
               "turned about Z and moved. Meshes (STL, 3MF) are references, not imports.")
    params = (
        P("file", "File", "text", "", help="a STEP file in the design's references folder"),
        P("op", "Operation", "choice", "add", choices=OPS),
        P("place", "Placement", "choice", "on_bed", choices=("on_bed", "as_in_file"),
          help="on_bed: centred on the origin, lowest point on the bed"),
        P("scale", "Scale", "ratio", 1.0, 0.001, 1000.0),
        P("rotate_z", "Rotate Z", "angle", 0.0, -360.0, 360.0),
        P("x", "Move X", "length", 0.0, -2000.0, 2000.0),
        P("y", "Move Y", "length", 0.0, -2000.0, 2000.0),
        P("z", "Move Z", "length", 0.0, -2000.0, 2000.0),
    )

    def build(self, ctx, p, refs):
        from pathlib import Path

        from ..geometry.meshio import ImportError_, load_step

        name = p["file"].strip()
        if not name:
            raise FeatureError("no file: use File > Import STEP, or name a STEP file in the references folder")
        if ctx.references_dir is None or Path(name).name != name:
            raise FeatureError("the file must be in the design's references folder")
        try:
            shape = load_step(Path(ctx.references_dir) / name)
        except ImportError_ as exc:
            raise FeatureError(str(exc)) from exc
        except Exception as exc:  # a malformed file: say which, not a stack trace
            raise FeatureError(f"can't read {name}: {exc}") from exc
        if p["scale"] != 1.0:
            shape = shape.scale(p["scale"])
        if p["rotate_z"]:
            shape = shape.rotate(Axis.Z, p["rotate_z"])
        if p["place"] == "on_bed":
            bb = shape.bounding_box()
            shape = Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * shape
        tool = Pos(p["x"], p["y"], p["z"]) * shape
        return _op_output(p["op"], tool, ctx, {"surface": list(tool.faces())})


# ---------------------------------------------------------------- text


def _glyphs(p: dict, font_file) -> list[Face]:
    """The text as faces in the XY plane, laid out per the alignment and letter spacing, centred on the origin
    vertically and on the alignment point horizontally."""
    content, size, spacing = p["content"], p["size"], p["spacing"]
    halign = {"left": TextAlign.LEFT, "center": TextAlign.CENTER, "right": TextAlign.RIGHT}[p["align"]]
    if not spacing:
        t = Text(content, font_size=size, font_path=str(font_file), text_align=(halign, TextAlign.CENTER))
        faces = list(t.faces())
    else:  # letter by letter, each advanced by its own width plus the spacing
        faces, x = [], 0.0
        for ch in content:
            if ch.isspace():
                x += size * 0.3 + spacing
                continue
            t = Text(ch, font_size=size, font_path=str(font_file), text_align=(TextAlign.LEFT, TextAlign.CENTER))
            bb = t.bounding_box()
            faces += [f.translate((x - bb.min.X, 0, 0)) for f in t.faces()]
            x += bb.size.X + spacing
        width = x - spacing
        shift = {"left": 0.0, "center": -width / 2, "right": -width}[p["align"]]
        faces = [f.translate((shift, 0, 0)) for f in faces]
    if not faces:
        raise FeatureError("the text is empty")
    if p["rotation"]:
        faces = [f.rotate(Axis.Z, p["rotation"]) for f in faces]
    if p["offset_u"] or p["offset_v"]:
        faces = [f.translate((p["offset_u"], p["offset_v"], 0)) for f in faces]
    return faces


def _upright_plane(at: Vector, n: Vector) -> Plane:
    """A plane through `at` facing `n` whose X runs level, so text on it reads upright from outside the face."""
    up = Vector(0, 0, 1) - n * n.Z
    if up.length < 1e-6:  # a level face: text reads along X, up along Y
        up = Vector(0, 1, 0) - n * n.Y
    x = up.normalized().cross(n)
    return Plane(origin=at, x_dir=x, z_dir=n)


class TextFeature(Feature):
    kind = "text"
    label = "Text"
    group = "Create"
    summary = ("Text on a picked face (flat or curved: it wraps), raised (emboss), sunk (deboss), cut through, or inlaid "
               "(sunk, and the letters filled flush as their own body, for a second colour), in any bundled or design "
               "font. With no face picked it stands on the bed as its own solid. The content can come from a text "
               "parameter.")
    params = (
        P("content", "Text", "text", "TEXT"),
        P("font", "Font", "text", "Bebas Neue"),
        P("size", "Height", "length", 10.0, 0.5, 500.0, help="the font size in mm"),
        P("spacing", "Letter spacing", "length", 0.0, -20.0, 100.0),
        P("bold", "Bold", "bool", False),
        P("italic", "Italic", "bool", False),
        P("align", "Align", "choice", "center", choices=("left", "center", "right")),
        P("mode", "Type", "choice", "emboss", choices=("emboss", "deboss", "cut_through", "inlay")),
        P("depth", "Depth", "length", 0.6, 0.05, 50.0, help="how far it stands out or sinks in"),
        P("into", "Text body", "text", "", help="put the letters in this body, for a second colour (inlay needs one): "
          "raised letters stand on the face, inlaid ones fill a pocket flush with it"),
        P("rotation", "Rotation", "angle", 0.0, -360.0, 360.0),
        P("offset_u", "Move across", "length", 0.0, -2000.0, 2000.0),
        P("offset_v", "Move up", "length", 0.0, -2000.0, 2000.0),
    )

    def build(self, ctx, p, refs):
        from ..fonts import families

        fams = families(ctx.fonts_dir)
        fam = fams.get(p["font"])
        if fam is None:
            raise FeatureError(f"no font {p['font']!r}; fonts: {', '.join(sorted(fams))}")
        glyphs = _glyphs(p, fam.file(p["bold"], p["italic"]))
        mode, depth = p["mode"], p["depth"]
        into = p["into"]
        if into:
            if mode not in ("emboss", "inlay"):
                raise FeatureError("only raised or inlaid letters can go in their own body")
            ctx.other_body(into, "the letters")
        elif mode == "inlay":
            raise FeatureError("an inlay puts the letters in their own body; choose it (or a new body)")
        info = _text_info(p, glyphs)
        if not ctx.face_picks:
            if mode != "emboss":
                raise FeatureError("pick a face to sink, inlay or cut the text into")
            tool = extrude(glyphs, amount=depth) if len(glyphs) > 1 else extrude(glyphs[0], amount=depth)
            return self._output(ctx, mode, into, tool, tool, info)
        (face,) = ctx.picked_faces(ctx.face_picks[:1])
        pick = ctx.face_picks[0]
        at = face.closest_point(Vector(*pick.at)) if hasattr(face, "closest_point") else Vector(*pick.at)
        n = face.normal_at(at)
        plane = _upright_plane(at, n)
        far = 2 * max(face.bounding_box().diagonal, 10.0) if mode == "cut_through" else depth
        if face.geom_type == GeomType.PLANE:
            placed = [plane.location * g for g in glyphs]
            if mode == "emboss":
                letters = cutters = [extrude(f, amount=depth, dir=n) for f in placed]
            else:  # the cutter starts just outside the face so the boolean has no coincident faces
                cutters = [extrude(f.translate(n * 0.01), amount=far + 0.01, dir=-n) for f in placed]
                letters = [extrude(f, amount=depth, dir=-n) for f in placed] if mode == "inlay" else cutters
        else:
            try:
                wrapped = [face.wrap(g, Location(plane)) for g in glyphs]
                letters = cutters = [Solid.thicken(w, depth if mode == "emboss" else -far) for w in wrapped]
            except Exception as exc:  # noqa: BLE001 - the kernel's wrap fails on some surfaces
                raise FeatureError(f"could not wrap the text onto that face ({exc}); try a flatter face or a smaller size") from exc
        return self._output(ctx, mode, into, _fuse(letters), _fuse(cutters), info)

    @staticmethod
    def _output(ctx, mode: str, into: str, letters: Shape, cutter: Shape, info: dict) -> FeatureOutput:
        named = {"text": list(letters.faces())}
        if not into:
            return FeatureOutput("add" if mode == "emboss" else "subtract", cutter, named=named, info=info, new_role="text")
        held = ctx.parts.get(into)
        bodies = {into: letters if held is None else held + letters}
        if mode == "inlay":
            return FeatureOutput("subtract", cutter, named=named, info=info, bodies=bodies, new_role="text")
        return FeatureOutput("none", None, named=named, info=info, bodies=bodies, new_role="text")

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        i = out.info
        thin = i["thin_glyphs"]
        return [
            Check(fid, "Letter height", i["height"], _status(i["height"], r.min_text_height, r.min_text_height * 0.6),
                  f"≥ {r.min_text_height:g} mm reads well"),
            Check(fid, "Stroke width", r.min_text_stroke if not thin else 0.0, "ok" if not thin else "warn",
                  f"strokes ≥ {r.min_text_stroke:g} mm (two nozzle widths)" + (f"; {int(thin)} letter(s) thinner" if thin else "")),
        ]


def _text_info(p: dict, glyphs: list[Face]) -> dict[str, float]:
    """Letter height and how many glyphs have a stroke thinner than the minimum (an inward offset of half the minimum
    stroke leaves almost nothing of them)."""
    ys = [g.bounding_box() for g in glyphs]
    height = max(b.max.Y for b in ys) - min(b.min.Y for b in ys)
    thin = 0
    for g in glyphs:
        try:
            inner = _offset([Face(g.outer_wire())], -0.4)
            hole_free = sum(f.area for f in inner)
            if hole_free < 0.02 * Face(g.outer_wire()).area:
                thin += 1
        except FeatureError:
            thin += 1
    return {"height": height, "thin_glyphs": float(thin), "glyphs": float(len(glyphs))}


def _fuse(shapes: list[Shape]) -> Shape:
    out = shapes[0]
    for s_ in shapes[1:]:
        out = out + s_
    return out


# ---------------------------------------------------------------- bodies


class Split(Feature):
    kind = "split"
    label = "Split"
    group = "Bodies"
    summary = ("Cut the body in two along a plane (a base plane moved by an offset, or a picked flat face): one side stays, "
               "the other goes into another body, to print in another colour (a vase's rim band: split at the band's "
               "height). Split again for a band in the middle. The cut faces are named cut.")
    params = (
        P("plane", "Plane", "choice", "XY", choices=("XY", "XZ", "YZ"), help="ignored when a face is picked"),
        P("offset", "Offset", "length", 10.0, -2000.0, 2000.0, help="along the plane's normal (Z for XY)"),
        P("keep", "Keep", "choice", "below", choices=("below", "above"),
          help="the side that stays in this body; above is the side the normal points to"),
        P("into", "Other side to", "text", "new", help="the body the other side goes into"),
    )

    def build(self, ctx, p, refs):
        if ctx.part is None:
            raise FeatureError("nothing to split yet")
        into = ctx.other_body(p["into"], "the other side")
        if ctx.face_picks:
            (face,) = ctx.picked_faces(ctx.face_picks[:1])
            if face.geom_type != GeomType.PLANE:
                raise FeatureError("split along a flat face")
            n = face.normal_at()
            plane = Plane(origin=face.center(), z_dir=n).offset(p["offset"])
        else:
            plane = BASE_PLANES[p["plane"]].offset(p["offset"])
        below = ctx.part.split(plane, keep=Keep.BOTTOM)
        above = ctx.part.split(plane, keep=Keep.TOP)
        stay, go = (below, above) if p["keep"] == "below" else (above, below)
        vol = ctx.part.volume
        if stay is None or go is None or min(stay.volume, go.volume) < 1e-6 * max(vol, 1.0):
            raise FeatureError("the plane misses the body; move it so it cuts through")
        held = ctx.parts.get(into)
        return FeatureOutput("replace", stay, bodies={into: go if held is None else held + go}, new_role="cut",
                             info={"stay": stay.volume, "go": go.volume})


class Combine(Feature):
    kind = "combine"
    label = "Combine"
    group = "Bodies"
    summary = ("Join another body into this one, cut it out of this one, or keep only where they overlap. The other body "
               "is used up unless kept.")
    params = (
        P("op", "Operation", "choice", "add", choices=OPS),
        P("tool", "With body", "text", "", help="the body to join, cut or intersect with"),
        P("keep_tool", "Keep that body", "bool", False),
    )

    def build(self, ctx, p, refs):
        if ctx.part is None:
            raise FeatureError("this body has nothing in it yet")
        tool = ctx.other_body(p["tool"], "the other body")
        other = ctx.parts.get(tool)
        if other is None:
            raise FeatureError(f"{tool} has nothing in it at this point of the tree")
        op = p["op"]
        result = ctx.part + other if op == "add" else ctx.part - other if op == "cut" else ctx.part & other
        if result is None or result.volume < 1e-9:
            raise FeatureError("nothing is left; the bodies do not overlap" if op == "intersect" else "nothing is left")
        return FeatureOutput("replace", result, bodies={} if p["keep_tool"] else {tool: None}, new_role="combined")


# ---------------------------------------------------------------- modify


class Shell(Feature):
    kind = "shell"
    label = "Shell"
    group = "Modify"
    summary = ("Hollow the body to a wall thickness, leaving the picked faces open (pick none for a closed hollow "
               "part).")
    params = (
        P("thickness", "Wall", "length", 1.6, 0.1, 100.0),
        P("direction", "Direction", "choice", "inside", choices=("inside", "outside")),
    )

    def build(self, ctx, p, refs):
        if ctx.part is None:
            raise FeatureError("nothing to shell yet")
        solids = ctx.part.solids()
        if len(solids) != 1:
            raise FeatureError("the body must be one connected solid to shell it")
        openings = ctx.picked_faces() if ctx.face_picks else []
        t = p["thickness"] if p["direction"] == "outside" else -p["thickness"]
        try:
            if openings:
                shelled = offset(solids[0], t, openings=openings, kind=Kind.INTERSECTION)
            elif t < 0:  # closed: the solid less its inward offset
                shelled = solids[0] - offset(solids[0], t, kind=Kind.INTERSECTION)
            else:
                shelled = offset(solids[0], t, kind=Kind.INTERSECTION) - solids[0]
        except Exception as exc:  # noqa: BLE001 - the kernel's offset fails on some shapes
            raise FeatureError(f"shell of {p['thickness']:g} mm failed; try a thinner wall ({exc})") from exc
        if not shelled.is_valid or shelled.volume <= 0:
            raise FeatureError(f"shell of {p['thickness']:g} mm did not make a valid solid; try a thinner wall")
        return FeatureOutput("replace", shelled, info={"thickness": p["thickness"], "openings": float(len(openings))},
                             new_role="inner")

    def checks(self, ctx, fid, p, out):
        return [_wall_check(ctx, fid, "Wall", p["thickness"])]


# ---------------------------------------------------------------- edge treatment


class _EdgeOp(Feature):
    """Blends the edges the engineer picked (kept by geometry, see EdgePick), or else every edge where two groups of
    named faces meet."""

    group = "Edge treatment"
    refs = (
        RefSpec("faces_a", "Faces A (feature or feature.role)", (), "feature"),
        RefSpec("faces_b", "Faces B (feature or feature.role)", (), "feature"),
    )

    def _edges(self, ctx, refs):
        from .design import attribute, edges_between, find_picked_edges

        if ctx.part is None:
            raise FeatureError("nothing to modify yet")
        names = attribute(ctx.part, ctx.outputs)
        if ctx.edge_picks:
            edges = find_picked_edges(ctx.part, names, ctx.edge_picks)
        elif not (refs.get("faces_a") and refs.get("faces_b")):
            raise FeatureError("pick the edges to blend on the part (or choose two face groups)")
        else:
            edges = edges_between(ctx.part, names, refs.get("faces_a", ""), refs.get("faces_b", ""))
        if not edges:
            raise FeatureError(f"no edges where {refs.get('faces_a')} meets {refs.get('faces_b')}")
        solids = ctx.part.solids()
        if len(solids) != 1:
            raise FeatureError("the part must be a single connected solid before blending")
        return solids[0], edges


class Fillet(_EdgeOp):
    kind = "fillet"
    label = "Fillet / blend"
    summary = "Constant-radius blend on picked edges, or on the edges where two groups of named faces meet."
    params = (P("radius", "Radius", "length", 1.0, 0.1, 100.0),)

    def build(self, ctx, p, refs):
        solid, edges = self._edges(ctx, refs)
        try:
            return FeatureOutput("replace", solid.fillet(p["radius"], edges), info={"edges": float(len(edges))})
        except Exception as exc:
            raise FeatureError(f"fillet of {p['radius']:g} mm failed on {len(edges)} edges; try a smaller radius") from exc


class Chamfer(_EdgeOp):
    kind = "chamfer"
    label = "Chamfer"
    summary = "45° chamfer on picked edges, or on the edges where two groups of named faces meet."
    params = (P("distance", "Distance", "length", 0.5, 0.1, 50.0),)

    def build(self, ctx, p, refs):
        solid, edges = self._edges(ctx, refs)
        try:
            return FeatureOutput("replace", solid.chamfer(p["distance"], None, edges), info={"edges": float(len(edges))})
        except Exception as exc:
            raise FeatureError(f"chamfer of {p['distance']:g} mm failed on {len(edges)} edges; try a smaller size") from exc


FEATURES: dict[str, Feature] = {
    cls.kind: cls()
    for cls in (Extrude, Revolve, Import, TextFeature, Shell, Fillet, Chamfer, Split, Combine, Plate, ProfilePlate, ProfileCut, Pad,
                SketchPocket)
}


def resolve_params(kind: str, given: dict, strict: set[str] | frozenset[str] = frozenset()) -> dict:
    """Defaults merged with given values, clamped to the declared bounds. A value in `strict` (one a formula set) is
    not clamped: out of bounds it raises, so the user sees the formula is wrong instead of a quietly different part."""
    out = {}
    for s in FEATURES[kind].params:
        v = given.get(s.name, s.default)
        if s.quantity in ("length", "ratio", "angle", "count"):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float(s.default)
            lo, hi = s.min, s.max
            if s.name in strict and ((lo is not None and v < lo - 1e-9) or (hi is not None and v > hi + 1e-9)):
                bound = f"at least {lo:g}" if lo is not None and v < lo else f"at most {hi:g}"
                raise FeatureError(f"{s.label} = {v:g} from its formula; it must be {bound}")
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            if s.quantity == "count":
                v = int(round(v))
        elif s.quantity == "choice" and v not in s.choices:
            v = s.default
        elif s.quantity == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        elif s.quantity == "text":
            v = str(v)
        elif s.quantity == "profile":
            try:
                v = [[float(x), float(y)] for x, y in v]
            except (TypeError, ValueError):
                v = list(s.default)
        elif s.quantity == "profiles":
            try:
                v = [[[float(x), float(y)] for x, y in poly] for poly in v]
            except (TypeError, ValueError):
                v = list(s.default)
        out[s.name] = v
    return out
