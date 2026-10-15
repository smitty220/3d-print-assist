"""The parametric feature library.

Each feature declares its parameters and references, builds a tool solid in the design frame (Z = 0 is
the mount face) and reports rule checks. The AI composes these same features; one definition drives the
Properties panel, the AI tool description and the validator.
"""

from __future__ import annotations

from typing import ClassVar

from build123d import (
    Align,
    Axis,
    Box,
    BuildSketch,
    Circle,
    CenterOf,
    Cylinder,
    Face,
    GeomType,
    Kind,
    Locations,
    Mode,
    Plane,
    Polygon,
    Pos,
    Shape,
    Vector,
    Wire,
    extrude,
    offset,
)

from .context import BuildContext, FeatureError
from .model import Check, FeatureOutput, ParamSpec, RefSpec

IN = 25.4
P = ParamSpec
MIN_Z = (Align.CENTER, Align.CENTER, Align.MIN)


def _status(value: float, good: float, limit: float, lower_is_better: bool = False):
    if lower_is_better:
        return "ok" if value <= good else "warn" if value <= limit else "fail"
    return "ok" if value >= good else "warn" if value >= limit else "fail"


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
    summary = "Rectangular base plate on the mount face, rounded vertical corners."
    params = (
        P("length", "Length X", "length", 2.8 * IN, 12.0, 400.0),
        P("width", "Width Y", "length", 2.8 * IN, 12.0, 400.0),
        P("thickness", "Thickness", "length", 0.25 * IN, 1.0, 50.0),
        P("corner_radius", "Corner radius", "length", 0.3 * IN, 0.0, 100.0),
        P("center_x", "Center X", "length", 0.0, -400.0, 400.0),
        P("center_y", "Center Y", "length", 0.0, -400.0, 400.0),
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
        r = ctx.rules
        t = p["thickness"]
        return [Check(fid, "Thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in")]


class Wall(Feature):
    kind = "wall"
    label = "Wall"
    group = "Base"
    summary = "Upright plate along one edge of a base plate (the second leg of an angle bracket)."
    params = (
        P("side", "Edge", "choice", "+x", choices=("+x", "-x", "+y", "-y")),
        P("height", "Height", "length", 2.5 * IN, 5.0, 400.0),
        P("thickness", "Thickness", "length", 0.25 * IN, 1.0, 50.0),
    )
    refs = (RefSpec("base", "Base plate", ("plate",), "feature"),)

    def build(self, ctx, p, refs):
        _, bp = ctx.feature_ref(refs, "base")
        h, t, side = p["height"], p["thickness"], p["side"]
        cx, cy, L, W = bp["center_x"], bp["center_y"], bp["length"], bp["width"]
        sign = 1 if side[0] == "+" else -1
        if side[1] == "x":
            box = Pos(cx + sign * (L / 2 - t / 2), cy, 0) * Box(t, W, h, align=MIN_Z)
            n = Vector(sign, 0, 0)
        else:
            box = Pos(cx, cy + sign * (W / 2 - t / 2), 0) * Box(L, t, h, align=MIN_Z)
            n = Vector(0, sign, 0)
        faces = box.faces()
        outer, inner = _planar(faces, n), _planar(faces, -n)
        top = _planar(faces, Vector(0, 0, 1))
        bottom = _planar(faces, Vector(0, 0, -1))
        ends = [f for f in faces if f not in outer + inner + top + bottom]
        return FeatureOutput("add", box, named={"outer": outer, "inner": inner, "top": top, "sides": top + ends}, info={"thickness": t})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        t, h = p["thickness"], p["height"]
        return [
            Check(fid, "Thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in"),
            Check(fid, "Height / thickness", h / t, _status(h / t, 0.8 * r.wall_height_thickness_max, r.wall_height_thickness_max, True),
                  f"h/t ≤ {r.wall_height_thickness_max:g} to machine"),
        ]


# ---------------------------------------------------------------- attachments


class BoltHoles(Feature):
    kind = "bolt_holes"
    label = "Bolt holes"
    group = "Attachments"
    summary = "Through holes matching a bolt-pattern interface, sized from its fastener, cut through a plate or wall."
    refs = (
        RefSpec("pattern", "Bolt pattern", ("bolt_pattern",), "intent"),
        RefSpec("target", "Plate or wall", ("plate", "wall"), "feature"),
    )

    def build(self, ctx, p, refs):
        from ..catalog import FASTENERS

        bp = ctx.intent_ref(refs, "pattern", ("bolt_pattern",))
        target, _ = ctx.feature_ref(refs, "target")
        depth = target.info["thickness"]
        fastener = FASTENERS.get(bp.fastener or "")
        holes = ctx.bolt_holes_local(bp)
        hole_d = fastener.hole_close * IN if fastener else holes[0][2]
        tool = None
        for centre, axis, _d in holes:
            # Start just behind the mating face and run through the target's thickness along the bolt axis.
            cyl = Plane(origin=centre - axis * 1.0, z_dir=axis).location * Cylinder(hole_d / 2, depth + 2.0, align=MIN_Z)
            tool = cyl if tool is None else tool + cyl
        edge = min(min(side.distance_to(c) for side in target.named["sides"]) for c, _a, _d in holes)
        return FeatureOutput(
            "subtract", tool,
            named={"holes": [f for f in tool.faces() if f.geom_type == GeomType.CYLINDER]},
            driven={"pattern": bp.id, "fastener": bp.fastener or "none", "count": float(len(holes)), "hole_d": round(hole_d, 4)},
            info={"edge": edge, "d": fastener.shank_d * IN if fastener else hole_d},
        )

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        ed = out.info["edge"] / out.info["d"]
        return [Check(fid, "Edge distance e/D", ed, _status(ed, r.bolt_edge_preferred, r.bolt_edge_min),
                      f"{r.bolt_edge_preferred:g}D preferred, {r.bolt_edge_min:g}D min")]


def _ear(ctx: BuildContext, centre: Vector, axis: Vector, z_base: float, offset: float, t: float, R: float, b: float, pin_d: float):
    """One lug ear: trapezoid from the base up to a crown of radius R around the pin, bored for the pin.

    Built on a plane whose normal is the pin axis; its sketch X runs horizontally across the ear and
    sketch Y is the design-frame Z, so (s, t) coordinates read as (across, height).
    """
    across = Vector(0, 0, 1).cross(axis).normalized()
    origin = Vector(centre.X, centre.Y, 0) + axis * offset
    plane = Plane(origin=origin, x_dir=across, z_dir=axis)
    zp = centre.Z
    if zp - z_base < R * 0.5:
        raise FeatureError("pin centre is too close to the base for this crown radius")
    with BuildSketch(plane) as sk:
        Polygon((-b, z_base), (b, z_base), (R, zp), (-R, zp), align=None)
        with Locations((0, zp)):
            Circle(R)
        with Locations((0, zp)):
            Circle(pin_d / 2, mode=Mode.SUBTRACT)
    return extrude(sk.sketch, amount=t)


def _ear_named(ear: Shape, centre: Vector, axis: Vector, pin_d: float) -> dict[str, list[Face]]:
    faces = ear.faces()
    planars = [f for f in faces if f.geom_type == GeomType.PLANE and abs(f.normal_at().dot(axis)) > 0.999]
    def side(f):  # sign of the face's offset from the pin centre along the axis
        return (f.center(CenterOf.MASS) - centre).dot(axis)
    # The face nearer the pin centre plane is the inner face.
    planars.sort(key=lambda f: abs(side(f)))
    cyl = [f for f in faces if f.geom_type == GeomType.CYLINDER]
    bore = [f for f in cyl if abs(f.radius - pin_d / 2) < 1e-3]
    crown = [f for f in cyl if f not in bore]
    others = [f for f in faces if f not in planars + cyl]
    return {"inner": planars[:1], "outer": planars[1:], "bore": bore, "crown": crown, "flanks": others}


class _LugBase(Feature):
    group = "Attachments"
    refs = (
        RefSpec("pin", "Pin bore", ("bore",), "intent"),
        RefSpec("base", "Base plate", ("plate",), "feature"),
    )

    def _pin(self, ctx, refs):
        b = ctx.intent_ref(refs, "pin", ("bore",))
        centre, axis = ctx.bore_local(b)
        if abs(axis.Z) > 0.02:
            raise FeatureError("the pin axis must be parallel to the mount face")
        base, _ = ctx.feature_ref(refs, "base")
        return b, centre, axis, base.info["top_z"]

    def _lug_checks(self, ctx, fid, out, t):
        r = ctx.rules
        D = out.info["pin_d"]
        ed, td = out.info["crown_r"] / D, t / D
        return [
            Check(fid, "Edge margin e/D", ed, _status(ed, r.lug_edge_full_bearing, r.lug_edge_min),
                  f"≥ {r.lug_edge_full_bearing:g} for full bearing, {r.lug_edge_min:g} min"),
            Check(fid, "Thickness t/D", td, _status(td, max(r.lug_td_min, 0.3), r.lug_td_min), f"≥ {r.lug_td_min:g} (D/t ≤ 5)"),
            Check(fid, "Ear thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in"),
        ]


class Clevis(_LugBase):
    kind = "clevis"
    label = "Clevis"
    summary = "Double lug straddling a male rod end; pin size, position and gap come from the bore interface."
    params = (
        P("ear_thickness", "Ear thickness", "length", 0.25 * IN, 1.0, 50.0),
        P("crown_radius", "Crown radius (e)", "length", 0.6 * IN, 2.0, 100.0),
        P("base_half_width", "Base half-width", "length", 0.6 * IN, 2.0, 150.0),
        P("gap_clearance", "Gap clearance (total)", "length", 0.025 * IN, 0.0, 5.0, help="Added to the rod-end width"),
    )

    def build(self, ctx, p, refs):
        b, c, a, zb = self._pin(ctx, refs)
        t, R, half, clr = p["ear_thickness"], p["crown_radius"], p["base_half_width"], p["gap_clearance"]
        gap = b.length + clr
        if R <= b.diameter / 2:
            raise FeatureError("crown radius must exceed the pin radius")
        ears = [_ear(ctx, c, a, zb, gap / 2, t, R, half, b.diameter), _ear(ctx, c, a, zb, -gap / 2 - t, t, R, half, b.diameter)]
        named: dict[str, list[Face]] = {}
        for e in ears:
            for role, fs in _ear_named(e, c, a, b.diameter).items():
                named.setdefault(role, []).extend(fs)
        return FeatureOutput(
            "add", ears[0] + ears[1], named=named,
            driven={"pin": b.id, "pin_d": round(b.diameter, 4), "gap": round(gap, 4)},
            info={"pin_d": b.diameter, "crown_r": R, "outer_offset": gap / 2 + t, "z_base": zb,
                  "cx": c.X, "cy": c.Y, "cz": c.Z, "ax": a.X, "ay": a.Y},
        )

    def checks(self, ctx, fid, p, out):
        return self._lug_checks(ctx, fid, out, p["ear_thickness"])


class Lug(_LugBase):
    kind = "lug"
    label = "Lug (single)"
    summary = "Single lug centred on the bore interface, for a forked (clevis) mating end."
    params = (
        P("thickness", "Thickness", "length", 0.3 * IN, 1.0, 60.0),
        P("crown_radius", "Crown radius (e)", "length", 0.6 * IN, 2.0, 100.0),
        P("base_half_width", "Base half-width", "length", 0.65 * IN, 2.0, 150.0),
    )

    def build(self, ctx, p, refs):
        b, c, a, zb = self._pin(ctx, refs)
        t, R, half = p["thickness"], p["crown_radius"], p["base_half_width"]
        if b.gap is None:
            raise FeatureError("a single lug needs a forked mating end; pick a bore through one tine of a clevis")
        if t >= b.gap:
            raise FeatureError(f"lug thickness must be less than the clevis gap ({b.gap / IN:.3f} in)")
        if R <= b.diameter / 2:
            raise FeatureError("crown radius must exceed the pin radius")
        ear = _ear(ctx, c, a, zb, -t / 2, t, R, half, b.diameter)
        named = _ear_named(ear, c, a, b.diameter)
        named["faces"] = named.pop("inner") + named.pop("outer")
        return FeatureOutput(
            "add", ear, named=named,
            driven={"pin": b.id, "pin_d": round(b.diameter, 4), "gap": round(b.gap, 4), "side_clearance": round((b.gap - t) / 2, 4)},
            info={"pin_d": b.diameter, "crown_r": R, "outer_offset": t / 2, "z_base": zb,
                  "cx": c.X, "cy": c.Y, "cz": c.Z, "ax": a.X, "ay": a.Y},
        )

    def checks(self, ctx, fid, p, out):
        return self._lug_checks(ctx, fid, out, p["thickness"])


# ---------------------------------------------------------------- stiffening


class GussetPair(Feature):
    kind = "gusset_pair"
    label = "Gusset pair"
    group = "Stiffening"
    summary = "Triangular ribs on the outer faces of a lug or clevis, tying it to the base."
    params = (
        P("height", "Height above mount face", "length", 1.6 * IN, 5.0, 300.0),
        P("length", "Length", "length", 0.68 * IN, 2.0, 200.0),
        P("thickness", "Thickness", "length", 0.2 * IN, 1.0, 50.0),
    )
    refs = (RefSpec("lug", "Lug or clevis", ("clevis", "lug"), "feature"),)

    def build(self, ctx, p, refs):
        lug, _ = ctx.feature_ref(refs, "lug")
        i = lug.info
        a = Vector(i["ax"], i["ay"], 0)
        across = Vector(0, 0, 1).cross(a).normalized()
        plane = Plane(origin=Vector(i["cx"], i["cy"], 0), x_dir=a, z_dir=-across)
        xo, zb, h, L, t = i["outer_offset"], i["z_base"], p["height"], p["length"], p["thickness"]
        if h <= zb + 1.0:
            raise FeatureError("gusset height must be above the base top")
        tools = []
        for s in (1, -1):
            with BuildSketch(plane) as sk:
                Polygon((s * xo, zb), (s * (xo + L), zb), (s * xo, h), align=None)
            tools.append(extrude(sk.sketch, amount=t / 2, both=True))
        tool = tools[0] + tools[1]
        return FeatureOutput("add", tool, named={"faces": list(tool.faces())}, info={"z_base": zb})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        t = p["thickness"]
        ht = (p["height"] - out.info["z_base"]) / t
        return [
            Check(fid, "Thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in"),
            Check(fid, "Height / thickness", ht, _status(ht, 0.8 * r.wall_height_thickness_max, r.wall_height_thickness_max, True),
                  f"h/t ≤ {r.wall_height_thickness_max:g} to machine"),
        ]


# ---------------------------------------------------------------- material removal


class Pocket(Feature):
    kind = "pocket"
    label = "Pocket"
    group = "Material removal"
    summary = "Rectangular pocket in the top of a plate, optionally mirrored, with tool-sized corner radii."
    params = (
        P("center_x", "Center X", "length", 0.0, -400.0, 400.0),
        P("center_y", "Center Y", "length", 0.9 * IN, -400.0, 400.0),
        P("length_x", "Length X", "length", 1.0 * IN, 2.0, 400.0),
        P("width_y", "Width Y", "length", 0.45 * IN, 2.0, 400.0),
        P("floor", "Floor thickness", "length", 0.08 * IN, 0.5, 50.0),
        P("corner_radius", "Corner radius", "length", 0.12 * IN, 0.5, 50.0),
        P("mirror", "Mirror", "choice", "y", choices=("none", "x", "y")),
    )
    refs = (RefSpec("base", "Plate", ("plate",), "feature"),)

    def build(self, ctx, p, refs):
        base, _ = ctx.feature_ref(refs, "base")
        top = base.info["top_z"]
        fl, r = p["floor"], p["corner_radius"]
        if fl >= top:
            raise FeatureError("floor thickness must be less than the plate thickness")
        lx, wy = p["length_x"], p["width_y"]
        if r >= min(lx, wy) / 2:
            raise FeatureError("corner radius must be less than half the pocket width")
        centres = [(p["center_x"], p["center_y"])]
        if p["mirror"] == "x":
            centres.append((-p["center_x"], p["center_y"]))
        elif p["mirror"] == "y":
            centres.append((p["center_x"], -p["center_y"]))
        tool = None
        for cx, cy in centres:
            box = Pos(cx, cy, fl) * Box(lx, wy, top - fl + 1.0, align=MIN_Z)
            box = box.solid().fillet(r, box.edges().filter_by(Axis.Z))
            tool = box if tool is None else tool + box
        depth = top - fl
        tool_d = max(2 * r - 2 * 0.254, 0.5)
        floors = _planar(tool.faces(), Vector(0, 0, -1))
        walls = [f for f in tool.faces() if f not in floors and not _planar([f], Vector(0, 0, 1))]
        return FeatureOutput("subtract", tool, named={"floor": floors, "walls": walls},
                             info={"depth": depth, "tool_d": tool_d, "floor": fl, "r": r})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        i = out.info
        dt = i["depth"] / i["tool_d"]
        return [
            Check(fid, "Floor thickness", i["floor"], _status(i["floor"], r.min_wall_preferred, r.min_floor), f"floor ≥ {r.min_floor / IN:.3f} in"),
            Check(fid, "Depth / tool Ø", dt, _status(dt, r.pocket_depth_tool_max - 1, r.pocket_depth_tool_max, True),
                  f"≤ {r.pocket_depth_tool_max:g}× tool Ø (tool = 2R − 0.020)"),
            Check(fid, "Corner R / depth", i["r"] / i["depth"], "ok" if i["r"] >= i["depth"] / 3 else "warn", "R ≥ depth / 3"),
        ]


# ---------------------------------------------------------------- 2.5D profiles (machined plates)


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
    summary = ("Plate cut to a closed outline (a machined profile), between two Z levels of the design frame, inside "
               "corners rounded to at least the cutter radius.")
    params = (
        P("outline", "Outline", "profile", [], help="closed outline, [x, y] points in mm, design frame"),
        P("z_min", "Bottom Z", "length", -0.25 * IN, -400.0, 400.0),
        P("z_max", "Top Z", "length", 0.25 * IN, -400.0, 400.0),
        P("corner_radius", "Corner radius", "length", 0.125 * IN, 0.0, 50.0),
    )

    def build(self, ctx, p, refs):
        z0, z1 = p["z_min"], p["z_max"]
        if z1 - z0 < 0.1:
            raise FeatureError("top Z must be above bottom Z")
        # The cutter limits only the inside corners of an outside profile; outside corners can stay sharp.
        plate = _prism(_profile_face(p["outline"], p["corner_radius"], round_convex=False), z0, z1)
        faces = plate.faces()
        bottom, top = _planar(faces, Vector(0, 0, -1)), _planar(faces, Vector(0, 0, 1))
        sides = [f for f in faces if f not in bottom and f not in top]
        return FeatureOutput("add", plate, named={"bottom": bottom, "top": top, "sides": sides},
                             info={"top_z": z1, "bottom_z": z0, "thickness": z1 - z0})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        t = out.info["thickness"]
        return [Check(fid, "Thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in")]


class ProfileCut(Feature):
    kind = "profile_cut"
    label = "Profile cut"
    group = "Material removal"
    summary = ("Pocket or through cut-out of any outline, milled from the top (+z) or bottom (-z) of a profile plate "
               "to a depth, walls straight, corners at least the cutter radius.")
    params = (
        P("outline", "Outline", "profile", [], help="closed outline, [x, y] points in mm, design frame"),
        P("islands", "Islands", "profiles", [], help="outlines of material left standing inside the pocket"),
        P("side", "Cut from", "choice", "+z", choices=("+z", "-z")),
        P("through", "Through", "choice", "no", choices=("no", "yes")),
        P("depth", "Depth", "length", 0.1 * IN, 0.1, 400.0),
        P("corner_radius", "Corner radius", "length", 0.125 * IN, 0.5, 50.0),
    )
    refs = (RefSpec("base", "Profile plate", ("profile_plate",), "feature"),)

    def build(self, ctx, p, refs):
        base, _ = ctx.feature_ref(refs, "base")
        top, bot = base.info["top_z"], base.info["bottom_z"]
        through = p["through"] == "yes"
        depth = top - bot if through else p["depth"]
        if not through and depth >= top - bot - 0.1:
            raise FeatureError("depth reaches the far face; make it a through cut")
        # Only the cutter's own corners need rounding: the pocket's outside corners.
        faces = _profile_face(p["outline"], p["corner_radius"], round_concave=False)
        if p["islands"]:
            # Material left standing: its inside corners take the cutter radius, its outside corners can stay sharp.
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
        tool_d = max(2 * p["corner_radius"] - 2 * 0.254, 0.5)
        return FeatureOutput("subtract", tool, named={"floor": floors, "walls": walls},
                             info={"depth": depth, "tool_d": tool_d, "floor_z": floor_z, "through": float(through)})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        i = out.info
        dt = i["depth"] / i["tool_d"]
        return [Check(fid, "Depth / tool Ø", dt, _status(dt, r.pocket_depth_tool_max - 1, r.pocket_depth_tool_max, True),
                      f"≤ {r.pocket_depth_tool_max:g}× tool Ø (tool = 2R − 0.020)")]


class PinBore(Feature):
    kind = "pin_bore"
    label = "Pin bore"
    group = "Attachments"
    summary = "Bore for a pin interface, sized and placed from the intent, drilled right through the part."
    refs = (RefSpec("pin", "Pin bore", ("bore",), "intent"), RefSpec("base", "Plate", ("profile_plate", "pad"), "feature"))

    def build(self, ctx, p, refs):
        b = ctx.intent_ref(refs, "pin", ("bore",))
        base, _ = ctx.feature_ref(refs, "base")
        centre, axis = ctx.bore_local(b)
        span = 2 * (abs(base.info["top_z"]) + abs(base.info["bottom_z"])) + 10.0
        tool = Plane(origin=centre - axis * span / 2, z_dir=axis).location * Cylinder(b.diameter / 2, span, align=MIN_Z)
        sides = base.named["sides"]
        edge = min(side.distance_to(centre) for side in sides) if sides else 0.0
        return FeatureOutput("subtract", tool, named={"bore": [f for f in tool.faces() if f.geom_type == GeomType.CYLINDER]},
                             driven={"pin": b.id, "diameter": round(b.diameter, 4)}, info={"edge": edge, "d": b.diameter})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        ed = out.info["edge"] / out.info["d"]
        return [Check(fid, "Lug edge e/D", ed, _status(ed, r.lug_edge_full_bearing, r.lug_edge_min),
                      f"{r.lug_edge_full_bearing:g}D full bearing, {r.lug_edge_min:g}D min")]


# ---------------------------------------------------------------- sketch-based (Pad / Pocket, as in CATIA)


def _sketch_faces(ctx, refs) -> tuple[list[Face], "object"]:
    from ..sketch.geometry import SketchGeometryError, regions

    sk = ctx.sketch_ref(refs)
    try:
        return regions(sk), sk
    except SketchGeometryError as exc:
        raise FeatureError(str(exc)) from exc


def _min_radius(sk) -> float | None:
    from ..sketch.model import Arc, Circle

    rs = []
    for e in sk.entities:
        if getattr(e, "construction", False):
            continue
        if isinstance(e, Circle):
            rs.append(e.r)
        elif isinstance(e, Arc):
            c, a = sk.point(e.c), sk.point(e.start)
            rs.append(((a.x - c.x) ** 2 + (a.y - c.y) ** 2) ** 0.5)
    return min(rs) if rs else None


class Pad(Feature):
    kind = "pad"
    label = "Pad"
    group = "Base"
    summary = "Extrude a sketch's closed profiles between two Z levels of the design frame (nested profiles are holes)."
    params = (
        P("z_min", "Bottom Z", "length", -0.25 * IN, -400.0, 400.0),
        P("z_max", "Top Z", "length", 0.25 * IN, -400.0, 400.0),
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
        r = ctx.rules
        t = out.info["thickness"]
        return [Check(fid, "Thickness", t, _status(t, r.min_wall_preferred, r.min_wall), f"wall ≥ {r.min_wall / IN:.3f} in")]


class SketchPocket(Feature):
    kind = "sketch_pocket"
    label = "Pocket"
    group = "Material removal"
    summary = ("Mill a sketch's closed profiles into a pad from the top (+z) or bottom (-z), to a depth or through; "
               "nested profiles inside are islands left standing.")
    params = (
        P("side", "Cut from", "choice", "+z", choices=("+z", "-z")),
        P("through", "Through", "choice", "no", choices=("no", "yes")),
        P("depth", "Depth", "length", 0.1 * IN, 0.1, 400.0),
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
        r = _min_radius(sk)
        return FeatureOutput("subtract", tool, named={"floor": floors, "walls": [f for f in tf if f not in caps]},
                             driven={"sketch": sk.id},
                             info={"depth": depth, "tool_d": max(2 * r - 2 * 0.254, 0.5) if r else 0.0})

    def checks(self, ctx, fid, p, out):
        r = ctx.rules
        i = out.info
        if not i["tool_d"]:
            return []
        dt = i["depth"] / i["tool_d"]
        return [Check(fid, "Depth / tool Ø", dt, _status(dt, r.pocket_depth_tool_max - 1, r.pocket_depth_tool_max, True),
                      f"≤ {r.pocket_depth_tool_max:g}× tool Ø (tool = 2R − 0.020, R the sketch's smallest radius)")]


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
    params = (P("radius", "Radius", "length", 0.06 * IN, 0.1, 50.0),)

    def build(self, ctx, p, refs):
        solid, edges = self._edges(ctx, refs)
        try:
            return FeatureOutput("replace", solid.fillet(p["radius"], edges), info={"edges": float(len(edges))})
        except Exception as exc:
            raise FeatureError(f"fillet of {p['radius'] / IN:.3f} in failed on {len(edges)} edges; try a smaller radius") from exc


class Chamfer(_EdgeOp):
    kind = "chamfer"
    label = "Chamfer"
    summary = "45° chamfer on picked edges, or on the edges where two groups of named faces meet."
    params = (P("distance", "Distance", "length", 0.02 * IN, 0.1, 20.0),)

    def build(self, ctx, p, refs):
        solid, edges = self._edges(ctx, refs)
        try:
            return FeatureOutput("replace", solid.chamfer(p["distance"], None, edges), info={"edges": float(len(edges))})
        except Exception as exc:
            raise FeatureError(f"chamfer of {p['distance'] / IN:.3f} in failed on {len(edges)} edges; try a smaller size") from exc


FEATURES: dict[str, Feature] = {
    cls.kind: cls()
    for cls in (Plate, Wall, BoltHoles, Clevis, Lug, GussetPair, Pocket, ProfilePlate, ProfileCut, PinBore, Pad, SketchPocket, Fillet, Chamfer)
}


def resolve_params(kind: str, given: dict) -> dict:
    """Defaults merged with given values, clamped to the declared bounds."""
    out = {}
    for s in FEATURES[kind].params:
        v = given.get(s.name, s.default)
        if s.quantity in ("length", "ratio", "angle"):
            v = float(v)
            if s.min is not None:
                v = max(s.min, v)
            if s.max is not None:
                v = min(s.max, v)
        elif s.quantity == "choice" and v not in s.choices:
            v = s.default
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

