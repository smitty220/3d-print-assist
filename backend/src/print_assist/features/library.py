"""The parametric feature library.

Each feature declares its parameters and references, builds a tool solid in world millimetres (Z up, the print
bed at Z = 0) and reports print-rule checks. The AI composes these same features; one definition drives the
edit dialog, the AI tool description and the validator.
"""

from __future__ import annotations

from typing import ClassVar

from build123d import Align, Axis, Box, Face, GeomType, Pos, Shape, Vector, Wire, extrude

from .context import BuildContext, FeatureError
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


# ---------------------------------------------------------------- material removal


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
    for cls in (Plate, ProfilePlate, ProfileCut, Pad, SketchPocket, Fillet, Chamfer)
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
