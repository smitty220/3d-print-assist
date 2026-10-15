"""Build a Design (ordered feature instances) into a solid, with named faces and rule checks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from build123d import GeomType, Shape, Vector
from OCP.BRep import BRep_Tool
from OCP.gp import gp_Pnt
from OCP.ShapeAnalysis import ShapeAnalysis_Surface

from ..rules import PrintRules
from .context import BuildContext, FeatureError
from .library import FEATURES, resolve_params
from .model import Check, Design, EdgePick, FeatureOutput

ATTR_TOL = 1e-3  # mm


Status = str  # "ok" | "warn" (built, a check fails) | "fail" | "skipped"


@dataclass
class FeatureResult:
    id: str
    kind: str
    params: dict
    driven: dict = field(default_factory=dict)
    error: str | None = None
    suppressed: bool = False
    status: Status = "ok"
    message: str | None = None  # why it was skipped, or what its failing check says

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BuildResult:
    local: Shape | None  # every body together; local and world are the same (world frame, Z up, bed at Z = 0)
    world: Shape | None
    features: list[FeatureResult]
    checks: list[Check]
    face_names: list[str | None]  # of `world`'s faces
    volume: float  # mm³
    mass: float  # kg
    parts: dict[str, Shape] = field(default_factory=dict)  # body id -> shape
    body_names: dict[str, list[str | None]] = field(default_factory=dict)  # body id -> its faces' names
    complete: bool = True  # no feature failed
    hashes: dict[str, str] = field(default_factory=dict)  # feature / sketch id -> input hash it was built from
    param_values: dict = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # formula errors (see params.graph.Resolved.errors)
    sketch_status: dict[str, str] = field(default_factory=dict)  # sketch id -> "ok" | "fail"
    sketch_errors: dict[str, str] = field(default_factory=dict)


def _sample_point(face) -> Vector:
    """A point on the face itself.

    Takes the centroid of the face's first mesh triangle (the area centroid can fall in a hole) and projects it
    onto the true surface, since a chord-triangle centroid sits slightly inside a curved face.
    """
    try:
        verts, tris = face.tessellate(0.2, 0.5)
    except AttributeError:  # a sliver face the mesher can't triangulate: use its parametric middle
        return face.position_at(0.5, 0.5)
    if not tris:
        return face.position_at(0.5, 0.5)
    a, b, c = tris[0]
    m = (verts[a] + verts[b] + verts[c]) * (1 / 3)
    surf = BRep_Tool.Surface_s(face.wrapped)
    uv = ShapeAnalysis_Surface(surf).ValueOfUV(gp_Pnt(m.X, m.Y, m.Z), 1e-7)
    p = surf.Value(uv.X(), uv.Y())
    return Vector(p.X(), p.Y(), p.Z())


def _box(tool_face) -> tuple[float, ...]:
    bb = tool_face.bounding_box()
    return (bb.min.X - 0.1, bb.min.Y - 0.1, bb.min.Z - 0.1, bb.max.X + 0.1, bb.max.Y + 0.1, bb.max.Z + 0.1)


def _same_surface(face, tool_face, pt: Vector, box: tuple[float, ...] | None = None) -> bool:
    b = box or _box(tool_face)
    if not (b[0] <= pt.X <= b[3] and b[1] <= pt.Y <= b[4] and b[2] <= pt.Z <= b[5]):
        return False
    if face.geom_type != tool_face.geom_type:
        return False
    if face.geom_type == GeomType.PLANE and abs(face.normal_at().dot(tool_face.normal_at())) < 0.999:
        return False
    return tool_face.distance_to(pt) < ATTR_TOL


def attribute(shape: Shape, outputs: dict[str, FeatureOutput], fallback: bool = True) -> list[str | None]:
    """Name every face of `shape` as "<feature>.<role>" from the tool faces that produced it.

    Later features win (a hole cut through a plate belongs to the hole). Faces no tool explains, such as
    fillet blends, go to the most recent edge-modifying feature as "<feature>.blend".
    """
    order = list(outputs.items())
    last_replace = next((fid for fid, out in reversed(order) if out.op == "replace"), None)
    # Every tool face, latest feature first. A face the boolean left untouched is the tool's own face: it is named by
    # hash, without measuring; only the rest need bounding boxes and a surface test (worked out once, when first needed).
    candidates = [(f"{fid}.{role}", tf) for fid, out in reversed(order) for role, tools in out.named.items() for tf in tools]
    same: dict[int, str] = {}
    for n, tf in reversed(candidates):
        same[hash(tf)] = n  # reversed, so the latest feature's name is the one kept
    boxes: list[tuple[float, ...]] | None = None
    names: list[str | None] = []
    for face in shape.faces():
        name = same.get(hash(face))
        if name is None:
            if boxes is None:
                boxes = [_box(tf) for _, tf in candidates]
            pt = _sample_point(face)
            name = next((n for (n, tf), box in zip(candidates, boxes) if _same_surface(face, tf, pt, box)), None)
        names.append(name or (f"{last_replace}.blend" if last_replace and fallback else None))
    return names


def _matches(name: str | None, spec: str) -> bool:
    if not name or not spec:
        return False
    return name == spec or name.startswith(spec + ".")


def _owners(shape: Shape) -> tuple[dict[int, list[int]], dict[int, object]]:
    """For each edge (by hash): the indices of the faces it bounds, and the edge."""
    owners: dict[int, list[int]] = {}
    edge_obj = {}
    for i, f in enumerate(shape.faces()):
        for e in f.edges():
            h = hash(e)
            if i not in owners.setdefault(h, []):
                owners[h].append(i)
            edge_obj[h] = e
    return owners, edge_obj


def edges_between(shape: Shape, names: list[str | None], spec_a: str, spec_b: str):
    """Edges shared by a face matching spec_a and a different face matching spec_b."""
    owners, edge_obj = _owners(shape)
    out = []
    for h, idx in owners.items():
        a = [i for i in idx if _matches(names[i], spec_a)]
        b = [j for j in idx if _matches(names[j], spec_b)]
        if any(i != j for i in a for j in b):
            out.append(edge_obj[h])
    return out


SMOOTH_DEG = 1.0  # faces meeting at less than this along an edge are tangent: nothing to blend
TANGENT_COS = math.cos(math.radians(SMOOTH_DEG))


def _v(p) -> list[float]:
    return [round(p.X, 4), round(p.Y, 4), round(p.Z, 4)]


def _sharp(shape: Shape, owner_ids: list[int], edge) -> bool:
    """Whether the two faces an edge joins meet at an angle there (a seam or a tangent joint is not sharp)."""
    if len(owner_ids) != 2:
        return False
    faces = shape.faces()
    m = edge.position_at(0.5)
    try:
        na, nb = (faces[i].normal_at(m) for i in owner_ids)
    except Exception:  # noqa: BLE001 - a degenerate face: let the blend itself decide
        return True
    return abs(na.dot(nb)) < TANGENT_COS


def describe_edge(shape: Shape, names: list[str | None], index: int) -> EdgePick:
    """The pick record for edge `index` of `shape`. Raises FeatureError for an edge with nothing to blend."""
    edges = shape.edges()
    if not 0 <= index < len(edges):
        raise FeatureError(f"no edge {index}")
    e = edges[index]
    owners, _ = _owners(shape)
    ids = owners.get(hash(e), [])
    if not _sharp(shape, ids, e):
        raise FeatureError("that edge joins two tangent faces (or is a seam): there is no corner to blend")
    return EdgePick(at=_v(e.position_at(0.5)), tangent=_v(e.tangent_at(0.5)), length=round(e.length, 4),
                    faces=[names[i] or "" for i in ids], points=[_v(e.position_at(t / 8)) for t in range(9)])


def describe_face(shape: Shape, names: list[str | None], index: int):
    """The pick record for face `index` of `shape`: its name, a point on it, its outward normal there and its area."""
    from ..sketch.model import FacePick

    faces = shape.faces()
    if not 0 <= index < len(faces):
        raise FeatureError(f"no face {index}")
    f = faces[index]
    pt = _sample_point(f)
    return FacePick(name=names[index] if index < len(names) else None, at=_v(pt), normal=_v(f.normal_at(pt)),
                    area=round(f.area, 4))


def find_picked_edges(shape: Shape, names: list[str | None], picks: list[EdgePick]) -> list:
    """The part's edges for the picks: an edge in the same place (midpoint within 0.05 mm, same length), or failing that
    the nearest edge between the same two named faces that runs the same way (a dimension changed upstream). The kernel
    carries a blend on along the edges that continue a picked one tangentially (as CATIA's tangency propagation)."""
    owners, edge_obj = _owners(shape)
    found: dict[int, object] = {}
    for k, pick in enumerate(picks, 1):
        at, t = Vector(*pick.at), Vector(*pick.tangent)
        want = sorted(pick.faces)
        best, near = None, []
        for h, e in edge_obj.items():
            if abs(e.tangent_at(0.5).dot(t)) < 0.98:
                continue
            d = (e.position_at(0.5) - at).length
            if d <= 0.05 and abs(e.length - pick.length) <= 0.05:
                best = h
                break
            if all(want) and sorted(names[i] or "" for i in owners[h]) == want:
                near.append((d, h))
        if best is None and near:
            near.sort(key=lambda x: x[0])
            # The nearest edge between the same faces, if it is clearly the one (a dimension upstream moved it).
            if near[0][0] <= max(1.0, 0.5 * pick.length) or len(near) == 1 or near[0][0] < 0.5 * near[1][0]:
                best = near[0][1]
        if best is None:
            where = " / ".join(f for f in pick.faces if f) or "?"
            raise FeatureError(f"picked edge {k} ({where}, {pick.length:.1f} mm long) is not on the part; "
                               f"pick it again (it may come from a later feature)")
        found[best] = edge_obj[best]
    return list(found.values())


def build_design(design: Design, rules: PrintRules | None = None,
                 on_feature: Callable[[FeatureResult, Shape | None], None] | None = None,
                 references_dir=None) -> BuildResult:
    """Build every feature in order (world frame), carrying on past a failed feature. `on_feature` is called after each
    one with its result and its body's shape so far. The incremental, stop-at-failure build is build.engine.Engine."""
    from ..build.engine import Engine

    return Engine(cache_size=0).build(design, rules or PrintRules(), stop_on_fail=False, on_feature=on_feature,
                                        references_dir=references_dir)


def apply_output(ctx: BuildContext, out: FeatureOutput) -> None:
    """Fuse, cut or replace a feature's output into its body, and set the other bodies it gives new shapes."""
    part = ctx.part
    if out.op == "add":
        part = out.shape if part is None else part + out.shape
    elif out.op == "subtract":
        if part is None:
            raise FeatureError("nothing to cut from yet")
        part = part - out.shape
    elif out.op == "replace":
        part = out.shape
    if out.op == "replace" or out.bodies:
        # The faces no earlier tool explains are the ones this operation created: blends, a shell's inside, a split's cut.
        made = out.named.setdefault(out.new_role, [])
        for shape in ([part] if out.op == "replace" and part is not None else []) + [s for s in out.bodies.values() if s is not None]:
            prior = attribute(shape, ctx.outputs, fallback=False)
            made += [f for f, n in zip(shape.faces(), prior) if n is None]
    ctx.part = part
    for bid, shape in out.bodies.items():
        if shape is None:
            ctx.parts.pop(bid, None)
        else:
            ctx.parts[bid] = shape
