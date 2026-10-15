"""Turn B-rep shapes into face-tagged meshes the browser can pick.

Every face is tessellated separately and keeps its index, so a click on a triangle
in the viewer maps straight back to a B-rep face. Each face also carries a
geometric signature (type, area, centroid, normal or axis) that lets picked faces
re-find the face after the source file is revised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from build123d import CenterOf, Edge, Face, GeomType, Shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

LINEAR_TOL_MM = 0.02
ANGULAR_TOL_RAD = 0.2


@dataclass
class FaceMesh:
    id: int
    type: str
    area: float
    centroid: list[float]
    positions: list[float]
    indices: list[int]
    normal: list[float] | None = None
    axis_origin: list[float] | None = None
    axis_dir: list[float] | None = None
    radius: float | None = None


@dataclass
class EdgePolyline:
    id: int
    points: list[float]


@dataclass
class ShapeMesh:
    name: str
    units: str
    bbox_min: list[float]
    bbox_max: list[float]
    faces: list[FaceMesh] = field(default_factory=list)
    edges: list[EdgePolyline] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _vec(v) -> list[float]:
    return [round(v.X, 6), round(v.Y, 6), round(v.Z, 6)]


def _signature(face: Face) -> dict:
    sig: dict = {"type": face.geom_type.name.lower()}
    if face.geom_type == GeomType.PLANE:
        sig["normal"] = _vec(face.normal_at())
    elif face.geom_type == GeomType.CYLINDER:
        cyl = BRepAdaptor_Surface(face.wrapped).Cylinder()
        loc, d = cyl.Axis().Location(), cyl.Axis().Direction()
        sig["axis_origin"] = [round(loc.X(), 6), round(loc.Y(), 6), round(loc.Z(), 6)]
        sig["axis_dir"] = [round(d.X(), 6), round(d.Y(), 6), round(d.Z(), 6)]
        sig["radius"] = round(cyl.Radius(), 6)
    return sig


def _triangulation(face: Face) -> tuple[list[float], list[int]] | None:
    """The face's existing triangulation (from meshing the whole shape), in world coordinates, wound outward."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    if tri is None:
        return None
    trsf = loc.Transformation()
    positions = []
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        positions += [round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6)]
    flip = face.wrapped.Orientation() == TopAbs_REVERSED
    indices = []
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        indices += [a - 1, c - 1, b - 1] if flip else [a - 1, b - 1, c - 1]
    return positions, indices


def _mesh_face(face_id: int, face: Face) -> FaceMesh:
    got = _triangulation(face)
    if got is None:
        try:
            verts, tris = face.tessellate(LINEAR_TOL_MM, ANGULAR_TOL_RAD)
            got = [c for v in verts for c in _vec(v)], [i for tri in tris for i in tri]
        except AttributeError:  # a sliver face the mesher cannot triangulate: nothing to draw
            got = [], []
    positions, indices = got
    return FaceMesh(
        id=face_id,
        area=round(face.area, 6),
        centroid=_vec(face.center(CenterOf.MASS)),
        positions=positions,
        indices=indices,
        **_signature(face),
    )


def _polyline(edge_id: int, edge: Edge) -> EdgePolyline:
    n = 1 if edge.geom_type == GeomType.LINE else max(8, min(96, int(edge.length / 1.5)))
    pts = edge.positions([i / n for i in range(n + 1)])
    return EdgePolyline(id=edge_id, points=[c for p in pts for c in _vec(p)])


def mesh_shape(shape: Shape, name: str) -> ShapeMesh:
    bb = shape.bounding_box()
    out = ShapeMesh(name=name, units="mm", bbox_min=_vec(bb.min), bbox_max=_vec(bb.max))
    # Mesh the whole shape once, in parallel; each face's tessellate() below then reuses its triangulation instead of
    # meshing face by face (a mesh import has a face per triangle).
    BRepMesh_IncrementalMesh(shape.wrapped, LINEAR_TOL_MM, False, ANGULAR_TOL_RAD, True)
    out.faces = [_mesh_face(i, f) for i, f in enumerate(shape.faces())]
    out.edges = [_polyline(i, e) for i, e in enumerate(shape.edges())]
    return out
