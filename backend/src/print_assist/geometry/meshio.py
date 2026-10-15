"""Files from elsewhere as solids: STEP as it is, STL and 3MF meshes sewn into solids (each triangle a face).

3MF objects are placed as the file places them: build items and nested components (including Bambu Studio's
components in other model files) with their transforms, in the file's unit, converted to millimetres.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from build123d import Compound, Mesher, Shape, Shell, Solid, import_step
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

FORMATS = (".step", ".stp", ".stl", ".3mf")
CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
PROD = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"
UNITS = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}
BIG = 50_000  # triangles: above this, booleans on the sewn solid get slow
LIMIT = 250_000  # triangles: above this, sewing takes minutes

Point = tuple[float, float, float]
Mesh = tuple[list[Point], list[tuple[int, int, int]]]


class ImportError_(ValueError):
    pass


_cache: dict[tuple[str, float], tuple[Shape, int]] = {}


def load(path: Path) -> tuple[Shape, int]:
    """The file's contents as one shape (a compound when it holds several solids), and its triangle count (0 for
    STEP). Kept in memory per file and modification time, since sewing a big mesh takes seconds."""
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    if key not in _cache:
        if len(_cache) > 8:
            _cache.clear()
        _cache[key] = _load(path)
    return _cache[key]


def _load(path: Path) -> tuple[Shape, int]:
    ext = path.suffix.lower()
    if ext not in FORMATS:
        raise ImportError_(f"can't import {path.suffix or 'that'} files; formats: STEP, STL, 3MF")
    if not path.exists():
        raise ImportError_(f"{path.name} is missing from the design's references folder")
    if ext in (".step", ".stp"):
        shape = import_step(str(path))
        solids = shape.solids()
        if not solids:
            raise ImportError_(f"{path.name} has no solids")
        return (solids[0] if len(solids) == 1 else Compound(children=list(solids))), 0
    if ext == ".stl":
        if path.stat().st_size > LIMIT * 50 + 84:  # binary STL: 50 bytes a triangle
            raise ImportError_(f"{path.name} is too big to import (over {LIMIT:,} triangles)")
        shapes = Mesher().read(str(path))
        tris = sum(len(s.faces()) for s in shapes)
    else:
        meshes = read_3mf(path)
        if not meshes:
            raise ImportError_(f"{path.name} has nothing to print in it")
        n = sum(len(m[1]) for m in meshes)
        if n > LIMIT:
            raise ImportError_(f"{path.name} has {n:,} triangles in {len(meshes)} parts; imports stop at {LIMIT:,}. "
                               "Export just the part you need (Bambu Studio: right-click it > Export as one STL)")
        shapes = [solid_from_mesh(*m) for m in meshes]
        tris = sum(len(m[1]) for m in meshes)
    open_ = [s for s in shapes if not isinstance(s, Solid)]
    if open_:
        raise ImportError_(f"{path.name}: the mesh is not closed (holes or loose triangles), so it can't become a "
                           "solid; repair it first (Bambu Studio: right-click > Fix model)")
    return (shapes[0] if len(shapes) == 1 else Compound(children=shapes)), tris


def solid_from_mesh(points: list[Point], triangles: list[tuple[int, int, int]]) -> Shape:
    """Sew the triangles into a solid (a shell if the mesh is not closed); inner shells become voids."""
    pts = [gp_Pnt(*p) for p in points]
    sew = BRepBuilderAPI_Sewing()
    for a, b, c in triangles:
        if len({a, b, c}) < 3 or _area2(points[a], points[b], points[c]) < 1e-18:
            continue
        poly = BRepBuilderAPI_MakePolygon(pts[a], pts[b], pts[c], True)
        sew.Add(BRepBuilderAPI_MakeFace(poly.Wire()).Face())
    sew.Perform()
    shells = []
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    while ex.More():
        shells.append(Shell(TopoDS.Shell(ex.Current())))
        ex.Next()
    if not shells:
        raise ImportError_("the mesh has no usable triangles")
    outer = max(shells, key=lambda s: math.prod(s.bounding_box().size))
    if not outer.is_manifold:
        return outer
    mk = BRepBuilderAPI_MakeSolid(outer.wrapped)
    for s in shells:
        if s is not outer:
            mk.Add(s.wrapped)
    solid = Solid(mk.Solid())
    if solid.volume < 0:  # triangles wound inward
        solid = Solid(solid.wrapped.Reversed())
    return solid


def _area2(a: Point, b: Point, c: Point) -> float:
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return n[0] ** 2 + n[1] ** 2 + n[2] ** 2


# ---------------------------------------------------------------- 3MF

Affine = tuple[float, ...]  # 12 numbers, 3MF order: m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32 (row vectors)
IDENTITY: Affine = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)


def _affine(s: str | None) -> Affine:
    if not s:
        return IDENTITY
    v = tuple(float(x) for x in s.split())
    if len(v) != 12:
        raise ImportError_("bad transform in the 3MF")
    return v


def _compose(inner: Affine, outer: Affine) -> Affine:
    """The transform applying `inner` then `outer` (3MF row-vector convention: p' = p * M)."""
    a = [[inner[0], inner[1], inner[2]], [inner[3], inner[4], inner[5]], [inner[6], inner[7], inner[8]], [inner[9], inner[10], inner[11]]]
    b = [[outer[0], outer[1], outer[2]], [outer[3], outer[4], outer[5]], [outer[6], outer[7], outer[8]]]
    out = []
    for r in range(4):
        row = [sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
        if r == 3:
            row = [row[c] + outer[9 + c] for c in range(3)]
        out.extend(row)
    return tuple(out)


def _apply(m: Affine, p: Point, scale: float) -> Point:
    x, y, z = p
    return tuple(scale * (x * m[c] + y * m[3 + c] + z * m[6 + c] + m[9 + c]) for c in range(3))


def read_3mf(path: Path) -> list[Mesh]:
    """One mesh per placed mesh object (each part of a multi-part object separately), in millimetres, with every
    component's and the build item's transform applied."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        main = _model_path(z, names)
        docs: dict[str, ET.Element] = {}

        def doc(p: str) -> ET.Element:
            p = p.lstrip("/")
            if p not in docs:
                if p not in names:
                    raise ImportError_(f"the 3MF refers to {p}, which is not in it")
                docs[p] = ET.fromstring(z.read(p))
            return docs[p]

        root = doc(main)
        scale = UNITS.get(root.get("unit", "millimeter"), 1.0)

        out: list[Mesh] = []

        def collect(model: str, oid: str, m: Affine, depth: int = 0) -> None:
            if depth > 16:
                raise ImportError_("the 3MF's components nest too deeply")
            obj = doc(model).find(f"{CORE}resources/{CORE}object[@id='{oid}']")
            if obj is None:
                raise ImportError_(f"the 3MF has no object {oid} in {model}")
            mesh = obj.find(f"{CORE}mesh")
            if mesh is not None:
                pts = [_apply(m, (float(v.get("x")), float(v.get("y")), float(v.get("z"))), scale)
                       for v in mesh.find(f"{CORE}vertices")]
                tris = [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))) for t in mesh.find(f"{CORE}triangles")]
                if tris:
                    out.append(_merge(pts, tris))
            comps = obj.find(f"{CORE}components")
            for c in comps if comps is not None else ():
                collect(c.get(f"{PROD}path") or model, c.get("objectid"), _compose(_affine(c.get("transform")), m),
                        depth + 1)

        for item in root.findall(f"{CORE}build/{CORE}item"):
            if item.get("printable") != "0":
                collect(main, item.get("objectid"), _affine(item.get("transform")))
        return out


def _model_path(z: zipfile.ZipFile, names: set[str]) -> str:
    rels = "_rels/.rels"
    if rels in names:
        for r in ET.fromstring(z.read(rels)):
            if r.get("Type", "").endswith("/3dmodel"):
                return r.get("Target", "").lstrip("/")
    return "3D/3dmodel.model"


def _merge(points: list[Point], tris: list[tuple[int, int, int]]) -> Mesh:
    """Merge coincident vertices (components often share seams) so the mesh sews closed."""
    index: dict[Point, int] = {}
    out: list[Point] = []
    remap = []
    for p in points:
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        if key not in index:
            index[key] = len(out)
            out.append(key)
        remap.append(index[key])
    return out, [(remap[a], remap[b], remap[c]) for a, b, c in tris]
