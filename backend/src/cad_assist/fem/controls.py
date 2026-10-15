"""Mesh controls the user (or Claude) sets before a solve: global size, edge divisions, mapped faces and spheres of
influence, plus the best-practice rules that propose a first set.

Edges and faces are picked on the built part by index, but an index changes whenever the part is rebuilt, so every
control also keeps the entity's centre of mass and size. Controls are re-found on the current part (and inside Gmsh,
which numbers entities its own way) by the nearest centre of mass.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from build123d import CenterOf, GeomType, Shape
from pydantic import BaseModel, Field

# Global size (mm), curvature (elements per circle), sphere size at a lug (fraction of the pin diameter). Bell crank,
# peak von Mises: coarse 33.3 ksi (246k DOF, 13 s), medium 33.4 (646k, 34 s), fine 33.4 (1.31M, 78 s).
PRESETS = {
    "coarse": (5.0, 20, 1 / 5),
    "medium": (3.0, 28, 1 / 8),
    "fine": (2.5, 36, 1 / 11),  # ~2x medium's DOF; memory grows with it (medium: ~4.6 GB peak)
}
LUG_RADIUS = 2.2  # sphere of influence radius, pin diameters
GB_PER_MDOF = 7.1  # measured: medium bell crank, 646k DOF, 4.6 GB peak

DEFAULT_PRACTICES = """\
1. Quadratic tetrahedra (C3D10). Keep the global size about a third of the smallest structural width.
2. Pin bores carry the peak stress (net section and bearing): at least 24-32 elements around each bore.
3. A sphere of influence about each lug, radius about 2 pin diameters, element size about D/8; let it blend
   into the global size rather than jumping more than 2:1.
4. At least 3 quadratic elements through the lug thickness at the bore, and 2 through thin webs.
5. At least 4-6 elements along any fillet that carries stress.
6. Mapped (structured) faces only on regular 3- or 4-sided faces; they rarely help a tetrahedral part.
7. Stay under about 1M degrees of freedom on a laptop (about 7 GB of memory per million).
8. Check convergence: the peak should move less than 5% between this mesh and a coarser one.
"""


class EdgeDivision(BaseModel):
    edge: int  # index in the part's edges when picked
    n: int = Field(ge=1, le=500)  # elements along the edge
    ref: list[float]  # centre of mass, world mm
    length: float
    label: str = ""


class MappedFace(BaseModel):
    face: int
    ref: list[float]
    area: float
    label: str = ""


class Sphere(BaseModel):
    center: list[float]
    radius: float = Field(gt=0)
    size: float = Field(gt=0)  # element size inside
    label: str = ""


class MeshSetup(BaseModel):
    size: float = Field(3.0, gt=0)  # largest element edge, mm
    curvature: int = Field(28, ge=0, le=120)  # elements per full circle on curved faces (0: off)
    min_size: float = Field(0.15, gt=0)
    edges: list[EdgeDivision] = Field(default_factory=list)
    faces: list[MappedFace] = Field(default_factory=list)
    spheres: list[Sphere] = Field(default_factory=list)
    source: str = "none"  # none | rules | ai | user
    notes: list[str] = Field(default_factory=list)  # why each control is there (rules or Claude)
    practices: str = DEFAULT_PRACTICES

    def key(self) -> str:
        d = self.model_dump(exclude={"source", "notes", "practices"})
        return hashlib.sha1(repr(d).encode()).hexdigest()[:12]

    def scaled(self, k: float) -> "MeshSetup":
        """Every size times k (divisions over k): a coarser (k > 1) or finer mesh with the same layout."""
        s = self.model_copy(deep=True)
        s.size *= k
        s.min_size *= k
        s.curvature = max(8, round(s.curvature / k)) if s.curvature else 0
        for e in s.edges:
            e.n = max(1, round(e.n / k))
        for sp in s.spheres:
            sp.size *= k
        return s


def _c(v) -> list[float]:
    return [round(v.X, 4), round(v.Y, 4), round(v.Z, 4)]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def edge_ref(part: Shape, i: int) -> dict:
    e = part.edges()[i]
    return {"ref": _c(e.center(CenterOf.MASS)), "length": round(e.length, 4)}


def face_ref(part: Shape, i: int) -> dict:
    f = part.faces()[i]
    return {"ref": _c(f.center(CenterOf.MASS)), "area": round(f.area, 4)}


def mappable(part: Shape, i: int) -> str | None:
    """Why a face cannot be mapped, or None if it can (one boundary of 3 or 4 edges)."""
    f = part.faces()[i]
    if len(f.wires()) != 1:
        return "it has holes in it"
    n = len(f.outer_wire().edges())
    if n not in (3, 4):
        return f"it has {n} edges; mapped meshing needs 3 or 4"
    return None


def _nearest(refs: np.ndarray, sizes: np.ndarray, ref, size) -> int | None:
    if not len(refs):
        return None
    d = np.linalg.norm(refs - np.asarray(ref, float), axis=1)
    d = d + 0.2 * np.abs(sizes - size) / max(size, 1e-9) * max(size, 1.0) ** 0.5  # prefer the same size
    k = int(np.argmin(d))
    tol = max(0.05, 0.02 * math.sqrt(max(size, 1e-9)))
    return k if np.linalg.norm(refs[k] - np.asarray(ref, float)) < tol + 1e-6 and abs(sizes[k] - size) < 0.02 * size + 1e-3 else None


def resolve(setup: MeshSetup, part: Shape) -> dict:
    """Current indices of every control's edge and face on `part` (None where the part has changed under it)."""
    E = part.edges()
    F = part.faces()
    er = np.array([_c(e.center(CenterOf.MASS)) for e in E]) if E else np.zeros((0, 3))
    el = np.array([e.length for e in E])
    fr = np.array([_c(f.center(CenterOf.MASS)) for f in F]) if F else np.zeros((0, 3))
    fa = np.array([f.area for f in F])
    return {"edges": [_nearest(er, el, c.ref, c.length) for c in setup.edges],
            "faces": [_nearest(fr, fa, c.ref, c.area) for c in setup.faces]}


def _bore_edges(part: Shape, b) -> tuple[list[int], list[int]]:
    """Circular edges on a bore (its rims) and the straight seams of its wall."""
    a = _unit(b.frame.z)
    o = np.asarray(b.frame.origin, float)
    r = b.diameter / 2
    rims, seams = [], []
    for i, e in enumerate(part.edges()):
        if e.geom_type == GeomType.CIRCLE and abs(e.radius - r) < 0.01 * r:
            c = np.asarray(_c(e.arc_center)) - o
            if np.linalg.norm(c - (c @ a) * a) < 0.01 * r:
                rims.append(i)
        elif e.geom_type == GeomType.LINE:
            p0, p1 = (np.asarray(_c(v)) for v in (e.position_at(0), e.position_at(1)))
            d = p1 - p0
            if np.linalg.norm(d) > 1e-6 and abs(abs(_unit(d) @ a) - 1) < 1e-4:
                v = p0 - o
                if abs(np.linalg.norm(v - (v @ a) * a) - r) < 0.01 * r:
                    seams.append(i)
    return rims, seams


def propose(intent, part: Shape, preset: str = "medium", practices: str = DEFAULT_PRACTICES) -> MeshSetup:
    """The best-practice rules as a first set of controls: global size from the preset, a sphere of influence about
    each lug, divisions around each bore rim and through its wall."""
    size, curvature, frac = PRESETS[preset]
    s = MeshSetup(size=size, curvature=curvature, min_size=round(size / 20, 3), source="rules", practices=practices)
    s.notes.append(f"Global size {size:.1f} mm and {curvature} elements per circle on curved faces ({preset} preset).")
    for b in (o for o in intent.objects if o.kind == "bore"):
        h = round(max(0.4, frac * b.diameter), 3)
        s.spheres.append(Sphere(center=[round(x, 4) for x in b.frame.origin], radius=round(LUG_RADIUS * b.diameter, 3), size=h,
                                label=f"{b.id} lug"))
        rims, seams = _bore_edges(part, b)
        around = max(24, 2 * math.ceil(math.pi * b.diameter / h / 2))
        for i in rims:
            ref = edge_ref(part, i)
            n = max(4, round(around * ref["length"] / (math.pi * b.diameter)))  # a split rim gets its share
            s.edges.append(EdgeDivision(edge=i, n=n, label=f"{b.id} bore rim", **ref))
        for i in seams:
            ref = edge_ref(part, i)
            s.edges.append(EdgeDivision(edge=i, n=max(3, math.ceil(ref["length"] / h)), label=f"{b.id} through thickness", **ref))
        s.notes.append(f"{b.id} (D {b.diameter:.2f} mm): sphere of influence R {LUG_RADIUS * b.diameter:.1f} mm at {h:.2f} mm, "
                       f"{around} elements around the bore, at least 3 through the wall.")
    return s


def estimate_gb(dof: int) -> float:
    return GB_PER_MDOF * dof / 1e6
