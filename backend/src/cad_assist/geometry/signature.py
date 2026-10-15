"""Face signatures and re-matching faces across revisions of a part."""

from __future__ import annotations

import math

from build123d import CenterOf, Face, GeomType, Shape
from OCP.BRepAdaptor import BRepAdaptor_Surface

from ..intent.models import FaceSignature


def _r(v: float) -> float:
    return round(v, 6)


def face_signature(face: Face) -> FaceSignature:
    c = face.center(CenterOf.MASS)
    sig = FaceSignature(type=face.geom_type.name.lower(), area=_r(face.area), centroid=(_r(c.X), _r(c.Y), _r(c.Z)))
    if face.geom_type == GeomType.PLANE:
        n = face.normal_at()
        sig.normal = (_r(n.X), _r(n.Y), _r(n.Z))
    elif face.geom_type == GeomType.CYLINDER:
        cyl = BRepAdaptor_Surface(face.wrapped).Cylinder()
        loc, d = cyl.Axis().Location(), cyl.Axis().Direction()
        sig.axis_origin = (_r(loc.X()), _r(loc.Y()), _r(loc.Z()))
        sig.axis_dir = (_r(d.X()), _r(d.Y()), _r(d.Z()))
        sig.radius = _r(cyl.Radius())
    return sig


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _dist(a, b) -> float:
    return math.dist(a, b)


def _line_dist(p, origin, direction) -> float:
    """Distance from point p to the infinite line through origin along unit direction."""
    v = [p[i] - origin[i] for i in range(3)]
    t = _dot(v, direction)
    return math.dist(v, [t * d for d in direction])


def match_score(sig: FaceSignature, cand: FaceSignature, scale: float) -> float | None:
    """Lower is better; None means incompatible. `scale` is a characteristic part size in mm."""
    if sig.type != cand.type:
        return None
    score = 0.0
    if sig.normal and cand.normal:
        if _dot(sig.normal, cand.normal) < math.cos(math.radians(10)):
            return None
        score += 1 - _dot(sig.normal, cand.normal)
    if sig.axis_dir and cand.axis_dir:
        if abs(_dot(sig.axis_dir, cand.axis_dir)) < math.cos(math.radians(10)):
            return None
        if sig.radius and cand.radius and abs(sig.radius - cand.radius) > 0.25 * sig.radius:
            return None
        score += _line_dist(cand.axis_origin, sig.axis_origin, sig.axis_dir) / scale
    score += _dist(sig.centroid, cand.centroid) / scale
    score += abs(math.log(max(cand.area, 1e-9) / max(sig.area, 1e-9))) * 0.5
    return score


def find_face(shape: Shape, sig: FaceSignature, max_score: float = 0.5) -> int | None:
    """Index of the face in `shape` that best matches `sig`, or None if nothing is close enough."""
    bb = shape.bounding_box()
    scale = max(bb.diagonal, 1e-6)
    best: tuple[float, int] | None = None
    for i, f in enumerate(shape.faces()):
        s = match_score(sig, face_signature(f), scale)
        if s is not None and (best is None or s < best[0]):
            best = (s, i)
    return best[1] if best and best[0] <= max_score else None
