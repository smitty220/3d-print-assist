"""Shared state while a design is built: print rules, each body's shape so far and earlier feature outputs.

Everything is in world millimetres, Z up, the print bed at Z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from build123d import Face, GeomType, Plane, Shape, Vector

from ..rules import PrintRules
from .model import FeatureOutput

if TYPE_CHECKING:
    from ..sketch.model import FacePick, Sketch


class FeatureError(ValueError):
    pass


BASE_PLANES = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}


@dataclass
class BuildContext:
    rules: PrintRules = field(default_factory=PrintRules)
    parts: dict[str, Shape] = field(default_factory=dict)  # body id -> its shape so far
    body: str = "B1"  # the body the feature being built works on
    outputs: dict[str, FeatureOutput] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)  # resolved params per feature id
    sketches: dict[str, "Sketch"] = field(default_factory=dict)  # solved, formulas applied
    edge_picks: list = field(default_factory=list)  # the feature being built's picked edges (EdgePick)
    face_picks: list = field(default_factory=list)  # the feature being built's picked faces (FacePick)
    fonts_dir: object = None  # the design's fonts folder, for fonts copied into the project
    body_ids: tuple[str, ...] = ()  # the design's bodies, for features that put shapes into another body

    @property
    def part(self) -> Shape | None:
        return self.parts.get(self.body)

    @part.setter
    def part(self, shape: Shape | None) -> None:
        if shape is None:
            self.parts.pop(self.body, None)
        else:
            self.parts[self.body] = shape

    def feature_ref(self, refs: dict[str, str], role: str) -> tuple[FeatureOutput, dict]:
        fid = refs.get(role)
        if not fid or fid not in self.outputs:
            raise FeatureError(f"needs a {role.replace('_', ' ')} feature built before it; {fid or 'none'} not available")
        return self.outputs[fid], self.params[fid]

    def other_body(self, bid: str, what: str = "it") -> str:
        """Check `bid` names a body other than the one being built into, for a feature that puts `what` there."""
        if not bid:
            raise FeatureError(f"choose the body to put {what} in")
        if self.body_ids and bid not in self.body_ids:
            raise FeatureError(f"no body {bid!r}")
        if bid == self.body:
            raise FeatureError(f"{what} must go into a different body than {bid}")
        return bid

    def sketch_ref(self, refs: dict[str, str], role: str = "sketch") -> "Sketch":
        sid = refs.get(role)
        s = self.sketches.get(sid or "")
        if s is None:
            raise FeatureError(f"needs a sketch; {sid or 'none'} not found")
        return s

    # ---------- faces and planes ----------

    def face_names(self) -> list[str | None]:
        from .design import attribute

        return attribute(self.part, self.outputs) if self.part is not None else []

    def picked_faces(self, picks: list["FacePick"] | None = None) -> list[Face]:
        """The current body's faces for the picks (see find_picked_faces)."""
        if self.part is None:
            raise FeatureError("nothing built yet to pick faces on")
        return find_picked_faces(self.part, self.face_names(), picks if picks is not None else self.face_picks)

    def sketch_plane(self, sk: "Sketch") -> Plane:
        """The world plane a sketch lies on (its X axis is the sketch's X)."""
        pl = sk.placement
        if pl.kind == "face":
            if pl.face is None:
                raise FeatureError(f"{sk.id} is placed on a face that was never picked")
            (face,) = self.picked_faces([pl.face])
            if face.geom_type != GeomType.PLANE:
                raise FeatureError(f"{sk.id} must sit on a flat face")
            n = face.normal_at()
            x = Vector(1, 0, 0) - n * n.X
            if x.length < 1e-6:
                x = Vector(0, 1, 0) - n * n.Y
            plane = Plane(origin=face.center(), x_dir=x.normalized(), z_dir=n)
        else:
            plane = BASE_PLANES[pl.plane]
        if pl.flip:
            plane = Plane(origin=plane.origin, x_dir=plane.x_dir, z_dir=-plane.z_dir)
        if pl.rotation:
            plane = plane.rotated((0, 0, pl.rotation))
        if pl.offset:
            plane = plane.offset(pl.offset)
        return plane


def find_picked_faces(shape: Shape, names: list[str | None], picks: list["FacePick"]) -> list[Face]:
    """The shape's faces for the picks: by name when a picked face had one (the face of that name nearest the picked
    point), else by geometry (a face through the picked point, facing the same way). A pick with no match raises: the
    user must pick again rather than the part being built on a guess."""
    faces = shape.faces()
    out: list[Face] = []
    for k, pick in enumerate(picks, 1):
        at, n = Vector(*pick.at), Vector(*pick.normal)

        def facing(f: Face) -> bool:
            if f.geom_type != GeomType.PLANE:
                return True
            try:
                return f.normal_at().dot(n) > 0.98
            except Exception:  # noqa: BLE001 - a degenerate face
                return False

        best = None
        if pick.name:
            named = [f for f, nm in zip(faces, names) if nm == pick.name and facing(f)]
            if named:
                best = min(named, key=lambda f: f.distance_to(at))
        if best is None:
            near = [(f.distance_to(at), f) for f in faces if facing(f)]
            near = [x for x in near if x[0] <= 0.05]
            if near:
                best = min(near, key=lambda x: x[0])[1]
        if best is None:
            raise FeatureError(f"picked face {k} ({pick.name or 'unnamed'}) is not on the part any more; pick it again")
        if not any(best.is_same(f) for f in out):
            out.append(best)
    return out


def faces_of(shape: Shape) -> list[Face]:
    return list(shape.faces())
