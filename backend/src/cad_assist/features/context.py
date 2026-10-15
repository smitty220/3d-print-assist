"""Shared state while a design is built: the intent, rules, the design frame and earlier feature outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from build123d import Face, Plane, Shape, Vector

from ..intent.models import Bore, BoltPattern, Frame, Intent, IntentObject, Rules
from .model import FeatureOutput

if TYPE_CHECKING:
    from ..sketch.model import Sketch


class FeatureError(ValueError):
    pass


def frame_plane(frame: Frame) -> Plane:
    return Plane(origin=frame.origin, x_dir=frame.x, z_dir=frame.z)


@dataclass
class BuildContext:
    intent: Intent
    frame: Frame  # design frame
    part: Shape | None = None
    outputs: dict[str, FeatureOutput] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)  # resolved params per feature id
    sketches: dict[str, "Sketch"] = field(default_factory=dict)
    edge_picks: list = field(default_factory=list)  # the feature being built's picked edges (EdgePick)

    @property
    def rules(self) -> Rules:
        return self.intent.rules

    # ---------- coordinates ----------

    def to_local(self, p) -> Vector:
        return frame_plane(self.frame).to_local_coords(Vector(*p))

    def dir_to_local(self, d) -> Vector:
        pl = frame_plane(self.frame)
        return Vector(Vector(*d).dot(pl.x_dir), Vector(*d).dot(pl.y_dir), Vector(*d).dot(pl.z_dir))

    # ---------- references ----------

    def intent_ref(self, refs: dict[str, str], role: str, kinds: tuple[str, ...]) -> IntentObject:
        rid = refs.get(role)
        obj = self.intent.get(rid) if rid else None
        if obj is None:
            raise FeatureError(f"needs a {role.replace('_', ' ')} ({' / '.join(kinds)}); {rid or 'none'} not found")
        if obj.kind not in kinds:
            raise FeatureError(f"{rid} is a {obj.kind}, expected {' / '.join(kinds)}")
        return obj

    def feature_ref(self, refs: dict[str, str], role: str) -> tuple[FeatureOutput, dict]:
        fid = refs.get(role)
        if not fid or fid not in self.outputs:
            raise FeatureError(f"needs a {role.replace('_', ' ')} feature built before it; {fid or 'none'} not available")
        return self.outputs[fid], self.params[fid]

    def sketch_ref(self, refs: dict[str, str], role: str = "sketch") -> "Sketch":
        """A sketch, checked against the intent: points tied to a pin must still sit on it."""
        sid = refs.get(role)
        s = self.sketches.get(sid or "")
        if s is None:
            raise FeatureError(f"needs a sketch; {sid or 'none'} not found")
        for e in s.entities:
            if getattr(e, "ref", None):
                obj = self.intent.get(e.ref)
                if obj is None:
                    raise FeatureError(f"{s.id}: {e.id} is tied to {e.ref}, which no longer exists")
                p = self.to_local(obj.frame.origin)
                if abs(p.X - e.x) > 1e-3 or abs(p.Y - e.y) > 1e-3:
                    raise FeatureError(f"{s.id} is out of date with {e.ref}; solve the sketch again")
        return s

    def bolt_holes_local(self, bp: BoltPattern) -> list[tuple[Vector, Vector, float]]:
        """(centre, axis, diameter) for each hole, in the design frame. Axis points away from the mating face."""
        pl = Plane(origin=bp.frame.origin, x_dir=bp.frame.x, z_dir=bp.frame.z)
        axis = self.dir_to_local(bp.frame.z)
        out = []
        for h in bp.holes:
            world = pl.from_local_coords(Vector(h.center[0], h.center[1], 0))
            out.append((self.to_local(world), axis, h.diameter))
        return out

    def bore_local(self, b: Bore) -> tuple[Vector, Vector]:
        return self.to_local(b.frame.origin), self.dir_to_local(b.frame.z).normalized()


def faces_of(shape: Shape) -> list[Face]:
    return list(shape.faces())
