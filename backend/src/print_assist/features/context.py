"""Shared state while a design is built: print rules, the part so far and earlier feature outputs.

Everything is in world millimetres, Z up, the print bed at Z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from build123d import Face, Shape

from ..rules import PrintRules
from .model import FeatureOutput

if TYPE_CHECKING:
    from ..sketch.model import Sketch


class FeatureError(ValueError):
    pass


@dataclass
class BuildContext:
    rules: PrintRules = field(default_factory=PrintRules)
    part: Shape | None = None
    outputs: dict[str, FeatureOutput] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)  # resolved params per feature id
    sketches: dict[str, "Sketch"] = field(default_factory=dict)
    edge_picks: list = field(default_factory=list)  # the feature being built's picked edges (EdgePick)

    def feature_ref(self, refs: dict[str, str], role: str) -> tuple[FeatureOutput, dict]:
        fid = refs.get(role)
        if not fid or fid not in self.outputs:
            raise FeatureError(f"needs a {role.replace('_', ' ')} feature built before it; {fid or 'none'} not available")
        return self.outputs[fid], self.params[fid]

    def sketch_ref(self, refs: dict[str, str], role: str = "sketch") -> "Sketch":
        sid = refs.get(role)
        s = self.sketches.get(sid or "")
        if s is None:
            raise FeatureError(f"needs a sketch; {sid or 'none'} not found")
        return s


def faces_of(shape: Shape) -> list[Face]:
    return list(shape.faces())
