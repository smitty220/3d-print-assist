"""Data model for a designed part: an ordered list of parametric feature instances.

The part is built in the design frame (the primary mount's frame: Z = 0 is the mount face, +Z away from
the mating part) and moved to world coordinates for display and checks. Lengths in mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from build123d import Face, Shape
from pydantic import BaseModel, Field

from ..sketch.model import Sketch

ParamValue = float | str | bool | list[list[float]] | list[list[list[float]]]
# "profile": a closed outline, a list of [x, y] points (mm) in the design frame's XY plane; "profiles": several.
Quantity = Literal["length", "ratio", "angle", "choice", "bool", "profile", "profiles"]
Status = Literal["ok", "warn", "fail"]


class EdgePick(BaseModel):
    """An edge the engineer picked for a fillet or chamfer, kept by its geometry in the design frame so it is found again
    on a rebuilt part: where its middle is, which way it runs there, its length and the two faces it joined."""

    at: list[float]  # mm, the edge's midpoint
    tangent: list[float]  # unit direction at the midpoint
    length: float  # mm
    faces: list[str] = Field(default_factory=list)  # the joined faces' names ("F1.top", "F3.walls") when picked
    points: list[list[float]] = Field(default_factory=list)  # samples along it, for drawing it


class FeatureInstance(BaseModel):
    id: str
    kind: str
    params: dict[str, ParamValue] = Field(default_factory=dict)
    refs: dict[str, str] = Field(default_factory=dict)  # role -> intent id or feature id
    edges: list[EdgePick] = Field(default_factory=list)  # picked edges (fillet, chamfer)
    locked: bool = False
    suppressed: bool = False


class Design(BaseModel):
    features: list[FeatureInstance] = Field(default_factory=list)
    sketches: list[Sketch] = Field(default_factory=list)  # consumed by pad / sketch_pocket features
    recipe: dict | None = None  # the plate recipe the AI drafted this design from (topo/recipe.py), if any
    recipe_key: str | None = None  # sketch_key() when it was drafted: differs once the engineer edits the sketches
    recipe_summary: str = ""  # what the AI said about its draft
    recipe_questions: list[str] = Field(default_factory=list)  # what it asked the engineer to decide

    def sketch_key(self) -> str:
        import hashlib
        import json

        blob = json.dumps([s.model_dump(include={"entities", "constraints"}) for s in self.sketches], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def sketch(self, sid: str) -> Sketch | None:
        return next((s for s in self.sketches if s.id == sid), None)

    def get(self, fid: str) -> FeatureInstance | None:
        return next((f for f in self.features if f.id == fid), None)

    def next_id(self) -> str:
        used = {int(f.id[1:]) for f in self.features if f.id[1:].isdigit()}
        n = 1
        while n in used:
            n += 1
        return f"F{n}"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    quantity: Quantity
    default: ParamValue
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class RefSpec:
    role: str
    label: str
    kinds: tuple[str, ...]  # intent kinds (planar_mount, bolt_pattern, bore) or feature kinds
    target: Literal["intent", "feature", "sketch"]


@dataclass
class Check:
    feature: str
    name: str
    value: float | None
    status: Status
    rule: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FeatureOutput:
    """What a feature contributes: a tool solid fused into or cut from the part, or a modified part."""

    op: Literal["add", "subtract", "replace"]
    shape: Shape
    named: dict[str, list[Face]] = field(default_factory=dict)  # role -> faces of the tool, for attribution
    driven: dict[str, float | str] = field(default_factory=dict)  # values taken from intent, shown read-only
    info: dict[str, float] = field(default_factory=dict)  # numbers the rule checks use
