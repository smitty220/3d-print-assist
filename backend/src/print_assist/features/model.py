"""Data model for a designed part: named parameters, sketches and an ordered list of parametric feature instances,
each building into one of the design's bodies.

Everything is in world millimetres, Z up, the print bed at Z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from build123d import Face, Shape
from pydantic import BaseModel, Field

from ..params.model import Parameter
from ..sketch.model import FacePick, Sketch

ParamValue = float | str | bool | list[list[float]] | list[list[list[float]]]
# "profile": a closed outline, a list of [x, y] points (mm) in the XY plane; "profiles": several.
Quantity = Literal["length", "ratio", "angle", "count", "choice", "bool", "text", "profile", "profiles"]
Status = Literal["ok", "warn", "fail"]


class EdgePick(BaseModel):
    """An edge the user picked for a fillet or chamfer, kept by its geometry so it is found again
    on a rebuilt part: where its middle is, which way it runs there, its length and the two faces it joined."""

    at: list[float]  # mm, the edge's midpoint
    tangent: list[float]  # unit direction at the midpoint
    length: float  # mm
    faces: list[str] = Field(default_factory=list)  # the joined faces' names ("F1.top", "F3.walls") when picked
    points: list[list[float]] = Field(default_factory=list)  # samples along it, for drawing it


class FeatureInstance(BaseModel):
    id: str
    kind: str
    params: dict[str, ParamValue] = Field(default_factory=dict)  # literal values (and the last value of each formula)
    exprs: dict[str, str] = Field(default_factory=dict)  # param -> formula ("height * 0.8"); wins over params
    refs: dict[str, str] = Field(default_factory=dict)  # role -> feature id or sketch id
    edges: list[EdgePick] = Field(default_factory=list)  # picked edges (fillet, chamfer)
    faces: list[FacePick] = Field(default_factory=list)  # picked faces (shell openings, text target)
    body: str = "B1"
    locked: bool = False
    suppressed: bool = False


class Body(BaseModel):
    id: str
    name: str = "Body"
    color: str | None = None  # "#rrggbb", shown when no filament is assigned
    filament: int | None = None  # the palette slot it prints in


class Filament(BaseModel):
    """One filament in the design's palette, numbered like the printer's slots (AMS slot 1 to 4 on an A1 + AMS lite)."""

    slot: int = Field(ge=1)
    name: str = ""
    color: str = "#d0d0d0"  # "#rrggbb"
    material: str = "PLA"
    nozzle: int = Field(default=0, ge=0)  # which nozzle it feeds, on a dual-nozzle printer


class Reference(BaseModel):
    """A file in the design's references folder shown beside the design to look at, measure and copy features from;
    never built, printed or exported. Placed like the Import feature: scaled, turned about Z, then on the bed centred
    (or as the file puts it), then moved."""

    id: str  # "R1"
    name: str
    file: str  # in the references folder
    plate: int | None = None  # a Bambu project's plate (1-based); None: the first
    visible: bool = True
    place: Literal["on_bed", "as_in_file"] = "on_bed"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotate_z: float = 0.0  # degrees
    scale: float = 1.0
    color: str = "#7fb2e5"


class Variant(BaseModel):
    """A row of the variants table: parameter values that differ from the design's ("cards": "60")."""

    id: str
    name: str
    overrides: dict[str, str] = Field(default_factory=dict)  # parameter -> formula or value


class Colorway(BaseModel):
    """A set of filament colours for the palette's slots; the slots it leaves out keep the palette's colour."""

    id: str
    name: str
    colors: dict[int, str] = Field(default_factory=dict)  # slot -> "#rrggbb"
    names: dict[int, str] = Field(default_factory=dict)  # slot -> filament name ("Galaxy Black")


def _default_bodies() -> list[Body]:
    return [Body(id="B1", name="Body")]


class Design(BaseModel):
    parameters: list[Parameter] = Field(default_factory=list)
    bodies: list[Body] = Field(default_factory=_default_bodies)
    features: list[FeatureInstance] = Field(default_factory=list)
    rollback: int | None = None  # features[rollback:] are past the end-of-part bar and not built
    sketches: list[Sketch] = Field(default_factory=list)  # consumed by pad / sketch_pocket features
    filaments: list[Filament] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=list)
    colorways: list[Colorway] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)

    def filament(self, slot: int | None) -> Filament | None:
        return next((f for f in self.filaments if f.slot == slot), None)

    def body_color(self, b: Body) -> str | None:
        """What a body is shown in: its filament's colour, else its own."""
        f = self.filament(b.filament)
        return f.color if f else b.color

    def variant(self, vid: str) -> Variant | None:
        return next((v for v in self.variants if v.id == vid), None)

    def reference(self, rid: str) -> Reference | None:
        return next((r for r in self.references if r.id == rid), None)

    def colorway(self, cid: str) -> Colorway | None:
        return next((c for c in self.colorways if c.id == cid), None)

    def sketch(self, sid: str) -> Sketch | None:
        return next((s for s in self.sketches if s.id == sid), None)

    def get(self, fid: str) -> FeatureInstance | None:
        return next((f for f in self.features if f.id == fid), None)

    def parameter(self, name: str) -> Parameter | None:
        return next((p for p in self.parameters if p.name == name), None)

    def body(self, bid: str) -> Body | None:
        return next((b for b in self.bodies if b.id == bid), None)

    def next_sketch_id(self) -> str:
        used = {int(s.id[1:]) for s in self.sketches if s.id[1:].isdigit()}
        n = 1
        while n in used:
            n += 1
        return f"S{n}"

    def next_body_id(self) -> str:
        used = {int(b.id[1:]) for b in self.bodies if b.id[1:].isdigit()}
        n = 1
        while n in used:
            n += 1
        return f"B{n}"

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
    kinds: tuple[str, ...]  # feature kinds the ref accepts (empty = any)
    target: Literal["feature", "sketch"]


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

    op: Literal["add", "subtract", "replace", "none"]  # "none": the feature's own body is left as it is
    shape: Shape | None
    named: dict[str, list[Face]] = field(default_factory=dict)  # role -> faces of the tool, for attribution
    driven: dict[str, float | str] = field(default_factory=dict)  # values taken from elsewhere, shown read-only
    info: dict[str, float] = field(default_factory=dict)  # numbers the rule checks use
    new_role: str = "blend"  # a "replace" op: the name given to the faces it created (a shell's are "inner")
    bodies: dict[str, Shape | None] = field(default_factory=dict)  # other bodies' new shapes (None removes one)
