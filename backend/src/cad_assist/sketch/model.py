"""2D sketches: points, lines, arcs, circles and splines with constraints and dimensions (mirrors
frontend/src/sketch/model.ts). Coordinates are mm in the design frame's XY plane; a sketch is stored solved."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    id: str
    kind: Literal["point"] = "point"
    x: float
    y: float
    fixed: bool = False
    ref: str | None = None  # an intent id that places this point (a pin centre); held there


class Line(BaseModel):
    id: str
    kind: Literal["line"] = "line"
    p1: str
    p2: str
    construction: bool = False


class Circle(BaseModel):
    id: str
    kind: Literal["circle"] = "circle"
    c: str
    r: float
    construction: bool = False


class Arc(BaseModel):
    """Counter-clockwise from `start` to `end` about centre `c` (the radius is the centre-to-start distance)."""

    id: str
    kind: Literal["arc"] = "arc"
    c: str
    start: str
    end: str
    construction: bool = False


class Spline(BaseModel):
    """A smooth curve through its points, in order (the solver moves the points; the curve follows)."""

    id: str
    kind: Literal["spline"] = "spline"
    points: list[str]
    construction: bool = False
    # Leave each end along the curve it joins there (smooth into a lug arc or a line), rather than free.
    tangent_start: bool = False
    tangent_end: bool = False


Entity = Annotated[Point | Line | Circle | Arc | Spline, Field(discriminator="kind")]

ConstraintKind = Literal["coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "concentric",
                         "equal", "point_on", "distance", "radius", "diameter", "angle", "fix"]
DIMENSIONS = {"distance", "radius", "diameter", "angle"}


class Constraint(BaseModel):
    id: str
    kind: ConstraintKind
    refs: list[str]
    value: float | None = None  # mm, or radians for an angle
    driving: bool = True  # False: a reference dimension, measured rather than imposed


class SolveReport(BaseModel):
    status: Literal["solved", "failed"]
    dof: int
    conflicting: list[str] = Field(default_factory=list)
    redundant: list[str] = Field(default_factory=list)


class Sketch(BaseModel):
    id: str
    name: str = ""
    entities: list[Entity] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    report: SolveReport | None = None  # from the last solve

    def get(self, eid: str):
        return next((e for e in self.entities if e.id == eid), None)

    def point(self, pid: str) -> Point:
        e = self.get(pid)
        if not isinstance(e, Point):
            raise KeyError(f"{pid} is not a point in {self.id}")
        return e
