"""The design-intent model: typed objects every pillar (CAD, FEM, DFM, GD&T) reads.

All lengths are millimetres. Each object owns a local frame (origin + Z axis + X axis); positions
inside it (hole centres, footprints) are in that frame's XY plane.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


class Frame(BaseModel):
    origin: Vec3
    z: Vec3
    x: Vec3


class FaceSignature(BaseModel):
    """Geometry that identifies a face independent of its index, used to re-find it after a revision."""

    type: str
    area: float
    centroid: Vec3
    normal: Vec3 | None = None
    axis_origin: Vec3 | None = None
    axis_dir: Vec3 | None = None
    radius: float | None = None


class FaceRef(BaseModel):
    part: str
    face_id: int
    signature: FaceSignature


class IntentBase(BaseModel):
    id: str
    label: str = ""
    strength: Literal["hard", "soft"] = "hard"
    source: Literal["picked", "typed", "ai"] = "picked"


class PlanarMount(IntentBase):
    kind: Literal["planar_mount"] = "planar_mount"
    frame: Frame
    footprint: list[Vec2]
    face: FaceRef
    # Boundary condition for analysis. "normal": the faying surface can't move along the mount normal (a
    # linear stand-in for contact; it also resists lift-off). "auto" = free.
    support: Literal["auto", "free", "normal"] = "auto"


class Hole(BaseModel):
    center: Vec2
    diameter: float
    depth: float | None = None  # length of the hole wall in the mating part, along -Z of the frame


class BoltPattern(IntentBase):
    kind: Literal["bolt_pattern"] = "bolt_pattern"
    frame: Frame
    holes: list[Hole]
    fastener: str | None = None
    fit: Literal["floating", "fixed"] = "floating"
    parent: str | None = None
    face: FaceRef
    # "fixed": the clamped area under each fastener is held in all directions. "auto" = fixed when the
    # pattern is on a mount face and no load case targets it, otherwise free.
    support: Literal["auto", "fixed", "free"] = "auto"


class Bore(IntentBase):
    """A pin bore on a mating part. For a forked end (two coaxial bores), `gap` is the clear space between
    the tines and the frame origin sits mid-gap; otherwise `gap` is None and the origin is mid-bore."""

    kind: Literal["bore"] = "bore"
    frame: Frame
    diameter: float
    length: float
    gap: float | None = None
    face: FaceRef
    # "auto" = free (bores are where loads come in). "pin": held by the pin but free to turn about it (a pivot).
    # "link": the pin joins a two-force link (a rod end), so the bore reacts only along the link: the line from the
    # pin to the mating part's centre of mass, square to the pin. "fixed" clamps the bore surface in every direction.
    support: Literal["auto", "fixed", "free", "pin", "link"] = "auto"


class KeepOut(IntentBase):
    kind: Literal["keep_out"] = "keep_out"
    part: str
    clearance: float = 3.048  # 0.120 in


class DesignSpace(IntentBase):
    """Box the part must stay inside, in the frame of the primary mount (min/max corners, frame coords)."""

    kind: Literal["design_space"] = "design_space"
    frame: Frame
    min: Vec3
    max: Vec3


class LoadCase(IntentBase):
    """A load applied at an interface. Force in N and moment in N·mm, global axes, at the target's frame origin."""

    kind: Literal["load_case"] = "load_case"
    target: str | None
    force: Vec3 = (0.0, 0.0, 0.0)
    moment: Vec3 = (0.0, 0.0, 0.0)
    category: Literal["limit", "ultimate"] = "limit"
    factor: float = 1.5  # limit-to-ultimate factor


IntentObject = Annotated[
    PlanarMount | BoltPattern | Bore | KeepOut | DesignSpace | LoadCase, Field(discriminator="kind")
]

ID_PREFIX = {
    "planar_mount": "IF",
    "bolt_pattern": "IF",
    "bore": "IF",
    "keep_out": "KO",
    "design_space": "DS",
    "load_case": "LC",
}

MATERIALS: dict[str, float] = {  # density, kg/m³
    "7075-T7351 plate": 2810.0,
    "2024-T351 plate": 2780.0,
    "6061-T651 plate": 2700.0,
    "Ti-6Al-4V annealed": 4430.0,
}


KSI = 6.894757  # MPa


class Material(BaseModel):
    """Stiffness and strength for analysis. Allowables from MIL-HDBK-5J (public domain; superseded by MMPDS, which
    is licensed and not used here): A-basis, long-transverse plate, quoted from the handbook tables and to be checked
    against the engineer's own copy. Editable per project through the rules."""

    E: float  # MPa
    nu: float
    ftu: float  # MPa
    fty: float  # MPa
    source: str


MATERIAL_PROPS: dict[str, Material] = {
    "7075-T7351 plate": Material(E=10.3e3 * KSI, nu=0.33, ftu=69 * KSI, fty=56 * KSI, source="MIL-HDBK-5J, plate 0.250-2.000 in, A-basis LT"),
    "2024-T351 plate": Material(E=10.7e3 * KSI, nu=0.33, ftu=63 * KSI, fty=42 * KSI, source="MIL-HDBK-5J, plate 0.500-1.000 in, A-basis LT"),
    "6061-T651 plate": Material(E=9.9e3 * KSI, nu=0.33, ftu=42 * KSI, fty=35 * KSI, source="MIL-HDBK-5J, plate 0.250-2.000 in, A-basis LT"),
    "Ti-6Al-4V annealed": Material(E=16.0e3 * KSI, nu=0.31, ftu=130 * KSI, fty=120 * KSI, source="MIL-HDBK-5J, annealed plate, A-basis LT"),
}


class Rules(BaseModel):
    """Design-rule defaults (seeded from the researched rule table; editable per project). Lengths in mm."""

    material: str = "7075-T7351 plate"
    min_wall: float = 1.143  # 0.045 in
    min_wall_preferred: float = 1.524  # 0.060 in
    min_floor: float = 1.143
    bolt_edge_min: float = 1.5  # e/D
    bolt_edge_preferred: float = 2.0
    lug_edge_full_bearing: float = 1.5  # e/D
    lug_edge_min: float = 1.0
    lug_td_min: float = 0.2  # t/D
    pocket_depth_tool_max: float = 4.0  # depth / tool Ø
    wall_height_thickness_max: float = 15.0
    mass_target: float | None = None  # kg, soft
    ftu: float | None = None  # MPa: overrides the material table's ultimate tensile allowable
    fty: float | None = None  # MPa: overrides the yield allowable

    def material_props(self) -> "Material":
        m = MATERIAL_PROPS.get(self.material, MATERIAL_PROPS["7075-T7351 plate"])
        if self.ftu is None and self.fty is None:
            return m
        return m.model_copy(update={k: v for k, v in (("ftu", self.ftu), ("fty", self.fty)) if v is not None}
                            | {"source": m.source + " (overridden in the rules)"})


def resolved_support(intent: "Intent", o: "IntentObject") -> str:
    """The boundary condition an interface gets in analysis, with "auto" worked out."""
    s = getattr(o, "support", "free")
    if s != "auto":
        return s
    loaded = any(lc.kind == "load_case" and lc.target == o.id for lc in intent.objects)
    if o.kind == "bolt_pattern" and o.parent is not None and not loaded:
        return "fixed"
    return "free"


def face_part(o: "IntentObject") -> str | None:
    """The mating part an object is attached to through a picked face, if any."""
    face = getattr(o, "face", None)
    return face.part if face is not None else None


class Intent(BaseModel):
    objects: list[IntentObject] = Field(default_factory=list)
    rules: Rules = Field(default_factory=Rules)

    def next_id(self, kind: str) -> str:
        prefix = ID_PREFIX[kind]
        used = {int(o.id.split("-")[1]) for o in self.objects if o.id.startswith(prefix + "-")}
        n = 0 if prefix == "IF" else 1
        while n in used:
            n += 1
        return f"{prefix}-{n:02d}"

    def get(self, obj_id: str) -> IntentObject | None:
        return next((o for o in self.objects if o.id == obj_id), None)
