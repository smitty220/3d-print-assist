"""Print rules: the limits every feature check and part check reads. Lengths in mm."""

from __future__ import annotations

from pydantic import BaseModel

MATERIALS: dict[str, float] = {  # density, kg/m³
    "PLA": 1240.0,
    "PETG": 1270.0,
    "ABS": 1040.0,
    "ASA": 1070.0,
    "TPU": 1210.0,
}


class PrintRules(BaseModel):
    material: str = "PLA"
    nozzle: float = 0.4
    layer: float = 0.2
    min_wall: float = 0.8  # two perimeters of a 0.4 mm nozzle
    min_wall_preferred: float = 1.2
    min_text_stroke: float = 0.8
    min_text_height: float = 5.0
    max_overhang: float = 45.0  # degrees from vertical

    @property
    def density(self) -> float:
        return MATERIALS.get(self.material, MATERIALS["PLA"])
