"""Printer profiles: bed size, nozzles and filament slots, read from the JSON files beside this module."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

HERE = Path(__file__).parent


class Nozzle(BaseModel):
    dia: float = 0.4
    slots: int = 4  # filament slots that can feed this nozzle


class PrinterProfile(BaseModel):
    id: str
    name: str
    bed: tuple[float, float, float]  # mm, X Y Z
    nozzles: list[Nozzle]

    @property
    def slots(self) -> int:
        return sum(n.slots for n in self.nozzles)


def profiles() -> dict[str, PrinterProfile]:
    out = {}
    for path in sorted(HERE.glob("*.json")):
        p = PrinterProfile.model_validate_json(path.read_text())
        out[p.id] = p
    return out


DEFAULT_PRINTER = "bambu_a1_ams_lite"
