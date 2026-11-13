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
    # Bambu Studio's project settings for this printer (a project_settings.config saved by Bambu Studio, relative to
    # this folder); with it, 3MF exports are Bambu Studio projects with the filaments preassigned.
    bambu_template: str | None = None

    @property
    def slots(self) -> int:
        return sum(n.slots for n in self.nozzles)

    def template_path(self) -> Path | None:
        return HERE / self.bambu_template if self.bambu_template else None


def _custom_dir() -> Path:
    from ..project import data_root

    return data_root() / "printers"


def profiles() -> dict[str, PrinterProfile]:
    out = {}
    for folder in (HERE, _custom_dir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            p = PrinterProfile.model_validate_json(path.read_text(encoding="utf-8"))
            out[p.id] = p
    return out


def save_custom_profile(profile: PrinterProfile) -> PrinterProfile:
    d = _custom_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{profile.id}.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


DEFAULT_PRINTER = "bambu_a1_ams_lite"
