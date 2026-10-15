"""Fonts for the Text feature: the open-licence fonts bundled with the app (each with its licence file, all allowing
use in products you sell), plus any font dropped into a design's own `fonts/` folder.

Font files are found by name: `<Family>-<Style>.ttf` / `.otf`, the style one of Regular, Bold, Italic, BoldItalic
(e.g. `Lato-BoldItalic.ttf`). A family without a bold or italic file uses its regular one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BUNDLED = Path(__file__).parent / "bundled"
STYLES = ("Regular", "Bold", "Italic", "BoldItalic")
DEFAULT_FONT = "Bebas Neue"


@dataclass
class FontFamily:
    name: str
    files: dict[str, Path] = field(default_factory=dict)  # style -> file
    license: str = ""
    source: str = "bundled"

    def file(self, bold: bool = False, italic: bool = False) -> Path:
        style = ("Bold" if bold else "") + ("Italic" if italic else "") or "Regular"
        return self.files.get(style) or self.files.get("Bold" if bold else "Regular") or next(iter(self.files.values()))

    def to_dict(self) -> dict:
        return {"name": self.name, "styles": sorted(self.files), "license": self.license, "source": self.source}


def _display(stem: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)  # "BebasNeue" -> "Bebas Neue"


def _scan(folder: Path, source: str, out: dict[str, FontFamily]) -> None:
    if not folder.exists():
        return
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in (".ttf", ".otf"):
            continue
        stem, _, style = path.stem.partition("-")
        style = style if style in STYLES else "Regular"
        name = _display(stem)
        fam = out.setdefault(name, FontFamily(name, source=source))
        fam.files.setdefault(style, path)
        lic = next((p for p in path.parent.iterdir() if p.name.upper().startswith(("OFL", "LICENSE"))), None)
        if lic and not fam.license:
            fam.license = "SIL Open Font License 1.1" if lic.name.upper().startswith("OFL") else lic.name


def families(project_fonts: Path | None = None) -> dict[str, FontFamily]:
    out: dict[str, FontFamily] = {}
    _scan(BUNDLED, "bundled", out)
    if project_fonts is not None:
        _scan(Path(project_fonts), "design", out)
    return out
