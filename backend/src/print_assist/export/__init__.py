"""Exports of a built design: 3MF (one object per body, placed on the printer's bed), STL and STEP.

The Bambu Studio project 3MF (filament slots and colours preassigned) is built in `bambu.py` from a reference project
saved by Bambu Studio; until that reference is in place the plain 3MF opens in Bambu Studio with colours to assign.
"""

from __future__ import annotations

import re
from pathlib import Path

from build123d import Color, Compound, Location, Mesher, Shape, Unit, export_step, export_stl

from ..printers import PrinterProfile

FORMATS = ("3mf", "stl", "step")
LINEAR = 0.01  # mm, mesh deflection: well under a 0.2 mm layer
ANGULAR = 0.1  # rad


class ExportError(ValueError):
    pass


def on_bed(shapes: list[Shape], printer: PrinterProfile) -> list[Shape]:
    """Move the bodies together so they sit on the bed (lowest point at Z = 0) centred on the plate."""
    whole = shapes[0] if len(shapes) == 1 else Compound(children=shapes)
    bb = whole.bounding_box()
    dx = printer.bed[0] / 2 - (bb.min.X + bb.max.X) / 2
    dy = printer.bed[1] / 2 - (bb.min.Y + bb.max.Y) / 2
    loc = Location((dx, dy, -bb.min.Z))
    return [loc * s for s in shapes]


def filename(name: str, fmt: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or "design"
    return f"{stem}.{fmt}"


def export(parts: dict[str, Shape], names: dict[str, str], printer: PrinterProfile, fmt: str, path: Path,
           title: str = "", colors: dict[str, str | None] | None = None) -> Path:
    """Write the bodies to `path`. A 3MF has one object per body, named and coloured (`colors`: body -> "#rrggbb")."""
    if fmt not in FORMATS:
        raise ExportError(f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}")
    if not parts:
        raise ExportError("nothing is built to export")
    ids = list(parts)
    shapes = on_bed([parts[b] for b in ids], printer)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "3mf":
        m = Mesher(unit=Unit.MM)
        for bid, shape in zip(ids, shapes):
            shape.label = names.get(bid, bid)
            if (colors or {}).get(bid):
                shape.color = Color(colors[bid])
            m.add_shape(shape, linear_deflection=LINEAR, angular_deflection=ANGULAR)
        m.add_meta_data("", "Title", title or "3D Print Assist design", "xs:string", False)
        m.add_meta_data("", "Application", "3D Print Assist", "xs:string", False)
        m.write(str(path))
    elif fmt == "stl":
        whole = shapes[0] if len(shapes) == 1 else Compound(children=shapes)
        if not export_stl(whole, str(path), tolerance=LINEAR, angular_tolerance=ANGULAR):
            raise ExportError("STL export failed")
    else:
        whole = shapes[0] if len(shapes) == 1 else Compound(children=shapes)
        whole.label = title or "design"
        if not export_step(whole, str(path)):
            raise ExportError("STEP export failed")
    return path
