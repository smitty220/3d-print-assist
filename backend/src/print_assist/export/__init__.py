"""Exports of a built design: 3MF (one object per body, placed on the printer's bed), STL and STEP.

When the printer has a Bambu Studio template and the design is given, the 3MF is a Bambu Studio project with each
body's filament and the palette's colours preassigned (`bambu.py`); otherwise it is a plain 3MF with coloured objects.
"""

from __future__ import annotations

import re
from pathlib import Path

from build123d import Color, Compound, Location, Mesher, Shape, Unit, export_step, export_stl

from ..features.model import Design
from ..printers import PrinterProfile

from . import bambu

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
           title: str = "", colors: dict[str, str | None] | None = None, design: Design | None = None) -> Path:
    """Write the bodies to `path`. A 3MF is a Bambu Studio project when `design` is given and the printer has a
    template; otherwise it has one object per body, named and coloured (`colors`: body -> "#rrggbb")."""
    if fmt not in FORMATS:
        raise ExportError(f"unknown format {fmt!r}; formats: {', '.join(FORMATS)}")
    if not parts:
        raise ExportError("nothing is built to export")
    ids = list(parts)
    shapes = on_bed([parts[b] for b in ids], printer)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "3mf" and design is not None and (template := bambu.load_template(printer)) is not None:
        return _bambu(ids, shapes, design, printer, template, path, title)
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


def _bambu(ids: list[str], shapes: list[Shape], design: Design, printer: PrinterProfile, template: dict, path: Path,
           title: str) -> Path:
    from ..geometry.render import render

    by_id = {b.id: b for b in design.bodies}
    colors = {bid: design.body_color(by_id[bid]) if bid in by_id else None for bid in ids}
    fallback = next((c for c in colors.values() if c), bambu.PLACEHOLDER)
    filaments = bambu.palette(design.filaments, fallback)
    parts = [bambu.Part(by_id[bid].name if bid in by_id else bid, shape, by_id[bid].filament if bid in by_id else None)
             for bid, shape in zip(ids, shapes)]
    try:
        thumb = render([(s, colors[b]) for b, s in zip(ids, shapes)], "iso", 512)
    except Exception:  # a thumbnail is a nicety; never fail the export over it
        thumb = None
    try:
        return bambu.write_project(parts, filaments, printer, template, path, title or "design", thumb)
    except bambu.BambuError as exc:
        raise ExportError(str(exc)) from exc
