"""Exports of a built design: 3MF (one object per body, placed on the printer's bed), STL and STEP.

When the printer has a Bambu Studio template and the design is given, the 3MF is a Bambu Studio project with each
body's filament and the palette's colours preassigned (`bambu.py`); otherwise it is a plain 3MF with coloured objects.
A design with a print layout (`layout.py`) exports its Bambu project laid out on its plates; the other formats stay
assembled.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "3mf" and design is not None and (template := bambu.load_template(printer)) is not None:
        if design.layout:
            return _bambu_layout(parts, design, printer, template, path, title)
        return _bambu(ids, on_bed([parts[b] for b in ids], printer), design, printer, template, path, title)
    shapes = on_bed([parts[b] for b in ids], printer)
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
        if len(shapes) > 1:
            _group_3mf_components(path, title or "3D Print Assist design")
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


def _group_3mf_components(path: Path, title: str) -> None:
    """Wrap multiple top-level `<build><item>`s into one multi-component `<object>` so slicers (e.g. PrusaSlicer)
    treat all bodies as one multi-material assembly and don't drop elevated inlays/parts down to Z = 0."""
    import xml.etree.ElementTree as ET
    import zipfile

    ns_core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ns_prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
    ET.register_namespace("", ns_core)
    ET.register_namespace("p", ns_prod)
    ET.register_namespace("m", "http://schemas.microsoft.com/3dmanufacturing/material/2015/02")

    with zipfile.ZipFile(path, "r") as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]

    updated = False
    out_entries: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename != "3D/3dmodel.model":
            out_entries.append((info, data))
            continue
        root = ET.fromstring(data)
        resources = root.find(f"{{{ns_core}}}resources")
        build = root.find(f"{{{ns_core}}}build")
        if resources is None or build is None:
            out_entries.append((info, data))
            continue
        items = list(build.findall(f"{{{ns_core}}}item"))
        if len(items) < 2:
            out_entries.append((info, data))
            continue
        max_id = max(
            (int(el.attrib["id"]) for el in resources.findall(f"{{{ns_core}}}object") if el.attrib.get("id", "").isdigit()),
            default=0,
        )
        asm_id = str(max_id + 1)
        asm = ET.SubElement(resources, f"{{{ns_core}}}object", {"id": asm_id, "name": title, "type": "model"})
        comps = ET.SubElement(asm, f"{{{ns_core}}}components")
        for it in items:
            attrs = {"objectid": it.attrib["objectid"]}
            if "transform" in it.attrib:
                attrs["transform"] = it.attrib["transform"]
            ET.SubElement(comps, f"{{{ns_core}}}component", attrs)
            build.remove(it)
        ET.SubElement(build, f"{{{ns_core}}}item", {"objectid": asm_id})
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        updated = True
        out_entries.append((info, data))

    if updated:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info, data in out_entries:
                zout.writestr(info.filename, data)


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
        from ..geometry.tessellate import triangles

        meshes = [(*triangles(s, 0.05, 0.3), colors[b]) for b, s in zip(ids, shapes)]  # a picture: coarse is fine
        thumb = render([], "iso", 512, meshes=meshes)
    except Exception:  # a thumbnail is a nicety; never fail the export over it
        thumb = None
    try:
        return bambu.write_project(parts, filaments, printer, template, path, title or "design", thumb)
    except bambu.BambuError as exc:
        raise ExportError(str(exc)) from exc


def _bambu_layout(parts: dict[str, Shape], design: Design, printer: PrinterProfile, template: dict, path: Path,
                  title: str) -> Path:
    from ..geometry.render import render
    from .layout import resolve

    lay = resolve(design, parts, (printer.bed[0], printer.bed[1]))
    if lay.problems:
        raise ExportError("the print layout is not ready: " + "; ".join(lay.problems[:6])
                          + (f" (and {len(lay.problems) - 6} more)" if len(lay.problems) > 6 else ""))
    by_id = {b.id: b for b in design.bodies}
    filaments = bambu.palette(design.filaments, next((design.body_color(b) for b in design.bodies
                                                       if design.body_color(b)), bambu.PLACEHOLDER))
    objects = [bambu.Obj(p.item.name, [bambu.Part(by_id[b].name if b in by_id else b, s,
                                                  by_id[b].filament if b in by_id else None) for b, s in p.parts],
                         p.item.plate, dict(p.item.settings)) for p in lay.placed]
    plates = [bambu.PlateInfo(pl.name, pl.tower) for pl in design.plates]
    try:
        from ..geometry.tessellate import triangles

        first = [(b, s) for p in lay.placed if p.item.plate == 1 for b, s in p.parts]
        meshes = [(*triangles(s, 0.05, 0.3), design.body_color(by_id[b]) if b in by_id else None) for b, s in first]
        thumb = render([], "iso", 512, meshes=meshes) if meshes else None
    except Exception:  # a thumbnail is a nicety; never fail the export over it
        thumb = None
    try:
        return bambu.write_project([], filaments, printer, template, path, title or "design", thumb, objects, plates)
    except bambu.BambuError as exc:
        raise ExportError(str(exc)) from exc
