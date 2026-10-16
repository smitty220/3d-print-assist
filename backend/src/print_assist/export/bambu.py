"""Bambu Studio project 3MF: the design as one object whose parts are its bodies, each part set to its filament, with
the printer's project settings and the palette's filaments, so it opens in Bambu Studio ready to slice.

The layout follows what Bambu Studio itself saves (checked against a two-colour A1 project):
- `3D/3dmodel.model` holds one object made of components that point into `3D/Objects/object_1.model`, which holds
  one mesh per body;
- `Metadata/model_settings.config` names the object and its parts and gives each part its `extruder`, which is the
  1-based project filament it prints in;
- `Metadata/project_settings.config` holds the printer, process and filament settings. It comes from the printer's
  template (a project_settings.config that Bambu Studio saved for that printer), resized to the design's filaments:
  each filament takes the per-filament values of the template filament with the same material, plus its own colour.

Project filament N is palette slot N, so a body in slot 3 prints from filament 3; slots the palette skips are filled
with grey placeholders that no part uses.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from xml.sax.saxutils import quoteattr

from build123d import Shape

from ..features.model import Filament
from ..printers import PrinterProfile

LINEAR = 0.01  # mm, as for the plain 3MF
ANGULAR = 0.1
PLACEHOLDER = "#808080"

# Lists in the template that are not one entry per filament, though some have as many entries as it has filaments:
# machine limits (normal and silent mode), bed geometry, per-plate wipe tower positions, per-extruder AMS counts.
NOT_PER_FILAMENT = {
    "start_end_points", "head_wrap_detect_zone", "printable_area", "upward_compatible_machine", "wipe_tower_x",
    "wipe_tower_y", "extruder_ams_count", "bed_exclude_area", "wrapping_exclude_area", "different_settings_to_system",
    "flush_volumes_matrix",
}
NOT_PER_FILAMENT_PREFIXES = ("machine_",)


class BambuError(ValueError):
    pass


@dataclass
class Part:
    """One body to write: its name, shape (already placed on the bed) and 1-based filament (None: the object's)."""

    name: str
    shape: Shape
    filament: int | None


def load_template(printer: PrinterProfile) -> dict | None:
    path = printer.template_path()
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _per_filament(key: str, value, n: int) -> bool:
    return (isinstance(value, list) and n > 0 and len(value) >= n and len(value) % n == 0
            and key not in NOT_PER_FILAMENT and not key.startswith(NOT_PER_FILAMENT_PREFIXES))


def _luminance(color: str) -> float:
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


PROCESS_DEFAULTS = {"skirt_loops": "0"}


def flush_volume(src: str, dst: str) -> int:
    """mm³ purged changing from colour `src` to `dst`: little going darker, much more going lighter (black to white
    is about 620, white to black 140, as Bambu Studio suggests for those two)."""
    if src.lower() == dst.lower():
        return 0
    return round(140 + 480 * max(0.0, _luminance(dst) - _luminance(src)))


def palette(filaments: list[Filament], fallback_color: str = PLACEHOLDER) -> list[Filament]:
    """The project filaments, 1..highest slot: the palette's, with grey placeholders in the slots it skips. An empty
    palette gives one filament in `fallback_color`."""
    if not filaments:
        return [Filament(slot=1, color=fallback_color)]
    by_slot = {f.slot: f for f in filaments}
    return [by_slot.get(s) or Filament(slot=s, name="(unused)", color=PLACEHOLDER) for s in range(1, max(by_slot) + 1)]


def project_settings(template: dict, filaments: list[Filament], printer: PrinterProfile) -> dict:
    """The template's settings with one filament per project filament (see the module docstring)."""
    n_t = len(template.get("filament_colour") or [])
    if n_t == 0:
        raise BambuError("the printer's Bambu template has no filaments")
    types = [t.upper() for t in template.get("filament_type", [])]
    source: list[int] = []
    for f in filaments:
        mat = (f.material or "PLA").upper()
        match = next((i for i, t in enumerate(types) if t == mat), None)
        if match is None:
            match = next((i for i, t in enumerate(types) if mat.startswith(t)), None)  # "PLA Silk" -> PLA
        if match is None:
            have = ", ".join(sorted(set(types)))
            raise BambuError(f"slot {f.slot} is {f.material}, but the {printer.name} template only has {have} "
                             "filament settings; save a Bambu Studio project for this printer that uses "
                             f"{f.material} as the template")
        source.append(match)

    out = dict(template)
    for key, value in template.items():
        if not _per_filament(key, value, n_t):
            continue
        k = len(value) // n_t
        out[key] = [x for i in source for x in value[i * k:(i + 1) * k]]
    colors = [f.color.upper() for f in filaments]
    out["filament_colour"] = colors
    out["filament_multi_colour"] = colors
    out["filament_colour_type"] = ["1"] * len(filaments)
    out["filament_self_index"] = [str(i + 1) for i in range(len(filaments))]
    if len(printer.nozzles) > 1:  # which nozzle (extruder) feeds each filament, set by the design
        out["filament_map"] = [str(f.nozzle + 1) for f in filaments]
        out["filament_map_mode"] = "Manual"
    out["flush_volumes_matrix"] = [str(flush_volume(a, b)) for a in colors for b in colors]
    # Process settings the template's maker chose that a design should not inherit: a skirt (the one-layer loop
    # around the parts) is Bambu Studio's default off; the A1 primes the nozzle with its own purge line.
    for key, value in PROCESS_DEFAULTS.items():
        if key in out:
            out[key] = value
    diff = template.get("different_settings_to_system")
    if isinstance(diff, list) and len(diff) == n_t + 2:  # process, one per filament, printer
        out["different_settings_to_system"] = [diff[0], *(diff[1 + i] for i in source), diff[-1]]
    return out


def mesh(shape: Shape) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """An indexed triangle mesh of the shape, with the vertices that faces share merged so the mesh is closed."""
    verts, tris = shape.tessellate(LINEAR, ANGULAR)
    index: dict[tuple[float, float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    remap = []
    for v in verts:
        key = (round(v.X, 6), round(v.Y, 6), round(v.Z, 6))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        remap.append(index[key])
    faces = []
    for a, b, c in tris:
        t = (remap[a], remap[b], remap[c])
        if len(set(t)) == 3:
            faces.append(t)
    return points, faces


def _n(x: float) -> str:
    return f"{x:.9g}"


def _uuid(*parts: object) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "print-assist:" + ":".join(map(str, parts))))


NS = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
      'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
      'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"')
HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n'
CONTENT_TYPES = HEAD + """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>
"""
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"


def _rels(rows: list[tuple[str, str]]) -> str:
    items = "".join(f' <Relationship Target="{t}" Id="rel-{i}" Type="{ty}"/>\n' for i, (t, ty) in enumerate(rows, 1))
    return HEAD + f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n{items}</Relationships>\n'


def write_project(parts: list[Part], filaments: list[Filament], printer: PrinterProfile, template: dict, path: Path,
                  title: str, thumbnail: bytes | None = None) -> Path:
    """Write the Bambu Studio project. `parts` are on the bed already; `filaments` are the project filaments
    (`palette()`), and each part's `filament` is one of their slots."""
    if not parts:
        raise BambuError("nothing to export")
    for p in parts:
        if p.filament is not None and not 1 <= p.filament <= len(filaments):
            raise BambuError(f"{p.name} prints in filament {p.filament}, the project has {len(filaments)}")
    settings = project_settings(template, filaments, printer)
    version = str(template.get("version") or "02.08.02.61")

    # The object's origin is the centre of the bodies' bounding box, as Bambu Studio keeps it; the build item puts
    # it back where the bodies sit on the bed.
    boxes = [p.shape.bounding_box() for p in parts]
    lo = [min(getattr(b.min, a) for b in boxes) for a in "XYZ"]
    hi = [max(getattr(b.max, a) for b in boxes) for a in "XYZ"]
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]

    meshes = []
    body_xml = []
    for i, p in enumerate(parts, 1):
        points, faces = mesh(p.shape)
        meshes.append(len(faces))
        vx = "".join(f'     <vertex x="{_n(x - c[0])}" y="{_n(y - c[1])}" z="{_n(z - c[2])}"/>\n' for x, y, z in points)
        tx = "".join(f'     <triangle v1="{a}" v2="{b}" v3="{d}"/>\n' for a, b, d in faces)
        body_xml.append(f'  <object id="{i}" p:UUID="{_uuid(title, "mesh", i)}" type="model">\n   <mesh>\n'
                        f'    <vertices>\n{vx}    </vertices>\n    <triangles>\n{tx}    </triangles>\n   </mesh>\n  </object>\n')
    objects_model = (HEAD + f'<model unit="millimeter" xml:lang="en-US" {NS}>\n'
                     ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
                     + "".join(body_xml) + ' </resources>\n <build/>\n</model>\n')

    oid = len(parts) + 1
    comps = "".join(f'    <component p:path="/3D/Objects/object_1.model" objectid="{i}" '
                    f'p:UUID="{_uuid(title, "component", i)}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
                    for i in range(1, len(parts) + 1))
    today = date.today().isoformat()
    meta = {"Application": f"BambuStudio-{version}", "BambuStudio:3mfVersion": "1", "CreationDate": today,
            "ModificationDate": today, "Title": title}
    if thumbnail:
        meta |= {"Thumbnail_Middle": "/Metadata/plate_1.png", "Thumbnail_Small": "/Metadata/plate_1_small.png"}
    meta_xml = "".join(f' <metadata name="{k}">{_esc(v)}</metadata>\n' for k, v in meta.items())
    main_model = (HEAD + f'<model unit="millimeter" xml:lang="en-US" {NS}>\n' + meta_xml + ' <resources>\n'
                  f'  <object id="{oid}" p:UUID="{_uuid(title, "object")}" type="model">\n   <components>\n{comps}'
                  '   </components>\n  </object>\n </resources>\n'
                  f' <build p:UUID="{_uuid(title, "build")}">\n'
                  f'  <item objectid="{oid}" p:UUID="{_uuid(title, "item")}" '
                  f'transform="1 0 0 0 1 0 0 0 1 {_n(c[0])} {_n(c[1])} {_n(c[2])}" printable="1"/>\n'
                  ' </build>\n</model>\n')

    default = next((p.filament for p in parts if p.filament), 1)
    part_xml = []
    for i, (p, nf) in enumerate(zip(parts, meshes), 1):
        ext = f'      <metadata key="extruder" value="{p.filament}"/>\n' if p.filament else ""
        part_xml.append(f'    <part id="{i}" subtype="normal_part">\n'
                        f'      <metadata key="name" value={quoteattr(p.name)}/>\n'
                        '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                        f'{ext}      <mesh_stat face_count="{nf}" edges_fixed="0" degenerate_facets="0" '
                        'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n    </part>\n')
    thumbs = ('    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>\n' if thumbnail else "")
    model_settings = (HEAD + '<config>\n'
                      f'  <object id="{oid}">\n    <metadata key="name" value={quoteattr(title)}/>\n'
                      f'    <metadata key="extruder" value="{default}"/>\n    <metadata face_count="{sum(meshes)}"/>\n'
                      + "".join(part_xml) + '  </object>\n'
                      '  <plate>\n    <metadata key="plater_id" value="1"/>\n'
                      f'    <metadata key="plater_name" value={quoteattr(title)}/>\n'
                      '    <metadata key="locked" value="false"/>\n'
                      f'    <metadata key="filament_map_mode" value={quoteattr(str(settings.get("filament_map_mode", "Auto For Flush")))}/>\n'
                      + thumbs +
                      '    <model_instance>\n'
                      f'      <metadata key="object_id" value="{oid}"/>\n'
                      '      <metadata key="instance_id" value="0"/>\n'
                      f'      <metadata key="identify_id" value="{oid * 100}"/>\n'
                      '    </model_instance>\n  </plate>\n'
                      '  <assemble>\n'
                      f'   <assemble_item object_id="{oid}" instance_id="0" '
                      f'transform="1 0 0 0 1 0 0 0 1 {_n(c[0])} {_n(c[1])} {_n(c[2])}" offset="0 0 0" />\n'
                      '  </assemble>\n</config>\n')

    slice_info = (HEAD + '<config>\n  <header>\n    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
                  f'    <header_item key="X-BBL-Client-Version" value="{version}"/>\n  </header>\n</config>\n')
    root_rels = [("/3D/3dmodel.model", MODEL_REL)]
    if thumbnail:
        root_rels += [
            ("/Metadata/plate_1.png", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"),
            ("/Metadata/plate_1.png", "http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"),
            ("/Metadata/plate_1_small.png", "http://schemas.bambulab.com/package/2021/cover-thumbnail-small"),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", _rels(root_rels))
        z.writestr("3D/3dmodel.model", main_model)
        z.writestr("3D/_rels/3dmodel.model.rels", _rels([("/3D/Objects/object_1.model", MODEL_REL)]))
        z.writestr("3D/Objects/object_1.model", objects_model)
        z.writestr("Metadata/model_settings.config", model_settings)
        z.writestr("Metadata/project_settings.config", json.dumps(settings, indent=4, ensure_ascii=False))
        z.writestr("Metadata/slice_info.config", slice_info)
        if thumbnail:
            z.writestr("Metadata/plate_1.png", thumbnail)
            z.writestr("Metadata/plate_1_small.png", thumbnail)
    return path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class ReadPart:
    name: str
    filament: int  # 1-based, the object's when the part sets none
    points: list[tuple[float, float, float]]  # on the bed (the build item's placement applied)
    triangles: list[tuple[int, int, int]]

    @property
    def volume(self) -> float:
        """Signed volume of the closed mesh (positive when its triangles face outward)."""
        p = self.points
        return sum(_det(p[a], p[b], p[c]) for a, b, c in self.triangles) / 6

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return tuple(min(q[i] for q in self.points) for i in range(3)), tuple(max(q[i] for q in self.points) for i in range(3))

    def open_edges(self) -> int:
        """Edges not shared by exactly two triangles in opposite directions (0 for a closed, consistent mesh)."""
        count: dict[tuple[int, int], int] = {}
        for a, b, c in self.triangles:
            for e in ((a, b), (b, c), (c, a)):
                count[e] = count.get(e, 0) + 1
        return sum(1 for (a, b), k in count.items() if k != 1 or count.get((b, a)) != 1)


def _det(a, b, c) -> float:
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))


def read_project(path: Path) -> tuple[list[ReadPart], dict, dict[str, str]]:
    """The parts, project settings and model metadata of a project written by `write_project` (one object whose
    components point into one objects file, translation-only placements)."""
    import xml.etree.ElementTree as ET

    core = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
    prod = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"
    with zipfile.ZipFile(path) as z:
        main = ET.fromstring(z.read("3D/3dmodel.model"))
        settings = json.loads(z.read("Metadata/project_settings.config"))
        ms = ET.fromstring(z.read("Metadata/model_settings.config"))
        meta = {m.get("name"): m.text or "" for m in main.findall(f"{core}metadata")}
        item = main.find(f"{core}build/{core}item")
        t = [float(x) for x in item.get("transform").split()][9:12]
        obj = main.find(f"{core}resources/{core}object[@id='{item.get('objectid')}']")
        conf = ms.find(f"object[@id='{item.get('objectid')}']")
        default = int(conf.find("metadata[@key='extruder']").get("value"))
        parts = []
        for comp, pc in zip(obj.find(f"{core}components"), conf.findall("part")):
            sub = ET.fromstring(z.read(comp.get(f"{prod}path").lstrip("/")))
            mo = sub.find(f"{core}resources/{core}object[@id='{comp.get('objectid')}']")
            ct = [float(x) for x in comp.get("transform").split()][9:12]
            pts = [tuple(float(v.get(k)) + t[i] + ct[i] for i, k in enumerate("xyz"))
                   for v in mo.find(f"{core}mesh/{core}vertices")]
            tris = [(int(x.get("v1")), int(x.get("v2")), int(x.get("v3"))) for x in mo.find(f"{core}mesh/{core}triangles")]
            ext = pc.find("metadata[@key='extruder']")
            parts.append(ReadPart(pc.find("metadata[@key='name']").get("value"),
                                  int(ext.get("value")) if ext is not None else default, pts, tris))
    return parts, settings, meta
