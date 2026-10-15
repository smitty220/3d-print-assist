"""Files from elsewhere: STEP as an exact solid (the Import feature), and STEP / STL / 3MF as reference meshes the user
and the AI look at, measure and slice, but never print (`Design.references`).

Reference meshes stay triangles (nothing is sewn into a solid), so any size loads quickly. 3MF parts keep their
names and placement: build items and nested components with their transforms (including Bambu Studio's components in
other model files), in the file's unit converted to millimetres, and only one plate of a multi-plate Bambu project.
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from build123d import Compound, Shape, import_step

STEP = (".step", ".stp")
MESH = (".stl", ".3mf")
FORMATS = STEP + MESH
CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
PROD = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"
UNITS = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}


class ImportError_(ValueError):
    pass


# ---------------------------------------------------------------- STEP solids (the Import feature)

_solids: dict[tuple[str, float], Shape] = {}


def load_step(path: Path) -> Shape:
    """The STEP file's solids as one shape (a compound when there are several). Kept in memory per file and
    modification time."""
    if path.suffix.lower() not in STEP:
        raise ImportError_(f"{path.name}: the Import feature takes STEP files; add meshes (STL, 3MF) as references")
    if not path.exists():
        raise ImportError_(f"{path.name} is missing from the design's references folder")
    key = (str(path), path.stat().st_mtime)
    if key not in _solids:
        solids = import_step(str(path)).solids()
        if not solids:
            raise ImportError_(f"{path.name} has no solids")
        if len(_solids) > 8:
            _solids.clear()
        _solids[key] = solids[0] if len(solids) == 1 else Compound(children=list(solids))
    return _solids[key]


# ---------------------------------------------------------------- reference meshes


@dataclass
class MeshPart:
    name: str
    points: np.ndarray  # (n, 3) float64, mm
    tris: np.ndarray  # (m, 3) int64

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(0), self.points.max(0)

    def volume(self) -> float | None:
        """Volume enclosed, if the mesh is closed (every edge shared by two triangles); else None."""
        e = np.sort(np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        if not (counts == 2).all():
            return None
        t = self.points[self.tris]
        return abs(float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6))

    def moved(self, m: np.ndarray) -> MeshPart:
        """The part through a 4x4 transform (column vectors)."""
        return MeshPart(self.name, self.points @ m[:3, :3].T + m[:3, 3], self.tris)


@dataclass
class MeshFile:
    parts: list[MeshPart]
    plates: list[str] = field(default_factory=list)  # a Bambu project's plate names; empty otherwise
    plate: int | None = None  # the plate read (1-based)

    @property
    def triangles(self) -> int:
        return sum(len(p.tris) for p in self.parts)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([p.bounds()[0] for p in self.parts], axis=0)
        hi = np.max([p.bounds()[1] for p in self.parts], axis=0)
        return lo, hi


_meshes: dict[tuple[str, float, int | None], MeshFile] = {}


def read_mesh(path: Path, plate: int | None = None) -> MeshFile:
    """A reference file as triangle meshes, one part per solid (STEP), mesh object (3MF) or file (STL)."""
    ext = path.suffix.lower()
    if ext not in FORMATS:
        raise ImportError_(f"can't read {path.suffix or 'that'} files; formats: STEP, STL, 3MF")
    if not path.exists():
        raise ImportError_(f"{path.name} is missing from the design's references folder")
    key = (str(path), path.stat().st_mtime, plate)
    if key not in _meshes:
        if len(_meshes) > 8:
            _meshes.clear()
        if ext in STEP:
            out = MeshFile([_step_part(s, f"solid {i + 1}") for i, s in enumerate(load_step(path).solids())])
        elif ext == ".stl":
            out = MeshFile([read_stl(path)])
        else:
            out = read_3mf(path, plate)
        if not out.parts:
            raise ImportError_(f"{path.name} has nothing in it" + (f" on plate {plate}" if plate else ""))
        _meshes[key] = out
    return _meshes[key]


def _step_part(solid: Shape, name: str) -> MeshPart:
    verts, tris = solid.tessellate(0.02, 0.2)
    return _merged(name, np.array([[v.X, v.Y, v.Z] for v in verts]), np.array(tris, dtype=np.int64).reshape(-1, 3))


def read_stl(path: Path) -> MeshPart:
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:  # binary
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n,
                                offset=84)
            return _merged(path.stem, rec["v"].reshape(-1, 3).astype(np.float64), np.arange(3 * n).reshape(-1, 3))
    text = data.decode("ascii", errors="replace")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not nums or len(nums) % 3:
        raise ImportError_(f"{path.name} is not a readable STL")
    v = np.array(nums, dtype=np.float64)
    return _merged(path.stem, v, np.arange(len(v)).reshape(-1, 3))


def _merged(name: str, points: np.ndarray, tris: np.ndarray) -> MeshPart:
    """Merge coincident vertices (STL repeats them per triangle) and drop degenerate triangles."""
    uniq, inv = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    t = inv.reshape(-1)[tris]
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return MeshPart(name, uniq, t[ok])


def _affine(s: str | None) -> np.ndarray:
    """A 3MF transform ("m00 m01 m02 m10 ... m32", row vectors) as a 4x4 matrix for column vectors."""
    m = np.eye(4)
    if s:
        v = [float(x) for x in s.split()]
        if len(v) != 12:
            raise ImportError_("bad transform in the 3MF")
        m[:3, :3] = np.array(v[:9]).reshape(3, 3).T
        m[:3, 3] = v[9:]
    return m


def read_3mf(path: Path, plate: int | None = None) -> MeshFile:
    """One part per placed mesh object (each part of a multi-part object separately), in millimetres, with every
    component's and the build item's transform applied. A Bambu project with plates gives the parts of one plate (the
    first unless `plate` says)."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        main = _model_path(z, names)
        docs: dict[str, ET.Element] = {}

        def doc(p: str) -> ET.Element:
            p = p.lstrip("/")
            if p not in docs:
                if p not in names:
                    raise ImportError_(f"the 3MF refers to {p}, which is not in it")
                docs[p] = ET.fromstring(z.read(p))
            return docs[p]

        root = doc(main)
        scale = UNITS.get(root.get("unit", "millimeter"), 1.0)
        plates, on_plate, part_names, object_names = _bambu_settings(z, names)
        chosen = None
        if plates:
            chosen = plate or 1
            if not 1 <= chosen <= len(plates):
                raise ImportError_(f"{path.name} has {len(plates)} plates, not {chosen}")
        parts: list[MeshPart] = []

        def collect(model: str, oid: str, m: np.ndarray, top: str, depth: int = 0) -> None:
            if depth > 16:
                raise ImportError_("the 3MF's components nest too deeply")
            obj = doc(model).find(f"{CORE}resources/{CORE}object[@id='{oid}']")
            if obj is None:
                raise ImportError_(f"the 3MF has no object {oid} in {model}")
            mesh = obj.find(f"{CORE}mesh")
            if mesh is not None:
                pts = np.array([(float(v.get("x")), float(v.get("y")), float(v.get("z")))
                                for v in mesh.find(f"{CORE}vertices")], dtype=np.float64).reshape(-1, 3)
                tris = np.array([(int(t.get("v1")), int(t.get("v2")), int(t.get("v3")))
                                 for t in mesh.find(f"{CORE}triangles")], dtype=np.int64).reshape(-1, 3)
                if len(tris):
                    pts = (pts @ m[:3, :3].T + m[:3, 3]) * scale
                    name = (part_names.get((top, oid)) or obj.get("name") or object_names.get(top)
                            or f"part {len(parts) + 1}")
                    parts.append(_merged(name, pts, tris))
            comps = obj.find(f"{CORE}components")
            for c in comps if comps is not None else ():
                collect(c.get(f"{PROD}path") or model, c.get("objectid"), m @ _affine(c.get("transform")), top,
                        depth + 1)

        for item in root.findall(f"{CORE}build/{CORE}item"):
            oid = item.get("objectid")
            if item.get("printable") == "0" or (chosen and oid not in on_plate.get(chosen, set())):
                continue
            collect(main, oid, _affine(item.get("transform")), oid)
        # Several parts with one name (copies of an object): number them.
        seen: dict[str, int] = {}
        for p in parts:
            seen[p.name] = seen.get(p.name, 0) + 1
        count: dict[str, int] = {}
        for p in parts:
            if seen[p.name] > 1:
                count[p.name] = count.get(p.name, 0) + 1
                p.name = f"{p.name} #{count[p.name]}"
        return MeshFile(parts, plates, chosen)


def _bambu_settings(z: zipfile.ZipFile, names: set[str]):
    """From a Bambu project's model_settings.config: plate names, each plate's objects, part names by (object, part)
    and object names. Empty for other 3MF files."""
    plates: list[str] = []
    on_plate: dict[int, set[str]] = {}
    part_names: dict[tuple[str, str], str] = {}
    object_names: dict[str, str] = {}
    p = "Metadata/model_settings.config"
    if p not in names:
        return plates, on_plate, part_names, object_names
    root = ET.fromstring(z.read(p))
    for obj in root.findall("object"):
        oid = obj.get("id")
        nm = obj.find("metadata[@key='name']")
        if nm is not None:
            object_names[oid] = nm.get("value", "")
        for part in obj.findall("part"):
            pn = part.find("metadata[@key='name']")
            if pn is not None:
                part_names[(oid, part.get("id"))] = pn.get("value", "")
    for i, pl in enumerate(root.findall("plate"), 1):
        nm = pl.find("metadata[@key='plater_name']")
        plates.append((nm.get("value") if nm is not None else "") or f"Plate {i}")
        on_plate[i] = {mi.find("metadata[@key='object_id']").get("value") for mi in pl.findall("model_instance")
                       if mi.find("metadata[@key='object_id']") is not None}
    return plates, on_plate, part_names, object_names


def _model_path(z: zipfile.ZipFile, names: set[str]) -> str:
    rels = "_rels/.rels"
    if rels in names:
        for r in ET.fromstring(z.read(rels)):
            if r.get("Type", "").endswith("/3dmodel"):
                return r.get("Target", "").lstrip("/")
    return "3D/3dmodel.model"


# ---------------------------------------------------------------- sections


def section(parts: list[MeshPart], axis: str, offset: float, tol: float = 0.05) -> list[dict]:
    """Where the plane `axis` = `offset` cuts the meshes: closed outlines in the plane's two other coordinates
    (Z: x, y; Y: x, z; X: y, z), simplified to within `tol` mm, largest first. Outlines close to a circle also give
    its centre and radius."""
    k = "XYZ".index(axis.upper())
    u, v = [i for i in range(3) if i != k]
    loops = []
    for part in parts:
        for loop in _chain(_cut(part, k, offset)):
            pts = _simplify(loop, tol)
            if len(pts) < 3:
                continue
            xy = np.array([[p[u], p[v]] for p in pts])
            ring = np.vstack([xy, xy[:1]])
            area = 0.5 * float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
            out = {"part": part.name, "points": np.round(xy, 3).tolist(), "area": round(abs(area), 3),
                   "length": round(float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()), 3)}
            c = _circle(np.array([[p[u], p[v]] for p in loop]))
            if c:
                out["circle"] = c
            loops.append(out)
    loops.sort(key=lambda o: -o["area"])
    return loops


def _cut(part: MeshPart, k: int, h: float) -> list[tuple[np.ndarray, np.ndarray]]:
    t = part.points[part.tris]  # (m, 3, 3)
    d = t[..., k] - h
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)  # a vertex on the plane counts as just above it
    side = d > 0
    cross = side.any(1) & ~side.all(1)
    segs = []
    for tri, dd, s in zip(t[cross], d[cross], side[cross]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if s[a] != s[b]:
                w = dd[a] / (dd[a] - dd[b])
                pts.append(tri[a] + w * (tri[b] - tri[a]))
        if len(pts) == 2:
            segs.append((pts[0], pts[1]))
    return segs


def _chain(segs: list[tuple[np.ndarray, np.ndarray]]) -> list[list[np.ndarray]]:
    """Join segments end to end into loops (rounded endpoints match)."""
    def key(p):
        return (round(p[0], 5), round(p[1], 5), round(p[2], 5))

    nbr: dict[tuple, list[int]] = {}
    for i, (a, b) in enumerate(segs):
        nbr.setdefault(key(a), []).append(i)
        nbr.setdefault(key(b), []).append(i)
    used = [False] * len(segs)
    loops = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        a, b = segs[i]
        loop = [a, b]
        start, cur = key(a), key(b)
        while cur != start:
            nxt = next((j for j in nbr.get(cur, []) if not used[j]), None)
            if nxt is None:
                break
            used[nxt] = True
            p, q = segs[nxt]
            far = q if key(p) == cur else p
            loop.append(far)
            cur = key(far)
        if cur == start:
            loop.pop()
        loops.append(loop)
    return loops


def _simplify(loop: list[np.ndarray], tol: float) -> list[np.ndarray]:
    """Drop points that lie within `tol` of the line through their neighbours (repeated until none do)."""
    pts = list(loop)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        i = 0
        while i < len(pts) and len(pts) > 3:
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            ac = c - a
            n = np.linalg.norm(ac)
            dist = np.linalg.norm(np.cross(b - a, ac)) / n if n > 1e-12 else np.linalg.norm(b - a)
            if dist < tol:
                pts.pop(i)
                changed = True
            else:
                i += 1
    return pts


def _circle(xy: np.ndarray) -> dict | None:
    """Centre and radius when the outline is a circle (every point within 2% of the fitted radius)."""
    if len(xy) < 8:
        return None
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(a, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    if r <= 0 or np.abs(np.hypot(x - cx, y - cy) - r).max() > 0.02 * r:
        return None
    return {"center": [round(float(cx), 3), round(float(cy), 3)], "radius": round(r, 3)}
