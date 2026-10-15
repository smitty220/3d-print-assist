"""References: files kept beside a design to look at, measure and copy features from (never built or exported).

`placed()` reads a reference's file and puts it where the design shows it; `summary()` describes it for the tree, the
AI and the API: its parts with their sizes, so "the lid's hinge" can be found and measured.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features.model import Reference
from .geometry.meshio import MeshFile, MeshPart, read_mesh


def transform(ref: Reference, mesh: MeshFile) -> np.ndarray:
    """The 4x4 placement: scale, turn about Z, then (on_bed) centre on the origin with the lowest point on the bed,
    then move."""
    m = np.eye(4)
    m[:3, :3] *= ref.scale
    a = math.radians(ref.rotate_z)
    rz = np.eye(4)
    rz[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    m = rz @ m
    if ref.place == "on_bed":
        lo, hi = MeshFile([p.moved(m) for p in mesh.parts]).bounds()
        t = np.eye(4)
        t[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
        m = t @ m
    mv = np.eye(4)
    mv[:3, 3] = [ref.x, ref.y, ref.z]
    return mv @ m


def placed(ref: Reference, folder: Path) -> MeshFile:
    mesh = read_mesh(folder / ref.file, ref.plate)
    m = transform(ref, mesh)
    return MeshFile([p.moved(m) for p in mesh.parts], mesh.plates, mesh.plate)


def pick_parts(mesh: MeshFile, part: str | None) -> list[MeshPart]:
    """The parts named `part` (exactly, or the one part whose name contains it), or all of them."""
    if not part:
        return mesh.parts
    exact = [p for p in mesh.parts if p.name == part]
    if exact:
        return exact
    near = [p for p in mesh.parts if part.lower() in p.name.lower()]
    if not near:
        raise KeyError(f"no part {part!r}; parts: {', '.join(p.name for p in mesh.parts)}")
    return near


def _box(lo, hi) -> dict:
    return {"min": np.round(lo, 3).tolist(), "max": np.round(hi, 3).tolist(), "size": np.round(hi - lo, 3).tolist()}


def summary(ref: Reference, folder: Path, parts: bool = True) -> dict:
    out = ref.model_dump()
    try:
        mesh = placed(ref, folder)
    except Exception as exc:  # noqa: BLE001 - a missing or unreadable file is reported, not raised
        return out | {"error": str(exc)}
    lo, hi = mesh.bounds()
    out |= {"triangles": mesh.triangles, "plates": mesh.plates, "bounds": _box(lo, hi), "part_count": len(mesh.parts)}
    if parts:
        rows = []
        for p in mesh.parts:
            plo, phi = p.bounds()
            vol = p.volume()
            rows.append({"name": p.name, "triangles": len(p.tris), **_box(plo, phi),
                         "volume": None if vol is None else round(vol, 2)})
        out["parts"] = rows
    return out
