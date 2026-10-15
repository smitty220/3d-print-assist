"""Tet10 meshes of a solid with Gmsh: second-order nodes on the true (curved) surfaces, sizes from curvature so bores and
fillets refine themselves, plus optional local refinement near given axes (pin bores carry the peak stresses)."""

from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from build123d import Shape, export_brep

_GMSH = threading.Lock()  # Gmsh keeps global state: one mesh at a time


@dataclass
class Mesh:
    nodes: np.ndarray  # (N, 3) mm
    tets: np.ndarray  # (M, 10) node indices, Gmsh order
    tris: np.ndarray  # (F, 6) boundary triangles
    tri_surface: np.ndarray  # (F,) the Gmsh surface each triangle lies on
    surfaces: dict[int, str] = field(default_factory=dict)  # Gmsh surface tag -> geometry type ("Plane", "Cylinder", ...)
    quality: np.ndarray | None = None  # (M,) scaled Jacobian (Gmsh minSICN) per tetrahedron: 1 ideal, <= 0 inverted
    notes: list[str] = field(default_factory=list)  # controls Gmsh could not apply, fallbacks taken

    @property
    def n_dof(self) -> int:
        return 3 * len(self.nodes)

    def corner_nodes(self) -> np.ndarray:
        return np.unique(self.tets[:, :4])


@dataclass
class Refine:
    """Finer elements within `radius` of the line through `origin` along `axis` (a pin bore and its lug)."""

    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    size: float


def mesh_shape(shape: Shape, size: float, curvature: int = 24, refine: list[Refine] | None = None, min_size: float | None = None,
               log=None, edges: list[dict] | None = None, faces: list[dict] | None = None, spheres: list[dict] | None = None) -> Mesh:
    """Mesh a solid. `size` is the largest element edge (mm); `curvature` elements per full circle on curved faces.

    Local controls (fem/controls.py), each found in Gmsh by its centre of mass: `edges` [{ref, length, n}] elements along
    an edge, `faces` [{ref, area}] mapped (structured) faces, `spheres` [{center, radius, size}] spheres of influence.

    Gmsh runs in a child process: it can abort the whole process on some geometries (a C++ exception), and that must
    not take the server down. Curved second-order elements first; if any comes out inverted, again with Gmsh's
    high-order optimisation, then with straight-sided quadratic elements (which cannot invert)."""
    import multiprocessing as mp

    with _GMSH, tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "part.brep"
        export_brep(shape, str(brep))
        attempts = [("curved", {"linear": 0, "optimize": 0}), ("curved, optimised", {"linear": 0, "optimize": 1}),
                    ("straight-sided", {"linear": 1, "optimize": 0})]
        if faces:
            attempts.append(("straight-sided, without mapped faces", {"linear": 1, "optimize": 0, "nomap": 1}))
        last = "no attempt"
        notes: list[str] = []
        for name, opts in attempts:
            out = Path(tmp) / f"mesh_{len(notes)}_{name.replace(' ', '_').replace(',', '')}.npz"
            controls = {"edges": edges or [], "faces": [] if opts.get("nomap") else faces or [], "spheres": spheres or []}
            args = (str(brep), str(out), size, curvature, [r.__dict__ for r in refine or []],
                    min_size if min_size is not None else size / 20, opts["linear"], opts["optimize"], controls)
            proc = mp.get_context("spawn").Process(target=_mesh_child, args=args, daemon=True)
            proc.start()
            proc.join(600)
            if proc.is_alive():
                proc.terminate()
                last = f"{name}: Gmsh timed out"
                continue
            if proc.exitcode != 0 or not out.exists():
                last = f"{name}: Gmsh failed (exit {proc.exitcode})"
                notes.append(f"Meshing with {name} elements failed")
                if log:
                    log(f"meshing ({name}) failed; trying again")
                continue
            z = np.load(out, allow_pickle=True)
            mesh = Mesh(z["nodes"], z["tets"], z["tris"], z["tri_surface"], dict(z["surfaces"].item()), z["quality"],
                        notes + [str(x) for x in z["notes"]])
            bad = inverted(mesh)
            if bad:
                last = f"{name}: {bad} inverted element(s)"
                notes.append(f"{bad} inverted element(s) with {name} elements")
                if log:
                    log(f"{bad} inverted element(s) with {name} elements; trying again")
                continue
            if name != "curved":
                mesh.notes.append(f"Used {name} elements")
            return mesh
        raise RuntimeError(f"Could not mesh the part ({last})")


def inverted(mesh: Mesh) -> int:
    """Elements with a non-positive Jacobian at any Gauss point."""
    from .element import DN_GAUSS

    bad = np.zeros(len(mesh.tets), bool)
    coords = mesh.nodes[mesh.tets]
    for g in range(4):
        J = np.einsum("mni,nj->mij", coords, DN_GAUSS[g])
        bad |= np.linalg.det(J) <= 0
    return int(bad.sum())


def _match(gmsh, dim: int, ref, measure: float) -> int | None:
    """The Gmsh entity of dimension `dim` whose centre of mass is nearest `ref` and whose length / area matches."""
    best, bd = None, float("inf")
    for _d, tag in gmsh.model.getEntities(dim):
        c = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag))
        m = gmsh.model.occ.getMass(dim, tag)
        d = float(np.linalg.norm(c - np.asarray(ref, float)))
        if abs(m - measure) < 0.02 * measure + 1e-3 and d < bd:
            best, bd = tag, d
    return best if bd < max(0.05, 0.02 * measure ** (1 / dim)) + 1e-6 else None


def _apply_controls(gmsh, controls: dict, size: float, notes: list[str]) -> list[int]:
    """Transfinite curves and surfaces, and Ball fields for the spheres. Returns the size fields made."""
    done: dict[int, int] = {}
    for i, e in enumerate(controls["edges"]):
        tag = _match(gmsh, 1, e["ref"], e["length"])
        if tag is None:
            notes.append(f"Edge divisions {i + 1} ({e.get('label') or 'edge'}): edge not found on the part; skipped")
            continue
        done[tag] = int(e["n"])
        gmsh.model.mesh.setTransfiniteCurve(tag, int(e["n"]) + 1)
    for i, f in enumerate(controls["faces"]):
        tag = _match(gmsh, 2, f["ref"], f["area"])
        if tag is None:
            notes.append(f"Mapped face {i + 1} ({f.get('label') or 'face'}): face not found on the part; skipped")
            continue
        curves = list(dict.fromkeys(abs(t) for _d, t in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=True)))
        if len(curves) not in (3, 4):
            notes.append(f"Mapped face {i + 1}: {len(curves)} boundary edges (needs 3 or 4); skipped")
            continue
        n = {c: done.get(c, max(1, math.ceil(gmsh.model.occ.getMass(1, c) / size))) for c in curves}
        if len(curves) == 4:  # opposite sides must match: boundary order is a loop, so 0-2 and 1-3
            for a, b in ((0, 2), (1, 3)):
                m = max(n[curves[a]], n[curves[b]])
                n[curves[a]] = n[curves[b]] = m
        for c, k in n.items():
            done[c] = k
            gmsh.model.mesh.setTransfiniteCurve(c, k + 1)
        gmsh.model.mesh.setTransfiniteSurface(tag)
    fields = []
    for s in controls["spheres"]:
        f = gmsh.model.mesh.field.add("Ball")
        for k, val in (("XCenter", s["center"][0]), ("YCenter", s["center"][1]), ("ZCenter", s["center"][2]),
                       ("Radius", s["radius"]), ("Thickness", 0.5 * s["radius"]), ("VIn", s["size"]), ("VOut", size)):
            gmsh.model.mesh.field.setNumber(f, k, float(val))
        fields.append(f)
    return fields


def _mesh_child(brep: str, out: str, size: float, curvature: int, refine: list[dict], min_size: float, linear: int, optimize: int,
                controls: dict | None = None) -> None:
    import math  # noqa: F401 - used by _apply_controls in this (spawned) process

    import gmsh

    controls = controls or {"edges": [], "faces": [], "spheres": []}
    notes: list[str] = []
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(brep)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", linear)  # 0: mid-side nodes on the true surfaces
        gmsh.option.setNumber("Mesh.HighOrderOptimize", optimize)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        fields = _apply_controls(gmsh, controls, size, notes)
        if refine or fields:
            for r in refine:
                f = gmsh.model.mesh.field.add("Cylinder")
                o, a = np.asarray(r["origin"], float), np.asarray(r["axis"], float)
                a = a / np.linalg.norm(a) * 1e3  # long enough to pass through the part
                for k, val in (("XCenter", o[0]), ("YCenter", o[1]), ("ZCenter", o[2]), ("XAxis", a[0]), ("YAxis", a[1]),
                               ("ZAxis", a[2]), ("Radius", r["radius"]), ("VIn", r["size"]), ("VOut", size)):
                    gmsh.model.mesh.field.setNumber(f, k, float(val))
                fields.append(f)
            mn = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(mn, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(mn)
        gmsh.model.mesh.generate(3)
        etags, _en = gmsh.model.mesh.getElementsByType(11)
        quality = np.asarray(gmsh.model.mesh.getElementQualities(etags, "minSICN"), float)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.full(int(tags.max()) + 1, -1, np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        nodes = coords.reshape(-1, 3)
        _et, tet_nodes = gmsh.model.mesh.getElementsByType(11)  # 10-node tetrahedra
        tets = index[tet_nodes.astype(np.int64)].reshape(-1, 10)
        tris, tri_surface, surfaces = [], [], {}
        for _dim, tag in gmsh.model.getEntities(2):
            surfaces[tag] = gmsh.model.getType(2, tag)
            _e, tn = gmsh.model.mesh.getElementsByType(9, tag)  # 6-node triangles on this surface
            if len(tn):
                t = index[tn.astype(np.int64)].reshape(-1, 6)
                tris.append(t)
                tri_surface.append(np.full(len(t), tag))
    finally:
        gmsh.finalize()
    used = np.unique(tets)  # drop nodes no tetrahedron uses
    remap = np.full(len(nodes), -1, np.int64)
    remap[used] = np.arange(len(used))
    tris_all = np.concatenate(tris) if tris else np.zeros((0, 6), np.int64)
    np.savez(out, nodes=nodes[used], tets=remap[tets], tris=remap[tris_all],
             tri_surface=np.concatenate(tri_surface) if tri_surface else np.zeros(0, np.int64), surfaces=np.array(surfaces, dtype=object),
             quality=quality, notes=np.array(notes, dtype=str))
