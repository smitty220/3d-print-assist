"""Part checks for printing: the build itself, feature rule checks, interference with reference bodies, minimum wall,
fit on the printer's bed, and mass. Lengths in mm, points in world coordinates (Z up, bed at Z = 0)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from build123d import Shape, Vector
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from .features.design import BuildResult
from .printers import PrinterProfile
from .rules import PrintRules

Status = Literal["ok", "warn", "fail"]
Vec3 = tuple[float, float, float]

VOL_TOL = 0.5  # mm³: overlaps smaller than this are boolean noise


@dataclass
class Finding:
    stage: Literal["preflight", "part"]
    code: str
    title: str
    status: Status
    detail: str
    value: float | None = None
    quantity: Literal["length", "area", "ratio", "percent", "mass", "count"] | None = None
    refs: list[str] = field(default_factory=list)
    points: list[Vec3] = field(default_factory=list)  # 1 = marker, 2 = dimension between them

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    preflight: list[Finding]
    part: list[Finding]

    def to_dict(self) -> dict:
        everything = self.preflight + self.part
        counts = {s: sum(f.status == s for f in everything) for s in ("ok", "warn", "fail")}
        return {
            "preflight": [f.to_dict() for f in self.preflight],
            "part": [f.to_dict() for f in self.part],
            "counts": counts,
            "preflight_ready": not any(f.status == "fail" for f in self.preflight),
        }


def _t(v: Vector) -> Vec3:
    return (round(v.X, 4), round(v.Y, 4), round(v.Z, 4))


def _vol(shape: Shape | None) -> float:
    if shape is None:
        return 0.0
    try:
        return max(0.0, shape.volume)
    except Exception:
        return 0.0


def _distance(a: Shape, b: Shape) -> tuple[float, Vector, Vector]:
    d = BRepExtrema_DistShapeShape()
    d.LoadS1(a.wrapped)
    d.LoadS2(b.wrapped)
    d.SetMultiThread(True)
    d.Perform()
    if not d.IsDone() or d.NbSolution() == 0:
        raise ValueError("distance computation failed")
    p1, p2 = d.PointOnShape1(1), d.PointOnShape2(1)
    return d.Value(), Vector(p1.X(), p1.Y(), p1.Z()), Vector(p2.X(), p2.Y(), p2.Z())


def check_part(build: BuildResult, rules: PrintRules, printer: PrinterProfile | None = None,
               references: dict[str, Shape] | None = None) -> list[Finding]:
    out: list[Finding] = []
    F = lambda *a, **k: out.append(Finding("part", *a, **k))  # noqa: E731

    errors = [f for f in build.features if f.error]
    for f in errors:
        F("build", f"{f.id} failed", "fail", f.error or "", refs=[f.id])
    part = build.world
    if part is None:
        F("build", "Part", "fail", "Nothing was built")
        return out
    built = [f for f in build.features if not f.error and not f.suppressed]
    if not errors:
        F("build", "Build", "ok", f"{len(built)} features built")

    bad = [c for c in build.checks if c.status != "ok"]
    for c in bad:
        F("rules", f"{c.feature} {c.name}", c.status, c.rule, value=c.value, refs=[c.feature])
    if not bad:
        F("rules", "Feature rule checks", "ok", f"All {len(build.checks)} pass")

    for name, ref in (references or {}).items():
        over = part & ref
        if _vol(over) > VOL_TOL:
            F("interference", f"Clear of {name}", "fail", f"Overlaps {name} by {_vol(over):.1f} mm³", points=[_t(over.center())])
        else:
            d, a, b = _distance(part, ref)
            F("interference", f"Clear of {name}", "ok", f"{d:.2f} mm at the closest point", value=d, quantity="length",
              points=[_t(a), _t(b)])

    if printer is not None:
        bb = part.bounding_box()
        size = (bb.size.X, bb.size.Y, bb.size.Z)
        over = [f"{ax} {s:.1f} > {lim:g}" for ax, s, lim in zip("XYZ", size, printer.bed) if s > lim + 1e-6]
        F("bed", "Fits the bed", "fail" if over else "ok",
          ("Too big for " + printer.name + ": " + ", ".join(over)) if over else
          f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm on a {printer.bed[0]:g} × {printer.bed[1]:g} × {printer.bed[2]:g} bed",
          value=max(size), quantity="length")

    out += check_wall(rules, printed(build))

    metres = build.volume / (3.14159265 * 0.875**2) / 1000  # 1.75 mm filament
    F("mass", "Mass", "ok", f"{build.mass * 1000:.1f} g of {rules.material}, about {metres:.1f} m of 1.75 mm filament "
      f"({build.volume / 1000:.1f} cm³ solid)", value=build.mass, quantity="mass")
    return out


def printed(build: BuildResult) -> Shape:
    """The part as it prints: bodies that touch (an inlay in its pocket, a colour band on its vase) fused together, so
    a pocket's floor is not mistaken for a thin wall. Bodies apart from the rest stay as they are."""
    from build123d import Compound

    parts = list((build.parts or {}).values()) or [build.world]
    if len(parts) == 1:
        return parts[0]
    boxes = [p.bounding_box() for p in parts]

    def touch(a, b) -> bool:
        return all(getattr(a.min, k) <= getattr(b.max, k) + 1e-6 and getattr(b.min, k) <= getattr(a.max, k) + 1e-6
                   for k in "XYZ")

    group = list(range(len(parts)))  # union-find over overlapping boxes

    def root(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if touch(boxes[i], boxes[j]):
                group[root(i)] = root(j)
    out = []
    for r in {root(i) for i in range(len(parts))}:
        members = [parts[i] for i in range(len(parts)) if root(i) == r]
        if len(members) == 1:
            out.append(members[0])
            continue
        try:
            out.append(members[0].fuse(*members[1:]).clean())
        except Exception:  # noqa: BLE001 - a fuse that fails: check the bodies as they are
            out.extend(members)
    return out[0] if len(out) == 1 else Compound(children=out)


WALL_REACH = 2.4  # mm: rays look this far through the material (twice the preferred wall); thicker is "thick enough"
WALL_SAMPLES = 8000  # per part, spread over its faces


def check_wall(rules: PrintRules, part: Shape) -> list[Finding]:
    """Sampled ray thickness: from points across every face, the distance through the material along the inward normal
    (corner readings excluded: the tip of an acute corner reads thin but is not a thin wall).

    Works on the part's triangle mesh, with the triangles bucketed in a grid, so a part with thousands of faces (text,
    knurling) is checked in about a second. Rays look WALL_REACH mm deep; anything thicker is reported as such."""
    import numpy as np
    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    from .geometry.tessellate import _triangulation

    reach = max(WALL_REACH, 2 * rules.min_wall_preferred)
    BRepMesh_IncrementalMesh(part.wrapped, 0.02, False, 0.2, True)
    tris, samples, normals, bounds = [], [], [], []  # bounds: (sample index range, the face's boundary segments)
    faces = part.faces()
    per_face = max(4, WALL_SAMPLES // max(len(faces), 1))
    for face in faces:
        got = _triangulation(face)
        if got is None or not got[1]:
            continue
        pts = np.array(got[0], dtype=np.float64).reshape(-1, 3)
        idx = np.array(got[1], dtype=np.int64).reshape(-1, 3)
        t = pts[idx]
        tris.append(t)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = np.linalg.norm(n, axis=1)
        keep = np.nonzero(area > 1e-9)[0]
        if len(keep) > per_face:  # the largest triangles spread the samples over the face
            keep = keep[np.argsort(-area[keep])[:per_face]]
        e = np.sort(np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        edge = pts[uniq[counts == 1]]  # the face's outline: triangle edges used once
        start = sum(len(x) for x in samples)
        samples.append(t[keep].mean(axis=1))
        normals.append(n[keep] / area[keep, None])
        bounds.append((start, start + len(keep), edge))
    if not tris:
        return []
    T = np.concatenate(tris)
    S = np.concatenate(samples)
    N = np.concatenate(normals)

    # Bucket triangles in a grid of cells as big as the reach, so each ray only meets triangles near it.
    cell = reach
    lo = T.reshape(-1, 3).min(0)
    tmin = np.floor((T.min(1) - lo) / cell).astype(int)
    tmax = np.floor((T.max(1) - lo) / cell).astype(int)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for i, (a, b) in enumerate(zip(tmin, tmax)):
        for x in range(a[0], b[0] + 1):
            for y in range(a[1], b[1] + 1):
                for z in range(a[2], b[2] + 1):
                    grid.setdefault((x, y, z), []).append(i)

    best = (np.inf, None, None)
    for start, stop, edge in bounds:
        for k in range(start, stop):
            p, d = S[k], -N[k]
            q = p + d * reach
            c0 = np.floor((np.minimum(p, q) - lo) / cell).astype(int)
            c1 = np.floor((np.maximum(p, q) - lo) / cell).astype(int)
            cand = {i for x in range(c0[0], c1[0] + 1) for y in range(c0[1], c1[1] + 1) for z in range(c0[2], c1[2] + 1)
                    for i in grid.get((x, y, z), ())}
            if not cand:
                continue
            tt = T[list(cand)]
            e1, e2 = tt[:, 1] - tt[:, 0], tt[:, 2] - tt[:, 0]  # Moller-Trumbore, all candidates at once
            h = np.cross(d, e2)
            a = (e1 * h).sum(1)
            ok = np.abs(a) > 1e-12
            f = np.where(ok, 1 / np.where(ok, a, 1), 0)
            sv = p - tt[:, 0]
            u = f * (sv * h).sum(1)
            qv = np.cross(sv, e1)
            v = f * (qv @ d)
            dist = f * (e2 * qv).sum(1)
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 1e-2) & (dist < reach)
            if not hit.any():
                continue
            th = float(dist[hit].min())
            if th >= best[0]:
                continue
            # Near a corner the ray meets the neighbouring face at once (an acute corner is not a thin wall): a reading
            # counts only if the point is farther from its face's outline than twice the thickness found.
            if len(edge):
                a0, b0 = edge[:, 0], edge[:, 1]
                ab = b0 - a0
                w = np.clip(((p - a0) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0, 1)
                if np.linalg.norm(a0 + ab * w[:, None] - p, axis=1).min() < 2 * th:
                    continue
            best = (th, p, p + d * th)
    t, p0, p1 = best
    if p0 is None:
        return [Finding("part", "wall", "Minimum wall", "ok", f"no wall thinner than {reach:g} mm (sampled)",
                        value=None, quantity="length")]
    status = "ok" if t >= rules.min_wall_preferred - 1e-6 else "warn" if t >= rules.min_wall - 1e-6 else "fail"
    return [Finding("part", "wall", "Minimum wall", status,
                    f"{t:.2f} mm thinnest (sampled); {rules.min_wall_preferred:g} mm preferred, {rules.min_wall:g} mm min",
                    value=t, quantity="length", points=[_t(Vector(*p0)), _t(Vector(*p1))])]


def validate(build: BuildResult | None, rules: PrintRules, printer: PrinterProfile | None = None,
             references: dict[str, Shape] | None = None, design=None) -> Report:
    if build is None:
        return Report([], [])
    part = check_part(build, rules, printer, references)
    if design is not None:
        from .variants import check_filaments

        part += check_filaments(design, list(build.parts), printer)
    return Report([], part)
