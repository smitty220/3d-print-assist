"""Part checks for printing: the build itself, feature rule checks, interference with reference bodies, minimum wall,
fit on the printer's bed, and mass. Lengths in mm, points in world coordinates (Z up, bed at Z = 0)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from build123d import Shape, Vector
from OCP.BRep import BRep_Tool
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.ShapeAnalysis import ShapeAnalysis_Surface

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

    out += check_wall(rules, part)

    F("mass", "Mass", "ok", f"{build.mass * 1000:.1f} g of {rules.material} ({build.volume / 1000:.1f} cm³)",
      value=build.mass, quantity="mass")
    return out


def check_wall(rules: PrintRules, part: Shape) -> list[Finding]:
    """Sampled ray thickness: from points across every face, the distance through the material along the inward normal
    (corner readings excluded: the tip of an acute corner reads thin but is not a thin wall)."""
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(part.wrapped, 1e-4)
    best: tuple[float, Vector, Vector] | None = None
    for face in part.faces():
        surf = BRep_Tool.Surface_s(face.wrapped)
        proj = ShapeAnalysis_Surface(surf)
        try:
            verts, tris = face.tessellate(0.3, 0.5)
        except AttributeError:  # a sliver face the mesher can't triangulate; it has no thickness to sample
            continue
        for a, b, c in tris:
            m = (verts[a] + verts[b] + verts[c]) * (1 / 3)
            uv = proj.ValueOfUV(gp_Pnt(m.X, m.Y, m.Z), 1e-7)
            gp = surf.Value(uv.X(), uv.Y())
            pt = Vector(gp.X(), gp.Y(), gp.Z())
            n = face.normal_at(pt)
            inter.Perform(gp_Lin(gp, gp_Dir(-n.X, -n.Y, -n.Z)), 1e-3, 1e9)
            hits = [inter.WParameter(i) for i in range(1, inter.NbPnt() + 1) if inter.WParameter(i) > 1e-2]
            if hits:
                t = min(hits)
                # Near a corner the ray meets the neighbouring face at once (an acute corner is not a thin wall): a
                # reading counts only if the point is farther from its face's edges than twice the thickness found.
                if t < rules.min_wall_preferred and min(e.distance_to(pt) for e in face.edges()) < 2 * t:
                    continue
                if best is None or t < best[0]:
                    best = (t, pt, pt - n * t)
    if best is None:
        return []
    t, p0, p1 = best
    status = "ok" if t >= rules.min_wall_preferred - 1e-6 else "warn" if t >= rules.min_wall - 1e-6 else "fail"
    return [Finding("part", "wall", "Minimum wall", status,
                    f"{t:.2f} mm thinnest (sampled); {rules.min_wall_preferred:g} mm preferred, {rules.min_wall:g} mm min",
                    value=t, quantity="length", points=[_t(p0), _t(p1)])]


def validate(build: BuildResult | None, rules: PrintRules, printer: PrinterProfile | None = None,
             references: dict[str, Shape] | None = None) -> Report:
    return Report([], check_part(build, rules, printer, references) if build is not None else [])
