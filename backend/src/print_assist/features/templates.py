"""Reference brackets: deterministic feature trees derived from a project's intent.

These are the "golden" designs the AI loop (M0.5) is measured against, and a quick way to get a
sensible starting part. Each template sizes the plate so every hole meets the preferred edge distance
and the attachment features sit fully on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..catalog import FASTENERS
from ..intent.models import Bore, BoltPattern, Intent
from .context import BuildContext, FeatureError
from .design import design_frame
from .model import Design, FeatureInstance as FI

IN = 25.4
STEP = 0.05 * IN  # sizes round up to 0.05 in


def _up(v: float) -> float:
    return math.ceil(v / STEP - 1e-9) * STEP


def _ctx(intent: Intent) -> BuildContext:
    return BuildContext(intent=intent, frame=design_frame(intent))


def _mount_pattern(ctx: BuildContext) -> BoltPattern:
    """The bolt pattern lying in the mount plane (holes along the design-frame Z axis)."""
    for o in ctx.intent.objects:
        if o.kind == "bolt_pattern" and abs(ctx.dir_to_local(o.frame.z).Z) > 0.99:
            return o
    raise FeatureError("needs a bolt pattern on the mount face")


def _shank(bp: BoltPattern) -> float:
    f = FASTENERS.get(bp.fastener or "")
    return f.shank_d * IN if f else bp.holes[0].diameter


def _hole_box(ctx: BuildContext, bp: BoltPattern) -> tuple[float, float, float, float]:
    """x0, x1, y0, y1 of the holes in the design frame, grown by the preferred edge distance."""
    e = ctx.rules.bolt_edge_preferred * _shank(bp) + 0.5
    pts = [c for c, _a, _d in ctx.bolt_holes_local(bp)]
    return min(p.X for p in pts) - e, max(p.X for p in pts) + e, min(p.Y for p in pts) - e, max(p.Y for p in pts) + e


def _lug_box(c, a, half_along: float, half_across: float) -> tuple[float, float, float, float]:
    across = (-a.Y, a.X)
    xs, ys = [], []
    for s in (-half_along, half_along):
        for t in (-half_across, half_across):
            xs.append(c.X + a.X * s + across[0] * t)
            ys.append(c.Y + a.Y * s + across[1] * t)
    return min(xs), max(xs), min(ys), max(ys)


def _plate(boxes, thickness: float = 0.25 * IN) -> dict:
    x0 = min(b[0] for b in boxes)
    x1 = max(b[1] for b in boxes)
    y0 = min(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    L, W = _up(x1 - x0), _up(y1 - y0)
    return {
        "length": L, "width": W, "thickness": thickness,
        "corner_radius": min(0.3 * IN, 0.2 * min(L, W)),
        "center_x": round((x0 + x1) / 2, 4), "center_y": round((y0 + y1) / 2, 4),
    }


def _pin_bore(ctx: BuildContext, forked: bool) -> Bore:
    for o in ctx.intent.objects:
        if o.kind == "bore" and (o.gap is not None) == forked:
            return o
    raise FeatureError("needs a bore on a forked (clevis) mating end" if forked else "needs a bore on a male rod end")


def clevis_bracket(intent: Intent) -> Design:
    ctx = _ctx(intent)
    bp, bore = _mount_pattern(ctx), _pin_bore(ctx, forked=False)
    c, a = ctx.bore_local(bore)
    t, clr = 0.25 * IN, 0.025 * IN
    R = max(0.6 * IN, ctx.rules.lug_edge_full_bearing * bore.diameter + 0.05 * IN)
    half = 0.6 * IN
    gap = bore.length + clr
    plate = _plate([_hole_box(ctx, bp), _lug_box(c, a, gap / 2 + t + 0.7 * IN, half + 1.0)])
    zb = plate["thickness"]
    return Design(features=[
        FI(id="F1", kind="plate", params=plate),
        FI(id="F2", kind="bolt_holes", refs={"pattern": bp.id, "target": "F1"}),
        FI(id="F3", kind="clevis", refs={"pin": bore.id, "base": "F1"},
           params={"ear_thickness": t, "crown_radius": R, "base_half_width": half, "gap_clearance": clr}),
        FI(id="F4", kind="gusset_pair", refs={"lug": "F3"},
           params={"height": round(zb + 0.49 * (c.Z - zb), 3), "length": 0.68 * IN, "thickness": 0.2 * IN}),
        FI(id="F5", kind="fillet", refs={"faces_a": "F1.top", "faces_b": "F3"}, params={"radius": 0.06 * IN}),
        FI(id="F6", kind="chamfer", refs={"faces_a": "F1.top", "faces_b": "F1.sides"}, params={"distance": 0.02 * IN}),
    ])


def single_lug_fitting(intent: Intent) -> Design:
    ctx = _ctx(intent)
    bp, bore = _mount_pattern(ctx), _pin_bore(ctx, forked=True)
    c, a = ctx.bore_local(bore)
    t = round(bore.gap - 2 * 0.015 * IN, 4)  # 0.015 in side clearance each side
    R = max(0.6 * IN, ctx.rules.lug_edge_full_bearing * bore.diameter + 0.05 * IN)
    half = 0.65 * IN
    plate = _plate([_hole_box(ctx, bp), _lug_box(c, a, t / 2 + 0.7 * IN, half + 1.0)])
    zb = plate["thickness"]
    return Design(features=[
        FI(id="F1", kind="plate", params=plate),
        FI(id="F2", kind="bolt_holes", refs={"pattern": bp.id, "target": "F1"}),
        FI(id="F3", kind="lug", refs={"pin": bore.id, "base": "F1"}, params={"thickness": t, "crown_radius": R, "base_half_width": half}),
        FI(id="F4", kind="gusset_pair", refs={"lug": "F3"},
           params={"height": round(zb + 0.45 * (c.Z - zb), 3), "length": 0.6 * IN, "thickness": 0.2 * IN}),
        FI(id="F5", kind="fillet", refs={"faces_a": "F1.top", "faces_b": "F3"}, params={"radius": 0.06 * IN}),
        FI(id="F6", kind="chamfer", refs={"faces_a": "F1.top", "faces_b": "F1.sides"}, params={"distance": 0.02 * IN}),
    ])


def angle_bracket(intent: Intent) -> Design:
    ctx = _ctx(intent)
    base_bp = _mount_pattern(ctx)
    wall_bp = next(
        (o for o in intent.objects if o.kind == "bolt_pattern" and abs(ctx.dir_to_local(o.frame.z).Z) < 0.01),
        None,
    )
    if wall_bp is None:
        raise FeatureError("needs a second bolt pattern on an upright face")
    axis = ctx.dir_to_local(wall_bp.frame.z)  # points from the upright mating face into the bracket
    n = (-axis.X, -axis.Y)  # wall outer-face normal
    if abs(n[0]) > 0.99:
        side, i_along = ("+x" if n[0] > 0 else "-x"), 0
    elif abs(n[1]) > 0.99:
        side, i_along = ("+y" if n[1] > 0 else "-y"), 1
    else:
        raise FeatureError("the upright bolt pattern must face along the design X or Y axis")
    wall_pts = [c for c, _a, _d in ctx.bolt_holes_local(wall_bp)]
    face_coord = (wall_pts[0].X, wall_pts[0].Y)[i_along]
    e2 = ctx.rules.bolt_edge_preferred * _shank(wall_bp) + 0.5
    x0, x1, y0, y1 = _hole_box(ctx, base_bp)
    across = [(p.Y if i_along == 0 else p.X) for p in wall_pts]
    if i_along == 0:
        x0, x1 = (x0, face_coord) if side == "+x" else (face_coord, x1)
        y0, y1 = min(y0, min(across) - e2), max(y1, max(across) + e2)
    else:
        y0, y1 = (y0, face_coord) if side == "+y" else (face_coord, y1)
        x0, x1 = min(x0, min(across) - e2), max(x1, max(across) + e2)
    plate = _plate([(x0, x1, y0, y1)])
    # Square corners: the wall runs the full edge, and a corner radius under its ends makes the root blend fail.
    plate["corner_radius"] = 0.0
    # Keep the wall's outer face on the mating face: snap the plate edge back to it after rounding.
    sign = 1 if side[0] == "+" else -1
    key, size = ("center_x", "length") if i_along == 0 else ("center_y", "width")
    plate[key] = round(face_coord - sign * plate[size] / 2, 4)
    height = _up(max(p.Z for p in wall_pts) + e2)
    return Design(features=[
        FI(id="F1", kind="plate", params=plate),
        FI(id="F2", kind="bolt_holes", refs={"pattern": base_bp.id, "target": "F1"}),
        FI(id="F3", kind="wall", refs={"base": "F1"}, params={"side": side, "height": height, "thickness": 0.25 * IN}),
        FI(id="F4", kind="bolt_holes", refs={"pattern": wall_bp.id, "target": "F3"}),
        FI(id="F5", kind="fillet", refs={"faces_a": "F1.top", "faces_b": "F3.inner"}, params={"radius": 0.12 * IN}),
        FI(id="F6", kind="chamfer", refs={"faces_a": "F1.top", "faces_b": "F1.sides"}, params={"distance": 0.02 * IN}),
    ])


TEMPLATES: dict[str, tuple[str, str, Callable[[Intent], Design]]] = {
    "clevis_bracket": ("Clevis bracket", "Double lug with gussets for a male rod end", clevis_bracket),
    "single_lug_fitting": ("Single-lug fitting", "Single lug with gussets for a forked actuator end", single_lug_fitting),
    "angle_bracket": ("Angle bracket", "Base plate plus upright wall, bolted on both legs", angle_bracket),
}
