"""The print layout: the built model split into the objects that print (`Design.layout`), each turned to lie on the
bed and set down on a plate.

An item takes the solids of some bodies whose bounding-box centre lies inside its `within` box of the assembled model
(all of them without one). Its solids keep their places relative to each other, so a piece in several colours stays
together; they are turned about the item's centre (X, then Y, then Z) and moved so the item's bounding box sits on
the bed with its front-left corner at (x, y) of its plate. Every solid of the model must be in exactly one item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from build123d import Axis, Compound, Location, Shape

from ..features.model import Design, PrintItem

TOWER = 35.0  # mm, the side of the purge tower kept clear when a plate names where its tower goes
GAP = 1.0  # mm, the least room between two objects (and the tower) on a plate


@dataclass
class Placed:
    """An item set down on its plate: its parts (body id, shape in plate coordinates) and its box on the plate."""

    item: PrintItem
    parts: list[tuple[str, Shape]]
    box: tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax on the plate


@dataclass
class Layout:
    placed: list[Placed] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    unplaced: list[dict] = field(default_factory=list)  # solids no item takes: body, index, centre, size

    @property
    def plates(self) -> int:
        return max((p.item.plate for p in self.placed), default=0)


def solids(parts: dict[str, Shape]) -> dict[str, list[Shape]]:
    """Each body's separate solids, in the kernel's order."""
    return {bid: list(shape.solids()) for bid, shape in parts.items() if shape is not None}


def _box(shapes: list[Shape]) -> tuple[float, ...]:
    bb = (shapes[0] if len(shapes) == 1 else Compound(children=shapes)).bounding_box()
    return (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)


def _r(v: float) -> float:
    return round(v, 2)


def resolve(design: Design, parts: dict[str, Shape], bed: tuple[float, float]) -> Layout:
    """Place every item of the design's layout on its plate; problems say what keeps it from printing as laid out."""
    out = Layout()
    by_body = solids(parts)
    taken: dict[tuple[str, int], str] = {}
    for item in design.layout:
        picked: list[tuple[str, Shape]] = []
        for bid in item.bodies:
            if bid not in by_body:
                out.problems.append(f"{item.id} {item.name}: body {bid} has nothing built")
                continue
            for i, s in enumerate(by_body[bid]):
                c = s.bounding_box().center()
                w = item.within
                if w is None or (w[0] <= c.X <= w[3] and w[1] <= c.Y <= w[4] and w[2] <= c.Z <= w[5]):
                    if (bid, i) in taken:
                        out.problems.append(f"{item.id} {item.name}: solid {i} of {bid} is also in {taken[(bid, i)]}")
                        continue
                    taken[(bid, i)] = item.id
                    picked.append((bid, s))
        if not picked:
            out.problems.append(f"{item.id} {item.name}: its box holds no solid of {', '.join(item.bodies)}")
            continue
        x0, y0, z0, x1, y1, z1 = _box([s for _, s in picked])
        centre = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        turned = []
        for bid, s in picked:
            for axis, deg in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), item.rotate):
                if deg:
                    s = s.rotate(Axis(centre, axis), deg)
            turned.append((bid, s))
        x0, y0, z0, *_ = _box([s for _, s in turned])
        move = Location((item.x - x0, item.y - y0, -z0))
        moved = [(bid, s.moved(move)) for bid, s in turned]
        box = tuple(_r(v) for v in _box([s for _, s in moved]))
        out.placed.append(Placed(item, moved, box))
        if box[0] < -0.01 or box[1] < -0.01 or box[3] > bed[0] + 0.01 or box[4] > bed[1] + 0.01:
            out.problems.append(f"{item.id} {item.name}: off plate {item.plate} (spans x {box[0]}..{box[3]}, "
                                f"y {box[1]}..{box[4]} on a {bed[0]:g} x {bed[1]:g} bed)")
    for bid, ss in by_body.items():
        for i, s in enumerate(ss):
            if (bid, i) not in taken:
                bb = s.bounding_box()
                out.unplaced.append({"body": bid, "index": i, "center": [_r(bb.center().X), _r(bb.center().Y),
                                                                          _r(bb.center().Z)],
                                     "size": [_r(bb.size.X), _r(bb.size.Y), _r(bb.size.Z)]})
    if design.layout and out.unplaced:
        out.problems.append(f"{len(out.unplaced)} solid(s) are on no plate: "
                            + ", ".join(f"{u['body']}#{u['index']} at {u['center']}" for u in out.unplaced[:8])
                            + (" ..." if len(out.unplaced) > 8 else ""))
    out.problems += _overlaps(out.placed, design)
    return out


def _overlaps(placed: list[Placed], design: Design) -> list[str]:
    """Objects on one plate whose footprints come closer than GAP, or that sit where the plate's tower goes."""
    out = []
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if a.item.plate == b.item.plate and _hit(a.box, b.box):
                out.append(f"{a.item.id} {a.item.name} and {b.item.id} {b.item.name} overlap on plate {a.item.plate}")
        n = a.item.plate
        tower = design.plates[n - 1].tower if n <= len(design.plates) else None
        if tower and _hit(a.box, (tower[0], tower[1], 0, tower[0] + TOWER, tower[1] + TOWER, 0)):
            out.append(f"{a.item.id} {a.item.name} is where plate {n}'s purge tower goes")
    return out


def _hit(a, b) -> bool:
    return not (a[3] + GAP <= b[0] or b[3] + GAP <= a[0] or a[4] + GAP <= b[1] or b[4] + GAP <= a[1])


def plate_origin(n: int, count: int, bed: tuple[float, float]) -> tuple[float, float]:
    """Where plate n (1-based) of `count` sits in a Bambu Studio project: plates are laid out in a grid with
    ceil(sqrt(count)) columns, 1.2 bed sizes apart, the rows going toward -Y (as Bambu Studio's PartPlateList does)."""
    cols = max(1, math.ceil(math.sqrt(count) - 1e-9))
    row, col = divmod(n - 1, cols)
    return col * bed[0] * 1.2, -row * bed[1] * 1.2


def report(layout: Layout, design: Design) -> dict:
    """The layout, compactly: plates with their objects' boxes, problems, and the solids no item takes."""
    count = max(layout.plates, len(design.plates))
    plates = []
    for n in range(1, count + 1):
        pl = design.plates[n - 1] if n <= len(design.plates) else None
        plates.append({"plate": n, "name": (pl.name if pl else "") or f"Plate {n}", "tower": pl.tower if pl else None,
                       "objects": [{"id": p.item.id, "name": p.item.name, "bodies": sorted({b for b, _ in p.parts}),
                                    "solids": len(p.parts), "box": list(p.box)}
                                   for p in layout.placed if p.item.plate == n]})
    return {"plates": plates, "problems": layout.problems, "unplaced": layout.unplaced[:60],
            "more_unplaced": max(0, len(layout.unplaced) - 60)}
