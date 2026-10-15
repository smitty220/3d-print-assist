"""Non-proprietary sample mating parts for development and tests.

Dimensions are authored in inches (IN = 25.4 mm) to match the bracket demo:
a T-section frame rib with a 4-hole pattern, an actuator rod end, and a hydraulic
line envelope used as an imported keep-out volume.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from build123d import (
    IN,
    Axis,
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Hole,
    Locations,
    Mode,
    Plane,
    Pos,
    Rot,
    Shape,
    Spline,
    sweep,
)

from .step_io import write_step

HOLE_PATTERN_IN = [(0.95, 0.95), (-0.95, 0.95), (-0.95, -0.95), (0.95, -0.95)]


def frame_rib(flange_t: float = 0.25, top_z: float = 0.0) -> Shape:
    """T-section rib: 6 x 4 x 0.25 flange (top face at Z=0) on a 0.25 x 2.0 web, 4x Ø0.201 holes.

    `flange_t` and `top_z` (inches) let tests build a revised rib to exercise face re-matching.
    """
    with BuildPart() as rib:
        with Locations(Pos(0, 0, (top_z - flange_t / 2) * IN)):
            Box(6 * IN, 4 * IN, flange_t * IN)
        with Locations(Pos(0, 0, -1.25 * IN)):
            Box(6 * IN, 0.25 * IN, 2 * IN)
        top = rib.faces().sort_by(Axis.Z)[-1]
        with Locations(Plane(top.center_location)):
            with Locations(*[(x * IN, y * IN) for x, y in HOLE_PATTERN_IN]):
                Hole(radius=0.1005 * IN, depth=flange_t * IN)
    return rib.part


def rod_end() -> Shape:
    """Rod-end eye (Ø0.84 x 0.60 wide, pin axis X at Z=3.0) with a Ø0.40 rod angled up and back."""
    with BuildPart() as re:
        with Locations(Pos(0, 0, 3 * IN) * Rot(0, 90, 0)):
            Cylinder(radius=0.42 * IN, height=0.6 * IN)
            Cylinder(radius=0.1885 * IN, height=0.6 * IN, mode=Mode.SUBTRACT)
        # Rod axis tilted 22.8° from +Z toward +Y; its centre sits 1.92 in from the pin along that axis.
        tilt = math.radians(22.8)
        reach = (0.42 + 1.5) * IN
        with Locations(Pos(0, reach * math.sin(tilt), 3 * IN + reach * math.cos(tilt)) * Rot(-22.8, 0, 0)):
            Cylinder(radius=0.2 * IN, height=3 * IN)
    return re.part


def hyd_line_envelope() -> Shape:
    """Swept Ø0.60 tube along a gentle spline: the keep-out volume around a hydraulic line."""
    pts = [(1.32, -3.2, 1.05), (1.3, -1.4, 1.22), (1.3, 0, 1.35), (1.33, 1.4, 1.5), (1.4, 3.2, 1.7)]
    with BuildLine() as path:
        Spline(*[(x * IN, y * IN, z * IN) for x, y, z in pts])
    start = path.line @ 0
    with BuildSketch(Plane(origin=start, z_dir=path.line % 0)) as profile:
        Circle(0.3 * IN)
    return sweep(profile.sketch, path=path.line)


def actuator_clevis_end() -> Shape:
    """Forked actuator end: two 0.20 in tines with a 0.40 in gap, Ø0.377 pin bores on axis X at Z=3.0, rod above."""
    gap, tine = 0.40, 0.20
    with BuildPart() as fork:
        for s in (1, -1):
            with Locations(Pos(s * (gap / 2 + tine / 2) * IN, 0, 3.15 * IN)):
                Box(tine * IN, 0.84 * IN, 1.1 * IN)
        with Locations(Pos(0, 0, 3.85 * IN)):
            Box((gap + 2 * tine) * IN, 0.84 * IN, 0.3 * IN)
        with Locations(Pos(0, 0, 5.0 * IN)):
            Cylinder(radius=0.2 * IN, height=2.0 * IN)
        with Locations(Pos(0, 0, 3 * IN) * Rot(0, 90, 0)):
            Cylinder(radius=0.1885 * IN, height=(gap + 2 * tine + 0.1) * IN, mode=Mode.SUBTRACT)
    return fork.part


def side_panel() -> Shape:
    """Upright panel whose -X face sits at X=1.5 in, with 2x Ø0.201 holes at Y=±0.6, Z=1.6 (angle-bracket reference)."""
    with BuildPart() as panel:
        with Locations(Pos(1.625 * IN, 0, 1.75 * IN)):
            Box(0.25 * IN, 4 * IN, 3 * IN)
        # Workplane on the -X face: x_dir -Y makes its local Y point up (+Z), so (y, z) offsets read naturally.
        with Locations(Plane(origin=(1.5 * IN, 0, 0), x_dir=(0, -1, 0), z_dir=(-1, 0, 0))):
            with Locations((0.6 * IN, 1.6 * IN), (-0.6 * IN, 1.6 * IN)):
                Hole(radius=0.1005 * IN, depth=0.25 * IN)
    return panel.part


def fork_end(gap: float, tine: float, width: float, bore_d: float, reach: float, rod_d: float | None) -> Shape:
    """Generic forked end, pin axis local X through the origin: two tines `reach` long toward local +Z, joined by a
    0.25 in web, then a rod (Ø`rod_d`, 1.5 in) or, with `rod_d` None, a 0.25 in base plate (a fitting on structure).
    Inches."""
    with BuildPart() as fork:
        for s in (1, -1):
            with Locations(Pos(s * (gap / 2 + tine / 2) * IN, 0, (reach / 2 - width / 4) * IN)):
                Box(tine * IN, width * IN, (reach + width / 2) * IN)
        span = gap + 2 * tine
        with Locations(Pos(0, 0, (reach + 0.125) * IN)):
            if rod_d is None:
                Box((span + 0.5) * IN, (width + 0.5) * IN, 0.25 * IN)
            else:
                Box(span * IN, width * IN, 0.25 * IN)
        if rod_d is not None:
            with Locations(Pos(0, 0, (reach + 0.25 + 0.75) * IN)):
                Cylinder(radius=rod_d / 2 * IN, height=1.5 * IN)
        with Locations(Rot(0, 90, 0)):
            Cylinder(radius=bore_d / 2 * IN, height=(span + 0.1) * IN, mode=Mode.SUBTRACT)
    return fork.part


def _placed(shape: Shape, pin: tuple[float, float, float], pin_axis, toward) -> Shape:
    """Move a `fork_end` so its pin sits at `pin` (inches) on `pin_axis`, the tines reaching along `toward`."""
    return shape.moved(Plane(origin=tuple(c * IN for c in pin), x_dir=pin_axis, z_dir=toward).location)


# Bell crank: pivot at the origin, pins along Z. The input rod end is 4.0 in out along +X and pulls along -Y; the output
# link is 2.5 in out along +Y and runs along -X. The crank is a male lug in each fork, up to 1.0 in thick.
BELL_CRANK_PINS_IN = {"pivot": (0.0, 0.0, 0.0), "input": (4.0, 0.0, 0.0), "output": (0.0, 2.5, 0.0)}


def pivot_fitting() -> Shape:
    """Fork fitting on structure: 0.75 in gap, 0.25 in tines 1.0 in wide, Ø0.377 pin bore; base 1.5 in below in -Y."""
    return _placed(fork_end(0.75, 0.25, 1.0, 0.377, 1.5, None), BELL_CRANK_PINS_IN["pivot"], (0, 0, 1), (0, -1, 0))


def input_rod_end() -> Shape:
    """Forked rod end: 0.40 in gap, 0.20 in tines, Ø0.377 bore; rod runs along -Y."""
    return _placed(fork_end(0.40, 0.20, 0.84, 0.377, 0.9, 0.40), BELL_CRANK_PINS_IN["input"], (0, 0, 1), (0, -1, 0))


def output_link_end() -> Shape:
    """Forked link end: 0.40 in gap, 0.20 in tines, Ø0.377 bore; the link runs along -X."""
    return _placed(fork_end(0.40, 0.20, 0.84, 0.377, 0.9, 0.40), BELL_CRANK_PINS_IN["output"], (0, 0, 1), (-1, 0, 0))


BELL_CRANK: dict[str, Callable[[], Shape]] = {
    "pivot_fitting": pivot_fitting,
    "input_rod_end": input_rod_end,
    "output_link_end": output_link_end,
}

REFERENCE_PARTS: dict[str, Callable[[], Shape]] = {
    "actuator_clevis_end": actuator_clevis_end,
    "side_panel": side_panel,
}

SAMPLES: dict[str, Callable[[], Shape]] = {
    "frame_rib_FS412": frame_rib,
    "actuator_rod_end": rod_end,
    "hyd_line_envelope": hyd_line_envelope,
}


def write_samples(out_dir: Path, samples: dict[str, Callable[[], Shape]] = SAMPLES) -> list[Path]:
    return [write_step(build(), out_dir / f"{name}.stp") for name, build in samples.items()]
