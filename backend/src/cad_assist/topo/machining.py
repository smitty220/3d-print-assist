"""Machining set-up for topology optimisation: a 4-axis mill with the rotary along one design-space axis.

The 2.5D parametrisation it drives lives in `prismatic.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES = "xyz"


@dataclass
class MachiningSpec:
    """How the part will be machined: 2.5D cuts from the faces of a 4-axis mill indexed every 90° about the
    rotary axis (plus the open end, if any), each with a few flat floor depths."""

    axis: str = "auto"  # "auto" (longest side of the design space) | "x" | "y" | "z", in the design-space frame
    depths: int = 3  # floor depths per setup face (equal steps inward from that face)
    free_end: str = "none"  # "none" | "+" | "-": the end away from the chuck, cut along the axis
    tool_diameter: float = 6.35  # mm; smallest cutter: smallest pocket width and corner radius

    def axis_index(self, grid_shape: tuple[int, int, int]) -> int:
        if self.axis == "auto":
            return int(np.argmax(grid_shape))
        return AXES.index(self.axis)

    def open_end(self) -> int:
        return {"none": 0, "+": 1, "-": -1}[self.free_end]

    def filter_radius(self, h: float) -> float:
        """Filter radius in elements: about the cutter radius, never below the default 1.5."""
        return max(1.5, self.tool_diameter / 2 / h)

    def label(self, grid_shape: tuple[int, int, int]) -> str:
        a = AXES[self.axis_index(grid_shape)].upper()
        end = "" if self.free_end == "none" else f" + {self.free_end}{a} end"
        return f"2.5D, 4-axis about {a}{end}, {self.depths} floor depths"
