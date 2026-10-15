"""Built-in fastener catalog.

Dimensions in inches, simplified for display and hole sizing; they are approximate and must be
checked against the governing NAS/MS standards before release use. Converted to mm at the API edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

IN = 25.4


@dataclass(frozen=True)
class Fastener:
    part_number: str
    description: str
    shank_d: float
    head: str  # "socket" | "hex"
    head_size: float  # socket: head diameter; hex: across flats
    head_height: float
    hole_close: float
    washer: str
    washer_od: float
    nut: str
    nut_af: float

    def to_mm_dict(self) -> dict:
        d = asdict(self)
        for k in ("shank_d", "head_size", "head_height", "hole_close", "washer_od", "nut_af"):
            d[k] = round(d[k] * IN, 4)
        return d


FASTENERS: dict[str, Fastener] = {
    f.part_number: f
    for f in [
        Fastener("NAS1352-3", "Socket head cap screw, #10-32", 0.190, "socket", 0.312, 0.190, 0.201, "NAS1149F0363P", 0.438, "MS21042L3", 0.3125),
        Fastener("NAS1352-4", "Socket head cap screw, 1/4-28", 0.250, "socket", 0.375, 0.250, 0.257, "NAS1149F0463P", 0.500, "MS21042L4", 0.375),
        Fastener("NAS6603", "Hex head bolt, close tolerance, 3/16", 0.1875, "hex", 0.3125, 0.125, 0.190, "NAS1149F0363P", 0.438, "MS21042L3", 0.3125),
        Fastener("NAS6604", "Hex head bolt, close tolerance, 1/4", 0.250, "hex", 0.375, 0.156, 0.250, "NAS1149F0463P", 0.500, "MS21042L4", 0.375),
    ]
}


def suggest_fastener(hole_d_mm: float) -> str | None:
    """Nearest catalog entry whose shank fits the hole with at most 0.040 in diametral clearance."""
    d_in = hole_d_mm / IN
    fits = [f for f in FASTENERS.values() if 0 <= d_in - f.shank_d <= 0.040]
    if not fits:
        return None
    return min(fits, key=lambda f: abs(f.hole_close - d_in)).part_number
