"""Variants and colorways: the same design in other sizes and other colours, for building and exporting in a batch.

A variant overrides some parameters (a 60-card box: `cards = 60`); a colorway swaps the palette's colours. Either
gives an ordinary design to build, so every check and export works on it unchanged.
"""

from __future__ import annotations

from .features.model import Colorway, Design, Variant
from .printers import PrinterProfile
from .validate import Finding


def with_variant(d: Design, v: Variant | None) -> Design:
    """The design with the variant's parameter values in place of its own."""
    out = d.model_copy(deep=True)
    if v is None:
        return out
    for name, expr in v.overrides.items():
        p = out.parameter(name)
        if p is not None:
            p.expr = expr
    return out


def with_colorway(d: Design, c: Colorway | None) -> Design:
    """The design with the colorway's filament colours (and names) in its palette."""
    out = d.model_copy(deep=True)
    if c is None:
        return out
    for f in out.filaments:
        if f.slot in c.colors:
            f.color = c.colors[f.slot]
        if f.slot in c.names:
            f.name = c.names[f.slot]
    return out


def check_filaments(d: Design, built: list[str], printer: PrinterProfile | None) -> list[Finding]:
    """Every built body prints in a palette filament the printer can load: its slot exists on the printer and feeds a
    nozzle the printer has, and the design needs no more filaments than the printer holds at once."""
    out: list[Finding] = []
    F = lambda *a, **k: out.append(Finding("part", *a, **k))  # noqa: E731
    bodies = [b for b in d.bodies if b.id in built]
    if not d.filaments:
        if len(bodies) > 1:
            F("filament", "Filaments", "warn", f"{len(bodies)} bodies but no palette: they will print in one colour "
              "unless you assign filaments", refs=[b.id for b in bodies])
        return out
    unassigned = [b for b in bodies if b.filament is None]
    if unassigned:
        F("filament", "Filament for every body", "warn",
          "No filament: " + ", ".join(f"{b.name} ({b.id})" for b in unassigned), refs=[b.id for b in unassigned])
    used = sorted({b.filament for b in bodies if b.filament is not None})
    if printer is not None:
        bad = []
        for slot in used:
            f = d.filament(slot)
            if f is None:
                continue
            if f.nozzle >= len(printer.nozzles):
                bad.append(f"slot {slot} feeds nozzle {f.nozzle + 1}, the {printer.name} has {len(printer.nozzles)}")
            elif slot > printer.slots:
                bad.append(f"slot {slot} is past the {printer.name}'s {printer.slots}")
        if len(used) > printer.slots:
            bad.append(f"{len(used)} filaments, the {printer.name} holds {printer.slots}")
        if bad:
            F("filament", "Filaments fit the printer", "fail", "; ".join(bad))
        elif used:
            names = ", ".join(f"{s}: {d.filament(s).name or d.filament(s).color}" for s in used)
            F("filament", "Filaments fit the printer", "ok", f"{len(used)} of {printer.slots} slots ({names})",
              value=float(len(used)), quantity="count")
    return out
