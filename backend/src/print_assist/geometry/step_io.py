"""STEP read/write. Shapes are held in millimetres internally (OCCT's native unit)."""

from __future__ import annotations

from pathlib import Path

from build123d import Shape, export_step, import_step


class StepImportError(ValueError):
    pass


def read_step(path: Path) -> Shape:
    try:
        shape = import_step(str(path))
    except Exception as exc:  # OCCT raises a mix of types for malformed files
        raise StepImportError(f"Could not read STEP file {path.name}: {exc}") from exc
    if not shape.faces():
        raise StepImportError(f"STEP file {path.name} contains no faces")
    return shape


def write_step(shape: Shape, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    export_step(shape, str(path))
    return path
