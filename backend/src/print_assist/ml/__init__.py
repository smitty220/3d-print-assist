"""Local, cross-platform (macOS Apple Silicon + Windows) Character Mode ML engine.

All binaries, Python ML environments, and open-weight models live in an on-demand folder outside the
app bundle (`ml_root()`), keeping the desktop installer lightweight and zero-cost for every LLM harness.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..project import data_root


def ml_root() -> Path:
    """Shared folder for local 2D/3D engines and weights (outside the PyInstaller bundle)."""
    if env := os.environ.get("PRINT_ASSIST_ML_DIR"):
        return Path(env).expanduser()
    # Keep beside the `designs/` folder (e.g. ~/Projects/3DPrinting/ml-engine)
    return data_root().parent / "ml-engine"
