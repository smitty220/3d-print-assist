# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for 3D Print Assist desktop app (macOS .app and Windows 11 .exe)."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
FRONTEND_DIST = SPEC_DIR.parent / "frontend" / "dist"
ICON_PATH = SPEC_DIR / "assets" / ("icon.icns" if sys.platform == "darwin" else "icon.ico")

datas = [
    (str(FRONTEND_DIST), "frontend_dist"),
    *collect_data_files("print_assist"),
]
binaries = []
hiddenimports = [
    *collect_submodules("uvicorn"),
    *collect_submodules("print_assist"),
    *collect_submodules("print_assist_mcp"),
]

for pkg in ("OCP", "build123d", "lib3mf", "manifold3d", "networkx", "pillow_heif", "resvg_py", "skimage", "trimesh", "webview"):
    pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_bins)
    hiddenimports.extend(pkg_hidden)

a = Analysis(
    [str(SPEC_DIR / "src" / "print_assist" / "desktop.py")],
    pathex=[str(SPEC_DIR / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="3DPrintAssist",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ICON_PATH),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="3DPrintAssist",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="3D Print Assist.app",
        icon=str(ICON_PATH),
        bundle_identifier="com.printassist.desktop",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.1.0",
        },
    )
