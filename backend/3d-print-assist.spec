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
    (str(SPEC_DIR / "src" / "print_assist" / "ml" / "pa_blender_lib.py"), "print_assist/ml"),
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

# Bundle standalone headless Blender 4.5 LTS if staged in `backend/vendor/blender/`.
# Copied post-BUNDLE/COLLECT to preserve Blender's own macOS code signature & symlinks intact.
import shutil
import subprocess

VENDOR_BLENDER = SPEC_DIR / "vendor" / "blender"
if sys.platform == "darwin":
    src_blender_app = VENDOR_BLENDER / "Blender.app"
    if src_blender_app.is_dir():
        dst_blender_app = Path(DISTPATH) / "3D Print Assist.app" / "Contents" / "Resources" / "blender" / "Blender.app"
        if dst_blender_app.exists():
            shutil.rmtree(dst_blender_app)
        dst_blender_app.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ditto", str(src_blender_app), str(dst_blender_app)], check=True)
        print(f"[3d-print-assist.spec] Bundled Blender.app -> {dst_blender_app}")
elif sys.platform == "win32":
    if (VENDOR_BLENDER / "blender.exe").is_file():
        dst_blender = Path(DISTPATH) / "3DPrintAssist" / "blender"
        if dst_blender.exists():
            shutil.rmtree(dst_blender)
        shutil.copytree(VENDOR_BLENDER, dst_blender)
        print(f"[3d-print-assist.spec] Bundled Windows Blender -> {dst_blender}")
else:
    if (VENDOR_BLENDER / "blender").is_file():
        dst_blender = Path(DISTPATH) / "3DPrintAssist" / "blender"
        if dst_blender.exists():
            shutil.rmtree(dst_blender)
        shutil.copytree(VENDOR_BLENDER, dst_blender, symlinks=True)
        print(f"[3d-print-assist.spec] Bundled Linux Blender -> {dst_blender}")
