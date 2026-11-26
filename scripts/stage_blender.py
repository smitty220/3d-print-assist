#!/usr/bin/env python3
"""Download and stage standalone headless Blender 4.5 LTS into `backend/vendor/blender/`.

Used by local packaging and GitHub Actions (`.github/workflows/desktop-build.yml`) so both
macOS (`3D Print Assist.app`) and Windows 11 (`3DPrintAssist`) bundles ship with a working,
offline, zero-conflict Blender 4.5 LTS organic & flexi modeling engine.
"""

from __future__ import annotations

import os
import platform
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BLENDER_VERSION = "4.5.9"
BLENDER_SERIES = "Blender4.5"
BLENDER_BASE_URL = f"https://download.blender.org/release/{BLENDER_SERIES}"


def target_asset() -> tuple[str, str]:
    """Return `(filename, platform_tag)` for the current OS/arch."""
    mach = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
        return f"blender-{BLENDER_VERSION}-macos-{arch}.dmg", f"macos-{arch}"
    if sys.platform == "win32":
        arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
        return f"blender-{BLENDER_VERSION}-windows-{arch}.zip", f"windows-{arch}"
    return f"blender-{BLENDER_VERSION}-linux-x64.tar.xz", "linux-x64"


def staged_executable(dest_dir: Path) -> Path | None:
    if sys.platform == "darwin":
        exe = dest_dir / "Blender.app" / "Contents" / "MacOS" / "Blender"
        return exe if exe.is_file() else None
    if sys.platform == "win32":
        exe = dest_dir / "blender.exe"
        return exe if exe.is_file() else None
    exe = dest_dir / "blender"
    return exe if exe.is_file() else None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "3DPrintAssist-Packager/1.0"})
    ctx = _ssl_context()
    with urllib.request.urlopen(req, context=ctx, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -5
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        print(f"  {pct:3d}% ({done / 1e6:.1f} / {total / 1e6:.1f} MB)", flush=True)


def stage_blender(dest_dir: Path, force: bool = False) -> Path:
    existing = staged_executable(dest_dir)
    if existing is not None and not force:
        print(f"Blender already staged at {existing}")
        return existing

    filename, tag = target_asset()
    url = f"{BLENDER_BASE_URL}/{filename}"
    print(f"Staging Blender {BLENDER_VERSION} LTS ({tag}) from {url} ...")

    with tempfile.TemporaryDirectory(prefix="stage_blender_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / filename
        download_file(url, archive)

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            mnt = tmp_path / "mnt"
            mnt.mkdir()
            subprocess.run(
                ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mnt), str(archive)],
                check=True,
            )
            try:
                src_app = mnt / "Blender.app"
                if not src_app.is_dir():
                    raise RuntimeError(f"Blender.app not found inside {filename}")
                subprocess.run(["ditto", str(src_app), str(dest_dir / "Blender.app")], check=True)
            finally:
                subprocess.run(["hdiutil", "detach", str(mnt), "-quiet"], check=False)
        elif sys.platform == "win32":
            extract_dir = tmp_path / "extract"
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_dir)
            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            src = roots[0] if len(roots) == 1 and (roots[0] / "blender.exe").is_file() else extract_dir
            for item in src.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))
        else:
            import tarfile

            extract_dir = tmp_path / "extract"
            with tarfile.open(archive, "r:xz") as tf:
                tf.extractall(extract_dir)
            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            src = roots[0] if len(roots) == 1 and (roots[0] / "blender").is_file() else extract_dir
            for item in src.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))

    exe = staged_executable(dest_dir)
    if exe is None:
        raise RuntimeError(f"Staged Blender executable not found in {dest_dir}")
    ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True)
    print(f"Verified: {ver.stdout.splitlines()[0]}")
    return exe


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dest = repo_root / "backend" / "vendor" / "blender"
    force = "--force" in sys.argv
    stage_blender(dest, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
