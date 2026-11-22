"""Tests for Character Mode: hardware tiering, watertight mesh repair/finishing, strict Photos separation,
Character Design Files, sculpting, print-prep updates, and STL/3MF export."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from print_assist import app as app_module
from print_assist.features.model import SculptPrep
from print_assist.geometry.mesh_prep import (
    export_prepared_meshes,
    prepare_print_mesh,
    render_prepared_meshes,
    to_viewer_mesh,
)
from print_assist.ml.hardware import select_tier
from print_assist.ml.installer import pick_sd_assets
from print_assist.printers import profiles


def _png_bytes(w: int = 300, h: int = 120) -> bytes:
    im = Image.new("RGB", (w, h), (180, 190, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _glb_bytes() -> bytes:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    return sphere.export(file_type="glb")


def test_hardware_tier_selection_mac_and_windows():
    mac_max = select_tier("darwin", "arm64", "mps", ram_gb=32.0, vram_gb=32.0)
    assert mac_max.tier == "high" and mac_max.sculptor_repo == "tencent/Hunyuan3D-2mv"

    win_rtx = select_tier("win32", "amd64", "cuda", ram_gb=32.0, vram_gb=16.0)
    assert win_rtx.tier == "high" and win_rtx.sculptor_repo == "tencent/Hunyuan3D-2mv"

    win_mid = select_tier("win32", "amd64", "cuda", ram_gb=16.0, vram_gb=8.0)
    assert win_mid.tier == "balanced" and win_mid.sculptor_repo == "tencent/Hunyuan3D-2mini"

    win_amd = select_tier("win32", "amd64", "vulkan", ram_gb=16.0, vram_gb=0.0)
    assert win_amd.tier == "balanced" and win_amd.accelerator == "vulkan"

    win_cpu = select_tier("win32", "amd64", "cpu", ram_gb=8.0, vram_gb=0.0)
    assert win_cpu.tier == "fast" and win_cpu.sculptor_repo == "tencent/Hunyuan3D-2mini"


def test_pick_sd_assets_for_macos_and_windows():
    assets = [
        {"name": "sd-master-123-bin-Darwin-macOS-26-arm64.zip", "browser_download_url": "https://ex/mac.zip"},
        {"name": "sd-master-123-bin-win-cuda12-x64.zip", "browser_download_url": "https://ex/cu.zip"},
        {"name": "cudart-sd-bin-win-cu12-x64.zip", "browser_download_url": "https://ex/cudart.zip"},
        {"name": "sd-master-123-bin-win-vulkan-x64.zip", "browser_download_url": "https://ex/vk.zip"},
        {"name": "sd-master-123-bin-win-cpu-x64.zip", "browser_download_url": "https://ex/cpu.zip"},
    ]
    mac = pick_sd_assets(assets, select_tier("darwin", "arm64", "mps", 32, 32))
    assert [a["name"] for a in mac] == ["sd-master-123-bin-Darwin-macOS-26-arm64.zip"]

    cu = pick_sd_assets(assets, select_tier("win32", "amd64", "cuda", 32, 16))
    assert len(cu) == 2 and any("win-cuda" in a["name"] for a in cu) and any("cudart" in a["name"] for a in cu)

    vk = pick_sd_assets(assets, select_tier("win32", "amd64", "vulkan", 16, 0))
    assert [a["name"] for a in vk] == ["sd-master-123-bin-win-vulkan-x64.zip"]

    cpu = pick_sd_assets(assets, select_tier("win32", "amd64", "cpu", 8, 0))
    assert [a["name"] for a in cpu] == ["sd-master-123-bin-win-cpu-x64.zip"]


def test_mesh_prep_repairs_broken_mesh_scales_adds_base_and_loop_and_exports(tmp_path: Path):
    # Create an intentionally broken open mesh (sphere missing several faces -> non-watertight)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    broken = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[:-12], process=False)
    assert not broken.is_watertight
    raw_path = tmp_path / "broken.stl"
    broken.export(str(raw_path), file_type="stl")

    prep = SculptPrep(
        target_height_mm=60.0,
        flatten_bottom_mm=0.5,
        base_style="round",
        base_height_mm=3.0,
        keychain_loop=True,
        keychain_hole_mm=4.5,
    )
    res = prepare_print_mesh(raw_path, prep, name="Hero Figurine")
    assert res.watertight is True
    assert res.volume_mm3 > 1000.0
    assert res.triangles > 100
    assert abs(res.bbox_min[2]) < 0.05  # placed cleanly on bed Z = 0
    assert res.bbox_max[2] > 60.0  # base + figurine + keychain loop

    vm = to_viewer_mesh(res.parts[0], "M1")
    assert vm["name"] == "M1" and len(vm["faces"][0]["positions"]) > 0

    png = render_prepared_meshes(res.parts, "iso", 256)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    printer = next(iter(profiles().values()))
    stl_out = export_prepared_meshes(res.parts, printer, "stl", tmp_path / "out.stl")
    assert stl_out.is_file() and stl_out.stat().st_size > 200

    mf_out = export_prepared_meshes(res.parts, printer, "3mf", tmp_path / "out.3mf", title="Hero")
    assert mf_out.is_file()
    with zipfile.ZipFile(mf_out) as zf:
        assert "3D/3dmodel.model" in zf.namelist()


def test_character_mode_end_to_end_api_and_strict_photos_separation(monkeypatch, tmp_path: Path):
    app_module.reset(tmp_path)
    c = TestClient(app_module.app)

    # Create project in Character Mode
    p = c.post("/api/projects", json={"name": "Moon Family", "creation_mode": "character"}).json()
    assert p["settings"]["creation_mode"] == "character"

    # Toggle Direct Build bypass in settings
    st = c.put("/api/settings", json={"direct_build": True}).json()
    assert st["direct_build"] is True and st["creation_mode"] == "character"
    c.put("/api/settings", json={"direct_build": False})

    # Upload a user reference photo -> goes into Photos (/api/images) with model == "upload"
    up = c.post("/api/images/upload", files={"file": ("family.png", _png_bytes(200, 200), "image/png")}).json()
    assert up["id"] == "I1" and up["model"] == "upload"

    # Mock local 2D Painter & 3D Sculptor
    from print_assist.ml import painter, sculptor

    monkeypatch.setattr(
        painter,
        "generate_character_design_image",
        lambda **kw: (_png_bytes(600, 200), ["front", "side", "back"]),
    )
    monkeypatch.setattr(sculptor, "sculpt_mesh_from_image", lambda *a, **kw: _glb_bytes())

    # Generate Character Mode Design File (turnaround)
    cd = c.post("/api/character/design", json={
        "title": "Family on Moon",
        "prompt": "cartoon family on the moon fighting ninja aliens",
        "reference_photos": ["I1"],
        "turnaround": True,
    })
    assert cd.status_code == 200
    concept = cd.json()
    assert concept["id"] == "K1" and concept["mode"] == "character" and concept["views"] == ["front", "side", "back"]

    # Confirm Design File image is served at /api/concepts/K1/picture and did NOT pollute Photos (/api/images)
    pic = c.get("/api/concepts/K1/picture")
    assert pic.status_code == 200 and pic.content[:8] == b"\x89PNG\r\n\x1a\n"
    imgs = c.get("/api/images").json()
    assert len(imgs) == 1 and imgs[0]["id"] == "I1" and imgs[0]["model"] == "upload"

    # Approve concept and sculpt 3D model
    assert c.post("/api/concepts/K1/answer", json={"decision": "approve", "note": ""}).status_code == 200
    sc = c.post("/api/character/sculpt", json={
        "concept_id": "K1",
        "target_height_mm": 75.0,
        "base_style": "round",
        "keychain_loop": False,
    })
    assert sc.status_code == 200
    sculpt = sc.json()["sculpt"]
    assert sculpt["id"] == "M1" and sculpt["watertight"] is True and sculpt["triangles"] > 0

    # Verify part(), checks(), render_preview, live prep update, and 3MF/STL export all work
    part = c.get("/api/part").json()
    assert len(part["bodies"]) == 1 and part["bodies"][0]["id"] == "M1" and part["volume"] > 0
    checks = c.get("/api/checks").json()
    assert any(f["title"] == "Watertight mesh" and f["status"] == "ok" for f in checks["part"])
    ren = c.get("/api/render", params={"view": "iso", "size": 256})
    assert ren.status_code == 200 and ren.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Instant print-prep update (add keychain loop & resize to 90 mm)
    up_prep = c.post("/api/sculpts/M1/prep", json={"target_height_mm": 90.0, "keychain_loop": True})
    assert up_prep.status_code == 200 and up_prep.json()["sculpt"]["prep"]["keychain_loop"] is True

    exp_3mf = c.get("/api/export/3mf")
    assert exp_3mf.status_code == 200 and len(exp_3mf.content) > 200
    exp_stl = c.get("/api/export/stl")
    assert exp_stl.status_code == 200 and len(exp_stl.content) > 200

    # ML status endpoint works cross-platform
    mls = c.get("/api/ml/status").json()
    assert "hardware" in mls and "ready" in mls
