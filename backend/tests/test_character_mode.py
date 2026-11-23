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


def test_flexi_print_in_place_articulation_magnet_and_hollow(tmp_path: Path):
    import manifold3d as m3d

    # Elongated shark-proportioned solid body (120 x 44 x 36 mm)
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([60.0, 22.0, 18.0])
    raw_path = tmp_path / "shark.stl"
    body.export(str(raw_path))

    flexi_prep = SculptPrep(
        target_height_mm=36.0,
        flatten_bottom_mm=3.0,
        flexi_segments=5,
        flexi_start_pct=32.0,
        flexi_end_pct=88.0,
        flexi_clearance_mm=0.45,
        flexi_angle_deg=24.0,
    )
    res = prepare_print_mesh(raw_path, flexi_prep, name="Flexi Bruce")
    assert res.watertight is True
    tm = trimesh.Trimesh(vertices=res.parts[0].points, faces=res.parts[0].tris, process=False)
    shells = tm.split(only_watertight=False)
    assert len(shells) == 6
    assert all(s.is_watertight for s in shells)
    # Verify pairwise zero-fusion (every pair of segments has strictly 0.0 intersection volume!)
    mfs = []
    for s in shells:
        m = m3d.Mesh(vert_properties=s.vertices.astype("float32"), tri_verts=s.faces.astype("uint32"))
        m.merge()
        mfs.append(m3d.Manifold(m))
    for i in range(len(mfs)):
        for j in range(i + 1, len(mfs)):
            assert abs((mfs[i] ^ mfs[j]).volume()) < 1e-4

    # Verify magnet pocket and hollow planter cavity reduce volume while staying watertight
    solid = prepare_print_mesh(raw_path, SculptPrep(target_height_mm=40.0, flatten_bottom_mm=2.0), name="Solid")
    with_mag = prepare_print_mesh(
        raw_path,
        SculptPrep(target_height_mm=40.0, flatten_bottom_mm=2.0, magnet_hole="bottom", magnet_dia_mm=8.2, magnet_depth_mm=3.2),
        name="Magnet",
    )
    assert with_mag.watertight is True and with_mag.volume_mm3 < solid.volume_mm3 - 50.0
    hollowed = prepare_print_mesh(
        raw_path,
        SculptPrep(target_height_mm=40.0, flatten_bottom_mm=2.0, hollow_cavity=True, hollow_wall_mm=3.0),
        name="Planter",
    )
    assert hollowed.watertight is True and hollowed.volume_mm3 < solid.volume_mm3 * 0.7


def test_async_character_job_deduplication_and_polling(monkeypatch, tmp_path: Path):
    import time

    app_module.reset(tmp_path)
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Async Test", "creation_mode": "character"})

    calls = {"n": 0}
    from print_assist.ml import painter

    def _slow_paint(**kw):
        calls["n"] += 1
        time.sleep(0.35)
        return _png_bytes(256, 256), ["front"]

    monkeypatch.setattr(painter, "generate_character_design_image", _slow_paint)

    # Start async job with wait_s=0.05 -> returns running immediately
    r1 = c.post("/api/character/design", json={"title": "Bruce", "prompt": "shark", "wait_s": 0.05}).json()
    assert r1["status"] == "running"
    # A duplicate retry attaches to the running job instead of spawning a second GPU process
    r2 = c.post("/api/character/design", json={"title": "Bruce", "prompt": "shark", "wait_s": 0.05}).json()
    assert r2["status"] == "running"
    assert calls["n"] == 1

    # Polling waits for completion and returns the finished Concept
    done = c.get("/api/character/job", params={"wait_s": 5}).json()
    assert done["status"] == "done" and done["result"]["id"] == "K1"
    assert calls["n"] == 1


def test_deferred_open_builds_in_background_and_returns_immediately(tmp_path: Path):
    import time
    from print_assist.build.workspace import Workspace
    from print_assist.project import Projects

    ps = Projects(tmp_path)
    p = ps.create("Box")
    w1 = Workspace(p)
    w1.change([
        {"op": "add_sketch", "name": "S", "placement": {"kind": "plane", "plane": "XY", "offset": 0},
         "entities": [
             {"id": "p1", "kind": "point", "x": 0, "y": 0}, {"id": "p2", "kind": "point", "x": 20, "y": 0},
             {"id": "p3", "kind": "point", "x": 20, "y": 20}, {"id": "p4", "kind": "point", "x": 0, "y": 20},
             {"id": "l1", "kind": "line", "p1": "p1", "p2": "p2"}, {"id": "l2", "kind": "line", "p1": "p2", "p2": "p3"},
             {"id": "l3", "kind": "line", "p1": "p3", "p2": "p4"}, {"id": "l4", "kind": "line", "p1": "p4", "p2": "p1"},
         ], "constraints": []},
        {"op": "add_feature", "kind": "extrude", "refs": {"sketch": "S1"}, "params": {"distance": 10}, "body": "B1"},
    ], message="box")
    w1.save("box")

    t0 = time.monotonic()
    w2 = Workspace(ps.open(p.slug), defer_build=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
    # Wait up to 5s for background thread to finish building
    deadline = time.monotonic() + 5.0
    while w2.tree()["building"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert w2.tree()["building"] is False
    assert w2.built_rev == 0
    assert w2.part()["volume"] > 3900.0


def test_character_archetype_resolution_and_flexi_landscape_framing():
    from print_assist.ml.painter import build_turnaround_prompt, resolve_character_style, style_canvas_size

    assert resolve_character_style("make a flexi shark like Bruce") == "flexi"
    assert resolve_character_style("cute articulated baby dragon") == "flexi"
    assert resolve_character_style("fridge magnet cat") == "magnet"
    assert resolve_character_style("succulent planter bulbasaur") == "planter"
    assert resolve_character_style("backpack keychain charm") == "keychain"
    assert resolve_character_style("character bust of a wizard") == "bust"
    assert resolve_character_style("wizard holding staff") == "figurine"
    assert resolve_character_style("Bruce shark", style="flexi") == "flexi"

    assert style_canvas_size("flexi", turnaround=False) == (768, 448)
    assert style_canvas_size("figurine", turnaround=False) == (640, 640)
    assert style_canvas_size("flexi", turnaround=True) == (960, 384)

    flexi_prompt = build_turnaround_prompt("Bruce cartoon great white shark", style="flexi")
    assert "elongated horizontal body" in flexi_prompt
    assert "NOT chubby or spherical" in flexi_prompt
    assert "no hinge cuts" in flexi_prompt


def test_sharp_edge_splitting_preserves_smooth_skin_and_splits_creases():
    import numpy as np
    from print_assist.geometry.mesh_prep import _split_sharp_edges

    # Cube has 8 vertices and 90° edges everywhere -> splitting should duplicate corner vertices (8 -> 24)
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    box.merge_vertices()
    pts_b = np.asarray(box.vertices, dtype=np.float64)
    tris_b = np.asarray(box.faces, dtype=np.int32)
    s_pts_b, s_tris_b = _split_sharp_edges(pts_b, tris_b, angle_deg=40.0)
    assert len(pts_b) == 8 and len(s_pts_b) == 24
    assert s_tris_b.shape == tris_b.shape

    # Smooth subdivided sphere has gentle angles (< 40°) -> zero splitting needed
    sph = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    pts_s = np.asarray(sph.vertices, dtype=np.float64)
    tris_s = np.asarray(sph.faces, dtype=np.int32)
    s_pts_s, _ = _split_sharp_edges(pts_s, tris_s, angle_deg=40.0)
    assert len(s_pts_s) == len(pts_s)


def test_web_search_mcp_tool_and_agy_allowance():
    from print_assist.ai import chat as chat_module

    tools = chat_module._tools()
    names = [t["name"] for t in tools]
    assert "web_search" in names
    assert "fetch_url" in names
    assert len(names) == len(set(names))  # no duplicates
    assert "search_web" in chat_module._AGY_SYSTEM_PREAMBLE
    assert "web_search" in chat_module._AGY_SYSTEM_PREAMBLE
    assert "search_web" in chat_module._AGY_TURN_REMINDER
