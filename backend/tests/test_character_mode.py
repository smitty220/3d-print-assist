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
    assert style_canvas_size("flexi", turnaround=True) == (768, 448)
    assert style_canvas_size("figurine", turnaround=True) == (896, 672)

    flexi_prompt = build_turnaround_prompt("Bruce cartoon great white shark", style="flexi")
    assert "elongated horizontal body" in flexi_prompt
    assert "NOT chubby or spherical" in flexi_prompt
    assert "no hinge cuts" in flexi_prompt

    quad_prompt = build_turnaround_prompt("Bruce cartoon great white shark", turnaround=True, style="figurine")
    assert "2x2 four-view" in quad_prompt


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
    assert "inspect_sculpt_anatomy" in names
    assert len(names) == len(set(names))  # no duplicates
    assert "search_web" in chat_module._AGY_SYSTEM_PREAMBLE
    assert "web_search" in chat_module._AGY_SYSTEM_PREAMBLE
    assert "read_file" in chat_module._AGY_SYSTEM_PREAMBLE
    assert "search_web" in chat_module._AGY_TURN_REMINDER
    assert "read_file" in chat_module._AGY_TURN_REMINDER


def test_anatomy_safe_flexi_solver_and_hybrid_cad_booleans(tmp_path: Path):
    import manifold3d as m3d
    import numpy as np

    # 1. Build a finned shark mesh with pectoral fins + dorsal fin at 40%..55% and a forked tail fin at 85%..100%
    body = m3d.Manifold.sphere(1.0, 36).scale((55.0, 18.0, 18.0)).translate((0.0, 0.0, 18.0))
    pect_l = m3d.Manifold.sphere(1.0, 24).scale((8.0, 18.0, 3.0)).translate((-5.0, 18.0, 8.0))
    pect_r = m3d.Manifold.sphere(1.0, 24).scale((8.0, 18.0, 3.0)).translate((-5.0, -18.0, 8.0))
    dorsal = m3d.Manifold.sphere(1.0, 24).scale((8.0, 3.0, 15.0)).translate((-5.0, 0.0, 35.0))
    tail_fin = m3d.Manifold.sphere(1.0, 24).scale((7.0, 2.5, 18.0)).translate((52.0, 0.0, 22.0))
    shark = body + pect_l + pect_r + dorsal + tail_fin
    sm = shark.to_mesh()
    raw_path = tmp_path / "finned_shark.stl"
    trimesh.Trimesh(vertices=sm.vert_properties[:, :3], faces=sm.tri_verts, process=False).export(str(raw_path))

    # Auto anatomy solver detects the pectoral+dorsal fin zone and avoids cutting inside it
    from print_assist.geometry.mesh_prep import inspect_sculpt_anatomy_file

    anat = inspect_sculpt_anatomy_file(raw_path, SculptPrep(target_height_mm=60.0, flexi_segments=4))
    assert len(anat["protected_fin_zones"]) >= 1
    for z in anat["protected_fin_zones"]:
        for cut_p in anat["recommended_cuts_pct"]:
            assert not (z["start_pct"] + 1.0 < cut_p < z["end_pct"] - 1.0)

    # Explicit flexi_cuts_pct and hybrid CAD boolean cut_bodies via API
    app_module.reset(tmp_path)
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Hybrid Shark", "creation_mode": "character"})
    up = c.post(
        "/api/sculpts/upload",
        params={"name": "Bruce", "target_height_mm": 60.0},
        files={"file": ("shark.stl", raw_path.read_bytes(), "model/stl")},
    ).json()
    assert up["sculpt"]["id"] == "M1"
    vol_before = up["sculpt"]["volume_mm3"]

    # Inspect anatomy via API endpoint
    anat_api = c.get("/api/sculpts/M1/anatomy").json()
    assert anat_api["id"] == "M1" and len(anat_api["recommended_cuts_pct"]) >= 2

    # Build a parametric CAD cutter box in B1 via apply_changes, then subtract it from M1 via cut_bodies=["B1"]
    ops_res = c.post("/api/ops", json={"message": "cutter slot", "ops": [
        {"op": "add_sketch", "name": "Slot", "placement": {"kind": "plane", "plane": "XY", "offset": 0},
         "entities": [
             {"id": "p1", "kind": "point", "x": -8, "y": -6}, {"id": "p2", "kind": "point", "x": 8, "y": -6},
             {"id": "p3", "kind": "point", "x": 8, "y": 6}, {"id": "p4", "kind": "point", "x": -8, "y": 6},
             {"id": "l1", "kind": "line", "p1": "p1", "p2": "p2"}, {"id": "l2", "kind": "line", "p1": "p2", "p2": "p3"},
             {"id": "l3", "kind": "line", "p1": "p3", "p2": "p4"}, {"id": "l4", "kind": "line", "p1": "p4", "p2": "p1"},
         ], "constraints": []},
        {"op": "add_feature", "kind": "extrude", "refs": {"sketch": "S1"}, "params": {"distance": 25}, "body": "B1"},
    ]})
    assert ops_res.status_code == 200

    # Same flexi settings without the CAD cut first (flexi seating/bed pads change volume on their own)
    base_res = c.post("/api/sculpts/M1/prep", json={"cut_bodies": [], "flexi_segments": 3}).json()
    vol_flexi = base_res["sculpt"]["volume_mm3"]
    assert vol_before > 0
    prep_res = c.post("/api/sculpts/M1/prep", json={"cut_bodies": ["B1"], "flexi_segments": 3}).json()
    assert prep_res["sculpt"]["watertight"] is True
    assert prep_res["sculpt"]["volume_mm3"] < vol_flexi - 100.0
    # Consumed tool body B1 is merged into M1 and hidden from standalone viewport bodies
    part = c.get("/api/part").json()
    assert [b["id"] for b in part["bodies"]] == ["M1"]


def test_pixar_to_flexi_design_conversion_and_clean_sculpt_inheritance(monkeypatch, tmp_path: Path):
    """First create a Pixar shark concept (K1), then convert it to a flexi-shark concept (K2) via from_concept_id,
    and verify K2.png shows the articulated flexi cuts while sculpt_character sculpts K2_clean.png and inherits flexi."""
    from PIL import Image, ImageDraw
    from print_assist.ml import painter, sculptor

    app_module.reset(tmp_path)
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Pixar to Flexi", "creation_mode": "character"})

    # Build a synthetic shark-like 2D concept image on a light studio background
    im = Image.new("RGB", (640, 400), (250, 250, 252))
    dr = ImageDraw.Draw(im)
    # Dorsal fin peak in the middle, blue upper body, cream belly
    dr.polygon([(270, 150), (335, 65), (360, 160)], fill=(85, 112, 138))
    dr.ellipse([60, 130, 560, 290], fill=(85, 112, 138))
    dr.chord([75, 160, 540, 295], start=10, end=170, fill=(232, 228, 212))
    # Pectoral fin hanging below cream belly
    dr.polygon([(275, 255), (385, 340), (330, 250)], fill=(82, 108, 134))
    # Tail fin
    dr.polygon([(520, 210), (605, 115), (585, 210), (605, 295)], fill=(85, 112, 138))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    shark_png = buf.getvalue()

    paint_calls = {"n": 0}

    def _mock_paint(**kw):
        paint_calls["n"] += 1
        return shark_png, ["front"]

    sculpted_from: list[Path] = []

    def _mock_sculpt(img_path: Path, **kw):
        sculpted_from.append(Path(img_path))
        return _glb_bytes()

    monkeypatch.setattr(painter, "generate_character_design_image", _mock_paint)
    monkeypatch.setattr(sculptor, "sculpt_mesh_from_image", _mock_sculpt)

    # Step 1: Propose initial Pixar-style figurine concept K1 ("It's ok to make it pixar the first time")
    k1 = c.post("/api/character/design", json={
        "title": "Bruce (Pixar Figurine)",
        "prompt": "Pixar style great white shark Bruce",
        "style": "figurine",
    }).json()
    assert k1["id"] == "K1"
    assert k1["style"] == "figurine"
    assert len(k1["views"]) == 4
    assert k1["flexi_segments"] == 0
    assert k1["flexi_cuts_pct"] == []
    assert paint_calls["n"] == 1

    # Step 2: User says "now make it a flexi-shark" -> convert K1 into 4-view bed-seated flexi concept K2 (repaint=False)
    k2 = c.post("/api/character/design", json={
        "title": "Bruce (Articulated Flexi Shark)",
        "prompt": "Articulated print-in-place flexi shark Bruce",
        "from_concept_id": "K1",
        "style": "flexi",
        "flexi_segments": 6,
        "repaint": False,
    }).json()
    assert k2["id"] == "K2"
    assert k2["style"] == "flexi"
    assert len(k2["views"]) == 4
    assert k2["flexi_segments"] == 6
    assert len(k2["flexi_cuts_pct"]) == 5
    # No second GPU paint call was needed!
    assert paint_calls["n"] == 1
    # None of the fin-safe cuts fall inside the dorsal fin (42..57%)
    for cp in k2["flexi_cuts_pct"]:
        assert not (42.5 <= cp <= 57.5), f"2D flexi concept cut {cp}% hit the dorsal fin!"

    # Verify K2.png is the 1280x900 4-view bed-seated engineering sheet, while K2_clean.png is the un-cut hero
    k2_preview_bytes = c.get("/api/concepts/K2/picture").content
    assert k2_preview_bytes != shark_png
    with Image.open(io.BytesIO(k2_preview_bytes)) as k2_im:
        assert k2_im.size == (1280, 900)
    ws = app_module.ws()
    clean_p = ws.concepts.clean_image_path("K2")
    assert clean_p is not None and clean_p.name == "K2_clean.png"
    assert clean_p.read_bytes() == shark_png

    # Step 3: User clicks "Approve and build" on K2 -> sculpts from K2_clean.png and inherits flexi_segments=6
    c.post("/api/concepts/K2/answer", json={"decision": "approve", "note": "Build flexi shark"})
    sc = c.post("/api/character/sculpt", json={"concept_id": "K2", "target_height_mm": 60.0}).json()
    assert sc["sculpt"]["watertight"] is True
    assert sc["sculpt"]["prep"]["flexi_segments"] == 6
    assert len(sculpted_from) == 1 and sculpted_from[0].name == "K2_clean.png"

    # Step 4: Verify 4-view 3D render_preview("sheet") and per-segment Z=0 bed-contact printability checks
    sheet_ren = c.get("/api/render", params={"view": "sheet", "size": 640})
    assert sheet_ren.status_code == 200 and sheet_ren.content[:8] == b"\x89PNG\r\n\x1a\n"
    checks = c.get("/api/checks").json()
    bed_check = next(f for f in checks["part"] if f["code"] == "bed_contact")
    assert bed_check["status"] in ("ok", "warn"), f"Unexpected floating segments: {bed_check}"
    assert "float above Z=0" not in bed_check["detail"]


def test_chubby_sculpt_auto_streamlining_and_zero_floating_z_pin_flexi(tmp_path: Path):
    """Verify a chubby (~1.42:1) finned figurine sculpt automatically streamlines into a sleek >= 2.15:1
    flexi profile with captive Z-axis swimming joints, zero fused segments, and 0 floating components."""
    import manifold3d as m3d
    from print_assist.geometry.mesh_prep import analyze_sculpt_printability

    # Chubby 1.42:1 finned character (length ~96 mm, width ~68 mm, height ~67 mm)
    torso = m3d.Manifold.sphere(1.0, 36).scale((44.0, 32.0, 28.0)).translate((0.0, 0.0, 28.0))
    dorsal = m3d.Manifold.sphere(1.0, 24).scale((8.0, 3.0, 14.0)).translate((0.0, 0.0, 53.0))
    pect_l = m3d.Manifold.sphere(1.0, 20).scale((7.0, 12.0, 2.8)).translate((0.0, 30.0, 14.0))
    pect_r = m3d.Manifold.sphere(1.0, 20).scale((7.0, 12.0, 2.8)).translate((0.0, -30.0, 14.0))
    tail = m3d.Manifold.sphere(1.0, 24).scale((8.0, 2.5, 16.0)).translate((44.0, 0.0, 30.0))
    chubby = torso + dorsal + pect_l + pect_r + tail
    cm = chubby.to_mesh()
    raw_path = tmp_path / "chubby_shark.stl"
    trimesh.Trimesh(vertices=cm.vert_properties[:, :3], faces=cm.tri_verts, process=False).export(str(raw_path))

    prep = SculptPrep(target_height_mm=65.0, flexi_segments=5)
    res = prepare_print_mesh(raw_path, prep, name="Streamlined Flexi")
    assert res.watertight is True
    sx = res.bbox_max[0] - res.bbox_min[0]
    sz = res.bbox_max[2] - res.bbox_min[2]
    assert sx / sz >= 2.10, f"Expected sleek >= 2.10:1 streamlined ratio, got {sx / sz:.2f}:1"
    assert any("streamlined torso/tail" in n for n in res.notes)

    pr = analyze_sculpt_printability(res.parts)
    assert pr["component_count"] >= 4
    assert pr["floating_count"] == 0, f"Expected all segments seated on Z=0, got: {pr}"

    sheet_png = render_prepared_meshes(res.parts, view="sheet", size=640)
    assert sheet_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_multiview_quad_splitting_and_multicolor_mmu3_3mf_export(tmp_path: Path):
    """Verify 2x2 painted turnaround sheets extract all 3 orthographic views for Hunyuan3D-2mv, and
    multi-color volumetric splitting produces watertight non-overlapping MMU3 parts + PrusaSlicer 3MF config."""
    import manifold3d as m3d
    from PIL import ImageDraw
    from print_assist.geometry.mesh_prep import analyze_sculpt_printability
    from print_assist.ml.sculptor import _split_turnaround

    # 1. Verify 2x2 painted quad sheet extracts front, left, back views for Hunyuan3D-2mv
    quad_p = tmp_path / "quad.png"
    Image.new("RGB", (896, 672), (240, 240, 242)).save(quad_p)
    views_out = _split_turnaround(quad_p, tmp_path / "mv", ["3/4 hero", "front", "side (Z=0)", "back"])
    assert set(views_out.keys()) == {"front", "left", "back"}

    # 2. Create a two-tone concept image (slate blue + cream belly) and a detailed >4k-face shark sculpt
    c_im = Image.new("RGB", (512, 320), (248, 248, 250))
    dr = ImageDraw.Draw(c_im)
    dr.ellipse([40, 60, 470, 240], fill=(78, 108, 140))
    dr.chord([50, 120, 460, 250], start=0, end=180, fill=(236, 230, 214))
    concept_p = tmp_path / "concept.png"
    c_im.save(concept_p)

    # Smooth elongated solid (5,120 faces > 4000 to exercise the full SDF Flexi-Factory + MMU3 pipeline!)
    body = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    body.apply_scale([62.0, 22.0, 20.0])
    body.apply_translation([0.0, 0.0, 20.0])
    raw_p = tmp_path / "hi_shark.stl"
    body.export(str(raw_p))

    prep = SculptPrep(target_height_mm=44.0, flexi_segments=4, flexi_angle_deg=24.0, color_mode="auto")
    res = prepare_print_mesh(raw_p, prep, name="Bruce", concept_image_path=concept_p, multi_extruder=True)
    assert res.watertight is True
    assert len(res.parts) == 2, f"Expected 2-color Body+Belly parts, got {[p.name for p in res.parts]}"
    assert res.parts[0].color != res.parts[1].color

    # Verify unified physical printability across multi-color parts (0 floating segments!)
    pr = analyze_sculpt_printability(res.parts)
    assert pr["component_count"] >= 4
    assert pr["floating_count"] == 0, f"Multi-color flexi segments should be seated on Z=0: {pr}"

    # Verify PrusaSlicer MMU3 extruder metadata is embedded in exported 3MF
    printer = profiles().get("prusa_mk4s_mmu3") or next(iter(profiles().values()))
    mf_out = export_prepared_meshes(res.parts, printer, "3mf", tmp_path / "bruce_mmu3.3mf", title="Flexi Bruce")
    with zipfile.ZipFile(mf_out) as zf:
        names = set(zf.namelist())
        assert "3D/3dmodel.model" in names
        assert "Metadata/Slic3r_PE_model.config" in names
        cfg_xml = zf.read("Metadata/Slic3r_PE_model.config").decode("utf-8")
        assert 'key="extruder" value="1"' in cfg_xml
        assert 'key="extruder" value="2"' in cfg_xml


def test_blender_engine_discovery_and_ml_status(tmp_path: Path):
    """Verify headless Blender discovery reports ready status in /api/ml/status and respects env overrides."""
    from print_assist.ml.blender import blender_status, find_blender_executable

    app_module.reset(tmp_path)
    c = TestClient(app_module.app)
    st = c.get("/api/ml/status").json()
    assert "blender_ready" in st
    assert "blender_version" in st
    assert "blender_source" in st
    exe = find_blender_executable()
    bl = blender_status()
    assert bl["ready"] == (exe is not None)
    assert st["blender_ready"] == bl["ready"]


def test_blender_multicolor_subsurf_flexi_end_to_end(tmp_path: Path):
    """Verify bundled headless Blender 4.5 LTS builds a glass-smooth 4-color SubSurf Flexi-Factory character,
    preserves every filament color & Z=0 seated link without voxel fallback, supports replace_sculpt_id,
    and exports multi-extruder MMU3 3MF."""
    import pytest
    from print_assist.ml.blender import find_blender_executable

    if find_blender_executable() is None:
        pytest.skip("Headless Blender not staged in this environment")

    app_module.reset(tmp_path)
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Blender Flexi Test", "creation_mode": "character"})

    script = """
body = pa_loft_profile("Body", [
    {"x": -55.0, "ry": 8.0,  "z_bot": 5.0, "z_top": 18.0, "belly_flat": 0.25},
    {"x": -35.0, "ry": 19.0, "z_bot": 0.0, "z_top": 32.0, "belly_flat": 0.72},
    {"x": -5.0,  "ry": 20.0, "z_bot": 0.0, "z_top": 34.0, "belly_flat": 0.75},
    {"x": 25.0,  "ry": 13.0, "z_bot": 0.0, "z_top": 22.0, "belly_flat": 0.75},
    {"x": 48.0,  "ry": 6.5,  "z_bot": 0.0, "z_top": 12.0, "belly_flat": 0.70},
], ring_verts=20, subsurf=2)
pa_assign_color(body, "Body", "#4b6e91", filament=1)

belly = pa_loft_profile("Belly", [
    {"x": -60.0, "ry": 24.0, "z_bot": -4.0, "z_top": 11.0, "belly_flat": 0.9},
    {"x": 0.0,   "ry": 24.0, "z_bot": -4.0, "z_top": 10.0, "belly_flat": 0.9},
    {"x": 45.0,  "ry": 14.0, "z_bot": -4.0, "z_top": 4.5,  "belly_flat": 0.9},
], ring_verts=16, subsurf=1)
pa_assign_color(belly, "Belly", "#f2ecdf", filament=2)

eyes = pa_quad_sphere("Eyes", (-40.0, 14.0, 21.0), (3.8, 3.2, 3.8), subsurf=2, mirror_y=True)
pa_assign_color(eyes, "Eyes", "#16191f", filament=3)

hl = pa_quad_sphere("Highlights", (-42.2, 15.8, 22.4), (1.2, 1.1, 1.2), subsurf=1, mirror_y=True)
pa_assign_color(hl, "Highlights", "#f8f8fa", filament=4)

pa_flexi_articulate(
    primary_body=body,
    detail_parts=[eyes, hl],
    color_zones=[belly],
    cuts_x=[-18.0, 5.0, 24.0],
    clearance_mm=0.45,
    angle_deg=24.0,
    bevel_mm=1.1,
    flatten_bottom_mm=0.6,
)
pa_export_glb(OUT_GLB, baked_flexi=True)
"""
    r = c.post("/api/character/blender", json={"name": "Flexi Critter", "script": script})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sculpt"]["id"] == "M1"
    assert data["sculpt"]["watertight"] is True
    assert len(data["parts"]) == 4
    colors = {p["color"].lower() for p in data["parts"]}
    assert {"#4b6e91", "#f2ecdf", "#16191f", "#f8f8fa"} == colors
    assert not any("voxel remesh" in n for n in data["notes"]), f"Unexpected voxel fallback: {data['notes']}"
    assert data["printability"]["component_count"] == 4
    assert data["printability"]["floating_count"] == 0

    # Verify Three.js /api/part exposes all 4 colored bodies cleanly
    part_res = c.get("/api/part").json()
    sculpt_bodies = [b for b in part_res["bodies"] if b["id"].startswith("M1")]
    assert len(sculpt_bodies) == 4

    # Verify 4-view sheet render & print checks pass cleanly
    ren = c.get("/api/render", params={"view": "sheet", "size": 640})
    assert ren.status_code == 200 and ren.content[:8] == b"\x89PNG\r\n\x1a\n"
    chk = c.get("/api/checks").json()
    bed_c = next(f for f in chk["part"] if f["code"] == "bed_contact")
    assert bed_c["status"] == "ok", bed_c

    # Verify 3MF export writes PrusaSlicer multi-extruder config for all 4 slots
    exp = c.get("/api/export/3mf")
    assert exp.status_code == 200
    mf_path = tmp_path / "blender_4color.3mf"
    mf_path.write_bytes(exp.content)
    with zipfile.ZipFile(mf_path) as zf:
        cfg_xml = zf.read("Metadata/Slic3r_PE_model.config").decode("utf-8")
        for slot in ("1", "2", "3", "4"):
            assert f'key="extruder" value="{slot}"' in cfg_xml

    # Verify replace_sculpt_id="M1" + color_mode="single" updates M1 cleanly in place
    r2 = c.post(
        "/api/character/blender",
        json={"name": "Flexi Critter Single", "script": script, "replace_sculpt_id": "M1", "color_mode": "single"},
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["sculpt"]["id"] == "M1"
    assert len(d2["tree"]["sculpts"]) == 1
    assert len(d2["parts"]) == 1



