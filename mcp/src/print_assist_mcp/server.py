"""MCP server: lets Claude Code design in 3D Print Assist.

It talks to the app's backend over HTTP (start it with ./dev.sh), so the user watches every change land in the app
and can edit alongside. Every change is sent as the actor "claude-code": the app builds and checks it first, refuses
it if it would break the part, and never lets it touch a feature the user locked.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server.mcpserver import Image, MCPServer

BASE_URL = os.environ.get("PRINT_ASSIST_URL") or f"http://127.0.0.1:{os.environ.get('PRINT_ASSIST_PORT', '8010')}"
ACTOR = {"X-Actor": os.environ.get("PRINT_ASSIST_ACTOR", "claude-code")}  # the in-app chat reuses these tools as "claude"

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=120)
    return _client


def set_client(c: httpx.Client) -> None:
    """Point the tools at another client (tests use the backend's in-process TestClient)."""
    global _client
    _client = c


def _guide() -> str:
    try:
        return client().get("/api/ai/guide", timeout=2).json()["guide"]
    except Exception:  # noqa: BLE001 - the app is not running yet
        return ("3D Print Assist is not running. Ask the user to start it (./dev.sh in the 3d-print-assist repo), then "
                "call get_guide for how to design with it.")


def _call(method: str, path: str, **kw) -> Any:
    try:
        r = client().request(method, path, **kw)
    except httpx.TimeoutException as exc:
        return {"error": f"Request to {path} timed out while 3D Print Assist was still working ({exc}). Check list_concepts or get_model before retrying."}
    except httpx.HTTPError as exc:
        return {"error": f"3D Print Assist is not reachable at {BASE_URL} ({exc}); ask the user to run ./dev.sh"}
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return {"error": detail, "status": r.status_code}
    if r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return r


def _text(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


mcp = MCPServer("3d-print-assist", instructions=None)


@mcp.tool()
def get_guide() -> str:
    """How to design with 3D Print Assist: units, the ops you can apply, sketch and feature formats, and the rules.
    Read this first in a session."""
    return _guide()


@mcp.tool()
def get_model() -> str:
    """The open design: parameters (with values, formulas and what they drive), sketches (geometry and constraints),
    features in build order (kind, params, formulas, refs, picks, body, update status ok / warn / fail / skipped /
    stale, and any error), bodies, and whether the model is up to date."""
    tree = _call("GET", "/api/tree")
    proj = _call("GET", "/api/project")
    if isinstance(tree, dict) and "error" in tree:
        return _text(tree)
    design = proj.get("design", {}) if isinstance(proj, dict) else {}
    settings = proj.get("settings", {}) if isinstance(proj, dict) else {}
    return _text({"tree": tree, "sketches": design.get("sketches", []), "printer": settings.get("printer"), "slicer": settings.get("slicer")})


@mcp.tool()
def get_feature_catalog() -> str:
    """Every feature kind with its parameters (name, quantity, default, bounds, choices) and refs."""
    return _text(_call("GET", "/api/features/catalog"))


@mcp.tool()
def apply_changes(ops: list[dict[str, Any]], message: str) -> str:
    """Change the design: a list of ops applied together (see get_guide for the op formats), with a one-line message
    for the user's log saying what and why. The change is built and checked before it is kept; if it would make a
    feature fail it is refused and nothing changes. Returns the new ids made and the updated model tree."""
    out = _call("POST", "/api/ops", json={"ops": ops, "message": message}, headers=ACTOR)
    if isinstance(out, dict) and "tree" in out:
        t = out["tree"]
        out["tree"] = {"features": [{k: f[k] for k in ("id", "kind", "status", "error", "message")} for f in t["features"]],
                       "pending_count": t["pending_count"], "failed": t["failed"]}
    return _text(out)


@mcp.tool()
def update() -> str:
    """Rebuild everything out of date (the user may have edited in manual-update mode). Returns the statuses."""
    t = _call("POST", "/api/update", headers=ACTOR)
    if isinstance(t, dict) and "features" in t:
        return _text({"features": [{k: f[k] for k in ("id", "kind", "status", "error", "message")} for f in t["features"]],
                      "failed": t["failed"]})
    return _text(t)


@mcp.tool()
def render_preview(view: str = "sheet", size: int = 960) -> Image | str:
    """A shaded picture of the last good build. Defaults to `view="sheet"` (a 2x2 4-view sheet showing 3/4 ISO, Side on
    Z=0 build plate, Top, and Bottom Z=0 bed contact!). Individual views also supported: iso, front, back, left, right,
    top, bottom, or a plate of the print layout ("plate1", "plate1-top", "plate1-bottom"). Look after every change."""
    r = _call("GET", "/api/render", params={"view": view, "size": size})
    if isinstance(r, dict):
        return _text(r)
    return Image(data=r.content, format="png")


@mcp.tool()
def print_layout() -> str:
    """The print layout as the export would lay it out: each plate's objects (bodies, solid count, box on the plate),
    problems (overlaps, off the plate, solids on no plate or on two), and the solids no item takes yet with their
    centres and sizes in the assembled model, to aim an item's `within` box at."""
    return _text(_call("GET", "/api/layout"))


@mcp.tool()
def run_checks() -> str:
    """Print checks on the last good build: watertight manifold check, per-segment Z=0 bed adhesion & floating-part check
    (`bed_contact`), overhang slope check (`overhang`), feature rule checks (walls, text), fit on the printer's bed,
    minimum wall thickness, mass and filament."""
    return _text(_call("GET", "/api/checks"))


@mcp.tool()
def list_faces(body: str = "B1") -> str:
    """The faces of a body as built: index, name (e.g. F1.end, F1.side, F2.inner), type, centre, normal, area. Use an
    index with pick_face to target a face (shell openings, text, a sketch on a face)."""
    return _text(_call("GET", "/api/faces", params={"body": body}))


@mcp.tool()
def pick_face(body: str, index: int) -> str:
    """The pick record for one face (from list_faces): put it in a feature's "faces" list or a sketch placement's
    "face"."""
    return _text(_call("GET", "/api/pick/face", params={"body": body, "face": index}))


@mcp.tool()
def list_fonts() -> str:
    """Fonts the Text feature can use (bundled open-licence fonts and any in the design's fonts folder)."""
    return _text(_call("GET", "/api/fonts"))


@mcp.tool()
def export(format: str = "3mf") -> str:
    """Export the design into its exports folder: 3mf (a Bambu Studio project with the filaments preassigned), stl or
    step. Only works when the model is up to date and nothing failed. Returns where the file was saved."""
    r = _call("GET", f"/api/export/{format}")
    if isinstance(r, dict):
        return _text(r)
    return _text({"saved": r.headers.get("x-saved-path"), "bytes": len(r.content)})


@mcp.tool()
def export_batch(format: str = "3mf", rows: list[dict[str, str | None]] | None = None) -> str:
    """Export several variants and colorways at once (each variant is built first). rows: [{"variant": "V1",
    "colorway": "C2"}, ...]; a missing key means the design as it is / the palette's colours. No rows: every variant in
    every colorway. Returns the folder the files were saved in and their names."""
    import io
    import zipfile

    r = _call("POST", "/api/export/batch", json={"fmt": format, "rows": rows or []})
    if isinstance(r, dict):
        return _text(r)
    return _text({"saved": r.headers.get("x-saved-path"), "files": zipfile.ZipFile(io.BytesIO(r.content)).namelist()})


@mcp.tool()
def list_concepts() -> str:
    """List the design's proposed concepts (id, title, status, summary, note), newest first."""
    r = _call("GET", "/api/concepts", params={"drawings": False})
    if isinstance(r, dict):
        return _text(r)
    return _text([{k: c.get(k) for k in ("id", "title", "status", "summary", "note")} for c in r])


@mcp.tool()
def look_at_concept(svg: str = "", id: str = "", width: int = 1200) -> Image | str:
    """See a concept picture as the user will: pass `id` (e.g. "K1") for an existing concept, or `svg` for a draft
    SVG you are about to propose. Omit both to inspect the newest concept if one exists. Look before every
    propose_concept and fix overlapping or cramped labels, things off the canvas and anything that does not match your numbers."""
    import re

    cid = id.strip()
    if not cid and svg and "<svg" not in svg.lower():
        m = re.search(r"\b(K\d+)\b", svg.strip(), re.I)
        if m:
            cid = m.group(1).upper()
    if not cid and not svg.strip():
        existing = _call("GET", "/api/concepts", params={"drawings": False})
        if isinstance(existing, dict):
            return _text(existing)
        if not existing:
            return _text({"concepts": [], "message": "No concepts have been proposed in this design yet."})
        cid = existing[0]["id"]
    if cid:
        r = _call("GET", f"/api/concepts/{cid}/picture", params={"width": width})
    else:
        r = _call("POST", "/api/concepts/picture", json={"svg": svg, "width": width})
    if isinstance(r, dict):
        return _text(r)
    return Image(data=r.content, format="png")


@mcp.tool()
def propose_concept(title: str, svg: str, summary: str = "", assumptions: list[str] | None = None,
                    questions: list[str] | None = None) -> str:
    """Before building anything in Engineer Mode, show the user a 4-view blueprint picture of the result and wait for
    their answer. `svg`: one complete <svg> document drawn to scale in mm, ALWAYS showing at least 4 views by default
    in a clean 2x2 layout (1: 3/4 Isometric, 2: Front, 3: Side showing the Z=0 build plate & overhangs, 4: Top or Bottom
    Z=0 bed contact), in real colours with hidden features ghosted and key sizes dimensioned in mm. Check it with
    look_at_concept first. summary: what changed since the last round and what you will build. assumptions: choices
    they did not state. questions: what they should decide."""
    return _text(_call("POST", "/api/concepts", json={"title": title, "svg": svg, "summary": summary,
                                                     "assumptions": assumptions or [], "questions": questions or [],
                                                     "views": ["3/4 iso", "front", "side (Z=0)", "top"]},
                       headers=ACTOR))


@mcp.tool()
def set_creation_mode(mode: str, direct_build: bool | None = None) -> str:
    """Switch the open design between 'engineer' (parametric CAD extrusions/sketches) and 'character'
    (local organic clay sculpting from 2D turnaround designs). Optionally toggle `direct_build` bypass."""
    patch: dict[str, Any] = {"creation_mode": mode}
    if direct_build is not None:
        patch["direct_build"] = bool(direct_build)
    return _text(_call("PUT", "/api/settings", json=patch))


@mcp.tool()
def propose_character_design(
    title: str,
    prompt: str,
    reference_photos: list[str] | None = None,
    summary: str = "",
    assumptions: list[str] | None = None,
    questions: list[str] | None = None,
    turnaround: bool = True,
    style: str = "auto",
    from_concept_id: str = "",
    repaint: bool = False,
    flexi_segments: int = 0,
    flexi_cuts_pct: list[float] | None = None,
) -> str:
    """Character Mode 4-View Design File: generates a 4-view Character/Engineering Design Sheet (`K#.png`, shown in the
    Design tab, never in Photos) by default (`turnaround=True`) while preserving a clean single-hero base (`K#_clean.png`)
    for 3D sculpting:
    - For standard characters (`"figurine"`, `"bust"`, `"keychain"`, `"magnet"`, `"planter"`, `"scene"`): generates a
      2x2 4-view turnaround sheet (`1: 3/4 Hero`, `2: Front`, `3: Side on Z=0 Bed`, `4: Back`).
    - For `"flexi"` (or `flexi_segments > 0`): automatically builds a 1280x900 4-View Bed-Seated Flexi Print-Engineering
      Sheet (`1: 3/4 Articulated Seated on Z=0`, `2: Side Orthographic on Z=0 Build Plate + X-Ray Internal Hinges`,
      `3: Top Orthographic Fin-Safe Cuts`, `4: Bottom Z=0 First-Layer Bed Contact Pads`), proving zero floating segments!
    CONVERTING AN EXISTING PIXAR CONCEPT TO FLEXI ("make it a flexi-shark" / "now make it flexi"):
    - Pass `from_concept_id="K#"` (e.g. `"K1"`), `style="flexi"`, `flexi_segments=6`, `repaint=False` to convert that
      character into a 4-view Z=0 bed-seated flexi engineering design in < 1 second! Set `repaint=True` only when
      also redrawing/elongating the character's shape/face.
    If this returns `status: "running"`, call `wait_for_character_job(wait_s=40)` until `status == "done"`."""
    out = _call(
        "POST",
        "/api/character/design",
        json={
            "title": title,
            "prompt": prompt,
            "reference_photos": reference_photos or [],
            "summary": summary,
            "assumptions": assumptions or [],
            "questions": questions or [],
            "turnaround": turnaround,
            "style": style,
            "from_concept_id": from_concept_id,
            "repaint": repaint,
            "flexi_segments": flexi_segments,
            "flexi_cuts_pct": flexi_cuts_pct or [],
            "wait_s": 40,
        },
        headers=ACTOR,
        timeout=75,
    )
    if isinstance(out, dict) and out.get("status") == "running":
        out["next"] = "Still rendering on local GPU. Call wait_for_character_job(wait_s=40) to wait for completion. Do NOT call propose_character_design again."
    return _text(out)


@mcp.tool()
def sculpt_character(
    concept_id: str = "",
    replace_sculpt_id: str = "",
    name: str = "",
    prompt: str = "",
    reference_photos: list[str] | None = None,
    target_height_mm: float = 80.0,
    flatten_bottom_mm: float = 0.4,
    base_style: str = "none",
    keychain_loop: bool = False,
    flexi_segments: int = 0,
    flexi_axis: str = "auto",
    flexi_start_pct: float = 35.0,
    flexi_end_pct: float = 88.0,
    flexi_clearance_mm: float = 0.45,
    flexi_angle_deg: float = 24.0,
    flexi_cuts_pct: list[float] | None = None,
    cut_bodies: list[str] | None = None,
    union_bodies: list[str] | None = None,
    magnet_hole: str = "none",
    magnet_dia_mm: float = 8.2,
    magnet_depth_mm: float = 3.2,
    hollow_cavity: bool = False,
    hollow_wall_mm: float = 3.0,
    multicolor: bool = False,
    color_mode: str = "auto",
) -> str:
    """PREFERRED Character Mode 3D Sculptor + Automatic Multi-Color MMU3 Inlay Projection + Flexi-Factory Engine:
    molds an authentic Pixar/organic watertight 3D mesh directly from an approved Character Design File (`concept_id`,
    e.g. "K8" or "K9") while AUTOMATICALLY combining ALL THREE:
    1. Authentic Pixar 3D character sculpting (`Hunyuan3D-2mv`, smooth skin, zero voxels) matching the approved concept's
       real cheeks, brow, smile, eyes, teeth, and silhouette.
    2. Automatic up-to-5-color Concept-to-3D MMU3/AMS Solid Inlay Projection (`color_mode="auto"|"multicolor"`):
       aligns the 3D sculpt to `concept_id`'s clean image and extrudes crisp watertight inlaid solids for Body, Belly,
       Eyes, Teeth/Highlights, and Mouth automatically in < 1 second!
    3. Fin-Safe Print-in-Place Flexi-Factory Articulation (`flexi_segments=5` or `6`): detects pectoral/dorsal/tail
       fins, cuts wide ±24° swimming V-gaps with captive vertical Z-axis pins, and seats 100% of segments flat on Z=0!
    Pass `replace_sculpt_id="M1"` when rebuilding an existing sculpt in place. Supports hybrid OpenCascade CAD booleans
    (`cut_bodies`, `union_bodies`), `magnet_hole`, `hollow_cavity`, `base_style`, and `keychain_loop`.
    If this returns `status: "running"`, call `wait_for_character_job(wait_s=40)` until `status == "done"`."""
    out = _call(
        "POST",
        "/api/character/sculpt",
        json={
            "concept_id": concept_id,
            "replace_sculpt_id": replace_sculpt_id,
            "name": name,
            "prompt": prompt,
            "reference_photos": reference_photos or [],
            "target_height_mm": target_height_mm,
            "flatten_bottom_mm": flatten_bottom_mm,
            "base_style": base_style,
            "keychain_loop": keychain_loop,
            "flexi_segments": flexi_segments,
            "flexi_axis": flexi_axis,
            "flexi_start_pct": flexi_start_pct,
            "flexi_end_pct": flexi_end_pct,
            "flexi_clearance_mm": flexi_clearance_mm,
            "flexi_angle_deg": flexi_angle_deg,
            "flexi_cuts_pct": flexi_cuts_pct or [],
            "cut_bodies": cut_bodies or [],
            "union_bodies": union_bodies or [],
            "magnet_hole": magnet_hole,
            "magnet_dia_mm": magnet_dia_mm,
            "magnet_depth_mm": magnet_depth_mm,
            "hollow_cavity": hollow_cavity,
            "hollow_wall_mm": hollow_wall_mm,
            "multicolor": multicolor,
            "color_mode": color_mode,
            "wait_s": 40,
        },
        headers=ACTOR,
        timeout=75,
    )
    if isinstance(out, dict) and out.get("status") == "running":
        out["next"] = "Still sculpting on local GPU. Call wait_for_character_job(wait_s=40) to wait for completion. Do NOT call sculpt_character again."
    return _text(out)


@mcp.tool()
def build_blender_character(
    name: str,
    script: str,
    concept_id: str = "",
    replace_sculpt_id: str = "",
    target_height_mm: float = 0.0,
    flatten_bottom_mm: float = 0.0,
    base_style: str = "none",
    keychain_loop: bool = False,
    flexi_segments: int = 0,
    flexi_clearance_mm: float = 0.45,
    flexi_angle_deg: float = 24.0,
    cut_bodies: list[str] | None = None,
    union_bodies: list[str] | None = None,
    magnet_hole: str = "none",
    magnet_dia_mm: float = 8.2,
    magnet_depth_mm: float = 3.2,
    hollow_cavity: bool = False,
    hollow_wall_mm: float = 3.0,
    color_mode: str = "auto",
) -> str:
    """Headless Blender 4.5 LTS SubSurf & Flexi-Factory Character Engine (~2-4 seconds, 100% local & free!).
    PREFERRED engine for stylized Pixar/cartoon characters, animals, collectibles, multi-color MMU3/AMS models,
    and print-in-place Flexi-Factory articulated toys with glass-smooth Catmull-Clark Subdivision Surfaces!
    (Use `sculpt_character` / Hunyuan3D-2mv instead only for uploaded real-world photos / portraits / pets.)

    Write a clean Python `script` (1 Blender unit = 1 mm, Z-up, print bed at Z=0, face character toward -X for flexis).
    `bpy` and all `pa_blender_lib` helpers are pre-imported automatically:
      - `pa_loft_profile(name, sections, ring_verts=28, subsurf=2)`: glass-smooth quad body loft along X; each section
        is `{"x", "ry", "z_bot", "z_top", "belly_flat": 0..0.9}` (`z_bot=0.0, belly_flat=0.75` seats flush on Z=0!).
      - `pa_sculpt_fin_strip(name, stations, ring_verts=16, subsurf=2, mirror_y=False)`: airfoil fins, tails, wings,
        ears, limbs; each station is `{"center": (x,y,z), "chord_vec": (dx,dy,dz), "thick_vec": (tx,ty,tz)}`.
      - `pa_quad_sphere(name, center, radius_xyz, rot_deg=(0,0,0), subsurf=2, mirror_y=False)`: smooth eyes, brows, cheeks, tongue.
      - `pa_cone(name, base_center, tip_point, radius_base, radius_tip=0.3, subsurf=1, mirror_y=False)`: teeth, horns, spikes.
      - `pa_box(name, center, size_xyz)`, `pa_cylinder(name, center, radius, height)`, `pa_extrude_polygon_z(name, pts_xy, z0, z1)`.
      - `pa_bool(target, tool, op="UNION"|"DIFFERENCE"|"INTERSECT", delete_tool=True)`: Blender 4.5 fast MANIFOLD boolean.
      - `pa_join(objs, name)`: merge multiple meshes of the same filament color into one object.
      - `pa_assign_color(obj, part_name, hex_color, filament=1..5)`: exact sRGB filament hex color & MMU3/AMS extruder slot!
      - `pa_flexi_articulate(primary_body, detail_parts=[mouth, eyes, teeth], color_zones=[belly], cuts_x=[...], clearance_mm=0.45, angle_deg=26.0, bevel_mm=1.4, flatten_bottom_mm=0.7)`:
        slices `primary_body` into smooth beveled flexi links with wide ±26° V-flank swing gaps + captive vertical Z-axis
        pins & chain-link tongues! Pass additive 3D features (Eyes, Teeth, Mouth) in `detail_parts` and volumetric paint
        masks (Belly, Stripes, Spots — wider than body so they paint the outer skin!) in `color_zones`.
      - `pa_partition_colors(primary_body, detail_parts=[...], color_zones=[...], flatten_bottom_mm=0.4)`: for non-flexi multi-color characters.
      - `pa_export_glb(OUT_GLB, baked_flexi=True|False)`: exports Z-up mm `.glb` preserving all colors & manifold solids.

    Pass `replace_sculpt_id="M1"` when revising an existing character so the viewport updates cleanly in place!
    Leave `target_height_mm=0.0` to keep your exact mm script coordinates (or set e.g. `80.0` to rescale uniformly).
    After calling this, ALWAYS call `render_preview(view="sheet")` + `run_checks` to visually inspect all 4 views!"""
    return _text(_call(
        "POST",
        "/api/character/blender",
        json={
            "name": name,
            "script": script,
            "concept_id": concept_id,
            "replace_sculpt_id": replace_sculpt_id,
            "target_height_mm": target_height_mm,
            "flatten_bottom_mm": flatten_bottom_mm,
            "base_style": base_style,
            "keychain_loop": keychain_loop,
            "flexi_segments": flexi_segments,
            "flexi_clearance_mm": flexi_clearance_mm,
            "flexi_angle_deg": flexi_angle_deg,
            "cut_bodies": cut_bodies or [],
            "union_bodies": union_bodies or [],
            "magnet_hole": magnet_hole,
            "magnet_dia_mm": magnet_dia_mm,
            "magnet_depth_mm": magnet_depth_mm,
            "hollow_cavity": hollow_cavity,
            "hollow_wall_mm": hollow_wall_mm,
            "color_mode": color_mode,
        },
        headers=ACTOR,
        timeout=140,
    ))


@mcp.tool()
def wait_for_character_job(wait_s: int = 40) -> str:
    """Wait up to `wait_s` seconds (max 45s) for the active Character Mode GPU painting or 3D sculpting job.
    Returns `status: "done"` with the finished Concept or Sculpt result, `status: "running"` with live progress
    (call again to continue waiting), or `status: "error"`."""
    sec = max(5, min(int(wait_s), 45))
    out = _call("GET", "/api/character/job", params={"wait_s": sec}, timeout=sec + 25)
    if isinstance(out, dict) and out.get("status") == "running":
        out["next"] = "Still running on local GPU. Call wait_for_character_job(wait_s=40) again."
    return _text(out)


@mcp.tool()
def inspect_sculpt_anatomy(id: str = "M1") -> str:
    """Inspect the 3D cross-sectional anatomy of a sculpted character (`id`, e.g. "M1"): returns exact bounding box
    and dimensions in mm, detected fin/limb protrusion zones (`protected_fin_zones`), tail waist percentage
    (`tail_waist_pct`), safe fin-free cut intervals (`safe_cut_intervals_pct`), recommended fin-safe flexi cut
    percentages (`recommended_cuts_pct`), and a 10% slice-by-slice width/height profile."""
    return _text(_call("GET", f"/api/sculpts/{id}/anatomy"))


@mcp.tool()
def update_mesh_prep(
    id: str,
    target_height_mm: float | None = None,
    flatten_bottom_mm: float | None = None,
    base_style: str | None = None,
    base_height_mm: float | None = None,
    keychain_loop: bool | None = None,
    flexi_segments: int | None = None,
    flexi_axis: str | None = None,
    flexi_start_pct: float | None = None,
    flexi_end_pct: float | None = None,
    flexi_clearance_mm: float | None = None,
    flexi_angle_deg: float | None = None,
    flexi_cuts_pct: list[float] | None = None,
    cut_bodies: list[str] | None = None,
    union_bodies: list[str] | None = None,
    magnet_hole: str | None = None,
    magnet_dia_mm: float | None = None,
    magnet_depth_mm: float | None = None,
    hollow_cavity: bool | None = None,
    hollow_wall_mm: float | None = None,
    rotate_x: float | None = None,
    rotate_y: float | None = None,
    rotate_z: float | None = None,
    max_colors: int | None = None,
    color_mode: str | None = None,
) -> str:
    """Instantly adjust hybrid CAD print-engineering & finishing on an existing Character Mode sculpt (`id`, e.g. "M1")
    in < 1s without re-running the 3D AI sculptor:
    - Fin-Safe Print-in-Place Flexi Articulation: set `flexi_segments` (auto-solves fin-free cut zones so pectoral fins,
      dorsal fins, limbs, and tail fins are never cut in half, with tail-safe bed seating) or pass `flexi_cuts_pct`
      (e.g. `[35.5, 60.1, 74.9]`, pass `[]` to use auto fin-safe solver) for explicit cut positions along the spine.
    - Hybrid CAD + Clay Booleans: pass `cut_bodies` (e.g. `["B1"]`) or `union_bodies` (e.g. `["B2"]`) to subtract or
      union parametric OpenCascade CAD bodies built via `apply_changes` directly into the sculpted mesh!
    - Resize (`target_height_mm`), flatten bottom (`flatten_bottom_mm`), cut magnet pockets (`magnet_hole`), hollow
      planter cavities (`hollow_cavity`), add display base (`base_style`), toggle `keychain_loop`, or rotate."""
    body = {k: v for k, v in {
        "target_height_mm": target_height_mm,
        "flatten_bottom_mm": flatten_bottom_mm,
        "base_style": base_style,
        "base_height_mm": base_height_mm,
        "keychain_loop": keychain_loop,
        "flexi_segments": flexi_segments,
        "flexi_axis": flexi_axis,
        "flexi_start_pct": flexi_start_pct,
        "flexi_end_pct": flexi_end_pct,
        "flexi_clearance_mm": flexi_clearance_mm,
        "flexi_angle_deg": flexi_angle_deg,
        "flexi_cuts_pct": flexi_cuts_pct,
        "cut_bodies": cut_bodies,
        "union_bodies": union_bodies,
        "magnet_hole": magnet_hole,
        "magnet_dia_mm": magnet_dia_mm,
        "magnet_depth_mm": magnet_depth_mm,
        "hollow_cavity": hollow_cavity,
        "hollow_wall_mm": hollow_wall_mm,
        "rotate_x": rotate_x,
        "rotate_y": rotate_y,
        "rotate_z": rotate_z,
        "max_colors": max_colors,
        "color_mode": color_mode,
    }.items() if v is not None}
    return _text(_call("POST", f"/api/sculpts/{id}/prep", json=body, headers=ACTOR))


@mcp.tool()
def wait_for_concept(id: str, timeout_s: int = 45) -> str:
    """Wait briefly for the user's answer on a concept: "approved" (build it, minding any note) or "revise" (their
    note says what to change: propose a new concept). If it times out still "pending", end your turn and let the user
    reply — NEVER loop or spawn shell/background scripts to keep polling."""
    import time

    deadline = time.time() + max(5, min(int(timeout_s), 90))
    while True:
        r = _call("GET", "/api/concepts", params={"drawings": False})
        if isinstance(r, dict):
            return _text(r)
        c = next((x for x in r if x["id"] == id), None)
        if c is None:
            return _text({"error": f"no concept {id!r}"})
        if c["status"] != "pending" or time.time() >= deadline:
            return _text({k: c[k] for k in ("id", "title", "status", "note")})
        time.sleep(2)


@mcp.tool()
def generate_image(prompt: str, quality: str = "standard", aspect: str = "1:1", size: str = "1K",
                   from_images: list[str] | None = None) -> list[Image | str] | str:
    """Make a picture from a prompt (Google Gemini, billed per image: draft or standard ~$0.03 at 1K, best ~$0.13)
    and keep it in the design's images folder as I1, I2, ... with its prompt. from_images: ids of earlier images to
    start from ("the same dragon, wings folded"). aspect: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9; size
    1K, 2K or 4K (best and standard only). Returns the record and the picture. Show it to the user in a concept with
    <image href="images/I3.png" .../> in the SVG."""
    r = _call("POST", "/api/images", json={"prompt": prompt, "quality": quality, "aspect": aspect, "size": size,
                                          "from_images": from_images or []}, headers=ACTOR, timeout=240)
    if not isinstance(r, dict) or "error" in r:
        return _text(r)
    pic = _call("GET", f"/api/images/{r['id']}")
    if isinstance(pic, dict):
        return _text(r | {"picture_error": pic.get("error")})
    fmt = "jpeg" if r["file"].endswith(".jpg") else r["file"].rsplit(".", 1)[-1]
    return [_text(r), Image(data=pic.content, format=fmt)]


@mcp.tool()
def list_images() -> str:
    """The design's generated images, newest first: id, file, prompt, quality, aspect, size, what each started from
    and its list price."""
    return _text(_call("GET", "/api/images"))


@mcp.tool()
def look_at_image(id: str) -> Image | str:
    """See generated image `id` again."""
    r = _call("GET", f"/api/images/{id}")
    if isinstance(r, dict):
        return _text(r)
    fmt = r.headers.get("content-type", "image/png").split("/")[-1]
    return Image(data=r.content, format="jpeg" if fmt == "jpg" else fmt)


@mcp.tool()
def list_references() -> str:
    """The design's references (files kept beside it to copy features from; never printed): id, name, file, plates,
    placement, overall bounds, and each part's name, triangle count, bounding box (min, max, size in mm) and volume."""
    return _text(_call("GET", "/api/references"))


@mcp.tool()
def look_at_reference(id: str, view: str = "iso", part: str | None = None, with_design: bool = False,
                      region: list[float] | None = None, size: int = 768) -> Image | str:
    """A picture of reference `id` as placed beside the design, or of one `part` (a name from list_references).
    Views: iso, front, back, left, right, top, bottom. with_design draws the design's last build in too, to compare.
    region [xmin, ymin, zmin, xmax, ymax, zmax] (mm) zooms in on what lies inside that box."""
    params: dict[str, Any] = {"view": view, "size": size, "with_design": with_design}
    if part:
        params["part"] = part
    if region:
        params["region"] = ",".join(str(v) for v in region)
    r = _call("GET", f"/api/references/{id}/render", params=params)
    if isinstance(r, dict):
        return _text(r)
    return Image(data=r.content, format="png")


@mcp.tool()
def section_reference(id: str, axis: str = "Z", offset: float = 0.0, part: str | None = None,
                      tolerance: float = 0.05) -> str:
    """Cut reference `id` (or one `part`) with the plane axis = offset (mm, as placed) and get its outlines: closed
    point lists in the plane's other two coordinates (Z: x, y; Y: x, z; X: y, z), largest first, each with area,
    length, and centre and radius when it is a circle. Use these numbers to draw sketches that rebuild a feature."""
    params: dict[str, Any] = {"axis": axis, "offset": offset, "tol": tolerance}
    if part:
        params["part"] = part
    return _text(_call("GET", f"/api/references/{id}/section", params=params))


@mcp.tool()
def add_reference(path: str, plate: int | None = None) -> str:
    """Add a STEP, STL or 3MF file from the user's computer (an absolute path) as a reference beside the open design.
    plate: which plate of a multi-plate Bambu project (default the first)."""
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_file():
        return _text({"error": f"no file at {p}"})
    params = {"plate": plate} if plate else {}
    with p.open("rb") as fh:
        r = _call("POST", "/api/references", params=params, files={"file": (p.name, fh)}, headers=ACTOR)
    if isinstance(r, dict) and "tree" in r:
        return _text({"made": r.get("made"), "references": r["tree"].get("references")})
    return _text(r)


@mcp.tool()
def save_design(message: str = "") -> str:
    """Save the design to disk and record a version with an optional message. Only save when the user asks you to;
    otherwise leave your changes unsaved for them to review (the app shows they are unsaved)."""
    return _text(_call("POST", "/api/file/save", json={"message": message}, headers=ACTOR))


@mcp.tool()
def undo() -> str:
    """Undo the last change (yours or the user's): the design goes back one step; redo is in the app."""
    return _text(_call("POST", "/api/undo", headers=ACTOR))


@mcp.tool()
def revert_last_ai_change() -> str:
    """Put the design back as it was before the last change an AI made."""
    return _text(_call("POST", "/api/revert-ai", headers=ACTOR))


@mcp.tool()
def list_designs() -> str:
    """The user's designs (slug, name), most recently changed first."""
    return _text(_call("GET", "/api/projects"))


@mcp.tool()
def open_design(slug: str) -> str:
    """Open another design in the app (the user's screen follows)."""
    return _text(_call("POST", f"/api/projects/{slug}/open"))


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for real-world character appearances (movies, animals, pop culture), 3D printing flexi/articulated
    toy designs, standard hardware sizes (M3 screws, 608 bearings, magnets), or real-world product dimensions."""
    return _text(_call("GET", "/api/web/search", params={"query": query, "max_results": max_results}))


@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch and read the plain-text content of an HTTP/HTTPS web page returned by `web_search`."""
    return _text(_call("GET", "/api/web/fetch", params={"url": url}))


@mcp.tool()
def create_design(name: str) -> str:
    """Start a new, empty design and open it."""
    return _text(_call("POST", "/api/projects", json={"name": name}))


def main() -> None:
    # The guide comes from the running app, so it always matches its features and ops.
    mcp._lowlevel_server.instructions = _guide()
    mcp.run(os.environ.get("MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
