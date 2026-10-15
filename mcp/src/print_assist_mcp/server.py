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

BASE_URL = os.environ.get("PRINT_ASSIST_URL", "http://127.0.0.1:8000")
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
    return _text({"tree": tree, "sketches": design.get("sketches", []), "printer": proj.get("settings", {}).get("printer")})


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
def render_preview(view: str = "iso", size: int = 768) -> Image | str:
    """A shaded picture of the last good build. Views: iso, front, back, left, right, top, bottom. Look after every
    change: it is how you see what you made."""
    r = _call("GET", "/api/render", params={"view": view, "size": size})
    if isinstance(r, dict):
        return _text(r)
    return Image(data=r.content, format="png")


@mcp.tool()
def run_checks() -> str:
    """Print checks on the last good build: build errors, feature rule checks (walls, text), fit on the printer's bed,
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
    """Export the design (3mf, stl or step) into the design's exports folder. Only works when the model is up to date
    and nothing failed. Returns where the file was saved."""
    r = _call("GET", f"/api/export/{format}")
    if isinstance(r, dict):
        return _text(r)
    return _text({"saved": r.headers.get("x-saved-path"), "bytes": len(r.content)})


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
def create_design(name: str) -> str:
    """Start a new, empty design and open it."""
    return _text(_call("POST", "/api/projects", json={"name": name}))


def main() -> None:
    # The guide comes from the running app, so it always matches its features and ops.
    mcp._lowlevel_server.instructions = _guide()
    mcp.run(os.environ.get("MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
