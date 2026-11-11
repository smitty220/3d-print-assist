# CLAUDE.md — 3D Print Assist

## What this is
A general-purpose parametric CAD app for 3D printing that the user and the AI edit together. The user designs original products to sell (geometric home decor, MTG accessories), but the software stays general: product knowledge lives in designs and saved templates, never in code. Claude writes nearly all of the code; the user is the product owner.

- Overview doc: https://claude.ai/code/artifact/393a79b3-dcb0-427c-85d7-a872fe865573
- UI mockup: https://claude.ai/artifact/FrQ8XXT6CR2EbyWormDQDu
- Plan: `~/.claude/plans/abundant-sprouting-rain.md`

## Core architecture (do not drift from these)
- **World frame, millimetres.** Z is up and the print bed is Z = 0. No other frame or unit inside the backend.
- **One feature definition** (`backend/src/print_assist/features/library.py`) drives the edit dialog, the AI tool description, the MCP tools and the checks.
- **Named faces, not face indices,** for anything downstream refers to (`features/design.py`: `attribute()`, `find_picked_edges`).
- **No custom geometry kernel.** build123d on OpenCascade (OCP).
- **Claude never writes geometry.** It edits the feature list through the same operations the user has; every AI change is built and checked before it is kept, and locked features are never changed.
- **Each design is a git-tracked folder** (`project.py`); every kept change is a checkpoint commit.

## Layout
| Path | What it is |
| --- | --- |
| `backend/src/print_assist/` | FastAPI app, features, sketches, build, checks, AI loop |
| `backend/src/print_assist/features/` | Feature model, library, build pipeline, face naming, edge picks |
| `backend/src/print_assist/sketch/` | 2D sketches: model, solve bridge, sketch → faces, faces → sketch |
| `backend/src/print_assist/rules.py`, `printers/` | Print rules (min wall, nozzle) and printer profiles (bed, nozzles, slots) |
| `frontend/` | React + TypeScript + plain three.js viewer, sketcher |
| `mcp/` | MCP server (`print-assist-mcp`): Claude Code's tools, talking to the backend over HTTP. Registered in `.mcp.json`; the in-app chat reuses the same tools in process |
| `backend/src/print_assist/build/` | Incremental build engine (`engine.py`), design ops (`ops.py`), the open design's state (`workspace.py`) |
| `backend/src/print_assist/ai/` | Shared AI guide (`guide.py`), API client, in-app chat |
| `frontend/sketch-cli.ts` | The planegcs solver run by Node for the backend. **Must stay at `frontend/`**: `sketch/solve.py` finds it via `parents[4]` |

## Conventions
- Backend: Python 3.12, uv, FastAPI, pytest. Frontend: React + TypeScript + plain three.js (no react-three-fiber); `erasableSyntaxOnly` is on, so no constructor parameter properties or enums.
- Mouse: left-drag orbits, right/middle-drag pans, wheel zooms toward the cursor.
- Every geometry function gets a test with a hand-checked number (volume, area, position), the formula in the test.
- Tests never call the real Anthropic or Gemini APIs (`tests/conftest.py` blocks them) and never touch the user's designs folder (`PRINT_ASSIST_DATA` points at a temp dir).
- Always pass `encoding="utf-8"` to `read_text` / `write_text` / `subprocess.run(text=True)`: OpenCascade's font loader (the Text feature) switches the process locale to ASCII.
- The backend sketch model (`sketch/model.py`) mirrors `frontend/src/sketch/model.ts`; keep them in step.

## Running
- `./dev.sh` starts the backend (:8010) and frontend (:5183); cad-assist uses 8000 / 5173, so both can run.
- Tests: `cd backend && uv run pytest`; `cd frontend && npm run build && npm run lint && npm test`.
- Designs live in `PRINT_ASSIST_DATA` (default `~/Projects/3DPrinting/designs`).
- AI chat key (optional; Claude Code over MCP is the preferred mode): `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in `backend/.env` (git-ignored). The provider and model are picked in the AI panel, or set with `PRINT_ASSIST_AI_PROVIDER=gemini|anthropic` / `PRINT_ASSIST_MODEL`; otherwise auto-detected from the keys (Claude if both). Defaults: `claude-opus-5-5`, `gemini-2.5-flash`.
- Image generation: `GEMINI_API_KEY` in `backend/.env` (pay-per-image, with a spend cap set in AI Studio).

## Status
- M1 done (2026-10-14): parameters and formulas, incremental build engine with update states (manual / auto, rollback), ops, Extrude / Revolve / Shell / Text / Fillet / Chamfer, sketch tools with formula dimensions, export (3MF / STL / STEP), renderer, MCP server, API chat, the full UI (tree, toolbars, edit dialogs, AI panel, Color & Export), end-to-end vase test.
- M2 done (2026-10-15): filament palette and body filaments, Split / Combine, inlay and separate-body text, variants and colorways, batch export, Color & Export screen, Bambu Studio project 3MF.
- Next: an X2D template (needs a project saved for the X2D in Bambu Studio); then M3 the rest of the toolbar and the commander deck box test.
- Known gaps: shell can fail on a body with a through hole (OCC offset); stroke check flags thin glyphs, not hairlines.

## Import and references
- **References** (`Design.references`, `references.py`, `geometry/meshio.py`): STEP / STL / 3MF files in the design's `references/` folder shown as see-through ghosts beside the design, never built or exported. They exist so the user can point the AI at "the hinge from this file": the AI measures them (`list_references`), looks (`look_at_reference`, with `region` zoom and `with_design` overlay) and slices them (`section_reference`: outlines with areas and fitted circles), then rebuilds the feature with our own sketches and features. Meshes stay triangles (no sewing), so size is not limited; 3MF keeps part names, placement and Bambu plates (plate 1 by default).
- **Import** (the `import` feature) is STEP only: an exact solid in a body of its own.
- Big meshes must stay fast: the renderer rasterises small triangles in vectorised batches and draws edges in one pass; `attribute()` names untouched faces by hash; `mesh_shape` meshes the whole shape once and reads each face's triangulation.

## Generated images
- `images.py`, `/api/images`, MCP `generate_image` / `list_images` / `look_at_image`: a prompt (and earlier images to
  vary) goes to Google's Gemini image models (Interactions API; quality draft / standard / best picks the model); the
  picture and a JSON record of its prompt are kept in the design's `images/` folder as I1, I2, ... A concept SVG shows
  one with `<image href="images/I3.png">`, inlined as a data URI when the concept is served or rendered.
- Tests never call Gemini (`conftest.py` replaces `images._post` and stops `backend/.env` being read).

## Bambu Studio export
- A 3MF export is a Bambu Studio project when the printer profile names a `bambu_template` (`printers/bambu/*.project_settings.config`, saved by Bambu Studio for that printer): one object, a part per body with its `extruder` (= palette slot), the template's settings resized to the palette (`export/bambu.py`). Project filament N is palette slot N.
- **Print layout** (`Design.layout` / `Design.plates`, `export/layout.py`; ops `set_print_item`, `set_plate`; MCP
  `print_layout`, render views `plateN` / `plateN-top`): items take the solids of some bodies inside a box of the
  assembled model, turn them and set them on a plate. With a layout the Bambu export is one object per item (a part per
  body, so multi-colour pieces stay together) on its plate; plate N sits where Bambu Studio puts it (ceil(sqrt(n))
  columns, 1.2 bed sizes apart, rows toward -Y) and each plate gets its own purge-tower position.
- **Plates tab** (`frontend/src/plates/PlatesPane.tsx`, `Viewer.setPlateMode` / `setPlate`): the layout is resolved only
  when the tab asks (`/api/layout`, `/api/layout/plates/{n}/mesh`), cached in the workspace until the part, layout or
  printer changes. The tab shows and exports the plates; the arrangement is changed through ops (by asking Claude).
- Per-filament settings come from the template filament with the same material; a material the template lacks is an export error. To support PETG etc., or to change the default process settings, save a Bambu Studio project for that printer with those filaments and copy its `Metadata/project_settings.config` over the template.
- The A1 template came from a MakerWorld project the user re-saved for their A1 (its process tweaks: 3 walls, gyroid, no brim, back seam). Bambu Studio is not installed on this Mac, so the format is checked by `tests/test_bambu.py` and by opening exports in Bambu Studio by hand.

## AI contract
- **Picture first.** For a request that makes or changes geometry, the AI proposes a concept (`concepts.py`, `POST /api/concepts`, MCP `propose_concept` / `wait_for_concept`): a picture it draws for that request (SVG), a summary, its assumptions and questions, and iterates on pictures until the user approves; only then does it build. Pictures are drawn to scale, show the assembled result from every side in real colours with hidden features ghosted, and the AI looks at each one before proposing (`look_at_concept`, `POST /api/concepts/picture`, rendered with resvg). After building, it checks the model against the approved picture before handing over. The app shows a concept large over the 3D view; the user approves or asks for changes. While one is waiting, AI changes are refused. Concepts are kept in the design's `concepts/` folder. Nothing in this flow is specific to a kind of part; the method is in `ai/guide.py`.
- The AI batches a whole step into one `apply_changes` call: each call is built and checked before it is kept.
- Every change is an op (`build/ops.py`) applied by the Workspace. Actors other than "user" are built and checked first; a change that makes a feature fail is refused and the design is untouched. Locked features refuse AI edits. Kept AI changes are git checkpoints ("AI: ..."), revertible.
- Claude Code: `./dev.sh`, then Claude Code in this repo picks up `.mcp.json` (server `3d-print-assist`).

## Deployment
Local while iterating; stable builds go on the user's home server. Plan placement with the `homelab-consultant` agent.
