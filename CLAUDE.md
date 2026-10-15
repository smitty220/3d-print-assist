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
- Tests never call the real Anthropic API (`tests/conftest.py` blocks it) and never touch the user's designs folder (`PRINT_ASSIST_DATA` points at a temp dir).
- Always pass `encoding="utf-8"` to `read_text` / `write_text` / `subprocess.run(text=True)`: OpenCascade's font loader (the Text feature) switches the process locale to ASCII.
- The backend sketch model (`sketch/model.py`) mirrors `frontend/src/sketch/model.ts`; keep them in step.

## Running
- `./dev.sh` starts the backend (:8000) and frontend (:5173).
- Tests: `cd backend && uv run pytest`; `cd frontend && npm run build && npm run lint`.
- Designs live in `PRINT_ASSIST_DATA` (default `~/Projects/3DPrinting/designs`).
- AI API key (optional; Claude Code over MCP is the preferred mode): `ANTHROPIC_API_KEY` in `backend/.env` (git-ignored).

## Status
- M1 backend done: parameters and formulas, incremental build engine with update states, ops, Extrude / Revolve / Shell / Text, export (3MF / STL / STEP), renderer, MCP server, API chat.
- Next: frontend layout (tree, toolbars, edit dialog, sketch tools, AI panel), end-to-end vase test; then M2 colours and variants (needs the user's Bambu Studio reference 3MF).

## AI contract
- Every change is an op (`build/ops.py`) applied by the Workspace. Actors other than "user" are built and checked first; a change that makes a feature fail is refused and the design is untouched. Locked features refuse AI edits. Kept AI changes are git checkpoints ("AI: ..."), revertible.
- Claude Code: `./dev.sh`, then Claude Code in this repo picks up `.mcp.json` (server `3d-print-assist`).

## Deployment
Local while iterating; stable builds go on the user's home server. Plan placement with the `homelab-consultant` agent.
