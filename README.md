# 3D Print Assist

A parametric CAD app for 3D printing where you and an AI edit the same model, and designs leave as print-ready Bambu Studio files.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Node 23.6+ (the backend runs the TypeScript sketch solver through Node).

```sh
./dev.sh
```

Then open http://127.0.0.1:5173. The backend serves on port 8000 and the frontend proxies `/api` to it.

Designs are stored as git-tracked folders under `~/Projects/3DPrinting/designs` (set `PRINT_ASSIST_DATA` to change it).

## Tests

```sh
cd backend && uv run pytest
cd frontend && npm run build && npm run lint
```

## Layout

| Path | What it is |
| --- | --- |
| `backend/` | FastAPI app, build123d / OpenCascade geometry, pytest suite |
| `frontend/` | React + TypeScript + three.js viewer and sketcher |

See `CLAUDE.md` for architecture and conventions.
