# 3D Print Assist

A parametric CAD app for 3D printing where you and an AI edit the same model, and designs leave as print-ready Bambu Studio files.

## Desktop App (macOS & Windows 11)

The desktop app bundles the backend, 3D CAD engine (`build123d` / OpenCascade), MCP server, and UI into a single native window — no terminal or Python/Node install required.

### macOS
1. Download `3D-Print-Assist-macOS.zip` from **Releases** (or the **Desktop App Build** GitHub Actions artifact) and unzip it.
2. Move `3D Print Assist.app` to `/Applications` (optional) and double-click to launch.
3. **First launch (unsigned build):** If macOS says the developer cannot be verified, **Control-click (or right-click) `3D Print Assist.app` → Open → Open** (or run `xattr -cr "/Applications/3D Print Assist.app"` once in Terminal).

### Windows 11
1. Download `3D-Print-Assist-Windows11.zip` from **Releases** (or the **Desktop App Build** GitHub Actions artifact) and **Extract All** to a folder (keep `3DPrintAssist.exe` together with its `_internal` folder).
2. Double-click `3DPrintAssist.exe` to launch (you can right-click it to **Pin to taskbar** or **Create shortcut**). Windows 11 includes Edge WebView2 by default.
3. **First launch (unsigned build):** If Windows SmartScreen shows *"Windows protected your PC"*, click **More info → Run anyway**.

### Data, AI & Slicers
- **Shared designs:** Both the desktop app and `./dev.sh` save designs under `~/Projects/3DPrinting/designs` (override with `PRINT_ASSIST_DATA`). Your last chosen **Assistant**, **Model**, and **Printer** are remembered automatically.
- **Antigravity (`agy`):** Works automatically if the `agy` CLI is installed and signed in.
- **Gemini / Claude API keys (optional):** Add `GEMINI_API_KEY=...` or `ANTHROPIC_API_KEY=...` to `~/Projects/3DPrinting/designs/.env` (no restart needed).

### Build the Desktop App from Source
```sh
cd frontend && npm ci && npm run build
cd ../backend && uv sync --extra desktop && uv run pyinstaller 3d-print-assist.spec --noconfirm
```
Outputs `backend/dist/3D Print Assist.app` on macOS or `backend/dist/3DPrintAssist/3DPrintAssist.exe` on Windows 11.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Node 23.6+ (the backend runs the TypeScript sketch solver through Node).

```sh
./dev.sh
```

Then open http://127.0.0.1:5183. The backend serves on port 8010 (clear of cad-assist's 8000 / 5173) and the frontend proxies `/api` to it.

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
