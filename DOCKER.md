# Self-Hosting 3D Print Assist (Docker / Unraid)

Containerizing the app is **optional** and does not affect local development (`./dev.sh`).
When `PRINT_ASSIST_STATIC_DIR` is set (as in the container), FastAPI serves both the `/api/*` backend and the built React frontend on a single port (`8010`).

---

## Option 1: Docker Compose (Unraid Compose Manager or any Docker host)

```bash
GEMINI_API_KEY=your_key_here docker compose up -d --build
```

Then open `http://<your-server-ip>:8010`.

To persist designs across container rebuilds, map `/data` to an Unraid appdata folder (for example `/mnt/user/appdata/3d-print-assist/data:/data` in `compose.yaml`).

---

## Option 2: Unraid Docker UI (Add Container)

1. Build or push the image from this repo (for example `docker build -t 3d-print-assist:latest .`).
2. In Unraid **Docker -> Add Container**:
   - **Repository**: `3d-print-assist:latest` (or your registry tag)
   - **Port**: Container `8010` → Host `8010` (TCP)
   - **Path**: Container `/data` → Host `/mnt/user/appdata/3d-print-assist` (Read/Write)
   - **Variables**:
     - `GEMINI_API_KEY` (for Google Gemini in-app chat)
     - `ANTHROPIC_API_KEY` (for Anthropic Claude in-app chat)

---

## Using Gemini CLI / Claude Code / Antigravity with the Container

- **In-App Chat**: Use **Google Gemini (API Key)** or **Anthropic Claude (API Key)** in the **Assistant** dropdown by setting `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` on the container. (Local Mac CLI binaries such as `agy` are not inside the Linux container.)
- **External MCP from your Mac/PC**: Your local Gemini CLI, Claude Code, or Antigravity IDE can still drive the Unraid-hosted app over MCP! Point the MCP server at your Unraid host in `.mcp.json` (or copy the generated snippet from the **External MCP** view inside the app):
  ```json
  {
    "mcpServers": {
      "3d-print-assist": {
        "command": "uv",
        "args": ["run", "--directory", "mcp", "print-assist-mcp"],
        "env": {
          "PRINT_ASSIST_URL": "http://<unraid-ip>:8010"
        }
      }
    }
  }
  ```
- **Slicers**: Because the container runs headless on Unraid, **Send to \<Slicer\>** is for local desktop runs. When hosted on Unraid, use the **3MF / STL / STEP** download buttons and open the file in PrusaSlicer / Bambu Studio on your computer.
