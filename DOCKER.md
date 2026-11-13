# Self-Hosting 3D Print Assist (Docker / Unraid)

Containerizing the app is **optional** and does not affect local development (`./dev.sh`).
When `PRINT_ASSIST_STATIC_DIR` is set (as in the container), FastAPI serves both the `/api/*` backend and the built React frontend on a single port (`8010`).

---

## Option 1: Docker Compose (Unraid Compose Manager or any Docker host)

```bash
GEMINI_API_KEY=your_key_here docker compose up -d --build
```

Then open `http://<your-server-ip>:8010`.

To check if `8010` is already in use on Unraid before starting:
```bash
ss -tulpn | grep :8010 || docker ps --format '{{.Names}}\t{{.Ports}}'
```
If `8010` is taken, pick any free host port (for example `PRINT_ASSIST_PORT=8310 docker compose up -d --build`). The in-app `PRINT_ASSIST_URL` helper automatically uses whichever host port you open in your browser.

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
- **External MCP from any machine (no repo clone needed)**: The server exposes a Streamable HTTP MCP endpoint at `/mcp/`. Any machine with Claude Code, Gemini CLI, or an HTTP-capable MCP client can connect directly by URL (no local repo, Python, or `uv` required):
  ```json
  {
    "mcpServers": {
      "3d-print-assist": {
        "type": "http",
        "url": "http://<unraid-ip>:8010/mcp/"
      }
    }
  }
  ```
  (You can also copy this exact snippet with your server's IP/port already filled in from the **External MCP** panel in the app.)
- **Slicers**: Because the container runs headless on Unraid, **Send to \<Slicer\>** is for local desktop runs. When hosted on Unraid, use the **3MF / STL / STEP** download buttons and open the file in PrusaSlicer / Bambu Studio on your computer.
