"""Native desktop launcher for 3D Print Assist (macOS WKWebView / Windows Edge WebView2)."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path


def _find_port(preferred: int = 8010) -> int:
    """Return `preferred` if available on 127.0.0.1, otherwise an ephemeral free port."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", candidate))
                return int(sock.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("Could not bind a local port for 3D Print Assist")


def _static_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "frontend_dist"
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _wait_for_server(port: int, server_thread: threading.Thread, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("Backend server exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Backend server did not start on port {port} within {timeout}s")


def _configure_macos_window(window: object) -> None:
    """Enable native macOS green-button fullscreen and HTML5 element fullscreen in WKWebView."""
    if sys.platform != "darwin":
        return
    try:
        import AppKit
        from PyObjCTools import AppHelper

        def _apply() -> None:
            ns_window = getattr(window, "native", None)
            if ns_window is None:
                return
            primary = getattr(AppKit, "NSWindowCollectionBehaviorFullScreenPrimary", 1 << 7)
            ns_window.setCollectionBehavior_(ns_window.collectionBehavior() | primary)
            content_view = ns_window.contentView()
            if content_view is not None and hasattr(content_view, "subviews"):
                for sub in content_view.subviews():
                    if hasattr(sub, "configuration"):
                        prefs = sub.configuration().preferences()
                        if hasattr(prefs, "setElementFullscreenEnabled_"):
                            prefs.setElementFullscreenEnabled_(True)

        AppHelper.callAfter(_apply)
    except Exception:
        pass


def main() -> None:
    os.environ.setdefault("PRINT_ASSIST_DESKTOP", "1")
    static_dir = _static_dir()
    if static_dir.is_dir():
        os.environ.setdefault("PRINT_ASSIST_STATIC_DIR", str(static_dir))

    port = int(os.environ.get("PRINT_ASSIST_PORT") or _find_port(8010))
    os.environ["PRINT_ASSIST_PORT"] = str(port)

    import uvicorn
    import webview

    from print_assist.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-server", daemon=True)
    thread.start()
    _wait_for_server(port, thread)

    try:
        window = webview.create_window(
            "3D Print Assist",
            f"http://127.0.0.1:{port}",
            width=1440,
            height=900,
            min_size=(1024, 680),
        )
        webview.start(_configure_macos_window, window)
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
