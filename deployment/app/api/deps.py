from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request, WebSocket


def get_server(request: Request) -> Any:
    """Get the VoxCPM server from app state (HTTP routes)."""
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Model server not ready")
    return cast(Any, server)


def get_server_ws(websocket: WebSocket) -> Any:
    """Get the VoxCPM server from app state (WebSocket routes).

    FastAPI cannot infer the same dependency for both Request and WebSocket,
    so we provide a dedicated version. WebSocket routes can't raise
    HTTPException so we return None when not ready - the route handler is
    responsible for closing the connection.
    """
    return getattr(websocket.app.state, "server", None)
