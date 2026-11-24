"""Shared web server test helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def open_stream(client: Any, session_id: str) -> Iterator[Any]:
    """Connect to a session's stream, past the snapshot every connection opens with."""
    with client.websocket_connect(f"/sessions/{session_id}/stream") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        yield ws
