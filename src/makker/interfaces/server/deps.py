"""What the routes share, read from ``app.state``, where ``create_app`` puts it.

A route asks for the pieces it needs by type: the session manager, the updater,
the launch's defaults, and the public base path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from starlette.requests import HTTPConnection

from makker.interfaces.server.manager import SessionManager
from makker.interfaces.server.updates import Updater


@dataclass(frozen=True)
class ServerDefaults:
    """What the launch fixed for every request: the bound simulator (``None`` lets
    a request choose), the model and approval mode a request may override, and
    the Julia project the version line reads the simulator's version from."""

    sim: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    julia_project: Path | None = None


# Async so that reading them never waits for a threadpool slot.
async def _manager(conn: HTTPConnection) -> SessionManager:
    return conn.app.state.manager


async def _updater(conn: HTTPConnection) -> Updater:
    return conn.app.state.updater


async def _defaults(conn: HTTPConnection) -> ServerDefaults:
    return conn.app.state.defaults


async def _base_path(conn: HTTPConnection) -> str:
    return getattr(conn.app.state, "base_path", "") or ""


ManagerDep = Annotated[SessionManager, Depends(_manager)]
UpdaterDep = Annotated[Updater, Depends(_updater)]
DefaultsDep = Annotated[ServerDefaults, Depends(_defaults)]
BasePathDep = Annotated[str, Depends(_base_path)]
