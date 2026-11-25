"""The version line and updates started from the browser; the update itself
runs in ``interfaces/server/updates.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from makker.interfaces.server.access import request_is_local
from makker.interfaces.server.deps import DefaultsDep, ManagerDep, UpdaterDep
from makker.interfaces.server.updates import BOOT_ID, UpdateRefused

router = APIRouter()


class UpdateRequest(BaseModel):
    # Upgrade Makker itself first, or only update the Julia environment.
    install: bool = False
    # What to do with this folder's system image, if it has one.
    sysimage: Literal["rebuild", "off"] | None = None


@router.get("/version")
def version_info(updater: UpdaterDep, defaults: DefaultsDep) -> dict[str, Any]:
    """What this server runs, for the quiet version line in the UI.

    The Makker version, a newer release when the update check knows of
    one, and the simulator version this folder's Julia environment resolved.
    Each part is best-effort: an unknown one is ``None``, never an error.
    """
    from makker import __version__
    from makker.paths import workspace_root
    from makker.simulators import registry
    from makker.update_check import is_enabled, pending_update, refresh_if_stale
    from makker.workspace import resolve_julia_project, resolved_package_version

    simulator = None
    if defaults.sim is not None:
        package = registry.get(defaults.sim).primary_package
        project = defaults.julia_project or resolve_julia_project(workspace_root())
        simulator = {
            "package": package,
            "version": resolved_package_version(project, package),
        }
    # A server can run for days, so it keeps its own view of releases current.
    refresh_if_stale()
    return {
        "makker": __version__.split("+")[0],
        "update": pending_update() if is_enabled() else None,
        "simulator": simulator,
        # The folder's Julia environment is older than this version supports.
        "env_update": updater.env_update_needed(),
        # Changes when the server restarts, e.g. after an update.
        "boot": BOOT_ID,
    }


@router.get("/update")
def update_status(updater: UpdaterDep, since: int = 0) -> dict[str, Any]:
    """What an update would do here, and the progress of one that is running.

    ``since`` skips output lines the page already has.
    """
    return updater.status(since=since)


@router.post("/update", status_code=202)
async def start_update(
    req: UpdateRequest, request: Request, manager: ManagerDep, updater: UpdaterDep
) -> dict[str, Any]:
    """Start updating Makker and this folder's Julia environment.

    Only this computer's own page may ask: the update runs commands and restarts
    the server. See ``interfaces/server/updates.py``.
    """
    if not request_is_local(request.headers):
        raise HTTPException(status_code=403, detail="updates are only accepted from this computer")
    try:
        await updater.start(manager, install=req.install, sysimage=req.sysimage)
    except UpdateRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return updater.status()
