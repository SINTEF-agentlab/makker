"""The FastAPI application: REST lifecycle plus the per-session turn WebSocket.

REST creates, lists, resumes, and closes sessions, and serves the files a
session produces. The WebSocket at ``/sessions/{id}/stream`` carries one turn at
a time: the client sends a prompt (or an approval decision, or a cancel), and
the server streams the agent's events back, serialized by ``protocol``.

``create_app`` takes a ``SessionManager`` so tests can inject one whose sessions
wrap fakes; the default manager stands up real sessions.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from makker.interfaces.server import live, stream
from makker.interfaces.server.access import OriginGuard, TokenGuard
from makker.interfaces.server.deps import ServerDefaults
from makker.interfaces.server.manager import SessionManager
from makker.interfaces.server.routes import artifacts, models, sessions, update
from makker.interfaces.server.updates import Updater
from makker.interfaces.server.urls import normalize_base_path, public_url

# The web UI ships pre-built next to this module: ``web_dist`` is the Vite build of
# ``webapp/`` (the React app), committed and shipped so an install needs no Node.
_SERVER_DIR = Path(__file__).resolve().parent
WEB_DIST_DIR = _SERVER_DIR / "web_dist"


def _ui_dir() -> Path | None:
    """The directory to serve the web UI from, or ``None`` if it is not built."""
    return WEB_DIST_DIR if (WEB_DIST_DIR / "index.html").is_file() else None


def _register_web_mime_types() -> None:
    """Force correct MIME types for the built UI's assets before serving.

    Vite loads its bundle via ``<script type="module">``, which a browser runs only
    when the file is served with a JavaScript MIME type. On Windows ``mimetypes``
    reads the registry, where ``.js`` is frequently ``text/plain`` — which makes the
    browser refuse the module and render a blank page. Registering the types in
    process makes serving correct regardless of the host registry. Idempotent.
    """
    import mimetypes

    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/svg+xml", ".svg")


def create_app(
    manager: SessionManager | None = None,
    *,
    ui: bool = True,
    default_sim: str | None = None,
    default_approval_mode: str | None = None,
    default_model: str | None = None,
    julia_project: Path | None = None,
    threads: str | None = None,
    add_dirs: Sequence[Path] = (),
    ephemeral_memory: bool = False,
    sysimage: bool | None = None,
    base_path: str = "",
    updater: Updater | None = None,
    local_only: bool = False,
    allowed_origins: Sequence[str] = (),
    token: str | None = None,
    restarted: bool = False,
) -> FastAPI:
    # The launch-wide knobs (folder-fixed) ride in the default manager's host
    # factory; an injected manager (tests) brings its own. The model is a default
    # a request can still override, so it is applied at the create/resume endpoint.
    base = normalize_base_path(base_path)
    if manager is None:
        from makker.interfaces.server.manager import (
            SessionLaunchDefaults,
            make_host_factory,
        )

        manager = SessionManager(
            host_factory=make_host_factory(
                SessionLaunchDefaults(
                    julia_project=julia_project,
                    threads=threads,
                    add_dirs=tuple(add_dirs),
                    ephemeral_memory=ephemeral_memory,
                    sysimage=sysimage,
                    base_path=base,
                )
            )
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.aclose()

    app = FastAPI(
        title="Makker",
        summary="Drive a Makker session over HTTP and WebSocket.",
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.base_path = base
    if updater is None:
        from makker.paths import workspace_root

        # Embedded or under test: report updates, but never run one.
        updater = Updater(workspace_root(), local=False)
    app.state.updater = updater
    app.state.defaults = ServerDefaults(
        sim=default_sim,
        model=default_model,
        approval_mode=default_approval_mode,
        julia_project=julia_project,
    )
    # Only those with the token, and only from this server's own pages, may drive
    # it; see interfaces/server/access.py. ``local_only`` (a server for this
    # computer only) also checks Host. Without a token (tests, an embedding host
    # that controls access itself) anyone who reaches the port may.
    if token is not None:
        # Temporary: the page that started an update waits for /version to say the
        # server is back. One from before the token (0.3.0) has no cookie, and
        # would wait out its ten minutes on a 401. A server restarted after an
        # update answers it; the page then reloads onto the one asking for the link.
        app.add_middleware(
            TokenGuard,
            token=token,
            base_path=base,
            open_prefixes=("/live/",),
            open_paths=("/version",) if restarted else (),
        )
    app.add_middleware(OriginGuard, local_only=local_only, allowed_origins=allowed_origins)

    for group in (models, update, sessions, artifacts, live, stream):
        app.include_router(group.router)

    # The bundled web UI is mounted last so the API routes above take precedence.
    ui_dir = _ui_dir()
    if ui and ui_dir is not None:
        _register_web_mime_types()
        index_html = (ui_dir / "index.html").read_text(encoding="utf-8")
        # Vite emits relative assets. Use public paths so /restricted and
        # /restricted/ both load, without relying on the proxy to redirect.
        assets_url = escape(public_url(base, "/assets/"), quote=True)
        index_html = re.sub(
            r"""((?:src|href)=["'])\./assets/""",
            lambda match: match[1] + assets_url,
            index_html,
        )
        # Inject the public base path so the SPA prefixes API/WebSocket URLs.
        inject = f"<script>window.__MAKKER_BASE_PATH__={json.dumps(base)};</script>"
        if "<!--makker-base-path-->" in index_html:
            index_html = index_html.replace("<!--makker-base-path-->", inject)
        elif "</head>" in index_html:
            index_html = index_html.replace("</head>", f"{inject}</head>", 1)
        else:
            index_html = inject + index_html

        @app.api_route("/", methods=["GET", "HEAD"])
        def serve_spa_index() -> HTMLResponse:
            return HTMLResponse(index_html)

        assets = ui_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

    return app
