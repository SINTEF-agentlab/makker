"""The page server behind an MCP session's figure links.

An MCP client shows text and images, not a live WebGL figure, so a plot's result
links to one: the web UI's own live-plot routes (``interfaces/server/live.py``),
served for this one session on a port of its own, bound to 127.0.0.1. Nothing
listens until the first live figure is linked.

Anything on the computer can reach a loopback port, so every address starts with
a random key only the links carry. Requests must also name the loopback host
itself and, where a browser states an origin, come from these pages
(``OriginGuard``).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from collections.abc import Callable, Iterator
from typing import Any

_LOOPBACK = "127.0.0.1"
_STARTUP_TIMEOUT = 10.0
_SHUTDOWN_TIMEOUT = 5.0


class _OneSession:
    """What the live-plot routes ask a session manager for: this server's session."""

    def __init__(self, host: Callable[[], Any]) -> None:
        self._host = host

    def get(self, session_id: str) -> Any:
        host = self._host()
        return host if host is not None and host.session.session_id == session_id else None


class Pages:
    """The live-plot routes for one session, served under ``/<key>`` once asked for."""

    def __init__(self, host: Callable[[], Any]) -> None:
        self._sessions = _OneSession(host)
        self._port = 0
        # Also the session's ``base_path``: the plot server writes its own links
        # under it, so they stay inside what this server answers.
        self.prefix = f"/{secrets.token_urlsafe(16)}"
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None
        self._starting = asyncio.Lock()

    async def base_url(self) -> str:
        """Where the pages are served, starting the server on first use. A server
        that failed to start is forgotten, so the next figure tries again."""
        async with self._starting:
            if self._task is None:
                try:
                    await self._start()
                except BaseException:
                    await self.aclose()
                    raise
        return f"http://{_LOOPBACK}:{self._port}{self.prefix}"

    async def _start(self) -> None:
        import uvicorn
        from fastapi import FastAPI
        from starlette.applications import Starlette
        from starlette.routing import Mount

        from makker.interfaces.server import live
        from makker.interfaces.server.access import OriginGuard

        routes = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        routes.state.manager = self._sessions
        routes.state.base_path = self.prefix
        routes.include_router(live.router)
        app = OriginGuard(Starlette(routes=[Mount(self.prefix, app=routes)]), local_only=True)
        config = uvicorn.Config(
            app,
            log_config=None,  # leave the hosting process's logging alone
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=2,
        )

        class Embedded(uvicorn.Server):
            """Leaves signal handling to the process hosting it."""

            @contextlib.contextmanager
            def capture_signals(self) -> Iterator[None]:
                yield

        # Bound first, so the port is known before anything links to it.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STARTUP_TIMEOUT
        try:
            sock.bind((_LOOPBACK, 0))
            self._server = Embedded(config)
            self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
            while not self._server.started:
                if self._task.done():
                    self._task.result()  # surface why it stopped
                    raise RuntimeError("the page server stopped while starting")
                if loop.time() > deadline:
                    raise RuntimeError("the page server did not start")
                await asyncio.sleep(0.01)
        except BaseException:
            sock.close()
            raise
        self._port = sock.getsockname()[1]

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        self._server.should_exit = True
        done, _ = await asyncio.wait({task}, timeout=_SHUTDOWN_TIMEOUT)
        if not done:
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved: a failed shutdown is not worth a warning
