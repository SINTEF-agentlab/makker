"""Who may use this server.

Whoever can drive a session can run code on this computer as the user who
started it: the agent runs Julia and shell commands. So the server asks for its
token (``TokenGuard``): anything else on this computer, another user's program
on a shared machine included, can reach the port, but can't read the token.

Browsers add a second risk: they let any site send requests to 127.0.0.1 and,
without CORS getting in the way, open WebSockets to it. So the server also
turns away pages that aren't its own (``OriginGuard``):

- A request from a browser page carries ``Origin``. A WebSocket handshake, or a
  request that changes something (anything but GET, HEAD and OPTIONS), is only
  answered when that origin is this server's: the ``Host`` the request was sent
  to, the ``X-Forwarded-Host`` a reverse proxy passed on, or an origin allowed
  with ``--allowed-origin``. Another site can't read the answer to a GET, since
  the server sends no CORS headers.
- A server that serves this computer only also checks ``Host`` on every request:
  it must be a loopback name, or the host of an allowed origin. Otherwise a site
  could point its own domain at 127.0.0.1 and pass as the server's own page.

Requests without ``Origin`` (scripts and other programs) are not affected: they
are not run by a page someone else wrote. The checks run as ASGI middleware, so
they cover every route and WebSocket, including the live-plot proxy.
"""

from __future__ import annotations

import contextlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_log = logging.getLogger("uvicorn.error")

TOKEN_ENV_VAR = "MAKKER_SERVER_TOKEN"
_COOKIE = "makker-token"
_COOKIE_MAX_AGE = 365 * 24 * 3600


def token_path() -> Path:
    from makker.paths import state_home

    return state_home() / "server-token"


def server_token() -> str:
    """The token this user's servers ask for.

    ``MAKKER_SERVER_TOKEN`` when set (for a launcher that builds the link
    itself), else the one in ``server-token`` in the state home, made on first
    use and readable by this user only. It stays the same across launches, so a
    browser that has opened the link once keeps working after a restart. Delete
    the file to make a new one.
    """
    if token := os.environ.get(TOKEN_ENV_VAR, "").strip():
        return token
    path = token_path()
    with contextlib.suppress(FileNotFoundError):
        if token := path.read_text(encoding="utf-8").strip():
            return token
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token


_ENTER_PAGE = """<!doctype html><meta charset="utf-8"><title>Makker</title>
<script>location.replace(__TARGET_JS__)</script>
<noscript><a href="__TARGET__">Open Makker</a></noscript>
"""

_LOCKED_PAGE = """<!doctype html><meta charset="utf-8"><title>Makker</title>
<p style="font-family: system-ui, sans-serif; margin: 3em auto; max-width: 34em">
This Makker server asks for its link. Open the link Makker printed when it
started; it ends in <code>?token=</code>.</p>
"""


class TokenGuard:
    """ASGI middleware letting through only requests that carry the token.

    The token comes in a cookie, an ``Authorization: Bearer`` header (scripts),
    or a ``token`` query parameter (the printed link, or a WebSocket from a
    program). A page opened from the link gets the cookie and is sent on to the
    same address without the token, so it leaves the address bar and history.

    The cookie is ``SameSite=Strict``: a browser never sends it with a request
    another site starts, so another site can't open the UI as the user, in a
    frame or a new tab.

    Paths under ``open_prefixes`` carry a capability of their own and pass
    without the token: a live plot's URL holds its session's plot secret
    (``plot_julia_src.viz_route``), so it can be shown where no cookie reaches,
    such as another application's frame or browser pane. ``open_paths`` are
    single paths that give nothing away and pass too.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        base_path: str = "",
        open_prefixes: Iterable[str] = (),
        open_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.token = token.encode()
        self.base_path = base_path
        self.open_prefixes = tuple(open_prefixes)
        self.open_paths = frozenset(open_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] not in ("http", "websocket")
            or scope["path"].startswith(self.open_prefixes)
            or scope["path"] in self.open_paths
        ):
            await self.app(scope, receive, send)
            return
        conn = HTTPConnection(scope)
        if self._matches(conn.cookies.get(_COOKIE)) or self._matches(_bearer(conn.headers)):
            await self.app(scope, receive, send)
        elif self._matches(conn.query_params.get("token")):
            if scope["type"] == "http" and scope["method"] == "GET" and _wants_html(conn):
                await self._enter(conn)(scope, receive, send)
            else:
                await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.close", "code": 1008})
        elif _wants_html(conn):
            await HTMLResponse(_LOCKED_PAGE, status_code=401)(scope, receive, send)
        else:
            await PlainTextResponse("missing or wrong token", status_code=401)(scope, receive, send)

    def _matches(self, given: str | None) -> bool:
        return bool(given) and hmac.compare_digest(given.encode(), self.token)

    def _enter(self, conn: HTTPConnection) -> Response:
        """Set the cookie, then load the same address without the token.

        A page rather than a redirect: the cookie is then sent on the next load
        even when the link was opened from another site.
        """
        rest = [(k, v) for k, v in conn.query_params.multi_items() if k != "token"]
        target = quote(self.base_path + conn.url.path) + (f"?{urlencode(rest)}" if rest else "")
        page = _ENTER_PAGE.replace("__TARGET_JS__", json.dumps(target)).replace(
            "__TARGET__", escape(target, quote=True)
        )
        response = HTMLResponse(page, headers={"Cache-Control": "no-store"})
        response.set_cookie(
            _COOKIE,
            self.token.decode(),
            max_age=_COOKIE_MAX_AGE,
            path=self.base_path or "/",
            httponly=True,
            samesite="strict",
        )
        return response


def _bearer(headers: Headers) -> str | None:
    scheme, _, value = headers.get("authorization", "").partition(" ")
    return value.strip() if scheme.lower() == "bearer" else None


def _wants_html(conn: HTTPConnection) -> bool:
    return "text/html" in conn.headers.get("accept", "")


def is_loopback_host(host: str) -> bool:
    """Whether a bind address or URL host name only reaches this computer."""
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _hostname(host: str) -> str:
    """The name part of a ``Host`` header (``[::1]:8742`` gives ``::1``)."""
    return urlsplit(f"//{host}").hostname or ""


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]`` in lower case, as browsers send it.

    Raises ``ValueError`` for anything else, so a typo in ``--allowed-origin``
    fails at launch rather than refusing the page it meant to let in.
    """
    parts = urlsplit(origin.strip().rstrip("/"))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"not an origin: {origin!r} (expected e.g. https://example.org)")
    if parts.path or parts.query or parts.fragment:
        raise ValueError(f"an origin has no path: {origin!r}")
    return f"{parts.scheme}://{parts.netloc}".lower()


def request_is_local(headers: Mapping[str, str]) -> bool:
    """Whether a request comes from this server's own page on this computer.

    Stricter than the middleware, for the few requests only someone at this
    computer should make (updating Makker): the page must be the server's
    own, and served under a loopback name.
    """
    host = headers.get("host", "")
    origin = headers.get("origin")
    if not host or not origin:
        return False
    return is_loopback_host(_hostname(host)) and urlsplit(origin).netloc == host


class OriginGuard:
    """ASGI middleware applying the checks in the module docstring."""

    def __init__(
        self, app: ASGIApp, *, local_only: bool = False, allowed_origins: Iterable[str] = ()
    ) -> None:
        self.app = app
        # Check Host too: the server answers to this computer only.
        self.local_only = local_only
        self.allowed = {normalize_origin(origin) for origin in allowed_origins}
        self.allowed_hosts = {urlsplit(origin).netloc for origin in self.allowed}
        self._reported: set[str] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        reason = self.refusal(scope["type"], scope.get("method", ""), Headers(scope=scope))
        if reason is None:
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            # Closing before accepting answers the handshake with a 403.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
        else:
            await PlainTextResponse(reason, status_code=403)(scope, receive, send)

    def refusal(self, kind: str, method: str, headers: Headers) -> str | None:
        """Why this request is refused, or ``None`` to let it through."""
        host = headers.get("host", "").lower()
        if self.local_only and host and not self._host_ok(host):
            self._report(
                f"host {host}",
                f"Refused a request for {host}: this server only answers to localhost. "
                "Open it at 127.0.0.1, or start it with --allowed-origin for that address.",
            )
            return "this server only answers to localhost"
        origin = headers.get("origin")
        if origin is None or (kind == "http" and method in _SAFE_METHODS):
            return None
        if self._origin_ok(origin.lower(), host, headers.get("x-forwarded-host", "")):
            return None
        self._report(
            origin,
            f"Refused a request from {origin}, which isn't this server's own page. If it "
            f"should use this server, start Makker with --allowed-origin {origin}",
        )
        return "requests from other sites are refused"

    def _host_ok(self, host: str) -> bool:
        return is_loopback_host(_hostname(host)) or host in self.allowed_hosts

    def _origin_ok(self, origin: str, host: str, forwarded_host: str) -> bool:
        if origin in self.allowed:
            return True
        netloc = urlsplit(origin).netloc
        # A page's own requests name its host; behind a proxy that rewrites Host,
        # the original rides in X-Forwarded-Host (the first one, if chained). A
        # browser page can't set either header on another site's request.
        forwarded = forwarded_host.split(",")[0].strip().lower()
        return bool(netloc) and netloc in (host, forwarded)

    def _report(self, key: str, message: str) -> None:
        """Say once per origin or host why its requests are refused."""
        if key not in self._reported:
            self._reported.add(key)
            _log.warning(message)
