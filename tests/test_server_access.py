"""Only the token's holder, from the server's own pages, may drive it (access.py)."""

from __future__ import annotations

import os
import stat

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from makker.interfaces.server.access import (
    TOKEN_ENV_VAR,
    OriginGuard,
    TokenGuard,
    normalize_origin,
    server_token,
    token_path,
)

SERVER = "http://127.0.0.1:8742"


async def _ok(_request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _echo(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text("hello")
    await websocket.close()


def _client(**guard) -> TestClient:
    app = Starlette(
        routes=[Route("/thing", _ok, methods=["GET", "POST"]), WebSocketRoute("/ws", _echo)]
    )
    return TestClient(OriginGuard(app, **guard), base_url=SERVER)


def _connects(client: TestClient, **headers) -> bool:
    # The test client sends Host: testserver on websockets, whatever its base URL.
    headers.setdefault("host", "127.0.0.1:8742")
    try:
        with client.websocket_connect("/ws", headers=headers) as ws:
            return ws.receive_text() == "hello"
    except WebSocketDisconnect as exc:
        assert exc.code == 1008
        return False


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (SERVER, True),  # its own page
        (None, True),  # a script or another program
        ("https://evil.example", False),  # another site in the same browser
        ("http://127.0.0.1:9999", False),  # another local server's page
        ("null", False),  # a sandboxed frame or a file:// page
    ],
)
def test_websockets_only_from_the_servers_own_page(origin, allowed) -> None:
    headers = {"origin": origin} if origin else {}
    assert _connects(_client(), **headers) is allowed


def test_requests_that_change_something_need_the_servers_own_page() -> None:
    client = _client()
    assert client.post("/thing", headers={"origin": SERVER}).status_code == 200
    assert client.post("/thing").status_code == 200
    refused = client.post("/thing", headers={"origin": "https://evil.example"})
    assert refused.status_code == 403
    # Another site can't read the answer to a GET, so it isn't refused.
    assert client.get("/thing", headers={"origin": "https://evil.example"}).status_code == 200


def test_a_local_server_answers_to_localhost_only() -> None:
    # A site that points its own domain at 127.0.0.1 is "same origin" with itself.
    client = _client(local_only=True)
    rebound = {"host": "evil.example:8742", "origin": "http://evil.example:8742"}
    assert client.get("/thing", headers=rebound).status_code == 403
    assert not _connects(client, **rebound)
    for host in ("127.0.0.1:8742", "localhost:8742", "[::1]:8742"):
        assert client.get("/thing", headers={"host": host}).status_code == 200


def test_an_allowed_origin_may_use_the_server() -> None:
    client = _client(local_only=True, allowed_origins=["http://jutul.test:8742/"])
    page = {"host": "jutul.test:8742", "origin": "http://jutul.test:8742"}
    assert client.post("/thing", headers=page).status_code == 200
    assert _connects(client, **page)
    # From a page on another origin, with this server's own host.
    other = _client(allowed_origins=["https://app.example.org"])
    assert _connects(other, origin="https://app.example.org")


def test_behind_a_reverse_proxy() -> None:
    # The proxy keeps the public Host: the page is the server's own.
    client = _client()
    public = {"host": "sso.example.org", "origin": "https://sso.example.org"}
    assert _connects(client, **public)
    # The proxy rewrites Host, and passes the original on.
    forwarded = {
        "host": "127.0.0.1:8742",
        "x-forwarded-host": "sso.example.org",
        "origin": "https://sso.example.org",
    }
    assert _connects(client, **forwarded)
    assert not _connects(client, **{**forwarded, "origin": "https://evil.example"})


@pytest.mark.parametrize(
    ("value", "normalized"),
    [("https://App.Example.org/", "https://app.example.org"), (SERVER, SERVER)],
)
def test_origin_values(value, normalized) -> None:
    assert normalize_origin(value) == normalized


@pytest.mark.parametrize("value", ["app.example.org", "https://example.org/app", "ftp://x"])
def test_bad_origin_values_fail_at_launch(value) -> None:
    from makker.interfaces.cli import web

    with pytest.raises(ValueError):
        normalize_origin(value)
    with pytest.raises(SystemExit):
        web.build_parser().parse_args(["--allowed-origin", value])


def test_the_server_is_guarded(tmp_path) -> None:
    from makker.interfaces.server.app import create_app
    from makker.interfaces.server.manager import SessionManager

    async def unused_factory(**_kwargs):
        raise AssertionError("no session should start")

    app = create_app(
        SessionManager(host_factory=unused_factory), default_sim="jutuldarcy", token=TOKEN
    )
    with TestClient(app) as client:
        assert client.get("/sessions").status_code == 401
        assert client.get("/sessions", headers=BEARER).status_code == 200
        # A live plot's URL is its own capability (the secret in its route), so the
        # proxy answers without the token; here there is no plot server to reach.
        assert client.get("/live/2026-01-01-0000-abcd/s3cret/viz/a").status_code == 404
        response = client.post(
            "/sessions", json={}, headers={"origin": "https://evil.example", **BEARER}
        )
        assert response.status_code == 403
        stream = "/sessions/2026-01-01-0000-abcd/stream"
        with pytest.raises(WebSocketDisconnect) as refused:  # noqa: SIM117
            with client.websocket_connect(stream, headers={"origin": "https://evil.example"}):
                pass
        assert refused.value.code == 1008


def test_a_server_restarted_after_an_update_says_it_is_back(tmp_path) -> None:
    # The page that started the update polls /version to learn the server is back.
    # One from before the token has no cookie, so only that route answers it.
    from makker.interfaces.server.app import create_app
    from makker.interfaces.server.manager import SessionManager

    async def unused_factory(**_kwargs):
        raise AssertionError("no session should start")

    def app(**kwargs):
        manager = SessionManager(host_factory=unused_factory)
        return create_app(manager, default_sim="jutuldarcy", token=TOKEN, **kwargs)

    with TestClient(app(restarted=True)) as client:
        assert client.get("/version").json()["boot"]
        assert client.get("/sessions").status_code == 401
        assert client.get("/", headers=HTML).status_code == 401
    with TestClient(app()) as client:
        assert client.get("/version").status_code == 401


# --- The token ----------------------------------------------------------------

TOKEN = "s3cret-token"
BEARER = {"authorization": f"Bearer {TOKEN}"}
HTML = {"accept": "text/html,application/xhtml+xml"}


def _token_client(base_path: str = "") -> TestClient:
    app = Starlette(routes=[Route("/thing", _ok), WebSocketRoute("/ws", _echo)])
    return TestClient(TokenGuard(app, token=TOKEN, base_path=base_path), base_url=SERVER)


def test_every_request_needs_the_token() -> None:
    client = _token_client()
    assert client.get("/thing").status_code == 401
    assert client.get("/thing", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/thing", headers=BEARER).text == "ok"
    assert client.get("/thing", params={"token": TOKEN}).text == "ok"  # a script's GET
    assert _connects(client) is False
    assert _connects(client, **BEARER) is True
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        assert ws.receive_text() == "hello"


def test_open_paths_pass_without_the_token() -> None:
    app = Starlette(routes=[Route("/live/s/x", _ok), Route("/thing", _ok)])
    guard = TokenGuard(app, token=TOKEN, open_prefixes=("/live/",))
    client = TestClient(guard, base_url=SERVER)
    assert client.get("/live/s/x").text == "ok"
    assert client.get("/thing").status_code == 401


def test_a_browser_without_the_token_is_told_to_open_the_link() -> None:
    page = _token_client().get("/thing", headers=HTML)
    assert page.status_code == 401
    assert "?token=" in page.text


@pytest.mark.parametrize(
    ("base_path", "target"), [("", "/thing"), ("/restricted", "/restricted/thing")]
)
def test_the_link_swaps_the_token_for_a_cookie(base_path, target) -> None:
    client = _token_client(base_path)
    page = client.get("/thing", params={"token": TOKEN, "data": "a b"}, headers=HTML)

    assert page.status_code == 200
    assert f'location.replace("{target}?data=a+b")' in page.text
    assert TOKEN not in page.text  # the address the page moves on to drops it
    cookie = page.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert f"path={base_path or '/'}" in cookie
    if not base_path:  # (behind a proxy, the browser sends it under the prefix)
        assert client.get("/thing").text == "ok"
    # The cookie opens WebSockets too, the live views' included. (The test client
    # opens them at another host, so it doesn't attach the saved cookie itself.)
    assert _connects(client, cookie=f"makker-token={TOKEN}") is True


def test_the_server_token_is_private_and_stays_the_same(monkeypatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    first = server_token()
    assert len(first) >= 32
    assert server_token() == first
    if os.name == "posix":
        assert stat.S_IMODE(token_path().stat().st_mode) == 0o600

    monkeypatch.setenv(TOKEN_ENV_VAR, "from-the-launcher")
    assert server_token() == "from-the-launcher"
