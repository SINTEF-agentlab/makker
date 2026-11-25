from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_sysimage_preflight_prepares_env_before_final_decision(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A bundled-source refresh must refuse before uvicorn serves a dead UI."""
    from makker import sysimage
    from makker.agent import capabilities
    from makker.interfaces.cli.web import _sysimage_ready
    from makker.simulators import env_setup, registry
    from makker.workspace import WorkspaceConfig, workspace_julia_env

    project = workspace_julia_env(tmp_path)
    project.mkdir(parents=True)
    image = tmp_path / "agent.dylib"
    decisions = iter(
        [
            sysimage.Decision(status=sysimage.CURRENT, path=image),
            sysimage.Decision(
                status=sysimage.DIVERGENT,
                reason="  edited since the image was built:\n    MakkerJutulDarcy",
            ),
        ]
    )
    prepared: list[tuple[object, Path, Path, str | None, list[Path]]] = []
    adapter = SimpleNamespace(name="jutuldarcy")

    monkeypatch.setattr(sysimage, "decide", lambda *args, **kwargs: next(decisions))
    monkeypatch.setattr(registry, "get", lambda name: adapter)
    monkeypatch.setattr(capabilities, "discover_extensions", lambda: [])
    monkeypatch.setattr(
        env_setup,
        "prepare_workspace_env",
        lambda selected, **kw: prepared.append(
            (
                selected,
                kw["workspace"],
                kw["julia_project"],
                kw["sim_name"],
                kw["dependencies"],
            )
        ),
    )

    args = Namespace(julia_project=None, sysimage=None)
    ready = _sysimage_ready(
        args,
        tmp_path,
        WorkspaceConfig(sysimage=True),
        sim="jutuldarcy",
    )

    assert ready is False
    assert prepared == [(adapter, tmp_path, project, "jutuldarcy", [])]
    stderr = capsys.readouterr().err
    assert "MakkerJutulDarcy" in stderr
    assert "makker sysimage build" in stderr


def test_sysimage_preflight_leaves_explicit_julia_project_alone(
    tmp_path: Path, monkeypatch
) -> None:
    from makker import sysimage
    from makker.interfaces.cli.web import _sysimage_ready
    from makker.simulators import env_setup
    from makker.workspace import WorkspaceConfig

    project = tmp_path / "shared-env"
    image = tmp_path / "agent.dylib"
    monkeypatch.setattr(
        sysimage,
        "decide",
        lambda *args, **kwargs: sysimage.Decision(status=sysimage.CURRENT, path=image),
    )

    def fail_if_prepared(*args, **kwargs):
        raise AssertionError("an explicit Julia project must not be prepared")

    monkeypatch.setattr(env_setup, "prepare_workspace_env", fail_if_prepared)
    assert _sysimage_ready(
        Namespace(julia_project=project, sysimage=None),
        tmp_path,
        WorkspaceConfig(sysimage=True),
        sim="jutuldarcy",
    )


def test_web_refuses_an_outdated_workspace_env_before_binding(
    tmp_path, monkeypatch, capsys
) -> None:
    """Without a system image the env is prepared per session, so check at launch."""
    from types import SimpleNamespace

    from makker.interfaces.cli import web
    from makker.simulators import env_setup

    def outdated(*_args):
        raise env_setup.EnvSetupError("has Foo 0.3.12, but requires Foo 0.4")

    monkeypatch.setattr(env_setup, "check_env_versions", outdated)
    args = SimpleNamespace(julia_project=None, sysimage=None)
    assert not web._env_versions_ok(args, tmp_path, sim="jutuldarcy")
    assert "requires Foo 0.4" in capsys.readouterr().err
    # A prepared env passed explicitly is the user's choice and is not checked.
    explicit = SimpleNamespace(julia_project=tmp_path, sysimage=None)
    assert web._env_versions_ok(explicit, tmp_path, sim="jutuldarcy")


@pytest.mark.parametrize(
    ("tty", "answer", "upgraded"),
    [(False, None, False), (True, "n", False), (True, "", True), (True, "y", True)],
)
def test_outdated_env_offers_an_upgrade_from_a_terminal(
    tmp_path, monkeypatch, capsys, tty, answer, upgraded
) -> None:
    """A terminal launch can upgrade in place and carry on; anything else stops."""
    from types import SimpleNamespace

    from makker.interfaces.cli import _helpers, upgrade
    from makker.simulators import env_setup

    def outdated(*_args):
        raise env_setup.EnvSetupError("has Foo 0.3.12, but requires Foo 0.4")

    calls = []
    monkeypatch.setattr(env_setup, "check_env_versions", outdated)
    monkeypatch.setattr(upgrade, "upgrade_workspace", lambda *a, **k: calls.append(k) or True)
    monkeypatch.setattr(_helpers.sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr(_helpers.sys, "stderr", SimpleNamespace(isatty=lambda: tty, write=print))
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    ok = _helpers.env_versions_ok(SimpleNamespace(), tmp_path, None, "jutuldarcy")

    assert ok is upgraded
    assert calls == ([{"sysimage": False}] if upgraded else [])


def test_a_newer_simulator_launches_with_a_note(tmp_path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    from makker.interfaces.cli import _helpers
    from makker.simulators import env_setup

    monkeypatch.setattr(env_setup, "check_env_versions", lambda *_a: None)
    monkeypatch.setattr(env_setup, "newer_simulator_note", lambda *_a: "has JutulDarcy 0.5.0")
    monkeypatch.setattr("builtins.input", lambda _p: pytest.fail("must not offer an update"))

    assert _helpers.env_versions_ok(SimpleNamespace(), tmp_path, None, "jutuldarcy")
    assert "note: has JutulDarcy 0.5.0" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "opens"),
    [
        ({}, True),
        ({"MAKKER_NO_OPEN": "1"}, False),  # turned off (and in tests)
        ({"SSH_CONNECTION": "10.0.0.2 51000 10.0.0.1 22"}, False),  # the browser is elsewhere
        ({"tty": False}, False),  # started by a program that shows the UI itself
        ({"display": False}, False),  # no desktop: a text browser would take the terminal
    ],
)
def test_the_browser_opens_only_in_front_of_the_user(monkeypatch, change, opens) -> None:
    from makker import display
    from makker.interfaces.cli import web

    monkeypatch.delenv("MAKKER_NO_OPEN", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    tty = change.get("tty", True)
    monkeypatch.setattr(web.sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr(web.sys, "stderr", SimpleNamespace(isatty=lambda: tty, write=print))
    monkeypatch.setattr(display, "has_display", lambda: change.get("display", True))
    for name, value in change.items():
        if name.isupper():
            monkeypatch.setenv(name, value)
    assert web._can_open_browser() is opens


def test_the_browser_opens_once_the_server_takes_connections(monkeypatch) -> None:
    import threading
    import webbrowser

    from makker.interfaces.cli import web

    opened = threading.Event()
    urls: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: urls.append(url) or opened.set())

    server = SimpleNamespace(started=False, should_exit=False)
    web._open_when_serving(server, "http://127.0.0.1:8742/?token=t")
    assert not opened.wait(0.3)  # not before it serves: the page would fail to load
    server.started = True
    assert opened.wait(5)
    assert urls == ["http://127.0.0.1:8742/?token=t"]

    # A server that stops before it serves opens nothing.
    import time

    gone = SimpleNamespace(started=False, should_exit=False)
    web._open_when_serving(gone, "http://127.0.0.1:8743/?token=t")
    gone.should_exit = True
    time.sleep(0.3)
    gone.started = True  # too late: the wait has ended
    time.sleep(0.3)
    assert urls == ["http://127.0.0.1:8742/?token=t"]
