"""Tests for the ``makker`` run command's process handling."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from types import SimpleNamespace

import pytest

from makker.interfaces.cli import run as run_cmd


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_signal_closes_the_session(name, monkeypatch) -> None:
    """A closed terminal (SIGHUP) or a kill (SIGTERM) used to end the process on
    the spot, leaving the session's Xvfb running; now the session is closed."""
    from makker.interfaces import tui
    from makker.julia import requirements
    from makker.session_host import SessionHost

    sig = getattr(signal, name)
    host = SimpleNamespace(
        session=SimpleNamespace(session_id="2026-11-25-1200-abcd"),
        agent=object(),
        backend=None,
        model="openai:gpt-5.4-mini",
        warmup_task=None,
        closed=False,
    )

    async def _start(**_kwargs):
        return host

    async def _aclose():
        host.closed = True

    host.aclose = _aclose

    class _TUI:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run_async(self) -> None:
            assert signal.getsignal(sig) is not signal.SIG_DFL, f"nothing handles {name}"
            os.kill(os.getpid(), sig)
            await asyncio.sleep(30)

    monkeypatch.setattr(SessionHost, "start", staticmethod(_start))
    monkeypatch.setattr(tui, "TUIApp", _TUI)
    monkeypatch.setattr(requirements, "require_julia", lambda: None)
    monkeypatch.setattr(run_cmd, "env_versions_ok", lambda *a, **k: True)
    monkeypatch.setattr(run_cmd, "_warn_if_plotting_unavailable", lambda: None)
    monkeypatch.setattr("makker.update_check.notify_at_launch", lambda: None)

    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        code = run_cmd.dispatch(run_cmd.build_parser().parse_args(["--sim", "jutuldarcy"]))
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)

    assert host.closed
    assert code == 1
