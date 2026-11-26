"""The host-owned turn loop, settle hooks, and titling (shared by every surface)."""

from __future__ import annotations

from pathlib import Path

import pytest

from makker.lab.fakes import FakeJulia, echo_agent, interrupt_agent, make_fake_adapter
from makker.session import Session
from makker.session_host import SessionHost


def _host(agent, tmp_path: Path, *, approval_mode: str = "ask") -> SessionHost:
    session = Session.create(
        julia=FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )
    return SessionHost(
        session=session,
        agent=agent,
        model="openai:gpt-test",
        approval_mode=approval_mode,
        surface="tui",
    )


async def test_drive_turn_auto_resumes_in_auto_mode(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path, approval_mode="auto")

    result = await host.drive_turn(lambda: host.runner.run_prompt("do it"))

    assert result.interrupts == []
    assert len(agent.resume_inputs) == 1  # the loop approved and resumed once


async def test_drive_turn_stops_for_a_human_in_ask_mode(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path)

    result = await host.drive_turn(lambda: host.runner.run_prompt("do it"))

    assert [i.interrupt_id for i in result.interrupts] == ["interrupt-1"]
    assert agent.resume_inputs == []


async def test_pending_interrupts_reads_the_persisted_state(tmp_path: Path) -> None:
    agent = interrupt_agent()
    host = _host(agent, tmp_path)
    await host.drive_turn(lambda: host.runner.run_prompt("do it"))

    pending = await host.pending_interrupts()

    assert [p.interrupt_id for p in pending] == ["interrupt-1"]


async def test_maybe_title_upgrades_once_after_the_first_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_title(model_id, conversation):
        assert "User:" in conversation
        return "A Fine Title"

    monkeypatch.setattr("makker.agent.titling.generate_session_title", fake_title)
    host = _host(echo_agent(), tmp_path)
    await host.runner.run_prompt("hello")  # records the first user message

    seen: list[str] = []
    task = host.maybe_title(seen.append)
    assert task is not None
    await task

    assert seen == ["A Fine Title"]
    assert host.session.title == "A Fine Title"
    assert host.maybe_title(seen.append) is None  # once per session


async def test_maybe_title_skips_later_turns(tmp_path: Path) -> None:
    host = _host(echo_agent(), tmp_path)
    await host.runner.run_prompt("one")
    await host.runner.run_prompt("two")

    assert host.maybe_title() is None
    assert host.titled is False  # never claimed, so a fresh host could still title


async def test_a_local_model_is_sized_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizing an unmeasured local model loads it, for seconds to minutes. On the
    event loop that stopped every session and the whole interface meanwhile."""
    import asyncio
    import threading
    import time

    from makker import models, session_host

    sized: list[threading.Thread] = []

    def slow_sizing(model_id: str, **_kwargs) -> int:
        sized.append(threading.current_thread())
        time.sleep(0.3)
        return 32_768

    monkeypatch.setattr(models, "local_context", slow_sizing)
    monkeypatch.setattr(session_host, "_MEASURE_NOTE_AFTER_S", 0.05)
    notes: list[str] = []
    turns = 0

    async def note() -> None:
        notes.append("measuring")

    async def keep_turning() -> None:
        nonlocal turns
        while True:
            await asyncio.sleep(0.01)
            turns += 1

    loop_alive = asyncio.create_task(keep_turning())
    await SessionHost.measure_model("ollama:qwen3.8:27b", on_wait=note)
    loop_alive.cancel()

    assert [thread is threading.current_thread() for thread in sized] == [False]
    assert turns >= 10  # the loop kept turning while the model was sized
    assert notes == ["measuring"]  # and the wait got its one word

    # A hosted model, or another provider's, has nothing to size.
    await SessionHost.measure_model("ollama:glm-5.1:cloud", on_wait=note)
    await SessionHost.measure_model("openai:gpt-test", on_wait=note)
    assert len(sized) == 1 and notes == ["measuring"]
