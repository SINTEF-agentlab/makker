"""Tests for the /context trigger figure and the manual /compact path.

Auto-compaction is deepagents' stock SummarizationMiddleware (covered by the
builder and recorder tests); this module covers the manual ``compact_thread``,
which drives the same engine non-mutatingly and recoverably.
"""

from __future__ import annotations

from pathlib import Path

from makker.agent.builder import build_agent
from makker.agent.summarization import compact_thread
from makker.agent.turns import TurnRunner
from makker.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from makker.session import Session


async def test_compact_thread_round_trip(tmp_path: Path) -> None:
    """/compact records a summarization event without rewriting the raw log."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    # Substantial replies so the summary is meaningfully smaller than the turns
    # it replaces (a few words would summarize to something larger).
    finals = [scripted_final(f"answer {i}: " + "detail " * 80) for i in range(8)]
    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, backend = build_agent(session, model=make_scripted_model(finals), checkpointer=saver)
        runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
        for i in range(6):
            await runner.run_prompt(f"question {i}")

        result = await compact_thread(
            agent,
            thread_id=session.session_id,
            model=make_scripted_model([scripted_final("SUMMARY of earlier work")]),
            backend=backend,
            trace=session.trace,
        )
        assert result is not None
        # 12 messages (6 turns), keep 8 → summarize the oldest 4.
        assert result.messages_summarized == 4
        assert result.messages_kept == 8
        assert result.freed_tokens > 0
        assert result.offloaded is True

        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        # Non-mutating: the raw conversation log is preserved in full...
        assert len(state.values["messages"]) == 12
        # ...and the compaction is recorded as an event the next turn applies,
        # anchored by message id and marking where the kept turns ended.
        event = state.values["_summarization_event"]
        assert event["cutoff_index"] == 4
        assert event["cutoff_message_id"] == state.values["messages"][4].id
        assert event["created_at_index"] == 12
        history_id = state.values["_summarization_session_id"]
        assert history_id

        for i in range(6, 8):
            await runner.run_prompt(f"question {i}")
        again = await compact_thread(
            agent,
            thread_id=session.session_id,
            model=make_scripted_model([scripted_final("SUMMARY of more work")]),
            backend=backend,
            trace=session.trace,
        )
        assert again is not None and again.offloaded
        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        assert state.values["_summarization_session_id"] == history_id
        assert state.values["_summarization_event"]["cutoff_index"] == 8

    # The offloaded turns were written somewhere recoverable under the session.
    offloaded = list((session.state_dir / "conversation_history").glob("*.md"))
    assert len(offloaded) == 1
    history = offloaded[0].read_text(encoding="utf-8")
    assert "question 0" in history and "question 2" in history

    # Each manual compaction is recorded once: the turns after it see the same
    # compaction in the state and must not record it again as an automatic one.
    events = [e for e in session.trace.iter_events() if e.kind == "context_compaction"]
    assert [e.payload["manual"] for e in events] == [True, True]
    assert [e.payload["cutoff"] for e in events] == [4, 8]
    session.finalize()


async def test_auto_compaction_is_announced_once_and_kept_out_of_the_chat(
    tmp_path: Path,
) -> None:
    """An automatic compaction reaches the front end as one note, not as chat text.

    The summarizer shares the scripted model with the agent, so it consumes one
    scripted reply; that reply must not stream to the chat. The compaction is
    recorded once, also after the agent is rebuilt (a resume or model switch).
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from makker.agent.turns import TurnCompaction, TurnTextDelta

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    replies = [scripted_final(f"reply-{i}: " + "detail " * 600) for i in range(30)]
    model = make_scripted_model(replies)
    # A small window, so a few long replies cross the summarizer's trigger.
    model.profile = {"max_input_tokens": 20_000}

    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, _ = build_agent(session, model=model, checkpointer=saver)
        runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

        async def turn(prompt: str, runner: TurnRunner) -> tuple[str, list[TurnCompaction]]:
            text: list[str] = []
            notes: list[TurnCompaction] = []

            def on_message(event: object) -> None:
                if isinstance(event, TurnTextDelta):
                    text.append(event.text)
                elif isinstance(event, TurnCompaction):
                    notes.append(event)

            await runner.run_prompt(prompt, on_message=on_message)
            return "".join(text), notes

        for i in range(12):
            text, notes = await turn(f"question {i}", runner)
            # One reply per turn reaches the chat, never the summary call's.
            assert text.count("reply-") == 1
            if notes:
                break
        assert len(notes) == 1, "compaction should have triggered within 12 turns"
        payload = notes[0].payload
        assert payload["manual"] is False and payload["offloaded"] is True
        assert payload["summarized"] == payload["cutoff"] > 0
        assert notes[0].text == "Context compacted."
        assert notes[0].payload["manual"] is False

        # The next turn and a rebuilt agent see the same compaction in the state.
        assert (await turn("next", runner))[1] == []
        rebuilt, _ = build_agent(session, model=model, checkpointer=saver)
        rerunner = TurnRunner(rebuilt, thread_id=session.session_id, trace=session.trace)
        assert (await turn("after rebuild", rerunner))[1] == []

    events = [e for e in session.trace.iter_events() if e.kind == "context_compaction"]
    assert len(events) == 1
    session.finalize()


async def test_compact_thread_skips_short_threads() -> None:
    class _NoState:
        pass

    assert (
        await compact_thread(_NoState(), thread_id="t", model=None, backend=None, trace=None)
    ) is None
