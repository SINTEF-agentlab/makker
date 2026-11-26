"""Tests for the turn runner and HITL trace recording."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from makker.agent.turns import (
    TurnReasoningDelta,
    TurnRunner,
    TurnTextDelta,
    TurnTextEnd,
    TurnToolEvent,
)
from makker.lab.fakes import (
    ScriptedV3Agent,
    interrupt_agent,
    tool_call_events,
    v3_message_event,
    v3_tool_event,
    v3_values_event,
)
from makker.trace import TraceLog


def _message_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_message_event(human),
            v3_message_event(assistant),
            v3_values_event([human, assistant]),
        ]
    )


def _chunk_message_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_message_event(AIMessageChunk(content="hel")),
            v3_message_event(AIMessageChunk(content="lo", chunk_position="last")),
            v3_values_event([human, assistant]),
        ]
    )


def _v3_reasoning_and_tool_agent() -> ScriptedV3Agent:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="hello")
    return ScriptedV3Agent(
        [
            v3_values_event([human]),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "reasoning-delta", "reasoning": "checking context"},
                }
            ),
            v3_message_event(
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": "hello"},
                }
            ),
            v3_message_event(
                {
                    "event": "content-block-finish",
                    "content": {
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "execute",
                        "args": {"path": "/"},
                    },
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "tool-1",
                    "tool_name": "execute",
                    "input": {"path": "/"},
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-finished",
                    "tool_call_id": "tool-1",
                    "output": ["a", "b"],
                }
            ),
            v3_message_event({"event": "message-finish"}),
            v3_values_event([human, assistant]),
        ]
    )


async def test_turn_runner_emits_text_as_deltas() -> None:
    """The runner surfaces text as TurnTextDelta events closed by a TurnTextEnd;
    full assembled messages live in ``result.messages``."""

    runner = TurnRunner(_message_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    deltas = [m for m in seen if isinstance(m, TurnTextDelta)]
    assert "".join(d.text for d in deltas) == "hello"
    assert isinstance(seen[-1], TurnTextEnd) or any(isinstance(m, TurnTextEnd) for m in seen)
    assert result.messages[-1].content == "hello"
    assert result.interrupts == []


async def test_turn_runner_adds_user_action_to_checkpoint_and_trace(tmp_path: Path) -> None:
    class UpdatingAgent:
        def __init__(self) -> None:
            self.updates: list[tuple[dict, dict]] = []

        async def aupdate_state(self, config: dict, update: dict) -> None:
            self.updates.append((config, update))

    agent = UpdatingAgent()
    log = TraceLog(tmp_path / "trace.sqlite")
    try:
        runner = TurnRunner(agent, thread_id="thread-1", trace=log)
        await runner.add_user_action("I replayed the plot.")

        config, update = agent.updates[0]
        assert config == {"configurable": {"thread_id": "thread-1"}}
        assert len(update["messages"]) == 1
        assert isinstance(update["messages"][0], HumanMessage)
        assert update["messages"][0].content == "I replayed the plot."
        [event] = list(log.iter_events())
        assert event.kind == "message_user"
        assert event.payload == {"content": "I replayed the plot."}
    finally:
        log.close()


async def test_turn_runner_streams_message_chunks() -> None:
    runner = TurnRunner(_chunk_message_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    deltas = [m for m in seen if isinstance(m, TurnTextDelta)]
    assert "".join(d.text for d in deltas) == "hello"
    assert result.messages[-1].content == "hello"
    assert result.interrupts == []


async def test_turn_runner_streams_v3_reasoning_and_tool_events() -> None:
    runner = TurnRunner(_v3_reasoning_and_tool_agent(), thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    reasoning = [m for m in seen if isinstance(m, TurnReasoningDelta)]
    assert any(r.text == "checking context" for r in reasoning)

    text_deltas = [m for m in seen if isinstance(m, TurnTextDelta) and m.text]
    assert "".join(d.text for d in text_deltas) == "hello"

    tool_events = [m for m in seen if isinstance(m, TurnToolEvent)]
    events_by_kind = {e.event for e in tool_events}
    assert "requested" in events_by_kind
    assert "started" in events_by_kind
    assert "finished" in events_by_kind
    finished = next(e for e in tool_events if e.event == "finished")
    assert finished.content == '["a", "b"]'

    assert result.messages[-1].content == "hello"


async def test_tool_result_does_not_leak_into_assistant_text() -> None:
    """A tool result reaches the UI only as a tool event, never as assistant
    prose, so file reads and skill and memory text don't get dumped."""

    tool_output = "---\nname: battmo-overview\ndescription: workflow\n---\nLOTS OF SKILL TEXT"
    agent = ScriptedV3Agent(
        tool_call_events(
            tool_name="read_file",
            tool_call_id="call_read_1",
            args={"file_path": "/skills/shared/battmo-overview/SKILL.md"},
            output=tool_output,
            final_text="Read the overview skill; ready to proceed.",
        )
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    await runner.run_prompt("read the skill", on_message=lambda msg: seen.append(msg))

    streamed_text = "".join(m.text for m in seen if isinstance(m, TurnTextDelta) and m.text)
    assert streamed_text == "Read the overview skill; ready to proceed."
    assert tool_output not in streamed_text
    assert "SKILL TEXT" not in streamed_text

    # The result still reaches the UI as a tool event (the tool card), not prose.
    finished = [m for m in seen if isinstance(m, TurnToolEvent) and m.event == "finished"]
    assert any(tool_output in (e.content or "") for e in finished)


async def test_a_message_another_node_writes_is_not_prose() -> None:
    # ``run.messages`` also carries a whole message that a middleware node writes.
    note = AIMessage(content="MIDDLEWARE NOTE")
    agent = ScriptedV3Agent(
        [
            v3_message_event(note, node="SomeMiddleware.before_model"),
            v3_message_event(AIMessage(content="hello")),
            v3_values_event([note]),
        ]
    )
    seen: list[object] = []

    await TurnRunner(agent, thread_id="thread-1").run_prompt("hi", on_message=seen.append)

    assert [m.text for m in seen if isinstance(m, TurnTextDelta)] == ["hello"]


async def test_turn_runner_parses_server_tool_call() -> None:
    agent = ScriptedV3Agent(
        [
            v3_message_event(
                {
                    "event": "content-block-finish",
                    "content": {
                        "type": "server_tool_call",
                        "id": "tool-2",
                        "name": "search",
                        "args": "query=jutul",
                    },
                }
            ),
            v3_values_event([]),
        ]
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    requested = [m for m in seen if isinstance(m, TurnToolEvent) and m.event == "requested"]
    assert len(requested) == 1
    assert requested[0].tool_name == "search"
    assert requested[0].args == {"value": "query=jutul"}


async def test_turn_runner_streams_tool_output_deltas() -> None:
    human = HumanMessage(content="hi")
    assistant = AIMessage(content="done")
    agent = ScriptedV3Agent(
        [
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "tool-julia",
                    "tool_name": "run_julia",
                    "input": {"code": "run()"},
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-output-delta",
                    "tool_call_id": "tool-julia",
                    "delta": "progress\n",
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-output-delta",
                    "tool_call_id": "tool-julia",
                    "delta": "→ 42\n",
                }
            ),
            v3_tool_event(
                {
                    "event": "tool-finished",
                    "tool_call_id": "tool-julia",
                    "output": "progress\n→ 42\n",
                }
            ),
            v3_values_event([human, assistant]),
        ]
    )
    runner = TurnRunner(agent, thread_id="thread-1")
    seen: list[object] = []

    result = await runner.run_prompt("hi", on_message=lambda msg: seen.append(msg))

    tool_events = [m for m in seen if isinstance(m, TurnToolEvent)]
    deltas = [e for e in tool_events if e.event == "delta"]
    assert [d.content for d in deltas] == ["progress\n", "→ 42\n"]
    finished = next(e for e in tool_events if e.event == "finished")
    assert finished.content == "progress\n→ 42\n"
    assert result.messages[-1].content == "done"


async def test_turn_runner_records_user_message_and_hitl_round_trip(
    tmp_path: Path,
) -> None:
    agent = interrupt_agent()
    log = TraceLog(tmp_path / "trace.sqlite")
    runner = TurnRunner(agent, thread_id="thread-1", trace=log)

    try:
        first = await runner.run_prompt("need approval")
        assert len(first.interrupts) == 1
        assert first.interrupts[0].interrupt_id == "interrupt-1"

        second = await runner.resume(
            {"interrupt-1": {"decisions": [{"type": "approve"}]}},
        )

        assert hasattr(agent.inputs[1], "resume")
        assert second.messages[-1].content == "approval handled"

        kinds = [event.kind for event in log.iter_events()]
        assert kinds == ["message_user", "hitl_request", "hitl_response"]
    finally:
        log.close()


async def test_a_tool_paused_by_an_interrupt_is_not_reported_finished() -> None:
    # langgraph closes a call still open when the run pauses with no output and
    # no error; it is neither finished nor failed until the run resumes.
    human = HumanMessage(content="hi")
    agent = ScriptedV3Agent(
        [
            v3_tool_event(
                {
                    "event": "tool-started",
                    "tool_call_id": "tool-task",
                    "tool_name": "task",
                    "input": {"description": "ask first"},
                }
            ),
            v3_values_event([human]),
        ]
    )
    seen: list[object] = []

    await TurnRunner(agent, thread_id="thread-1").run_prompt(
        "hi", on_message=lambda msg: seen.append(msg)
    )

    events = [(m.event, m.tool_call_id) for m in seen if isinstance(m, TurnToolEvent)]
    assert events == [("started", "tool-task")]


async def test_a_call_is_announced_when_the_model_starts_writing_it() -> None:
    """A call with long arguments takes a while to arrive whole, so a front end
    hears of it at its first chunk and gets the arguments once it is complete."""
    events = tool_call_events(
        tool_name="run_julia", tool_call_id="c1", args={"code": "1"}, output="1", final_text="ok"
    )
    seen: list[TurnToolEvent] = []
    await TurnRunner(ScriptedV3Agent(events), thread_id="t").run_prompt(
        "go", on_message=lambda m: seen.append(m) if isinstance(m, TurnToolEvent) else None
    )
    said = [e for e in seen if e.event in ("composing", "requested")]
    assert [(e.event, e.args) for e in said] == [("composing", None), ("requested", {"code": "1"})]


async def test_a_tool_that_fails_without_raising_reaches_the_stream_as_an_error(
    tmp_path: Path,
) -> None:
    """The stream hands over a result's text without its status, so the runner
    reads the status from the trace.

    That has to hold for a Julia error that follows printed output, for a refused
    plot argument, for a file that is not there, and for calls made together,
    whose results land before the loop reaches them.
    """
    from makker.agent.builder import build_agent
    from makker.julia.session import EvalResult
    from makker.lab import fakes
    from makker.session import Session

    def julia(code: str) -> EvalResult:
        if "boom" in code:
            return EvalResult(output="printed first", error="UndefVarError: `boom` not defined")
        return EvalResult(output="2")

    def calls(*specs: tuple[str, str, dict]) -> AIMessage:
        return AIMessage(
            content="", tool_calls=[{"id": i, "name": n, "args": a} for i, n, a in specs]
        )

    script = [
        calls(("ok", "run_julia", {"code": "1 + 1", "description": "Add two numbers"})),
        calls(
            ("gone", "read_file", {"file_path": "/nonexistent/x.jl"}),
            ("boom", "run_julia", {"code": "boom"}),
            ("fine", "run_julia", {"code": "2"}),
        ),
        calls(("plot", "plot_julia", {"code": "lines(1:3)", "size": [1]})),
        fakes.scripted_final("done"),
    ]
    session = Session.create(
        julia=fakes.FakeJulia(eval_handler=julia),
        state_root=tmp_path,
        simulator=fakes.make_fake_adapter(tmp_path),
    )
    seen: list[TurnToolEvent] = []
    try:
        agent, _ = build_agent(session, model=fakes.make_scripted_model(script))
        runner = TurnRunner(agent, thread_id="t", trace=session.trace)
        await runner.run_prompt(
            "go", on_message=lambda m: seen.append(m) if isinstance(m, TurnToolEvent) else None
        )
    finally:
        session.finalize()

    ended = {e.tool_call_id: e for e in seen if e.event in ("finished", "error")}
    assert {cid: e.event for cid, e in ended.items()} == {
        "ok": "finished",
        "gone": "error",
        "boom": "error",
        "fine": "finished",
        "plot": "error",
    }
    # The model still reads the reply as the tool wrote it.
    assert ended["boom"].content == "printed first\nERROR: UndefVarError: `boom` not defined"
    assert ended["plot"].content.startswith("ERROR: size must be")


async def test_a_result_written_late_still_marks_its_call_failed(tmp_path: Path) -> None:
    """Middleware can write a result a little after the stream reports the call's
    end: a call the trace has seen waits for its result, briefly. A call it has
    not seen (nothing records the run) has no result coming, and does not wait."""
    import asyncio
    import time

    from makker.agent.turns import RESULT_WAIT, _ResultStatus
    from makker.trace.schema import TOOL_CALL, TOOL_RESULT

    log = TraceLog(tmp_path / "trace.sqlite")
    try:
        results = _ResultStatus(log)
        for cid in ("early", "late", "lost"):
            log.append(TOOL_CALL, {"id": cid, "name": "run_julia", "args": {}})
        log.append(TOOL_RESULT, {"tool_call_id": "early", "status": "success"})

        async def write_later() -> None:
            await asyncio.sleep(0.05)
            log.append(TOOL_RESULT, {"tool_call_id": "late", "status": "error"})

        writer = asyncio.create_task(write_later())
        assert await results.failed("late") is True
        await writer
        assert await results.failed("early") is False  # read once, kept until asked
        assert await results.failed("lost") is False  # waited for, never came

        began = time.monotonic()
        assert await results.failed("unrecorded") is False
        assert time.monotonic() - began < RESULT_WAIT / 2
    finally:
        log.close()
