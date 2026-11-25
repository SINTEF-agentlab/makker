"""Tests for TraceRecorder middleware."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt

from makker.trace import TraceLog, TraceRecorder


def _request(tool_call_id: str = "call-1", name: str = "echo") -> object:
    return type(
        "Req",
        (),
        {"tool_call": {"id": tool_call_id, "name": name, "args": {"value": "x"}}},
    )()


async def test_trace_recorder_logs_reasoning_and_assistant(tmp_path) -> None:
    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": "checking the workspace"},
                    {"type": "text", "text": "ready to help"},
                ]
            ),
        ]
    }
    await recorder.aafter_model(state, runtime=None)

    kinds = [event.kind for event in log.iter_events()]
    assert kinds == ["message_reasoning", "message_assistant"]
    log.close()


async def test_trace_recorder_records_compaction_once(tmp_path) -> None:
    """The stock summarizer's _summarization_event surfaces as one trace event."""
    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    summary = HumanMessage(content="a summary", id="sum-1")
    event = {"cutoff_index": 4, "summary_message": summary, "file_path": "/conv/t.md"}
    # The leading four messages were folded into the summary, the model saw the
    # summary plus the fifth, and the sixth is the reply it just produced.
    state = {"messages": [AIMessage(content=str(n)) for n in range(6)]}
    state["_summarization_event"] = event

    await recorder.aafter_model(state, runtime=None)
    await recorder.aafter_model(state, runtime=None)  # same event: must not re-record
    # A recorder built later for the same session (resume, model switch) reads
    # the trace, so it does not record the compaction still in the state again.
    await TraceRecorder(log).aafter_model(state, runtime=None)

    compactions = [e for e in log.iter_events() if e.kind == "context_compaction"]
    assert len(compactions) == 1
    assert compactions[0].payload == {
        "summarized": 4,
        "kept": 1,
        "offloaded": True,
        "manual": False,
        "cutoff": 4,
    }

    # The next compaction extends the cutoff; it counts only the newly folded messages.
    state["messages"] += [AIMessage(content=str(n)) for n in range(6, 12)]
    state["_summarization_event"] = {**event, "cutoff_index": 9}
    await recorder.aafter_model(state, runtime=None)
    compactions = [e for e in log.iter_events() if e.kind == "context_compaction"]
    assert [(e.payload["summarized"], e.payload["kept"]) for e in compactions] == [(4, 1), (5, 2)]
    log.close()


async def test_trace_recorder_skips_a_compaction_already_traced_by_compact(tmp_path) -> None:
    """A manual /compact writes its own event; the next turn must not add an automatic one."""
    log = TraceLog(tmp_path / "trace.sqlite")
    log.append(
        "context_compaction",
        {"summarized": 4, "kept": 8, "offloaded": True, "manual": True, "cutoff": 4},
    )
    state = {
        "messages": [AIMessage(content=str(n)) for n in range(13)],
        "_summarization_event": {
            "cutoff_index": 4,
            "summary_message": HumanMessage(content="a summary"),
            "file_path": "/conv/t.md",
        },
    }
    await TraceRecorder(log).aafter_model(state, runtime=None)
    compactions = [e for e in log.iter_events() if e.kind == "context_compaction"]
    assert [e.payload["manual"] for e in compactions] == [True]
    log.close()


async def test_trace_recorder_logs_tool_round_trip(tmp_path) -> None:
    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    request = _request()

    async def handler(_request):
        return ToolMessage(content="x", tool_call_id="call-1", name="echo")

    result = await recorder.awrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)

    kinds = [event.kind for event in log.iter_events()]
    assert kinds == ["tool_call", "tool_result"]
    log.close()


async def test_raising_tool_is_converted_to_error_message(tmp_path) -> None:
    """A tool that raises must not abort the turn; the model gets the error."""

    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)
    request = _request(name="read_file")

    async def handler(_request):
        raise ValueError("Path: /etc/passwd outside root directory: /ws")

    result = await recorder.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-1"
    assert "read_file" in result.content
    assert "outside root directory" in result.content

    events = list(log.iter_events())
    assert [e.kind for e in events] == ["tool_call", "tool_result"]
    assert events[1].payload["status"] == "error"
    log.close()


async def test_control_flow_exceptions_propagate(tmp_path) -> None:
    """Interrupts (approval, etc.) must bubble up, not be swallowed as errors."""

    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)
    request = _request()

    async def handler(_request):
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        await recorder.awrap_tool_call(request, handler)
    log.close()


async def test_trace_recorder_logs_model_usage(tmp_path) -> None:
    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    state = {
        "messages": [
            AIMessage(
                content="done",
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 15,
                    "total_tokens": 135,
                },
            ),
        ]
    }
    await recorder.aafter_model(state, runtime=None)

    events = {event.kind: event.payload for event in log.iter_events()}
    log.close()
    assert events["model_usage"]["input_tokens"] == 120
    assert events["model_usage"]["output_tokens"] == 15


async def test_trace_recorder_normalizes_openai_reasoning_summary(tmp_path) -> None:
    """OpenAI keeps reasoning summaries under a raw ``summary`` key; the
    recorder reads the normalized content blocks so the trace still gets a
    ``message_reasoning`` event."""
    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    state = {
        "messages": [
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "thinking about primes"}],
                        "content": [],
                    },
                    {"type": "text", "text": "There are 10."},
                ],
                response_metadata={"model_provider": "openai"},
            )
        ]
    }
    await recorder.aafter_model(state, runtime=None)

    events = {event.kind: event.payload for event in log.iter_events()}
    assert "thinking about primes" in events["message_reasoning"]["content"]
    assert events["message_assistant"]["content"] == "There are 10."
    log.close()


async def test_command_tool_result_is_recorded(tmp_path) -> None:
    """A tool that returns a Command (the task subagent, write_todos, read_file on
    media) records the ToolMessage it carries, so transcripts show what came back."""

    from langgraph.types import Command

    from makker.transcript import render_html, render_markdown

    log = TraceLog(tmp_path / "trace.sqlite")
    recorder = TraceRecorder(log)

    async def handler(_request):
        # Shaped like deepagents' task tool: the message carries no tool name.
        return Command(
            update={
                "todos": [],
                "messages": [
                    ToolMessage("the subagent found 3 wells", tool_call_id="other"),
                    ToolMessage("the subagent found 7 wells", tool_call_id="call-1"),
                    HumanMessage("attached media"),
                ],
            }
        )

    result = await recorder.awrap_tool_call(_request(name="task"), handler)
    assert isinstance(result, Command)

    events = list(log.iter_events())
    assert [e.kind for e in events] == ["tool_call", "tool_result"]
    assert events[1].payload == {
        "tool_call_id": "call-1",
        "name": "task",
        "content": "the subagent found 7 wells",
        "status": "success",
    }
    assert "### Tool result · `task`" in render_markdown(events)
    assert "the subagent found 7 wells" in render_markdown(events)
    assert "the subagent found 7 wells" in render_html(events)
    log.close()
