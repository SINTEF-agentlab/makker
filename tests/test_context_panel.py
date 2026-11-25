"""The context view: what a request holds, and the meter and report read from the trace."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.notification import add_notification_metadata
from langchain_core.tools import tool

from makker.agent.context_measure import PARTS, measure_request, scale_to_usage
from makker.context_panel import (
    context_meter,
    context_report,
    format_tokens,
    render_context_markdown,
    status_label,
    usage_alert,
)
from makker.trace import TraceLog
from makker.trace.schema import (
    CONTEXT_COMPACTION,
    CONTEXT_OFFLOAD,
    MODEL,
    MODEL_USAGE,
    compaction_payload,
    model_payload,
    offload_payload,
)


@tool
def run_julia(code: str) -> str:
    """Run Julia code in the persistent REPL and return what it printed."""
    return code


def _request(system: str, messages: list) -> SimpleNamespace:
    return SimpleNamespace(
        system_message=SystemMessage(content=system), tools=[run_julia], messages=messages
    )


def test_format_tokens_scales() -> None:
    assert format_tokens(950) == "950"
    assert format_tokens(1_400) == "1.4k"
    assert format_tokens(29_400) == "29k"
    assert format_tokens(1_200_000) == "1.2M"


def test_a_request_is_measured_part_by_part() -> None:
    system = "You are a careful agent. " * 40
    memory = (
        "<agent_memory>\n- note: units are SI\n</agent_memory>\n"
        "<memory_guidelines>use it</memory_guidelines>"
    )
    summary = add_notification_metadata(
        HumanMessage(content="Earlier: the user asked for a sweep. " * 20), source="summarization"
    )
    messages = [
        summary,
        HumanMessage(content="Now plot it"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "read_file", "args": {"file_path": "src/model.jl"}}],
        ),
        ToolMessage(content="x" * 4_000, tool_call_id="c1", name="read_file"),
        AIMessage(
            content="", tool_calls=[{"id": "c2", "name": "run_julia", "args": {"code": "1"}}]
        ),
        ToolMessage(content="2", tool_call_id="c2", name="run_julia"),
    ]
    request = _request(system + memory, messages)
    finished = _request(system + memory + "\n\n## Skills\n- plotting: how to plot" * 5, messages)

    measured = measure_request(request, finished=finished)

    parts = measured["parts"]
    assert set(parts) == set(PARTS)
    assert all(parts[name] > 0 for name in PARTS)
    assert parts["tool_results"] > parts["user"]
    # The largest results first, named by the tool and what it was asked about.
    assert measured["largest"][0]["label"] == "Read · src/model.jl"
    assert [r["tokens"] for r in measured["largest"]] == sorted(
        (r["tokens"] for r in measured["largest"]), reverse=True
    )


def test_the_parts_are_scaled_to_the_measured_total() -> None:
    measured = {
        "parts": {"system": 300, "tools": 100, "user": 7},
        "largest": [{"label": "x", "tokens": 50}],
    }
    scaled = scale_to_usage(measured, {"input_tokens": 1_000, "output_tokens": 40})
    # The parts add up to the call's input, plus its reply on the agent's side.
    assert sum(scaled["parts"].values()) == 1_040
    assert scaled["parts"]["assistant"] == 40
    assert scaled["largest"] == [{"label": "x", "tokens": round(50 * 1_000 / 407)}]


def _model(trace: TraceLog, name: str = "openai:gpt-6", window: int = 200_000) -> None:
    trace.append(
        MODEL,
        model_payload(
            model=name,
            window=window,
            window_source="provider",
            compact_at=int(window * 0.85),
            offload_at=round(int(window * 0.85) * 0.7),
            settings={},
        ),
    )


def _usage(trace: TraceLog, input_tokens: int, output_tokens: int = 1_000, **extra) -> None:
    trace.append(
        MODEL_USAGE,
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            **extra,
        },
    )


def test_the_meter_is_the_last_call_and_what_changed_since(tmp_path: Path) -> None:
    with TraceLog(tmp_path / "trace.sqlite") as trace:
        assert context_meter(trace) is None
        _model(trace)
        meter = context_meter(trace)
        assert meter["used"] is None and meter["window"] == 200_000
        assert meter["compact_at"] == 170_000 and meter["offload_at"] == 119_000
        assert status_label(meter) is None

        _usage(trace, 40_000, 2_000)
        assert context_meter(trace)["used"] == 42_000
        assert status_label(context_meter(trace)) == "ctx 21%"

        # A resume rebuilds the same model: still measured.
        _model(trace)
        assert context_meter(trace)["estimated"] is False

        # /compact made it smaller; no call has measured how much yet.
        trace.append(
            CONTEXT_COMPACTION,
            compaction_payload(
                summarized=30, kept=6, offloaded=True, manual=True, freed_tokens=20_000
            ),
        )
        meter = context_meter(trace)
        assert meter["used"] == 22_000 and meter["estimated"] is True
        assert status_label(meter) == "ctx ~11%"

        # The next call measures it again.
        _usage(trace, 25_000, 500)
        assert context_meter(trace)["estimated"] is False

        # Another model counts the same text differently.
        _model(trace, "anthropic:claude-opus-5-5", window=400_000)
        meter = context_meter(trace)
        assert meter["estimated"] is True and meter["window"] == 400_000


def test_alerts_by_share_of_the_window() -> None:
    meter = {"window": 100_000, "used": 50_000}
    assert usage_alert(meter) == "ok"
    assert usage_alert({**meter, "used": 75_000}) == "warn"
    assert usage_alert({**meter, "used": 95_000}) == "high"
    assert usage_alert(None) == "ok"


def test_the_report_has_the_split_and_what_moved_out(tmp_path: Path) -> None:
    with TraceLog(tmp_path / "trace.sqlite") as trace:
        _model(trace)
        trace.append(CONTEXT_OFFLOAD, offload_payload(count=4, freed_tokens=30_000))
        trace.append(
            CONTEXT_COMPACTION,
            compaction_payload(summarized=40, kept=6, offloaded=True, freed_tokens=50_000),
        )
        _usage(
            trace,
            60_000,
            2_000,
            input_token_details={"cache_read": 50_000},
            output_token_details={"reasoning": 800},
            context={
                "parts": {"system": 4_000, "tools": 6_000, "tool_results": 30_000, "user": 22_000},
                "largest": [{"label": "Read · model.jl", "tokens": 9_000}],
            },
        )
        report = context_report(trace)

    assert [p["key"] for p in report["breakdown"]] == ["system", "tools", "user", "tool_results"]
    assert report["breakdown"][0]["label"] == "System prompt"
    assert report["largest"] == [{"label": "Read · model.jl", "tokens": 9_000}]
    assert report["moved"] == {"results": 4, "compactions": 1}

    text = render_context_markdown(report)
    assert "**62k of 200k tokens** (31%)" in text
    assert "`openai:gpt-6`" in text and "assumed" not in text
    assert "| System prompt | 4.0k | 2% |" in text
    assert "| Free space | 108k | 54% |" in text
    assert "| Compaction buffer | 30k | 15% |" in text
    assert "Moved out: 4 tool results to files, 1 compaction." in text


def test_a_call_recorded_without_parts_shows_one_total(tmp_path: Path) -> None:
    with TraceLog(tmp_path / "trace.sqlite") as trace:
        _model(trace)
        _usage(trace, 10_000, 500)
        report = context_report(trace)
    assert report["breakdown"] == [{"key": "all", "label": "In context", "tokens": 10_500}]


def test_the_report_before_any_call_says_so_and_flags_an_assumed_window(tmp_path: Path) -> None:
    with TraceLog(tmp_path / "trace.sqlite") as trace:
        trace.append(
            MODEL,
            model_payload(
                model="ollama:new-model",
                window=200_000,
                window_source="assumed",
                compact_at=170_000,
                offload_at=119_000,
                settings={},
            ),
        )
        text = render_context_markdown(context_report(trace))
    lines = text.splitlines()
    assert lines[:2] == ["No model calls yet.", "`ollama:new-model` · window size assumed"]


async def test_an_agent_turn_measures_each_call_and_reports_it_as_it_goes(tmp_path: Path) -> None:
    """End to end: each call's usage carries the request's parts (the skills index
    included, though it joins the prompt after the recorder), and the runner
    reports each call before the next tool runs."""
    from makker.agent.builder import build_agent
    from makker.agent.turns import TurnRunner, TurnToolEvent, TurnUsage
    from makker.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model
    from makker.session import Session

    data = tmp_path / "data.txt"
    data.write_text("\n".join(f"row {n}" for n in range(60)))
    usage = {"input_tokens": 12_000, "output_tokens": 40, "total_tokens": 12_040}
    model = make_scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "read_file", "args": {"file_path": str(data)}}],
                usage_metadata=usage,
            ),
            AIMessage(content="done", usage_metadata={**usage, "input_tokens": 12_500}),
        ]
    )
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    agent, _ = build_agent(session, model=model)
    seen: list[str] = []

    def on_message(event: object) -> None:
        if isinstance(event, TurnUsage):
            seen.append("usage")
        elif isinstance(event, TurnToolEvent) and event.event == "started":
            seen.append("tool")

    await TurnRunner(agent, thread_id=session.session_id, trace=session.trace).run_prompt(
        "read it", on_message=on_message
    )

    assert seen == ["usage", "tool", "usage"]
    first, second = [e.payload["context"] for e in session.trace.events_of(MODEL_USAGE)]
    assert all(first["parts"][name] > 0 for name in ("system", "tools", "skills", "memory"))
    assert sum(second["parts"].values()) == 12_540
    assert second["largest"][0]["label"].startswith("Read · ")
    report = context_report(session.trace)
    assert report["meter"]["used"] == 12_540
    session.finalize()
