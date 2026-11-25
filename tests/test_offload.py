"""Older tool results move to files as the context fills, instead of being cleared."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from makker.agent.offload import ToolResultOffloadMiddleware, context_used
from makker.context_budget import ContextBudget
from makker.trace import TraceLog

# Compaction at 17k tokens, so offloading starts at 70% of that (11.9k) and stops
# at 45% (7.65k); a window under 100k keeps the newest 3 tool results.
_BUDGET = ContextBudget(window=20_000, source="provider")


@dataclass
class _Written:
    path: str | None = None
    error: str | None = None


class _Backend:
    artifacts_root = "/state"

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def awrite(self, path: str, content: str) -> _Written:
        if path in self.files:
            return _Written(error="File exists")
        self.files[path] = content
        return _Written(path=path)


def _conversation(results: int, size: int = 6_000) -> list:
    messages: list = [HumanMessage(content="go", id="h")]
    for i in range(results):
        messages.append(
            AIMessage(
                content="",
                id=f"a{i}",
                tool_calls=[{"id": f"c{i}", "name": "run_julia", "args": {}}],
            )
        )
        body = "\n".join(f"line {n} of result {i} " + "x" * 40 for n in range(size // 60))
        messages.append(
            ToolMessage(content=body, tool_call_id=f"c{i}", name="run_julia", id=f"t{i}")
        )
    return messages


async def test_moves_the_oldest_results_to_files_and_keeps_the_newest(tmp_path: Path) -> None:
    backend = _Backend()
    with TraceLog(tmp_path / "trace.sqlite") as trace:
        middleware = ToolResultOffloadMiddleware(_BUDGET, backend, trace)
        messages = _conversation(10)
        assert context_used(messages) > _BUDGET.offload_at

        update = await middleware.abefore_model({"messages": messages}, runtime=None)

        moved = update["messages"]
        # Oldest first, the same messages (by id) so the conversation replaces them,
        # and never one of the newest three.
        assert [m.id for m in moved] == [f"t{i}" for i in range(len(moved))]
        assert len(moved) <= 7
        after = {m.id: m for m in messages} | {m.id: m for m in moved}
        assert context_used([after[m.id] for m in messages]) <= _BUDGET.offload_to
        # Each leaves its path and a preview; the file has the full text.
        first = moved[0]
        assert first.content.startswith("[Moved out of the conversation to save context.")
        assert "/state/large_tool_results/c0" in first.content
        assert "line 0 of result 0" in first.content
        assert backend.files["/state/large_tool_results/c0"] == messages[2].content
        (event,) = [e.payload for e in trace.iter_events() if e.kind == "context_offload"]
        assert event["count"] == len(moved) and event["freed_tokens"] > 0

        # Next call: the moved results are not moved again, and nothing else needs to be.
        assert await middleware.abefore_model({"messages": list(after.values())}, None) is None


async def test_leaves_the_conversation_alone_below_the_mark(tmp_path: Path) -> None:
    middleware = ToolResultOffloadMiddleware(_BUDGET, _Backend(), None)
    assert await middleware.abefore_model({"messages": _conversation(3)}, runtime=None) is None


async def test_skips_small_results_and_ones_referred_to_by_value() -> None:
    middleware = ToolResultOffloadMiddleware(_BUDGET, _Backend(), None)
    small = ToolMessage(content="ok", tool_call_id="s", name="run_julia", id="s")
    attempt = ToolMessage(content="x" * 9_000, tool_call_id="r", name="record_attempt", id="r")
    assert await middleware._move(small) is None
    assert await middleware._move(attempt) is None


def test_context_used_counts_from_the_last_measured_call() -> None:
    measured = AIMessage(
        content="",
        usage_metadata={"input_tokens": 9_000, "output_tokens": 100, "total_tokens": 9_100},
    )
    later = ToolMessage(content="y" * 400, tool_call_id="c", id="t")
    assert context_used([HumanMessage(content="hi"), measured, later]) == 9_100 + context_used(
        [later]
    )


async def test_an_agent_offloads_mid_turn(tmp_path: Path) -> None:
    """End to end: large tool results in a small window move to files during the turn."""
    from langgraph.checkpoint.memory import InMemorySaver

    from makker.agent.builder import build_agent
    from makker.agent.turns import TurnRunner
    from makker.lab.fakes import (
        FakeJulia,
        make_fake_adapter,
        make_scripted_model,
        scripted_final,
        scripted_tool_call,
    )
    from makker.session import Session

    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"row {n}: " + "v" * 60 for n in range(300)))
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    reads = [
        scripted_tool_call(tool_name="read_file", args={"file_path": str(big)}) for _ in range(14)
    ]
    model = make_scripted_model([*reads, scripted_final("done")])
    model.profile = {"max_input_tokens": 30_000}
    agent, _ = build_agent(session, model=model, checkpointer=InMemorySaver())
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    await runner.run_prompt("read it")

    assert session.trace.events_of("context_offload")
    state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
    results = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert results[0].content.startswith("[Moved out of the conversation")
    assert not results[-1].content.startswith("[Moved")  # the newest stay
    moved = sorted((session.state_dir / "large_tool_results").iterdir())
    assert moved and "row 99:" in moved[0].read_text()  # the full result read_file gave
    session.finalize()


async def test_the_budget_sized_summarizer_replaces_the_default(
    tmp_path: Path, monkeypatch
) -> None:
    """One summarizer runs per model call: ours replaced deepagents' default by name."""
    from deepagents.middleware.summarization import SummarizationMiddleware

    from makker.agent.builder import build_agent
    from makker.agent.turns import TurnRunner
    from makker.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
    from makker.session import Session

    calls: list[object] = []
    original = SummarizationMiddleware.awrap_model_call

    async def counting(self, request, handler):
        calls.append(self)
        return await original(self, request, handler)

    monkeypatch.setattr(SummarizationMiddleware, "awrap_model_call", counting)
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    agent, _ = build_agent(session, model=make_scripted_model([scripted_final("hi")]))
    await TurnRunner(agent, thread_id=session.session_id).run_prompt("hello")

    (summarizer,) = calls
    trigger = summarizer._lc_helper.trigger
    assert trigger == ("tokens", session.context_budget.compact_at)
    session.finalize()
