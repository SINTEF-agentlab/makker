"""Canaries over the private upstream surface Makker deliberately touches.

A failure here means an upstream bump moved something we compose with. Some
contacts degrade gracefully at runtime; others can break request construction.
These tests make the change fail loudly at test time, with a map of what to fix.
"""

from __future__ import annotations

import pytest


def test_summarization_engine_private_methods_still_exist() -> None:
    from deepagents.middleware.summarization import SummarizationMiddleware

    from makker.agent.summarization import _ENGINE_METHODS, manual_compaction_available

    missing = [name for name in _ENGINE_METHODS if not hasattr(SummarizationMiddleware, name)]
    assert manual_compaction_available(), (
        f"deepagents no longer exposes {missing}; manual /compact degrades to "
        "'unavailable' at runtime. Recompose agent/summarization.py against the "
        "new engine."
    )


def test_langgraph_tool_call_writer_still_exposed() -> None:
    from makker.agent import tools

    assert tools._tool_call_writer is not None, (
        "langgraph moved pregel._tools._tool_call_writer; live run_julia output "
        "streaming silently disappears. Update the import guard in agent/tools.py."
    )


def test_deepagents_validate_path_still_patchable() -> None:
    import deepagents.backends.utils as backend_utils
    from deepagents.middleware import _skill_tools
    from deepagents.middleware import filesystem as fs_middleware

    for module in (backend_utils, fs_middleware, _skill_tools):
        assert callable(getattr(module, "validate_path", None)), (
            f"{module.__name__} no longer binds validate_path; the real-path shim in "
            "agent/real_paths.py misses it, and relative paths may land at '/'. "
            "Re-point the patch."
        )


def test_deepagents_harness_profile_resolver_still_exists() -> None:
    from deepagents.profiles.harness import harness_profiles

    assert hasattr(harness_profiles, "_harness_profile_for_model"), (
        "deepagents moved _harness_profile_for_model; tests/test_builder.py "
        "asserts profile resolution through it."
    )


def test_openrouter_message_serialization_hook_still_exists() -> None:
    from langchain_openrouter import ChatOpenRouter

    hook = getattr(ChatOpenRouter, "_create_message_dicts", None)
    assert callable(hook), "Update agent/openrouter.py against the new serialization path."


@pytest.mark.filterwarnings("ignore:The v3 streaming protocol")
async def test_v3_projections_have_the_shape_turns_reads_and_the_fakes_copy(tmp_path) -> None:
    import asyncio

    from langchain_core.messages import HumanMessage, ToolMessage

    from makker.agent.builder import build_agent
    from makker.lab import fakes
    from makker.session import Session

    async def observe(agent) -> tuple[list, list, list]:
        """What each projection yields, read the way agent/turns.py reads it."""

        run = await agent.astream_events(
            {"messages": [{"role": "user", "content": "go"}]},
            config={"configurable": {"thread_id": "t"}},
            version="v3",
        )
        messages, tools = [], []

        async def read_messages() -> None:
            async for stream in run.messages:
                text = [delta async for delta in stream.text]
                _ = [delta async for delta in stream.reasoning]
                _ = [chunk async for chunk in stream.tool_calls]
                calls = [(c["name"], c["id"], c["args"]) for c in await stream.tool_calls]
                messages.append((stream.node, "".join(text), calls))

        async def read_tool_calls() -> None:
            async for call in run.tool_calls:
                at_start = (
                    call.tool_name,
                    call.tool_call_id,
                    call.input,
                    call.completed,
                    call.output,
                )
                deltas = [delta async for delta in call.output_deltas]
                tools.append((*at_start, deltas, call.completed, call.output, call.error))

        await asyncio.gather(read_messages(), read_tool_calls())
        assert await run.interrupts() == []
        final = [type(m).__name__ for m in (await run.output())["messages"]]
        return messages, tools, final

    args = {"code": "2 + 2"}
    ask = fakes.scripted_tool_call(tool_name="run_julia", args=args, tool_call_id="j1")
    done = fakes.scripted_final("Done: 4")
    session = Session.create(
        julia=fakes.FakeJulia(answers={"2 + 2": "4"}, stream_chunks=["progress\n"]),
        state_root=tmp_path,
        simulator=fakes.make_fake_adapter(tmp_path),
    )
    try:
        real, _ = build_agent(session, model=fakes.make_scripted_model([ask, done]))
        seen_real = await observe(real)
    finally:
        session.finalize()

    human, result = HumanMessage("go"), ToolMessage("4", tool_call_id="j1", name="run_julia")
    started = {"tool_call_id": "j1", "tool_name": "run_julia", "input": args}
    fake = fakes.ScriptedV3Agent(
        [
            fakes.v3_message_event(human),
            fakes.v3_message_event(ask),
            fakes.v3_tool_event({"event": "tool-started", **started}),
            fakes.v3_tool_event(
                {"event": "tool-output-delta", "tool_call_id": "j1", "delta": "progress\n"}
            ),
            fakes.v3_tool_event({"event": "tool-finished", "tool_call_id": "j1", "output": "4"}),
            fakes.v3_message_event(result),
            fakes.v3_message_event(done),
            fakes.v3_values_event([human, ask, result, done]),
        ]
    )

    # One stream per model call, from the model node; the tool result is not among
    # them. A tool call is yielded when it starts and has its output once its deltas end.
    expected = (
        [("model", "", [("run_julia", "j1", args)]), ("model", "Done: 4", [])],
        [("run_julia", "j1", args, False, None, ["progress\n"], True, "4", None)],
        ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"],
    )
    assert seen_real == expected, (
        "langgraph's v3 projections changed shape. Update agent/turns.py, then "
        "make lab/fakes.py copy the new shape."
    )
    assert await observe(fake) == expected, "lab/fakes.py no longer copies the v3 projections."
