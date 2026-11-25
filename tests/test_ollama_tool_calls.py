"""Parallel tool calls from a local Ollama model reach the agent as separate calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from langchain_core.language_models._compat_bridge import achunks_to_events, chunks_to_events
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGenerationChunk
from langchain_ollama import ChatOllama

from makker.agent.ollama import ChatOllamaParallelTools

# Two calls as Ollama streams them: each whole, in its own chunk, with no index.
CALLS = [("ls", {"path": "/tmp"}), ("run_julia", {"code": "1 + 1"})]


def _chunks() -> list[ChatGenerationChunk]:
    calls = [tool_call(name=name, args=args, id=name) for name, args in CALLS]
    return [ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=[c])) for c in calls]


@pytest.fixture
def ollama_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    def stream(self: Any, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        yield from _chunks()

    async def astream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in _chunks():
            yield chunk

    monkeypatch.setattr(ChatOllama, "_stream", stream)
    monkeypatch.setattr(ChatOllama, "_astream", astream)


def _finished_calls(events: list[Any]) -> list[dict[str, Any]]:
    return [
        e["content"]
        for e in events
        if e.get("event") == "content-block-finish" and "tool_call" in e["content"]["type"]
    ]


@pytest.mark.usefixtures("ollama_streams")
async def test_parallel_tool_calls_stay_separate() -> None:
    model = ChatOllamaParallelTools(model="qwen3.8:27b")
    expected = [
        {"type": "tool_call", "id": name, "name": name, "args": args} for name, args in CALLS
    ]
    assert _finished_calls(list(chunks_to_events(model._stream([])))) == expected
    events = [e async for e in achunks_to_events(model._astream([]))]
    assert _finished_calls(events) == expected


@pytest.mark.usefixtures("ollama_streams")
def test_canary_the_stream_bridge_still_joins_unnumbered_calls() -> None:
    """When this fails, langchain-core keeps index-less tool calls apart on its own,
    and ``agent/ollama.py`` (``ChatOllamaParallelTools``) can go."""
    calls = _finished_calls(list(chunks_to_events(ChatOllama(model="qwen3.8:27b")._stream([]))))
    assert [c["type"] for c in calls] == ["invalid_tool_call"]
    assert calls[0]["args"] == '{"path": "/tmp"}{"code": "1 + 1"}'
