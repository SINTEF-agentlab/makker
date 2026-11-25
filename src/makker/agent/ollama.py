"""Ollama's parallel tool calls, kept apart while they stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from itertools import count
from typing import Any

from langchain_core.outputs import ChatGenerationChunk
from langchain_ollama import ChatOllama


class ChatOllamaParallelTools(ChatOllama):
    """Number each streamed tool call so parallel calls stay separate.

    Ollama streams every tool call whole, one per chunk, without an index.
    langchain-core's stream bridge (``language_models._compat_bridge``) keys a
    block without an index by its position in the chunk, so each call lands on
    the first one's block: the arguments are joined into one string that is not
    JSON (``{...}{...}``), under the last call's name, and none of the calls runs.
    Numbering the calls gives each its own block. Remove this shim when the
    bridge keeps index-less tool calls apart (a canary test says when).
    """

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        numbers = count()
        for chunk in super()._stream(*args, **kwargs):
            yield _numbered(chunk, numbers)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        numbers = count()
        async for chunk in super()._astream(*args, **kwargs):
            yield _numbered(chunk, numbers)


def _numbered(chunk: ChatGenerationChunk, numbers: Iterator[int]) -> ChatGenerationChunk:
    for call in getattr(chunk.message, "tool_call_chunks", None) or []:
        if call.get("index") is None:
            call["index"] = next(numbers)
    return chunk
