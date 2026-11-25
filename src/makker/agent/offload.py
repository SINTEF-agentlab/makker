"""Move older tool results out of the conversation, into files, as the context fills.

This replaces clearing them. langchain's clearing turned every tool result but the
newest few into ``[cleared]`` on each request once the history passed a threshold:
a cliff, silent, with nothing to read back, after which an agent sometimes filled
the gap from memory. Here, once the context passes the budget's ``offload_at``,
the oldest tool results go to files until it is back under ``offload_to``. Each
leaves a short preview and the file's path, so the agent can read back what it
needs. The change is stored in the conversation (the full text stays in the trace
and in the file), which keeps the prompt the same between calls for provider
caching, and each step is one ``context_offload`` trace event that the front ends
announce.
"""

from __future__ import annotations

from typing import Any

from deepagents.backends.utils import sanitize_tool_call_id
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from makker.context_budget import ContextBudget
from makker.trace import TraceLog
from makker.trace.schema import CONTEXT_OFFLOAD, offload_payload

# A result this short frees too little to be worth a file and a read-back.
_MIN_CHARS = 2_000
# Results the agent refers back to by value (an attempt id passed on as a parent).
_NEVER_OFFLOAD = frozenset({"record_attempt"})
_PREVIEW_LINES = 5
_PREVIEW_LINE_CHARS = 200
# Opens every replacement, so a result is never moved twice.
_MOVED = "[Moved out of the conversation to save context."
# deepagents' own replacement for a result too large to keep in the first place.
_EVICTED = "was saved in the filesystem at this path"


class ToolResultOffloadMiddleware(AgentMiddleware):
    """Before each model call, move the oldest tool results to files when the context is full."""

    def __init__(self, budget: ContextBudget, backend: Any, trace: TraceLog | None) -> None:
        super().__init__()
        self._budget = budget
        self._backend = backend
        self._trace = trace
        root = str(getattr(backend, "artifacts_root", "") or "").rstrip("/")
        self._folder = f"{root}/large_tool_results"

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages: list[AnyMessage] = list(state.get("messages") or [])
        # Messages a compaction already replaced with its summary are not sent.
        event = state.get("_summarization_event")
        start = event.get("cutoff_index", 0) if isinstance(event, dict) else 0
        live = messages[start:]
        used = context_used(live)
        if used <= self._budget.offload_at:
            return None
        results = [m for m in live if isinstance(m, ToolMessage)]
        keep = self._budget.keep_tool_results
        moved: list[ToolMessage] = []
        freed = 0
        for message in results[:-keep] if keep else results:
            if used - freed <= self._budget.offload_to:
                break
            replacement = await self._move(message)
            if replacement is None:
                continue
            freed += count_tokens_approximately([message]) - count_tokens_approximately(
                [replacement]
            )
            moved.append(replacement)
        if not moved:
            return None
        if self._trace is not None:
            self._trace.append(
                CONTEXT_OFFLOAD, offload_payload(count=len(moved), freed_tokens=freed)
            )
        return {"messages": moved}

    async def _move(self, message: ToolMessage) -> ToolMessage | None:
        """Write one result to its file and return the message that replaces it."""
        content = message.content
        if (
            message.id is None
            or not isinstance(content, str)
            or len(content) < _MIN_CHARS
            or content.startswith(_MOVED)
            or _EVICTED in content[:400]
            or message.name in _NEVER_OFFLOAD
        ):
            return None
        path = f"{self._folder}/{sanitize_tool_call_id(message.tool_call_id or message.id)}"
        written = await self._backend.awrite(path, content)
        if written is None or getattr(written, "error", None):
            return None
        return message.model_copy(
            update={"content": _replacement(getattr(written, "path", None) or path, content)}
        )


def context_used(messages: list[AnyMessage]) -> int:
    """Tokens the conversation takes now: the last call's measured size, plus what came after.

    The provider's count for the last model call covers the system prompt, the
    tools and every message it was sent; the messages added since are estimated.
    Without a measured call yet, everything is estimated.
    """
    for i in range(len(messages) - 1, -1, -1):
        usage = getattr(messages[i], "usage_metadata", None)
        if isinstance(messages[i], AIMessage) and usage:
            measured = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
            return measured + count_tokens_approximately(messages[i + 1 :])
    return count_tokens_approximately(messages)


def _replacement(path: str, content: str) -> str:
    lines = content.splitlines()
    clip = [line[:_PREVIEW_LINE_CHARS] for line in lines]
    if len(clip) > 2 * _PREVIEW_LINES:
        preview = [*clip[:_PREVIEW_LINES], "…", *clip[-_PREVIEW_LINES:]]
    else:
        preview = clip
    return (
        f"{_MOVED} The full result ({len(lines)} lines) is in {path}; read it with "
        "read_file (an offset and limit read part of it) rather than recalling it.]\n\n"
        + "\n".join(preview)
    )


def describe_offload(payload: dict[str, Any]) -> str:
    """The note every front end shows for a ``context_offload`` event."""
    from makker.context_panel import format_tokens

    count = payload.get("count") or 0
    freed = payload.get("freed_tokens")
    text = f"Moved {count} older tool result{'s' if count != 1 else ''} to files to free context"
    if isinstance(freed, int) and freed > 0:
        text += f" (about {format_tokens(freed)} tokens)"
    return text + ". The agent can read them back."
