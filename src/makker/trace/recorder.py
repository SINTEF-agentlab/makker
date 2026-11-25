"""Middleware that records model and tool events to a `TraceLog`."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.internal_call_transformer import InternalCallTransformer
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from makker.agent.context_measure import measure_request, scale_to_usage
from makker.trace import TraceLog
from makker.trace.messages import content_to_str, reasoning_to_str
from makker.trace.schema import (
    CONTEXT_COMPACTION,
    MESSAGE_ASSISTANT,
    MESSAGE_REASONING,
    MODEL_USAGE,
    TOOL_CALL,
    TOOL_RESULT,
    compaction_payload,
)


class TraceRecorder(AgentMiddleware):
    """Append model responses and tool round-trips to a trace log."""

    # deepagents' summarizer makes its summary model call inside the model node and
    # tags it as internal, but never declares the filter that keeps tagged calls out
    # of ``run.messages``. Without it the summary streams into the chat as if the
    # assistant had written it. Any middleware may declare the filter (langchain
    # registers it once), so it sits on this one, which every agent has.
    transformers = (InternalCallTransformer,)

    def __init__(
        self,
        trace: TraceLog,
        *,
        finish_request: Callable[[ModelRequest], ModelRequest] | None = None,
    ) -> None:
        """``finish_request`` adds what middleware running after this one add to a
        request (the skills index), so the measured request is the one sent."""
        super().__init__()
        self._trace = trace
        self._traced_cutoff: int | None = None
        self._finish_request = finish_request
        self._measured: dict[str, Any] | None = None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # What the request holds, part by part, for the context view; recorded
        # with the call's usage once the provider has reported it.
        self._measured = None
        with contextlib.suppress(Exception):
            finished = self._finish_request(request) if self._finish_request else None
            self._measured = measure_request(request, finished=finished)
        return await handler(request)

    def _record_compaction(self, state: Any) -> None:
        """Emit a ``context_compaction`` event when the stock summarizer compacts.

        deepagents' SummarizationMiddleware is non-mutating: on the turns it
        compacts, it records a ``_summarization_event`` (a fresh summary message,
        a cutoff index, and the offload path), which then stays in the state
        until the next compaction replaces it. The cutoff only grows, so it
        identifies the compaction. The trace is the record of which ones are
        already written: manual ``/compact`` and earlier agents built for this
        session (a resume, a model switch) write to it too.
        """
        event = state.get("_summarization_event") if isinstance(state, dict) else None
        if not isinstance(event, dict):
            return
        cutoff = event.get("cutoff_index")
        if not isinstance(cutoff, int):
            return
        if self._traced_cutoff is not None and cutoff <= self._traced_cutoff:
            return
        prior = _traced_cutoff(self._trace)
        self._traced_cutoff = max(prior, cutoff)
        if cutoff <= prior:
            return
        # The state still holds the full list (the middleware is non-mutating), so
        # what the model saw after the cut is directly computable. The reply just
        # produced came after the compaction: it is new, not kept.
        messages = state.get("messages")
        kept = freed = None
        if isinstance(messages, list) and len(messages) >= cutoff:
            tail = messages[cutoff:]
            if tail and isinstance(tail[-1], AIMessage):
                tail = tail[:-1]
            kept = len(tail)
            with contextlib.suppress(Exception):
                summary = event.get("summary_message")
                freed = max(
                    0,
                    int(count_tokens_approximately(messages[prior:cutoff]))
                    - int(count_tokens_approximately([summary] if summary else [])),
                )
        self._trace.append(
            CONTEXT_COMPACTION,
            compaction_payload(
                summarized=cutoff - prior,
                kept=kept,
                offloaded=event.get("file_path") is not None,
                cutoff=cutoff,
                freed_tokens=freed,
            ),
        )

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._record_compaction(state)
        messages = (
            state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        )
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        # ``content_blocks`` normalizes provider-specific shapes (e.g. OpenAI
        # keeps reasoning summaries under a raw ``summary`` key that the
        # projection helpers don't know about).
        blocks = getattr(last, "content_blocks", None) or last.content
        reasoning = reasoning_to_str(blocks)
        if reasoning.strip():
            self._trace.append(MESSAGE_REASONING, {"content": reasoning})
        content = content_to_str(blocks)
        if content.strip():
            self._trace.append(MESSAGE_ASSISTANT, {"content": content})
        usage = getattr(last, "usage_metadata", None)
        if usage:
            # Token accounting per model turn; cost and efficiency analyses
            # read these events instead of re-deriving counts from text. The
            # context view reads what the request held from ``context``.
            payload = dict(usage)
            if self._measured is not None:
                payload["context"] = scale_to_usage(self._measured, payload)
            self._trace.append(MODEL_USAGE, payload)
        self._measured = None
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        self._trace.append(
            TOOL_CALL,
            {"id": call.get("id"), "name": call.get("name"), "args": call.get("args")},
        )
        try:
            result = await handler(request)
        except GraphBubbleUp:
            # Control-flow signals (approval interrupts, parent commands,
            # graph-drain bubbles) must propagate untouched. (Cancellation and
            # KeyboardInterrupt are BaseException, so `except Exception` below
            # never catches them.)
            raise
        except Exception as exc:
            # A tool raised instead of returning a result. Don't let one failed
            # tool call abort the entire turn: hand the error back to the model
            # as a tool result so it can recover (retry, pick another tool, fix
            # its arguments).
            result = _tool_error_result(call, exc)

        # Tools that update state (write_todos, the task subagent, read_file on
        # media) return a Command whose update carries the call's ToolMessage.
        message = result if isinstance(result, ToolMessage) else _command_tool_message(result, call)
        if message is not None:
            self._trace.append(
                TOOL_RESULT,
                {
                    "tool_call_id": message.tool_call_id,
                    "name": message.name or call.get("name"),
                    "content": content_to_str(message.content),
                    "status": message.status,
                },
            )
        return result


def _traced_cutoff(trace: TraceLog) -> int:
    """The furthest compaction cutoff already in the trace (0 when none is)."""

    cutoffs = [event.payload.get("cutoff") for event in trace.events_of(CONTEXT_COMPACTION)]
    return max((c for c in cutoffs if isinstance(c, int)), default=0)


def _command_tool_message(result: Any, call: dict[str, Any]) -> ToolMessage | None:
    """The ToolMessage answering ``call`` in a Command's update, if it has one."""

    update = result.update if isinstance(result, Command) else None
    messages = update.get("messages") if isinstance(update, dict) else None
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, ToolMessage) and message.tool_call_id == call.get("id"):
            return message
    return None


def _tool_error_result(call: dict[str, Any], exc: Exception) -> ToolMessage:
    """Turn a raised tool exception into an error ``ToolMessage`` for the model."""

    name = call.get("name") or "tool"
    content = f"Error running tool `{name}`: {type(exc).__name__}: {exc}"
    return ToolMessage(
        content=content,
        tool_call_id=call.get("id") or "",
        name=call.get("name"),
        status="error",
    )
