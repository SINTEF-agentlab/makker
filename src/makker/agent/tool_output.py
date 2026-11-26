"""Normalize streamed tool output for display and interrupt detection."""

from __future__ import annotations

import functools
import json
import re
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.types import Command

from makker.trace.messages import content_to_str

# Some langgraph paths stringify ``[ToolMessage(content='...', ...)]``
# instead of returning the structured object. The regex below pulls the
# original content back out of that repr.
_TOOL_MESSAGE_CONTENT = re.compile(
    r"content=(?P<quote>['\"])(?P<body>(?:\\.|(?!\1).)*)\1",
    re.DOTALL,
)


def error_reply_tool(fn):
    """``@tool`` for a tool that says it failed by replying ``ERROR: …``.

    The model reads the reply unchanged. Raising it gives the result the error
    status the framework's own tools set, which is how a front end knows the call
    failed. A tool whose error follows other output raises ``ToolException`` itself.
    """

    @functools.wraps(fn)
    async def run(*args: Any, **kwargs: Any) -> Any:
        reply = await fn(*args, **kwargs)
        if isinstance(reply, str) and reply.startswith("ERROR:"):
            raise ToolException(reply)
        return reply

    built = tool(run)
    built.handle_tool_error = True
    return built


def normalize_tool_output(value: Any) -> str:
    """Return human-readable tool output for display."""

    if value is None:
        return ""
    if isinstance(value, ToolMessage):
        return normalize_tool_output(value.content)
    if isinstance(value, Command):
        # A tool that updates state answers with a command; its reply to the
        # model is the tool message in the update.
        update = value.update if isinstance(value.update, dict) else {}
        messages = [m for m in update.get("messages") or () if isinstance(m, ToolMessage)]
        return normalize_tool_output(messages)
    if isinstance(value, list):
        if value and all(isinstance(item, ToolMessage) for item in value):
            parts = [normalize_tool_output(item) for item in value]
            return "\n".join(part for part in parts if part)
        # A multimodal content list: a tool handing the model something to look at
        # returns blocks tagged with ``type``, the image one holding base64 that
        # would flood the transcript. Keep the text; the image reaches the user as
        # the artifact card built from the same file. Any other list is data, and
        # reads better as JSON.
        if value and all(isinstance(item, dict) and "type" in item for item in value):
            return content_to_str(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        if is_interrupt_payload(value):
            return value
        # Only a whole repr: text that merely contains one (a file the agent read
        # that builds a ToolMessage, say) is output in its own right.
        if value.lstrip().startswith(("[ToolMessage(content=", "ToolMessage(content=")):
            extracted = _extract_tool_messages_from_repr(value)
            if extracted:
                return extracted
        return value
    return str(value)


def is_interrupt_payload(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("interrupt(") or "interrupt(value=" in lowered


# One ``read_file`` gutter row: a right-justified line marker, optionally with a
# ``.<n>`` suffix marking the continuation of an over-long line, then the
# separator. Two spaces is the separator the framework emits today; the tab is
# the older ``cat -n`` form, which stored transcripts still carry. The separator
# belongs to the framework and has changed before, so this pattern lives in one
# place and every surface that shows ``read_file`` output strips through it.
READ_FILE_GUTTER = re.compile(r"^ *\d+(?:\.\d+)?(?:\t|  )")


def strip_read_file_gutter(text: str) -> str:
    """Drop the line-number gutter from each ``read_file`` row that carries one."""

    return "\n".join(READ_FILE_GUTTER.sub("", line, count=1) for line in text.splitlines())


def _extract_tool_messages_from_repr(text: str) -> str:
    """Pull content strings back out of a ``[ToolMessage(content='...', …)]`` repr."""

    parts: list[str] = []
    for match in _TOOL_MESSAGE_CONTENT.finditer(text):
        body = match.group("body")
        quote = match.group("quote")
        if quote == "'":
            body = body.replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")
        else:
            body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        parts.append(body)
    return "\n".join(parts)
