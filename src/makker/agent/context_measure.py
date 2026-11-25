"""What a model request holds, part by part, for the context view.

The provider reports one number per call: the input tokens. To show what fills
the context, the request as sent is counted part by part (by characters, the way
the summarizer estimates), and the counts are then scaled so they add up to that
measured total. The total is exact; the split is a close estimate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from makker.tool_labels import tool_label

# The parts of a request, in the order the context view lists them.
PARTS = ("system", "tools", "skills", "memory", "summary", "user", "assistant", "tool_results")

# How many of the largest tool results the view names, and how much of what
# each was asked about.
_LARGEST = 5
_HINT = 60
# Opens every tool result moved out to a file (``agent/offload.py`` writes it), so a
# result is never moved twice and is not listed among the largest.
MOVED_PREFIX = "[Moved out of the conversation to save context."
# deepagents marks the summary that replaces compacted turns with this source.
_SUMMARY_SOURCE = "summarization"
# The memory index and its guidelines, as the memory middleware writes them.
_MEMORY_OPEN = "<agent_memory>"
_MEMORY_CLOSE = ("</memory_guidelines>", "</agent_memory>")


def measure_request(request: Any, *, finished: Any = None) -> dict[str, Any]:
    """Approximate tokens per part of ``request``, and its largest tool results.

    ``finished`` is the request as a later middleware passes it on, when that one
    adds to the system prompt (the skills index): what it adds counts as skills.
    """
    system = _text(getattr(request, "system_message", None))
    final = _text(getattr(finished, "system_message", None)) if finished is not None else system
    parts = dict.fromkeys(PARTS, 0)
    parts["skills"] = _tokens(final[len(system) :]) if final.startswith(system) else 0
    memory = _memory_section(system)
    parts["memory"] = _tokens(memory)
    parts["system"] = _tokens(system) - parts["memory"] if system else 0
    parts["tools"] = sum(_tool_tokens(tool) for tool in getattr(request, "tools", None) or ())

    calls: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for message in getattr(request, "messages", None) or ():
        tokens = int(count_tokens_approximately([message]))
        if isinstance(message, ToolMessage):
            parts["tool_results"] += tokens
            if not _text(message).startswith(MOVED_PREFIX):  # a moved result is small now
                call = calls.get(message.tool_call_id) or {}
                results.append({"label": _result_label(message.name, call), "tokens": tokens})
        elif isinstance(message, AIMessage):
            parts["assistant"] += tokens
            for call in message.tool_calls:
                if call.get("id"):
                    calls[call["id"]] = call
        elif isinstance(message, HumanMessage):
            source = str((message.additional_kwargs or {}).get("lc_source") or "")
            if _SUMMARY_SOURCE in source:
                parts["summary"] += tokens
            elif "skill" in source:
                parts["skills"] += tokens
            else:
                parts["user"] += tokens
        else:
            parts["user"] += tokens
    results.sort(key=lambda r: r["tokens"], reverse=True)
    return {"parts": parts, "largest": results[:_LARGEST]}


def scale_to_usage(measured: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """The measured parts scaled to the provider's reported input tokens.

    The call's own reply (its output tokens) joins the conversation, so it adds
    to ``assistant``: the parts then sum to input plus output, the figure every
    context display shows.
    """
    parts = measured.get("parts") or {}
    total = sum(parts.values())
    input_tokens = int(usage.get("input_tokens") or 0)
    factor = input_tokens / total if total and input_tokens else 1.0
    scaled = {name: round(tokens * factor) for name, tokens in parts.items()}
    if input_tokens and scaled:
        # Rounding leaves a few tokens over or under; the biggest part absorbs them.
        biggest = max(scaled, key=lambda name: scaled[name])
        scaled[biggest] += input_tokens - sum(scaled.values())
    scaled["assistant"] = scaled.get("assistant", 0) + int(usage.get("output_tokens") or 0)
    largest = [
        {"label": item["label"], "tokens": round(item["tokens"] * factor)}
        for item in measured.get("largest") or ()
    ]
    return {"parts": scaled, "largest": largest}


def _text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
            if isinstance(block, str | dict)
        )
    return str(content)


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def _memory_section(system: str) -> str:
    start = system.find(_MEMORY_OPEN)
    if start < 0:
        return ""
    for close in _MEMORY_CLOSE:
        end = system.find(close, start)
        if end >= 0:
            return system[start : end + len(close)]
    return ""


def _tool_tokens(tool: Any) -> int:
    try:
        schema = tool if isinstance(tool, dict) else convert_to_openai_tool(tool)
        return _tokens(json.dumps(schema, default=str))
    except Exception:
        return 0


def _result_label(name: str | None, call: dict[str, Any]) -> str:
    """``Read · src/model.jl``: the tool and what it was asked about."""
    label = tool_label(name) if name else "Tool result"
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    for key in ("file_path", "path", "pattern", "command", "code", "query", "description"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            if key in ("file_path", "path"):
                return f"{label} · {_short_path(value.strip())}"
            first = value.strip().splitlines()[0]
            return f"{label} · {first[:_HINT] + '…' if len(first) > _HINT else first}"
    return label


def _short_path(path: str) -> str:
    """A path inside the workspace relative to it; any other by its last parts."""
    from makker.paths import workspace_root

    try:
        return str(Path(path).relative_to(workspace_root()))
    except ValueError:
        parts = Path(path).parts
        return path if len(parts) <= 4 else str(Path("…", *parts[-3:]))
