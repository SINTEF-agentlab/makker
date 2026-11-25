"""The context view: how full the context is, what fills it, and what was moved out.

Everything here reads the session's trace, so a resumed session shows the same as
a live one and both front ends show the same figures:

- the ``model`` event holds the window, where it came from, and the points where
  offloading and compaction run;
- each ``model_usage`` event holds what the provider measured for a call, and
  (under ``context``) what the request held, part by part;
- ``context_offload`` and ``context_compaction`` events hold what was moved out.

The context in use is the last call's input plus its output: the reply joins the
conversation the next call sends. The summarizer compares the same figure with
the compaction point.
"""

from __future__ import annotations

from typing import Any

from makker.agent.context_measure import PARTS
from makker.trace import TraceLog
from makker.trace.schema import CONTEXT_COMPACTION, CONTEXT_OFFLOAD, MODEL, MODEL_USAGE

_BAR_CELLS = 24
# The meter warns as the context nears the compaction point.
_WARN_SHARE = 0.75
_HIGH_SHARE = 0.9

# Shown instead of the split while the figure is an estimate.
ESTIMATE_NOTE = "The breakdown updates after the next model call."

PART_LABELS = {
    "system": "System prompt",
    "tools": "Tool definitions",
    "skills": "Skills",
    "memory": "Memory",
    "summary": "Conversation summary",
    "user": "Your messages",
    "assistant": "Agent messages",
    "tool_results": "Tool results",
}


def format_tokens(count: int) -> str:
    """Token counts as compact figures: 1,500 is "1.5k", 1,050,000 "1.05M". A
    power-of-two context size keeps the name it goes by: 65,536 is "64k"."""

    def millions(value: float) -> str:
        return f"{value:.2f}".rstrip("0").removesuffix(".") + "M"

    if count >= 8192 and count & (count - 1) == 0:
        kib = count // 1024
        return millions(kib / 1024) if kib >= 1024 else f"{kib}k"
    if count >= 1_000_000:
        return millions(count / 1_000_000)
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    if count >= 1_000:
        return f"{count / 1000:.1f}k"
    return str(count)


def tokens_after_call(usage: dict[str, Any]) -> int:
    """The context a call leaves: its input plus its output."""
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


def context_meter(trace: TraceLog) -> dict[str, Any] | None:
    """How full the context is now, or None before the agent was first built.

    ``used`` is None until a model call has measured it. It is ``estimated`` when
    something changed since that call: a compaction made the context smaller (by
    about its ``freed_tokens``), or the model was switched (another tokenizer
    counts the same text differently).
    """
    model = trace.last_event((MODEL,))
    if model is None:
        return None
    info = model.payload
    meter: dict[str, Any] = {
        "model": info.get("model"),
        "window": info.get("window"),
        "window_source": info.get("window_source"),
        "compact_at": info.get("compact_at"),
        "offload_at": info.get("offload_at"),
        "used": None,
        "estimated": False,
    }
    usage = trace.last_event((MODEL_USAGE,))
    if usage is None:
        return meter
    used = tokens_after_call(usage.payload)
    estimated = False
    if model.id > usage.id:
        measured_on = trace.last_event((MODEL,), before=usage.id)
        estimated = measured_on is None or measured_on.payload.get("model") != info.get("model")
    for event in trace.events_after(usage.id, kinds=(CONTEXT_COMPACTION,)):
        used -= int(event.payload.get("freed_tokens") or 0)
        estimated = True
    meter["used"] = max(0, used)
    meter["estimated"] = estimated
    return meter


def context_report(trace: TraceLog) -> dict[str, Any]:
    """Everything the context view shows, as data both front ends render."""
    meter = context_meter(trace)
    usage_event = trace.last_event((MODEL_USAGE,))
    usage = usage_event.payload if usage_event else {}
    measured = usage.get("context") or {}
    parts = measured.get("parts") or {}

    breakdown = [
        {"key": key, "label": PART_LABELS[key], "tokens": int(parts[key])}
        for key in PARTS
        if parts.get(key)
    ]
    if not breakdown and usage:
        # Recorded before calls were measured part by part.
        breakdown = [{"key": "all", "label": "In context", "tokens": tokens_after_call(usage)}]

    moved = {
        "results": sum(int(e.payload.get("count") or 0) for e in trace.events_of(CONTEXT_OFFLOAD)),
        "compactions": len(trace.events_of(CONTEXT_COMPACTION)),
    }
    return {
        "meter": meter,
        "breakdown": breakdown,
        "largest": list(measured.get("largest") or ()),
        "moved": moved,
    }


def usage_alert(meter: dict[str, Any] | None) -> str:
    """``ok`` / ``warn`` / ``high`` for status-bar styling, by how near compaction is."""
    if not meter or meter.get("used") is None:
        return "ok"
    limit = meter.get("compact_at") or meter.get("window")
    if not limit:
        return "ok"
    share = meter["used"] / limit
    if share >= _HIGH_SHARE:
        return "high"
    if share >= _WARN_SHARE:
        return "warn"
    return "ok"


def status_label(meter: dict[str, Any] | None) -> str | None:
    """Short ``ctx …`` figure for the status bar, or None before the first call."""
    if not meter or meter.get("used") is None:
        return None
    fraction = _fraction(meter)
    approx = "~" if meter.get("estimated") else ""
    if fraction is None:
        return f"ctx {approx}{format_tokens(meter['used'])}"
    return f"ctx {approx}{fraction:.0%}"


def render_context_markdown(report: dict[str, Any]) -> str:
    """The context view as markdown, for the terminal's ``/context``."""
    meter = report.get("meter") or {}
    window = int(meter.get("window") or 0)
    used = meter.get("used")
    compact_at = int(meter.get("compact_at") or 0)
    lines: list[str] = []

    if used is None:
        lines.append("No model calls yet.")
    else:
        approx = "~" if meter.get("estimated") else ""
        if window:
            lines.append(
                f"**{approx}{format_tokens(used)} of {format_tokens(window)} tokens**"
                f" ({_share(used / window)})"
            )
        else:
            lines.append(f"**{approx}{format_tokens(used)} tokens**")
    if meter.get("model"):
        assumed = " · window size assumed" if meter.get("window_source") == "assumed" else ""
        lines.append(f"`{meter['model']}`{assumed}")
    if used is not None and window:
        lines.append(f"`{_bar(min(1.0, used / window))}`")
    if meter.get("estimated"):
        # The split is the last call's, from before the compaction or switch.
        lines += ["", ESTIMATE_NOTE]

    breakdown = [] if meter.get("estimated") else report.get("breakdown") or []
    if breakdown:
        lines += ["", "| | Tokens | |", "|---|---:|---:|"]
        lines += [_row(part["label"], int(part["tokens"]), window) for part in breakdown]
        if window and compact_at and used is not None:
            lines.append(_row("Free space", max(0, compact_at - used), window))
            lines.append(_row("Compaction buffer", max(0, window - compact_at), window))

    moved = report.get("moved") or {}
    bits = []
    if moved.get("results"):
        bits.append(f"{moved['results']} tool results to files")
    if moved.get("compactions"):
        count = moved["compactions"]
        bits.append(f"{count} compaction{'s' if count != 1 else ''}")
    if bits:
        lines += ["", "Moved out: " + ", ".join(bits) + "."]
    return "\n".join(lines)


def _fraction(meter: dict[str, Any] | None) -> float | None:
    if not meter or meter.get("used") is None or not meter.get("window"):
        return None
    return min(1.0, meter["used"] / meter["window"])


def _bar(fraction: float) -> str:
    filled = round(fraction * _BAR_CELLS)
    return "█" * filled + "░" * (_BAR_CELLS - filled)


def _row(label: str, tokens: int, window: int) -> str:
    share = _share(tokens / window) if window else ""
    return f"| {label} | {format_tokens(tokens)} | {share} |"


def _share(fraction: float) -> str:
    return "<1%" if 0 < fraction < 0.01 else f"{fraction:.0%}"
