"""Tiny display-policy helpers shared by approval and tool rendering.

Also the single home for the streamed tool-output display policy: every
surface accumulates a bounded tail of the raw stream and re-renders it through
the terminal emulator on a throttle, so a progress bar's carriage returns
collapse to one updating block and a chatty tool cannot burn the event loop.
The scheduling machinery differs per surface (a Textual timer in the TUI, an
asyncio task on the server); the policy numbers must not.
"""

from __future__ import annotations

import re
from typing import Any

# Largest raw-stream tail kept per tool call for live re-rendering.
TOOL_STREAM_TAIL_CAP = 256 * 1024
# Minimum seconds between re-renders of one tool's stream.
TOOL_STREAM_RENDER_INTERVAL = 0.1

_MAX_PREVIEW_LINES = 120
_MAX_PREVIEW_CHARS = 4000


def truncate_preview(
    text: str,
    *,
    max_lines: int = _MAX_PREVIEW_LINES,
    max_chars: int = _MAX_PREVIEW_CHARS,
    marker: str = "\n... [preview truncated]",
) -> str:
    """Trim ``text`` to fit within ``max_lines``/``max_chars`` for inline previews."""

    lines = text.splitlines()
    truncated = False

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip()
        truncated = True

    if truncated:
        rendered += marker
    return rendered


def fenced_block(text: str, *, language: str = "") -> str:
    """Wrap ``text`` in a backtick fence long enough to survive embedded backticks."""

    runs = re.findall(r"`+", text)
    fence = "`" * max(3, max((len(run) for run in runs), default=0) + 1)
    head = f"{fence}{language}" if language else fence
    return f"{head}\n{text}\n{fence}"


def shorten(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, appending ``...`` if it was clipped."""

    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def shorten_single_line(text: str, limit: int) -> str:
    """Collapse whitespace and trim to ``limit`` characters."""

    return shorten(" ".join(text.split()), limit)


# Under this, a duration is not worth showing: a number that flashes up is noise.
WORTH_TIMING = 2.0


def format_seconds(seconds: float) -> str:
    """How long something took or has run: "12s", "1:05"."""
    whole = max(0, int(seconds))
    return f"{whole}s" if whole < 60 else f"{whole // 60}:{whole % 60:02d}"


def short_path(path: str) -> str:
    """A path as its last two segments: enough to tell files apart, short enough
    for a line."""
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    return "/".join(parts[-2:]) if len(parts) > 2 else path


# The arguments that say what a call acts on, in the order they are looked for.
_WHAT_KEYS = (
    "title",
    "caption",
    "description",
    "code",
    "command",
    "pattern",
    "query",
    "file_path",
    "path",
    "slot",
)


def step_what(args: dict[str, Any] | None) -> str:
    """What a tool call acts on, in one line: its description, else its code's
    first line, the command, the pattern, or the file (as its last two segments)."""
    if not args:
        return ""
    for key in _WHAT_KEYS:
        value = args.get(key)
        if not value:
            continue
        line = next((ln.strip() for ln in str(value).splitlines() if ln.strip()), "")
        return short_path(line) if key in ("file_path", "path") else line
    first = next(iter(args.values()), None)
    return "" if first is None or isinstance(first, dict | list) else str(first).split("\n")[0]
