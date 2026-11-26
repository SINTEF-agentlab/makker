"""How a tool call reads in the TUI: the line of a step, and what is behind it."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from makker.agent.tool_output import strip_read_file_gutter
from makker.preview import (
    fenced_block,
    truncate_preview,
)


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


_READ_RANGE = re.compile(r"^@@ lines (\d+)-(\d+) of (\d+)")
_EN_DASH, _MINUS = "\u2013", "\u2212"


def _line_count(value: Any) -> int:
    return str(value).count("\n") + 1 if value else 0


def step_note(tool_name: str, args: dict[str, Any], output: str) -> str:
    """A few words on how a finished call went ("222 lines", "3 matches", an
    edit's lines added and removed), or "" for a tool whose result says nothing
    in a few words."""
    text = output.strip()
    if tool_name == "read_file":
        m = _READ_RANGE.match(text)
        if m:
            first, last, total = (int(g) for g in m.groups())
            whole = first == 1 and last == total
            return _count(total, "line") if whole else f"lines {first}{_EN_DASH}{last}"
        return _count(len([ln for ln in text.splitlines() if ln.strip()]), "line") if text else ""
    if tool_name == "grep":
        if text.startswith(("No matches found", _NO_MATCHES)):
            return "no matches"
        hits = sum(1 for ln in text.splitlines() if re.match(r"^\s+\d+:", ln))
        files = len([ln for ln in text.splitlines() if ln.strip()])
        return _count(hits, "match", "matches") if hits else _count(files, "file")
    if tool_name in ("glob", "ls"):
        n = len(_parse_path_list(output))
        return _count(n, "file") if tool_name == "glob" else _count(n, "entry", "entries")
    if tool_name == "write_file":
        return _count(str(args.get("content") or "").count("\n") + 1, "line")
    if tool_name == "edit_file":
        added, removed = (_line_count(args.get(k)) for k in ("new_string", "old_string"))
        return f"+{added} {_MINUS}{removed}"
    if tool_name == "record_attempt":
        return text.split(" · id=")[0].splitlines()[0] if text else ""
    return ""


def display_tool_body(tool_name: str, args: dict[str, Any], *, output: str, expanded: bool) -> str:
    """What a finished call returned, under the code it ran: a preview (the
    end of a Julia run, where its result and timing are), or all of it when
    ``expanded``."""
    code_section = julia_code_section(tool_name, args)
    full = strip_read_file_gutter(output) if tool_name == "read_file" else output
    language = _TOOL_LANGUAGES.get(tool_name, "")
    meta = _summarize_output(full)
    if expanded:
        block = fenced_block(full, language=language)
        return _join_nonempty(code_section, f"_result · {meta} · full output_", "", block)

    expandable = is_expandable(full, tool_name=tool_name)
    if not expandable:
        preview = full
    elif tool_name == "run_julia":
        preview = _truncate_preview_tail(full, tool_name=tool_name)
    else:
        preview = _truncate_preview(full, tool_name=tool_name)
    rendered = (
        fenced_block(preview, language=language)
        if _prefer_fenced_preview(tool_name)
        else _quote_block(preview)
    )
    kind = "output" if code_section else "result"
    header = f"_{kind} · {meta}{' · preview' if expandable else ''}_"
    return _join_nonempty(code_section, header, "", rendered)


def julia_code_section(tool_name: str, args: dict[str, Any]) -> str:
    """Return a fenced ``Code`` block for run_julia / plot_julia args.

    Empty string for any other tool. The TUI uses this so the agent's code
    stays visible alongside the simulator output instead of being replaced
    by it (and so something is shown while the call is still running).
    """

    if tool_name not in {"run_julia", "plot_julia"}:
        return ""
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return ""
    label = "**Code**"
    return "\n".join([label, "", fenced_block(code, language="julia"), ""])


def _truncate_preview_tail(text: str, *, tool_name: str | None = None) -> str:
    """Like ``_truncate_preview`` but keeps the trailing lines.

    Julia outputs put the meaningful bit (the ``→ return value`` and the
    ``[X.YZs]`` timing) at the bottom. Head-truncation hides exactly the
    part the agent; and the human reading the TUI; care about.
    """

    line_limit, char_limit = _preview_limits(tool_name)
    lines = text.splitlines()
    if len(lines) <= line_limit and len(text) <= char_limit:
        return text
    kept: list[str] = []
    chars = 0
    for line in reversed(lines):
        new_chars = chars + len(line) + (1 if kept else 0)
        if len(kept) >= line_limit or new_chars > char_limit:
            break
        kept.append(line)
        chars = new_chars
    kept.reverse()
    return "… [output truncated above]\n" + "\n".join(kept)


def _join_nonempty(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


# What `ls` and `glob` render instead of an empty list. Without this the card
# counts the sentence itself and an empty directory reads as "1 entry".
_NO_MATCHES = "No files found"


def _parse_path_list(output: str) -> list[str]:
    candidate = output.strip()
    if not candidate or candidate == _NO_MATCHES:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [line for line in candidate.splitlines() if line.strip()]


_PATH_DISPLAY_BUDGET = 40


def display_path(path_str: str) -> str:
    """A short path label: the trailing segments that fit, filename first.

    The filename matters most, so long paths keep their tail and mark dropped
    parents with a leading ``…/``; a single over-long name is truncated in the
    middle so its extension survives. Keeping the label short and on one line
    matters because a wrapped link can lose its click target past the first row.
    """

    normalized = str(path_str or "file").replace("\\", "/")
    if normalized in {"", "."}:
        return "."
    if normalized == "/":
        return "workspace"
    segments = [s for s in normalized.split("/") if s]
    if not segments:
        return "/"
    tail: list[str] = []
    width = 0
    for segment in reversed(segments):
        extra = len(segment) + (1 if tail else 0)
        if tail and width + extra > _PATH_DISPLAY_BUDGET:
            break
        tail.insert(0, segment)
        width += extra
    label = "/".join(tail)
    if len(tail) < len(segments):
        label = "…/" + label
    if len(label) > _PATH_DISPLAY_BUDGET + 4:
        label = _middle_truncate(segments[-1], _PATH_DISPLAY_BUDGET)
    return label


def _middle_truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` keeping both ends (so a file extension shows)."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = keep // 2
    return f"{text[:head]}…{text[-(keep - head) :]}"


_TOOL_LANGUAGES: dict[str, str] = {
    "run_julia": "julia",
    "execute": "sh",
}
_PREVIEW_LINES = 6
_PREVIEW_CHARS = 600
_TOOL_PREVIEW_LIMITS: dict[str, tuple[int, int]] = {
    "execute": (12, 2000),
    # run_julia limits are sized so a typical Jutul summary table plus the
    # return value and elapsed marker fit without a click-to-expand.
    "run_julia": (40, 6000),
    "read_file": (14, 1600),
}


def _preview_limits(tool_name: str | None) -> tuple[int, int]:
    if tool_name is None:
        return (_PREVIEW_LINES, _PREVIEW_CHARS)
    return _TOOL_PREVIEW_LIMITS.get(tool_name, (_PREVIEW_LINES, _PREVIEW_CHARS))


def _truncate_preview(text: str, *, tool_name: str | None = None) -> str:
    line_limit, char_limit = _preview_limits(tool_name)
    return truncate_preview(
        text,
        max_lines=line_limit,
        max_chars=char_limit,
        marker="\n... [output truncated]",
    )


def is_expandable(text: str, *, tool_name: str | None = None) -> bool:
    """Whether ``text`` exceeds the tool's preview budget (so expanding shows more).

    The single source of the preview limits: ``ToolBlock.expandable`` and the
    body renderer both call this, so the toggle is offered exactly when the
    expanded body actually differs from the preview.
    """
    line_limit, char_limit = _preview_limits(tool_name)
    return len(text) > char_limit or text.count("\n") + 1 > line_limit


def _summarize_output(text: str) -> str:
    line_count = max(1, text.count("\n") + 1)
    line_label = "line" if line_count == 1 else "lines"
    return f"{line_count} {line_label}"


def _quote_block(text: str) -> str:
    if not text:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _prefer_fenced_preview(tool_name: str) -> bool:
    # Program output is literal text: as Markdown, `__init__.py` turns bold.
    return tool_name in {"run_julia", "read_file", "execute"}
