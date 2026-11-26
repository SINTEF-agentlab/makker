"""Numbers the agent typed into a file that nothing it was shown holds.

An agent that has lost a result from its context can fill the gap from memory: it
writes a plausible value into a result file, and the session looks fine. A value
computed in the session was shown in some tool output first (a printed result, a
file read), so a precise number in a typed file that no earlier tool output or user
message holds was not read from anywhere.

Only what the agent typed is checked (``write_file`` and ``edit_file``), not files
its code wrote, and not code files, whose constants are knowledge rather than
results. A written number matches an earlier one at the precision it was written
with, so a rounded copy counts as read.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePath

from makker.review.findings import Finding
from makker.trace import Event
from makker.trace.schema import MESSAGE_USER, TOOL_CALL, TOOL_RESULT

# Fewer significant digits than this are too easily typed from the task itself or
# rounded by hand (0.25, 1.5e-3, 300.0) to say anything.
MIN_DIGITS = 5
_WRITE_TOOLS = {"write_file": "content", "edit_file": "new_string"}
_CODE_SUFFIXES = frozenset(
    [
        ".jl",
        ".py",
        ".r",
        ".m",
        ".sh",
        ".toml",
        ".js",
        ".ts",
        ".tsx",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".f",
        ".f90",
        ".rs",
        ".go",
        ".java",
    ]
)
# A decimal number, as written; a digit run with no point or exponent is a count.
_DECIMAL = re.compile(r"(?<![\w.])(\d*\.\d+|\d+\.)(?:[eE]([-+]?\d+))?")
# Any number shown; Fortran-style output (an optimizer's log) writes 2.30735D-02.
_ANY_NUMBER = re.compile(r"\d*\.?\d+(?:[eEdD][-+]?\d+)?")


@dataclass(frozen=True)
class UngroundedWrite:
    """A typed file whose precise numbers were not all shown to the agent first."""

    path: str
    numbers: tuple[str, ...]
    checked: int

    def finding(self) -> Finding:
        """The write as a review finding."""
        shown = ", ".join(self.numbers[:5]) + (", …" if len(self.numbers) > 5 else "")
        return Finding(
            category="agent-error",
            severity="high",
            title=f"Numbers written to {PurePath(self.path).name} that no output showed",
            evidence=(
                f"{self.path}: {shown} ({len(self.numbers)} of the {self.checked} precise "
                "numbers typed into it) appear in no earlier tool output or user message."
            ),
            suggestion=(
                "Check where the values came from: an agent that lost a result from its "
                "context may have written it from memory."
            ),
            fix_target="other",
            detail="Found by a check of the trace, not by the reviewer model.",
        )


def ungrounded_writes(events: Iterable[Event]) -> list[UngroundedWrite]:
    """The typed files, in order, holding numbers no earlier output or message holds."""
    seen: list[float] = []  # sorted, once a write needs it
    new: list[float] = []
    flagged: list[UngroundedWrite] = []
    for event in events:
        payload = event.payload
        if event.kind == TOOL_RESULT or event.kind == MESSAGE_USER:
            text = str(payload.get("content") or "")
            new += (float(n.replace("D", "e").replace("d", "e")) for n in _ANY_NUMBER.findall(text))
            continue
        if event.kind != TOOL_CALL or payload.get("name") not in _WRITE_TOOLS:
            continue
        args = payload.get("args") or {}
        path = str(args.get("file_path") or "")
        if PurePath(path).suffix.lower() in _CODE_SUFFIXES:
            continue
        if new:
            seen, new = sorted(seen + new), []
        text = str(args.get(_WRITE_TOOLS[payload["name"]]) or "")
        precise = [m for m in _DECIMAL.finditer(text) if _digits(m.group(1)) >= MIN_DIGITS]
        missing = [m.group(0) for m in precise if not _shown(m, seen)]
        if missing:
            flagged.append(UngroundedWrite(path, tuple(dict.fromkeys(missing)), len(precise)))
    return flagged


def _digits(mantissa: str) -> int:
    """Significant digits, not counting zeros that only pad (0.00120, 1.50000)."""
    digits = mantissa.replace(".", "").lstrip("0")
    if "." in mantissa:
        digits = digits.rstrip("0")
    return len(digits)


def _shown(match: re.Match[str], seen: list[float]) -> bool:
    """Whether an earlier number equals this one at the precision it was written."""
    mantissa, exponent = match.group(1), int(match.group(2) or 0)
    value = float(match.group(0))
    decimals = len(mantissa.partition(".")[2])
    # Rounding or truncating to the written digits: within one unit of the last one.
    unit = 10.0 ** (exponent - decimals) * (1 + 1e-9)
    for scale in (1.0, 100.0):  # a fraction written as a percentage
        target = value / scale
        tolerance = unit / scale
        i = bisect.bisect_left(seen, target - tolerance)
        if i < len(seen) and seen[i] <= target + tolerance:
            return True
    return False
