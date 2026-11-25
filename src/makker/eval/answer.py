"""The answer line: how a task asks for its values, and how scorers read them.

Scorers used to pick numbers out of the final prose, which passed wrong
answers (a number echoed from the prompt, like the 20 in "1x1x20") and failed
right ones ("150,200,250" read as one number). A sample that grades values now
names them in its metadata (``answer``: key -> what the value is), the solver
asks the agent to end with one line ``ANSWER: {...}`` holding them as JSON, and
the scorers read only that line. No line, or one that doesn't parse, is no
answer: that also fails an empty reply or a run that timed out.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

PREFIX = "ANSWER:"


def instruction(keys: Mapping[str, str]) -> str:
    """What the solver adds to the prompt of a sample that names its answer ``keys``."""

    skeleton = ", ".join(f'"{key}": ...' for key in keys)
    meanings = "\n".join(f"- {key}: {meaning}" for key, meaning in keys.items())
    return (
        "\n\nWhen you are done, end your final reply with one line of exactly this "
        "form, the values as JSON numbers (or lists of numbers) without units, and "
        "null for a value you could not obtain:\n"
        f"{PREFIX} {{{skeleton}}}\n{meanings}"
    )


def parse(completion: str) -> dict[str, Any] | None:
    """The values on the reply's last answer line, or ``None`` if there is none.

    Markdown around the line (bold, a code span) and text after the JSON object
    are tolerated; a line whose JSON doesn't parse to an object is no answer.
    """

    for line in reversed(completion.splitlines()):
        text = line.strip(" \t`*")
        if not text.startswith(PREFIX):
            continue
        rest = text[len(PREFIX) :].strip(" \t`*")
        try:
            values, _end = json.JSONDecoder().raw_decode(rest)
        except ValueError:
            return None
        return values if isinstance(values, dict) else None
    return None


def number(value: Any) -> float | None:
    """``value`` as a float: a JSON number, or a numeric string (a Unicode minus too)."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace("\u2212", "-"))
        except ValueError:
            return None
    return None
