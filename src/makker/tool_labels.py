"""Friendly, user-facing names for the agent's tools.

One source of truth shared by every surface that shows a tool to a person (the
live TUI steps, the approval prompt, and the HTML transcript), so a tool reads
the same everywhere. The agent-facing tool *id* (``run_julia``, ``execute``,
...) is never shown raw to the user. The web UI keeps its own copy of the
activity phrases (``webapp/src/steps.ts``).
"""

from __future__ import annotations

_TOOL_LABELS: dict[str, str] = {
    "run_julia": "Julia",
    "plot_julia": "Julia plot",
    "recapture_plot": "Recapture plot",
    "close_plots": "Close plots",
    "execute": "Shell",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "delete": "Delete",
    "glob": "Find files",
    "grep": "Search",
    "ls": "List",
    "write_todos": "Plan",
    "task": "Delegate",
    "record_attempt": "Record attempt",
    "write_report": "Report",
    "reset_julia": "Reset Julia",
    "remember": "Remember",
}


def tool_label(name: str) -> str:
    """The user-facing name for a tool id; unknown tools show their raw id."""
    return _TOOL_LABELS.get(name, name)


# What a tool did, for one call and for several, as a group's summary says it
# ("Read 4 files, ran Julia"). A tool not listed reads "Used <label>".
_TOOL_ACTIVITY: dict[str, tuple[str, str]] = {
    "read_file": ("Read a file", "Read {n} files"),
    "grep": ("Searched", "Searched {n} times"),
    "glob": ("Looked for files", "Looked for files {n} times"),
    "ls": ("Listed a folder", "Listed {n} folders"),
    "run_julia": ("Ran Julia", "Ran Julia {n} times"),
    "plot_julia": ("Made a plot", "Made {n} plots"),
    "recapture_plot": ("Captured a plot", "Captured {n} plots"),
    "close_plots": ("Closed plots", "Closed plots {n} times"),
    "execute": ("Ran a command", "Ran {n} commands"),
    "write_file": ("Wrote a file", "Wrote {n} files"),
    "edit_file": ("Edited a file", "Made {n} edits"),
    "delete": ("Deleted a file", "Deleted {n} files"),
    "task": ("Delegated a task", "Delegated {n} tasks"),
    "record_attempt": ("Recorded an attempt", "Recorded {n} attempts"),
    "write_report": ("Wrote a report", "Wrote {n} reports"),
    "reset_julia": ("Restarted Julia", "Restarted Julia {n} times"),
    "remember": ("Saved a note", "Saved {n} notes"),
}


def tool_activity(name: str, count: int) -> str:
    """What ``count`` calls of a tool did, in a few words ("Read 4 files")."""
    label = tool_label(name)
    one, many = _TOOL_ACTIVITY.get(name, (f"Used {label}", f"Used {label} {{n}} times"))
    return one if count == 1 else many.format(n=count)


def activity_summary(names: list[str]) -> str:
    """A run of calls in one line, in the order each tool first ran:
    ``["read_file", "grep", "read_file"]`` reads "Read 2 files, searched"."""
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    phrases = [tool_activity(name, n) for name, n in counts.items()]
    return ", ".join(p if i == 0 else p[0].lower() + p[1:] for i, p in enumerate(phrases))
