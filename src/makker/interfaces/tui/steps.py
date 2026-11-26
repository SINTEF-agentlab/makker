"""How the TUI condenses the agent's work.

Everything the agent does between two things it says or shows (its tool calls
and its reasoning) is one ``StepGroup``: a single line once the steps in it are
done ("Read 4 files, ran Julia · 1 failed · 12s"), the steps one click away. The
step under way is not in the log: the ``LiveStatus`` line above the prompt says
what it is and how long the turn has run, with the newest lines a long step
prints, and the ``PlanView`` above it shows the agent's plan while it works.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import events
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Static

from makker.interfaces.tui.tool_display import display_path
from makker.interfaces.tui.widgets import (
    ReasoningBlock,
    StepLine,
    StepRow,
    ToolBlock,
    todo_text,
)
from makker.preview import WORTH_TIMING, format_seconds, shorten_single_line
from makker.tool_labels import activity_summary


class ConversationLog(VerticalScroll):
    """The scrolling log. Appending anything but steps after a group of steps
    closes that group: it is done, and shows itself in full (see ``settle``)."""

    def mount(self, *widgets: Any, before: Any = None, after: Any = None) -> Any:
        if before is None and after is None and widgets:
            last = self.children[-1] if self.children else None
            if isinstance(last, StepGroup) and not isinstance(widgets[0], StepGroup):
                last.settle()
        return super().mount(*widgets, before=before, after=after)


class StepGroup(Vertical):
    """One stretch of the agent's work: a line summing up its steps, the steps
    (one line each) behind it. A stretch of one step is just that step's line."""

    DEFAULT_CSS = """
    StepGroup {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
    }
    StepGroup.empty {
        display: none;
    }
    StepGroup > #steps {
        height: auto;
        margin: 0 0 0 1;
        padding: 0 0 0 1;
        border-left: solid $surface-lighten-1;
    }
    StepGroup.lone > .group-line {
        display: none;
    }
    StepGroup.lone > #steps {
        margin: 0;
        padding: 0;
        border-left: none;
    }
    StepGroup.closed > #steps {
        display: none;
    }
    """

    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__()
        self._open = verbose
        self._verbose = verbose
        # Still being added to: a group of nothing but a thought waits for what
        # follows, so its line isn't replaced by a summary a moment later.
        self.live = True
        self._line = StepLine()
        self._line.add_class("group-line")
        self._steps = Vertical(id="steps")

    def compose(self):
        yield self._line
        yield self._steps

    def on_mount(self) -> None:
        self.refresh_view()

    @property
    def rows(self) -> list[StepRow]:
        return [row for row in self._steps.children if isinstance(row, StepRow)]

    async def add(self, row: StepRow) -> None:
        self.live = True
        await self._steps.mount(row)
        self.refresh_view()

    def settle(self) -> None:
        """The stretch of work is over: show what it holds."""
        self.live = False
        self.refresh_view()

    async def on_click(self, event: events.Click) -> None:
        if event.widget is not None and self._line in event.widget.ancestors_with_self:
            await self.set_open(not self._open)

    async def toggle(self) -> None:
        await self.set_open(not self._open)

    async def set_open(self, open_: bool, *, verbose: bool | None = None) -> None:
        """Open the group on its steps; ``verbose`` also opens every step in full."""
        self._open = open_
        if verbose is not None:
            self._verbose = verbose
            for row in self.rows:
                await row.set_expanded(verbose, verbose=verbose)
        self.refresh_view()

    def refresh_view(self) -> None:
        shown = [row for row in self.rows if not row.under_way]
        tools = [row for row in shown if isinstance(row, ToolBlock)]
        empty = not shown or (self.live and not tools)
        self.set_class(empty, "empty")
        self.set_class(len(shown) == 1, "lone")
        self.set_class(not self._open and len(shown) != 1, "closed")
        if empty or len(shown) == 1 or not self.is_mounted:
            return
        self._line.update(*group_line(shown, open_=self._open))


def group_line(rows: list[StepRow], *, open_: bool) -> tuple[Content, Content]:
    """A group's line: what its steps did; how many failed and how long it took."""
    tools = [row for row in rows if isinstance(row, ToolBlock)]
    chevron = ("▾ " if open_ else "▸ ", "dim")
    if len(tools) == 1:
        # One call and the thinking around it: the call's own line says more.
        what, how = tools[0].summary_parts()
        return Content.assemble(chevron, what), how
    ran = [row.tool_name for row in tools if row.ran]
    summary = activity_summary(ran) if ran else ("Thought" if not tools else "Nothing ran")
    what = Content.assemble(chevron, summary)
    how: list[tuple[str, str]] = []
    failed = sum(1 for row in tools if row.status == "error")
    if failed:
        how.append((f"  {failed} failed", "$error"))
    ends = [row.ended_at for row in tools if row.ended_at is not None]
    if tools and ends:
        took = max(ends) - min(row.started_at for row in tools)
        if took >= WORTH_TIMING:
            how.append((f"  {format_seconds(took)}", "dim"))
    return what, Content.assemble(*how)


@dataclass(frozen=True)
class Activity:
    """What the status line says: the step under way (its label and what it acts
    on), how long the turn has run, and the newest lines the step printed."""

    label: str
    what: str = ""
    since: float | None = None
    tail: str = ""
    thinking: bool = False


# Braille spinner frames, one every tenth of a second.
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TAIL_LINES = 5


def current_activity(
    rows: list[ToolBlock],
    thought: ReasoningBlock | None,
    *,
    since: float | None,
    cancelling: bool = False,
) -> Activity:
    """The status line's content: the newest step under way, else the thought
    still streaming, else just that the agent is working."""
    if cancelling:
        return Activity("Cancelling", since=since)
    running = next((row for row in reversed(rows) if row.status == "running"), None)
    if running is not None:
        tail = ""
        if running.output and time.monotonic() - running.started_at >= WORTH_TIMING:
            lines = [ln for ln in running.output.splitlines() if ln.strip()]
            tail = "\n".join(lines[-_TAIL_LINES:])
        return Activity(running.label, running.what, since, tail)
    if thought is not None and thought.under_way and thought.content_text.strip():
        return Activity("Thinking", thought.newest_line, since, thinking=True)
    return Activity("Working", since=since)


class LiveStatus(Static):
    """The line above the prompt while the agent works: a spinner, what it is
    doing, the turn's clock, and the newest lines of a step that prints."""

    DEFAULT_CSS = """
    LiveStatus {
        height: auto;
        max-height: 7;
        padding: 0 1;
        color: $text-muted;
        display: none;
    }
    LiveStatus.active {
        display: block;
    }
    """

    def __init__(self, activity: Callable[[], Activity | None], *, id: str | None = None) -> None:
        super().__init__("", id=id, markup=False)
        self._activity = activity
        self._frame = 0

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPIN)
        self.refresh_activity()

    def refresh_activity(self) -> None:
        activity = self._activity()
        self.set_class(activity is not None, "active")
        if activity is not None:
            width = self.content_size.width or None
            self.update(self.render_activity(activity, frame=_SPIN[self._frame], width=width))

    @staticmethod
    def render_activity(
        activity: Activity, *, frame: str = _SPIN[0], width: int | None = None
    ) -> Content:
        """The status line, then the step's newest lines. Within ``width`` each
        line is cut short rather than wrapped, and the clock always shows."""
        what = Content.assemble((f"{frame} ", "$accent"), (activity.label, "bold $text"))
        if activity.what:
            style = "italic" if activity.thinking else "$text"
            what = Content.assemble(what, "  ", (activity.what, style))
        clock = ""
        if activity.since is not None:
            clock = f"  {format_seconds(time.monotonic() - activity.since)}"
        if width and what.cell_length + len(clock) > width:
            what = what.truncate(max(width - len(clock), 1), ellipsis=True)
        lines = [Content.assemble(what, (clock, "dim"))]
        for line in activity.tail.splitlines():
            tail = Content.styled(f"  {line}", "dim")
            lines.append(tail.truncate(width, ellipsis=True) if width else tail)
        return Content("\n").join(lines)


_MARK = {"completed": ("✓ ", "$success"), "in_progress": ("▸ ", "$accent")}


class PlanView(Static):
    """The agent's plan, above the status line while it works through it."""

    DEFAULT_CSS = """
    PlanView {
        height: auto;
        padding: 0 1;
        color: $text-muted;
        display: none;
    }
    PlanView.active {
        display: block;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id, markup=False)

    def set_plan(self, todos: list[dict[str, Any]] | None) -> None:
        """Show ``todos``; hide the plan when there is none or it is all done."""
        active = bool(todos) and any(t.get("status") != "completed" for t in todos or ())
        self.set_class(active, "active")
        if not active:
            return
        parts: list[tuple[str, str] | str] = []
        for i, todo in enumerate(todos or ()):
            status = todo.get("status") or "pending"
            mark, mark_style = _MARK.get(status, ("○ ", "dim"))
            text_style = {"completed": "strike dim", "in_progress": "bold $text"}.get(status, "")
            parts.append(("\n" if i else "", ""))
            parts.append((mark, mark_style))
            parts.append((shorten_single_line(todo_text(todo), 100), text_style))
        self.update(Content.assemble(*parts))


_ARTIFACT_KIND = {"plot": "Plot", "report": "Report", "snapshot": "Snapshot"}


class ArtifactLine(Static):
    """A figure, image or report the agent made: a line of its own in the log,
    like what the agent says, which opens the file when clicked."""

    DEFAULT_CSS = """
    ArtifactLine {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
        color: $text-muted;
    }
    ArtifactLine:hover {
        background: $surface;
    }
    """

    class Opened(Message):
        """The user clicked the line: open its file."""

        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path

    def __init__(self, path: Path, *, kind: str | None, mime: str | None, caption: str) -> None:
        super().__init__("", markup=False)
        self.path = path
        image = (mime or "").startswith("image/")
        label = _ARTIFACT_KIND.get(kind or "", "Image" if image else "File")
        self.update(
            Content.assemble(
                ("▣ ", "$accent"),
                (label, "bold $text"),
                ("  ", ""),
                (shorten_single_line(caption, 120), "$text") if caption else ("", ""),
                ("  ", ""),
                (display_path(str(path)), "dim underline"),
            )
        )

    def on_click(self) -> None:
        self.post_message(self.Opened(self.path))
