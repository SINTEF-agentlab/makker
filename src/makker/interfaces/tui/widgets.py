"""Local Textual widgets for the Makker TUI."""

from __future__ import annotations

import ast
import json
import time
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt
from rich.text import Text
from textual import events
from textual.containers import Vertical
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Markdown, Static

from makker.agent.tool_output import is_interrupt_payload, normalize_tool_output
from makker.approval_preview import ApprovalCard
from makker.interfaces.tui.tool_display import (
    display_path,
    display_tool_body,
    julia_code_section,
    step_note,
)
from makker.juliakernel.text import render_terminal_output
from makker.paths import is_dated_session_id
from makker.preview import (
    TOOL_STREAM_RENDER_INTERVAL,
    TOOL_STREAM_TAIL_CAP,
    WORTH_TIMING,
    format_seconds,
    shorten,
    shorten_single_line,
    step_what,
)
from makker.tool_labels import tool_label

if TYPE_CHECKING:
    from textual.timer import Timer
    from textual.widgets._markdown import MarkdownStream

# Cap on the live-streamed buffer re-rendered per delta, so a long solve's output
# can't make each refresh cost grow without bound.
# Streamed tool output is re-rendered at most this often. Rendering runs the
# whole buffer through the terminal emulator and re-parses the markdown body,
# so doing it per delta makes a chatty tool freeze the UI.


def _link_friendly_parser() -> MarkdownIt:
    """A gfm-like Markdown parser that also renders ``file://`` links.

    Textual's default parser blocks the ``file:`` scheme, so a local file link
    shows as raw ``[text](file://…)`` instead of a clickable link. Allowing it
    lets a reported path open in the OS viewer (the click handler routes
    ``file://`` to the local opener); other unsafe schemes stay blocked.
    """
    md = MarkdownIt("gfm-like")
    default_validate = md.validateLink
    md.validateLink = lambda url: url.startswith("file:") or default_validate(url)
    return md


def display_session_id(session_id: str) -> str:
    """A session id as shown in the UI: dated ids in full, UUIDs shortened."""
    return session_id if is_dated_session_id(session_id) else session_id[:8]


class MessageBlock(Vertical):
    """Generic conversation card for user, assistant, system, and error messages."""

    DEFAULT_CSS = """
    MessageBlock {
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
        background: $surface;
    }

    MessageBlock.user {
        background: $surface-darken-1;
    }

    MessageBlock.assistant {
        background: $panel;
    }

    MessageBlock.reasoning {
        background: $surface-darken-1;
        border: solid $surface;
    }

    MessageBlock.reasoning .message-body {
        color: $text-muted;
    }

    MessageBlock.system {
        background: $surface;
    }

    MessageBlock.error {
        border: solid $error;
    }

    MessageBlock.welcome {
        background: $surface;
    }

    MessageBlock Markdown,
    MessageBlock .message-body {
        padding: 0;
        margin: 0;
        height: auto;
    }
    """

    def __init__(
        self,
        role_label: str,
        role_class: str,
        content: str = "",
        *,
        markdown: bool = False,
    ) -> None:
        super().__init__()
        self.add_class(role_class)
        self.border_title = role_label
        self._content = content
        self._markdown = markdown
        self._body_widget: Static | Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self):
        if self._markdown:
            yield Markdown(self._content, id="message-body", parser_factory=_link_friendly_parser)
            return
        yield Static(self._content, classes="message-body", id="message-body", markup=False)

    @property
    def content_text(self) -> str:
        """The message's raw text (markdown source for assistant replies)."""
        return self._content

    def on_mount(self) -> None:
        self._body_widget = self.query_one("#message-body")
        self.refresh_for_width()

    async def _ensure_body(self) -> None:
        if self._body_widget is not None:
            return
        if not self.is_mounted:
            await self._mounted_event.wait()
        self._body_widget = self.query_one("#message-body")

    def _get_markdown(self) -> Markdown:
        if self._body_widget is None:
            raise RuntimeError("MessageBlock body is not mounted yet")
        if not isinstance(self._body_widget, Markdown):
            raise TypeError("MessageBlock markdown stream requested for non-markdown body")
        return self._body_widget

    def _ensure_stream(self) -> MarkdownStream:
        if self._stream is None:
            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)

    async def append_content(self, content: str) -> None:
        if not content:
            return
        await self._ensure_body()
        if not self._markdown:
            await self.set_content(self._content + content)
            return
        self._content += content
        stream = self._ensure_stream()
        await stream.write(content)

    async def stop_stream(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
            await self._get_markdown().update(self._content)
            self.refresh_for_width()

    async def set_content(self, content: str) -> None:
        self._content = content
        await self._ensure_body()
        if isinstance(self._body_widget, Markdown):
            if self._stream is not None:
                await self._stream.stop()
                self._stream = None
            await self._body_widget.update(content)
        else:
            self._body_widget.update(content)
        self.refresh_for_width()


class StepLine(Widget):
    """One line of the log that opens on what is behind it: what it says (cut
    short with an ellipsis when the terminal is narrow) and how it went (a
    failure, a time), which always stays in view."""

    DEFAULT_CSS = """
    StepLine {
        height: 1;
        width: 1fr;
        color: $text-muted;
    }
    StepLine:hover {
        background: $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._what = Content("")
        self._how = Content("")

    @property
    def content(self) -> Content:
        return Content.assemble(self._what, self._how)

    def update(self, what: Content, how: Content) -> None:
        self._what, self._how = what, how
        self.refresh()

    def render(self) -> Content:
        room = self.size.width - self._how.cell_length
        what = self._what
        if self.size.width and what.cell_length > room:
            what = what.truncate(max(room, 1), ellipsis=True)
        return Content.assemble(what, self._how)


class StepRow(Vertical):
    """One step of the agent's work as a line: what it did, with what it showed
    behind the line (click it, or Ctrl+O for every step).

    A step still under way is not shown here: the status line above the prompt
    carries it until it ends. ``StepGroup`` (``tui.steps``) holds the rows of one
    stretch of work and sums them up in its own line.
    """

    DEFAULT_CSS = """
    StepRow {
        height: auto;
        margin: 0;
        padding: 0;
    }
    StepRow.under-way {
        display: none;
    }
    StepRow > .step-body {
        height: auto;
        margin: 0 0 1 2;
        padding: 0;
        color: $text-muted;
    }
    StepRow Markdown {
        padding: 0;
        margin: 0;
        link-color: $text-muted;
        link-style: underline;
        link-color-hover: $accent;
        link-style-hover: underline;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._expanded = False  # the line is open, showing what is behind it
        self._verbose = False  # Ctrl+O: everything, in full
        self._line = StepLine()
        self._body_widget: Markdown | Static | None = None
        self.started_at = time.monotonic()
        self.ended_at: float | None = None

    def compose(self):
        yield self._line

    async def on_mount(self) -> None:
        await self._refresh()

    @property
    def under_way(self) -> bool:
        return False

    @property
    def took(self) -> float | None:
        """Seconds from start to end, once ended."""
        return None if self.ended_at is None else self.ended_at - self.started_at

    async def on_click(self, event: events.Click) -> None:
        if event.widget is not None and self._line in event.widget.ancestors_with_self:
            await self.set_expanded(not self._expanded)

    async def set_expanded(self, expanded: bool, *, verbose: bool | None = None) -> None:
        """Open the line (what the step showed) or close it; ``verbose`` shows it in full."""
        if verbose is not None:
            self._verbose = verbose
        self._expanded = expanded
        await self._refresh()

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)

    def line_parts(self) -> tuple[Content, Content]:
        """The line: what the step did, and how it went."""
        raise NotImplementedError

    def line_content(self) -> Content:
        return Content.assemble(*self.line_parts())

    async def body_widget(self) -> Markdown | Static:
        raise NotImplementedError

    async def _render_body(self) -> None:
        raise NotImplementedError

    async def _refresh(self) -> None:
        self.set_class(self.under_way, "under-way")
        if not self.is_mounted:
            return
        self._line.update(*self.line_parts())
        if self._expanded and not self.under_way:
            if self._body_widget is None:
                self._body_widget = await self.body_widget()
                self._body_widget.add_class("step-body")
                await self.mount(self._body_widget)
            self._body_widget.display = True
            await self._render_body()
        elif self._body_widget is not None:
            self._body_widget.display = False
        group = self.parent.parent if self.parent is not None else None
        if group is not None and hasattr(group, "refresh_view"):
            group.refresh_view()


def _chevron(open_: bool) -> tuple[str, str]:
    return ("▾ " if open_ else "▸ ", "dim")


class ReasoningBlock(StepRow):
    """A stretch of the model's reasoning, as a step: "Thought" and its first
    line, the whole text behind it. While it streams it is on the status line."""

    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__()
        self._content = ""
        self._finished = False
        self._verbose = verbose

    @property
    def content_text(self) -> str:
        return self._content

    @property
    def under_way(self) -> bool:
        return not self._finished

    @property
    def newest_line(self) -> str:
        lines = [line.strip() for line in self._content.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""

    async def append_content(self, content: str) -> None:
        if content:
            self._content += content

    async def finish(self, *, expanded: bool | None = None) -> None:
        """The thought is complete: show its line (open in verbose mode)."""
        if not self._finished:
            self._finished = True
            self.ended_at = time.monotonic()
        expanded = self._verbose if expanded is None else expanded
        await self.set_expanded(expanded, verbose=expanded)

    def line_parts(self) -> tuple[Content, Content]:
        first = next((ln.strip() for ln in self._content.strip().splitlines() if ln.strip()), "")
        what = Content.assemble(
            _chevron(self._expanded), "Thought  ", (first.replace("**", ""), "italic")
        )
        took = self.took
        how = f"  {format_seconds(took)}" if took is not None and took >= WORTH_TIMING else ""
        return what, Content.styled(how, "dim")

    async def body_widget(self) -> Static:
        return Static("", markup=False)

    async def _render_body(self) -> None:
        if isinstance(self._body_widget, Static):
            self._body_widget.update(self._content.strip())


class WelcomeBlock(MessageBlock):
    """Landing card shown when the TUI starts or the visible log is cleared."""

    def __init__(self, *, simulator_label: str) -> None:
        super().__init__(
            "Welcome",
            "welcome",
            _render_welcome_message(simulator_label=simulator_label),
            markdown=True,
        )


class ApprovalBlock(Vertical):
    """Card for a pending approval: the tool and what it will do.

    Controls (approve / reject / reply) live in the approval menu at the prompt,
    so the request is shown once here and resolved in one place below.
    """

    DEFAULT_CSS = """
    ApprovalBlock {
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
        background: $surface;
    }

    ApprovalBlock Markdown {
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, card: ApprovalCard) -> None:
        super().__init__()
        self._card = card
        self._body_widget: Markdown | None = None
        self.border_title = card.title

    @property
    def tool_name(self) -> str:
        return self._card.tool_name

    def compose(self):
        yield Markdown(self._card.body, id="approval-body", parser_factory=_link_friendly_parser)

    def on_mount(self) -> None:
        self._body_widget = self.query_one("#approval-body", Markdown)
        self.border_subtitle = "awaiting review"
        self.refresh_for_width()

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)


def _render_welcome_message(*, simulator_label: str) -> str:
    # Keep it short: the session id, model, and approval mode are all in the status
    # bar, the prompt placeholder already invites a question, and the footer lists
    # the keys. The card only adds the two non-obvious controls and points at /help.
    return (
        f"**Makker** is ready for **{simulator_label}**.\n"
        "**Ctrl+C** interrupts a turn (twice to exit) · `/help` for commands."
    )


class PromptGuide(Static):
    """Dynamic helper line under the prompt for commands, history, and activity."""

    DEFAULT_CSS = """
    PromptGuide {
        height: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self._message = ""
        self._activity = "ready"

    @property
    def message(self) -> str:
        return self._message

    def set_message(self, message: str) -> None:
        self._message = message
        self.refresh()

    def set_activity(self, activity: str) -> None:
        self._activity = activity
        self.refresh()

    def render(self) -> Text:
        text = Text()
        if self._activity:
            text.append(self._activity, style=_status_style(self._activity))
        if self._message:
            if self._activity:
                text.append(" · ", style="dim")
            text.append(self._message, style="dim")
        return text


class StatusBar(Static):
    """Compact bottom status bar with session context and live state."""

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        simulator_label: str,
        session_id: str,
        model_label: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, markup=False)
        self._simulator_label = simulator_label
        self._session_id = session_id
        self._model_label = model_label
        self._pending_count = 0
        self._tool_toggle_available = False
        self._tools_expanded = False
        self._approval_mode_label = "default"
        self._context_label: str | None = None
        self._context_alert = "ok"
        self._warming = False

    def set_state(
        self,
        *,
        pending_count: int = 0,
        tool_toggle_available: bool = False,
        tools_expanded: bool = False,
        approval_mode_label: str = "default",
    ) -> None:
        self._pending_count = pending_count
        self._tool_toggle_available = tool_toggle_available
        self._tools_expanded = tools_expanded
        self._approval_mode_label = approval_mode_label
        self.refresh()

    def set_model(self, model_label: str | None) -> None:
        self._model_label = model_label
        self.refresh()

    def set_context(self, label: str | None, *, alert: str = "ok") -> None:
        self._context_label = label
        self._context_alert = alert
        self.refresh()

    def set_warming(self, warming: bool) -> None:
        """Show or clear the Julia warm-up chip.

        It lives here, not in the bottom activity line, so running a turn or
        a pending approval never replaces it while the kernel still compiles.
        """
        self._warming = warming
        self.refresh()

    def render(self) -> Text:
        text = Text()
        text.append("Makker", style="bold")
        text.append(" · ", style="dim")
        text.append(self._simulator_label, style="bold cyan")

        if self._model_label:
            text.append(" · ", style="dim")
            text.append(shorten(self._model_label, 28), style="dim")

        text.append(" · ", style="dim")
        text.append(display_session_id(self._session_id), style="dim")

        if self._approval_mode_label:
            text.append(" · ", style="dim")
            text.append(self._approval_mode_label, style="cyan")

        if self._context_label:
            style = {"warn": "yellow", "high": "bold red"}.get(self._context_alert, "dim")
            text.append(" · ", style="dim")
            text.append(self._context_label, style=style)

        if self._warming:
            text.append(" · ", style="dim")
            text.append("warming Julia", style="yellow")

        if self._pending_count:
            label = "approval" if self._pending_count == 1 else "approvals"
            text.append(" · ", style="dim")
            text.append(f"{self._pending_count} {label}", style="yellow")

        # Live state (turn status, warm-up, Ctrl+G cancel) lives in the bottom
        # bar next to the input; see ``_activity_label`` / the prompt guide.
        if self._tools_expanded:
            text.append(" · ", style="dim")
            text.append("verbose", style="cyan")
        elif self._tool_toggle_available:
            text.append(" · ", style="dim")
            text.append("details available", style="dim")

        return text


# A finished step's status, as its line says it. Success says nothing.
_STATUS_WORD = {
    "error": ("failed", "$error"),
    "rejected": ("declined", "$warning"),
    "cancelled": ("stopped", "$warning"),
    "responded": ("answered by you", "$accent"),
}


class ToolBlock(StepRow):
    """One tool call as a step: its line says what it acted on and how it went,
    and opens on its code, edit or written file, and output."""

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        expanded: bool = False,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args or {}
        self._tool_call_id = tool_call_id
        self._status = "running"
        self._output = ""
        self._streamed = ""
        self._expanded = expanded
        self._verbose = expanded
        self._is_error = False
        self._reject_reason: str | None = None
        self._stream_refresh_timer: Timer | None = None
        # The call has begun to run; until then the model is still writing it.
        self.started = False

    def mark_started(self) -> None:
        """The call runs from now: its clock starts here."""
        self.started = True
        self.started_at = time.monotonic()

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def label(self) -> str:
        return tool_label(self._tool_name)

    @property
    def what(self) -> str:
        return step_what(self._args)

    @property
    def tool_call_id(self) -> str | None:
        return self._tool_call_id

    @property
    def expandable(self) -> bool:
        return not self.under_way

    @property
    def has_output(self) -> bool:
        return bool(self._output or self._streamed)

    @property
    def output(self) -> str:
        """What the call has printed so far (terminal-rendered), or its result."""
        return self._output

    @property
    def status(self) -> str:
        return self._status

    @property
    def under_way(self) -> bool:
        return self._status in ("running", "approval")

    @property
    def ran(self) -> bool:
        """Whether the call ran: one declined, stopped or answered did not."""
        return self._status in ("success", "error")

    def set_args(self, args: dict[str, Any] | None) -> None:
        if args:
            self._args = args

    def line_parts(self) -> tuple[Content, Content]:
        what, how = self.summary_parts()
        return Content.assemble(_chevron(self._expanded), what), how

    def summary_parts(self) -> tuple[Content, Content]:
        """What the call acted on, and how it went: its line without the chevron."""
        what = Content.assemble(f"{self.label}  ", (shorten_single_line(self.what, 200), "$text"))
        how: list[tuple[str, str]] = []
        if self._status == "success":
            note = step_note(self._tool_name, self._args, self._output)
            if note:
                how.append((f"  {note}", "dim"))
        word = _STATUS_WORD.get(self._status)
        if word is not None:
            text, style = word
            if self._reject_reason:
                text = f"{text}: {shorten_single_line(self._reject_reason, 72)}"
            how.append((f"  {text}", style))
        took = self.took
        if took is not None and took >= WORTH_TIMING and self.ran:
            how.append((f"  {format_seconds(took)}", "dim"))
        return what, Content.assemble(*how)

    async def body_widget(self) -> Markdown:
        return Markdown("", parser_factory=_link_friendly_parser)

    async def _render_body(self) -> None:
        if isinstance(self._body_widget, Markdown):
            await self._body_widget.update(self.detail())

    def detail(self) -> str:
        """What is behind the line: the call itself, then what it returned."""
        return _step_detail(
            self._tool_name,
            self._args,
            output=self._output,
            verbose=self._verbose,
            is_error=self._is_error,
            reason=self._reject_reason,
        )

    async def set_running(self) -> None:
        self._status = "running"
        self._reject_reason = None
        self.started_at = time.monotonic()
        await self._refresh()

    async def set_pending_approval(self) -> None:
        self._status = "approval"
        await self._refresh()

    async def _end(self, status: str, reason: str | None = None) -> None:
        self._stop_stream_refresh()
        self._status = status
        self._reject_reason = reason or None
        self.ended_at = time.monotonic()
        await self._refresh()

    async def set_rejected(self, reason: str | None = None) -> None:
        await self._end("rejected", reason)

    async def set_responded(self, message: str | None = None) -> None:
        """The user answered instead of running the tool."""
        await self._end("responded", message)

    async def set_cancelled(self, reason: str | None = None) -> None:
        await self._end("cancelled", None if reason == "turn cancelled" else reason)

    async def append_output(self, delta: str) -> None:
        if not delta:
            return
        # Keep only the tail so each render stays bounded. Rendering tolerates a
        # truncated escape at the cut; it sits far up in scrollback anyway.
        self._streamed = (self._streamed + delta)[-TOOL_STREAM_TAIL_CAP:]
        self._status = "running"
        self.started = True
        if not self.is_running:
            # No message pump (plain widget under unit test): render in line.
            self._render_streamed()
            return
        if self._stream_refresh_timer is None:
            self._stream_refresh_timer = self.set_timer(
                TOOL_STREAM_RENDER_INTERVAL, self._flush_streamed
            )

    def _render_streamed(self) -> None:
        # The raw buffer as a terminal shows it, so carriage returns and cursor
        # moves (ProgressMeter/Jutul bars) collapse to one updating line, matching
        # the final EvalResult.output. The status line reads it from here.
        self._output = render_terminal_output(self._streamed)

    async def _flush_streamed(self) -> None:
        self._stream_refresh_timer = None
        if self._status != "running" or not self._streamed:
            return  # a final result landed first and already rendered
        self._render_streamed()

    def _stop_stream_refresh(self) -> None:
        if self._stream_refresh_timer is not None:
            self._stream_refresh_timer.stop()
            self._stream_refresh_timer = None

    async def set_result(self, content: str, *, is_error: bool = False) -> None:
        self._output = normalize_tool_output(content)
        self._streamed = ""
        self._is_error = is_error
        if is_error and is_interrupt_payload(self._output):
            self._stop_stream_refresh()
            self._status = "approval"
            await self._refresh()
            return
        await self._end("error" if is_error else "success")


def _status_style(message: str) -> str:
    lowered = message.lower()
    if "approval" in lowered:
        return "bold yellow"
    if (
        "thinking" in lowered
        or "running" in lowered
        or "resuming" in lowered
        or "warming" in lowered
    ):
        return "yellow"
    if "error" in lowered or "failed" in lowered:
        return "red"
    return "green"


def _fenced(text: str, language: str = "") -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{language}\n{text}\n{fence}"


def _step_detail(
    tool_name: str,
    args: dict[str, Any],
    *,
    output: str,
    verbose: bool,
    is_error: bool,
    reason: str | None,
) -> str:
    """The markdown behind a step's line: an edit as its diff, a written file as
    its contents, else the code and what the call returned (a preview, or all
    of it with ``verbose``)."""
    parts: list[str] = []
    path = str(args.get("file_path") or args.get("path") or "")
    if tool_name == "edit_file" and (args.get("old_string") or args.get("new_string")):
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") or "")
        diff = [f"- {ln}" for ln in old.splitlines()] + [f"+ {ln}" for ln in new.splitlines()]
        parts.append(f"{display_path(path)}\n\n{_fenced(chr(10).join(diff), 'diff')}")
        if is_error and output:
            parts.append(_fenced(output))
    elif tool_name == "write_file" and args.get("content") is not None:
        language = "julia" if path.endswith(".jl") else ""
        parts.append(f"{display_path(path)}\n\n{_fenced(str(args['content']), language)}")
        if is_error and output:
            parts.append(_fenced(output))
    elif output:
        parts.append(display_tool_body(tool_name, args, output=output, expanded=verbose))
    else:
        request = julia_code_section(tool_name, args) or _render_request_preview(tool_name, args)
        parts.append(request)
    if reason:
        parts.append(f"> {reason}")
    return "\n\n".join(part for part in parts if part)


def _render_request_preview(tool_name: str, args: dict[str, Any]) -> str:
    return f"_{_format_tool_display(tool_name, args)}_"


def _format_tool_display(tool_name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"{tool_name}()"
    rendered_args = ", ".join(
        f"{key}={shorten_single_line(str(value), 30)}" for key, value in list(args.items())[:3]
    )
    if len(args) > 3:
        rendered_args += ", ..."
    return f"{tool_name}({rendered_args})"


def parse_todos(value: Any) -> list[dict[str, Any]] | None:
    """The plan's items from ``write_todos`` arguments or reply, or None when
    they don't parse. Entries that are not items are dropped."""
    items = _parse_todo_items(value)
    if items is None:
        return None
    return [item for item in items if isinstance(item, dict)]


def _parse_todo_items(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        todos = value.get("todos")
        if isinstance(todos, list):
            return todos
        return None

    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return []

    for raw_candidate in (candidate, _extract_serialized_collection(candidate)):
        if raw_candidate is None:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw_candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("todos"), list):
                return parsed["todos"]
    return None


def _extract_serialized_collection(text: str) -> str | None:
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return None

    start = min(starts)
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    end = text.rfind(closer)
    if end <= start:
        return None
    return text[start : end + 1]


def todo_text(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content")
        if isinstance(content, str):
            return content
    return str(item)
