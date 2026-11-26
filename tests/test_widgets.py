"""Tests for the TUI's step rows, groups, status line and plan."""

# The lines use typographic dashes and minus signs on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import time

from textual.app import App

from makker.interfaces.tui.steps import (
    ConversationLog,
    LiveStatus,
    PlanView,
    StepGroup,
    current_activity,
)
from makker.interfaces.tui.tool_display import step_note
from makker.interfaces.tui.widgets import MessageBlock, ReasoningBlock, ToolBlock
from makker.preview import step_what
from makker.tool_labels import activity_summary


def test_a_step_says_what_it_acted_on_and_how_it_went() -> None:
    assert (
        step_what({"code": "x = 1\ny", "description": "Run the base case"}) == "Run the base case"
    )
    assert step_what({"code": '\n  include("a.jl")\nmore'}) == 'include("a.jl")'
    assert step_what({"pattern": "setup_well", "path": "/a/b/c"}) == "setup_well"
    assert step_what({"file_path": "/Users/me/pkg/examples/wells.jl"}) == "examples/wells.jl"
    assert step_what({"nested": {"a": 1}}) == ""

    assert step_note("read_file", {}, "@@ lines 1-74 of 74 @@\na") == "74 lines"
    assert step_note("read_file", {}, "@@ lines 101-200 of 222 | next offset 200 @@\na") == (
        "lines 101–200"
    )
    assert step_note("read_file", {"file_path": "plot.png"}, "") == ""  # an image
    assert step_note("grep", {}, "/a.jl:\n  12: x\n  40: y") == "2 matches"
    assert step_note("grep", {}, "/a.jl\n/b.jl") == "2 files"
    assert step_note("grep", {}, "No matches found") == "no matches"
    assert step_note("glob", {}, "['/a.jl', '/b.jl']") == "2 files"
    assert step_note("ls", {}, "No files found") == "0 entries"
    assert step_note("write_file", {"content": "a\nb\nc"}, "Updated file") == "3 lines"
    assert step_note("edit_file", {"old_string": "a", "new_string": "b\nc"}, "ok") == "+2 −1"
    assert step_note("record_attempt", {}, "attempt #2 (parent #1) · rmse=0.1 · id=ab") == (
        "attempt #2 (parent #1) · rmse=0.1"
    )
    assert step_note("run_julia", {"code": "1"}, "2") == ""


def test_a_stretch_of_work_reads_as_what_it_did() -> None:
    names = ["read_file", "grep", "read_file", "run_julia", "pick_horizon", "pick_horizon"]
    assert activity_summary(names) == (
        "Read 2 files, searched, ran Julia, used pick_horizon 2 times"
    )


async def test_tool_block_streams_like_a_terminal_and_keeps_the_result() -> None:
    """Streamed deltas are rendered like a terminal: a progress bar that
    overwrites itself with carriage returns collapses to one line, matching the
    kernel's final output; the result then replaces the stream."""
    block = ToolBlock("run_julia", {"code": "run()"}, tool_call_id="call-1")
    await block.append_output("Progress   0%|        |\r")
    await block.append_output("Progress  50%|####    |\r")
    await block.append_output("Progress 100%|########|\n")
    assert block.output == "Progress 100%|########|"
    assert block.under_way

    await block.set_result("Progress 100%|########|\n→ 42\n")
    assert block._streamed == ""
    assert block.status == "success" and not block.under_way
    assert "→ 42" in block.output


async def test_tool_block_coalesces_streamed_renders() -> None:
    """A mounted block renders streamed output on a timer, so a chatty tool
    can't re-render per delta; a final result cancels any pending render."""

    class _Host(App[None]):
        def compose(self):
            yield ConversationLog(id="log")

    app = _Host()
    async with app.run_test() as pilot:
        block = ToolBlock("run_julia", {"code": "run()"}, tool_call_id="call-3")
        await app.query_one("#log").mount(block)
        await pilot.pause()

        await block.append_output("Progress   0%|        |\r")
        await block.append_output("Progress  50%|####    |\r")
        assert block.output == ""  # not yet: the render is pending
        assert block.has_output  # streamed text still counts as output

        await block._flush_streamed()
        assert block.output.count("Progress") == 1 and "50%" in block.output

        await block.append_output("Progress 100%|########|\n")
        await block.set_result("Progress 100%|########|\n")
        await block._flush_streamed()  # a late render after the result is a no-op
        assert "100%" in block.output


async def test_a_step_line_says_how_it_went_and_opens_on_what_it_did() -> None:
    class _Host(App[None]):
        def compose(self):
            yield ConversationLog(id="log")

    app = _Host()
    async with app.run_test() as pilot:
        edit = ToolBlock(
            "edit_file", {"file_path": "/ws/v/a.jl", "old_string": "x = 1", "new_string": "x = 2"}
        )
        julia = ToolBlock("run_julia", {"code": "sqrt(-1)"})
        await app.query_one("#log").mount(edit, julia)
        await edit.set_result("Successfully replaced 1 instance")
        await julia.set_result("DomainError with -1.0", is_error=True)
        await pilot.pause()

        assert edit.line_content().plain == "▸ Edit  v/a.jl  +1 −1"
        assert julia.line_content().plain == "▸ Julia  sqrt(-1)  failed"

        # Opened, an edit shows its change and a failed run its code and error.
        await edit.set_expanded(True)
        assert "- x = 1\n+ x = 2" in edit.detail()
        assert "sqrt(-1)" in julia.detail() and "DomainError" in julia.detail()

        declined = ToolBlock("execute", {"command": "rm -rf old"})
        await app.query_one("#log").mount(declined)
        await declined.set_rejected("keep [the] old run")
        assert declined.line_content().plain.endswith("declined: keep [the] old run")
        assert not declined.ran


async def test_a_group_shows_its_finished_steps_as_one_line() -> None:
    class _Host(App[None]):
        def compose(self):
            yield ConversationLog(id="log")

    app = _Host()
    async with app.run_test() as pilot:
        log = app.query_one("#log", ConversationLog)
        group = StepGroup()
        await log.mount(group)
        thought = ReasoningBlock()
        await group.add(thought)
        await thought.append_content("Look at the examples first.")
        await pilot.pause()
        # Nothing finished yet: the status line carries what is under way.
        assert group.has_class("empty")

        await thought.finish()
        # A thought alone waits for what follows, so its line isn't replaced
        # by a summary a moment later.
        assert group.has_class("empty")

        reads = [ToolBlock("read_file", {"file_path": f"/x/{n}.jl"}) for n in "ab"]
        boom = ToolBlock("run_julia", {"code": "boom"})
        for row in (*reads, boom):
            await group.add(row)
        await reads[0].set_result("@@ lines 1-3 of 3 @@\nx")
        assert not group.has_class("empty")  # the thought and the read
        # One call and the thinking around it: the call's own line says more.
        assert group._line.content.plain == "▸ Read  /x/a.jl  3 lines"

        await reads[1].set_result("@@ lines 1-3 of 3 @@\nx")
        await boom.set_result("ERROR: boom", is_error=True)
        await pilot.pause()
        assert not group.has_class("lone") and group.has_class("closed")
        assert group._line.content.plain == "▸ Read 2 files, ran Julia  1 failed"

        await group.toggle()
        assert not group.has_class("closed")

        # Whatever is said next closes the stretch of work.
        await log.mount(MessageBlock("Assistant", "assistant", "Done."))
        assert not group.live

        # A stretch of one step is just that step's line.
        single = StepGroup()
        await log.mount(single)
        row = ToolBlock("run_julia", {"code": "1 + 1"})
        await single.add(row)
        await row.set_result("2")
        assert single.has_class("lone") and not single.has_class("empty")


async def test_the_status_line_says_what_is_under_way() -> None:
    started = time.monotonic() - 65
    julia = ToolBlock("run_julia", {"code": "simulate()", "description": "Run the base case"})
    await julia.append_output("Step 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n")
    # Too soon for its output: a quick step would only flash it.
    assert current_activity([julia], None, since=started).tail == ""
    julia.started_at -= 3
    activity = current_activity([julia], None, since=started)
    assert (activity.label, activity.what) == ("Julia", "Run the base case")
    assert activity.tail == "Step 2\nStep 3\nStep 4\nStep 5\nStep 6"
    text = LiveStatus.render_activity(activity, frame="*").plain
    assert text.startswith("* Julia  Run the base case  1:05\n  Step 2")

    await julia.set_result("done")
    thought = ReasoningBlock()
    await thought.append_content("First idea.\nThen check the wells")
    activity = current_activity([julia], thought, since=started)
    assert (activity.label, activity.what) == ("Thinking", "Then check the wells")
    assert current_activity([julia], None, since=None).label == "Working"
    assert current_activity([], None, since=None, cancelling=True).label == "Cancelling"


async def test_the_plan_shows_while_something_is_left_on_it() -> None:
    class _Host(App[None]):
        def compose(self):
            yield PlanView(id="plan")

    app = _Host()
    async with app.run_test():
        plan = app.query_one(PlanView)
        plan.set_plan(
            [
                {"content": "Write the script", "status": "completed"},
                {"content": "Run it", "status": "in_progress"},
                {"content": "Plot it", "status": "pending"},
            ]
        )
        assert plan.has_class("active")
        assert plan.content.plain == "✓ Write the script\n▸ Run it\n○ Plot it"
        plan.set_plan([{"content": "Run it", "status": "completed"}])
        assert not plan.has_class("active")
        plan.set_plan(None)
        assert not plan.has_class("active")
