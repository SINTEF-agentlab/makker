"""Tests for the HTML transcript renderer."""

from __future__ import annotations

from makker.lab.fakes import make_event
from makker.transcript import render_html


def test_render_html_is_complete_document(snapshot) -> None:
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "jutuldarcy"}),
        make_event(2, "message_user", {"content": "what is 2+2?"}),
        make_event(3, "session_end", {"session_id": "abc"}),
    ]
    html = render_html(events)
    assert html.startswith("<!doctype html>")
    assert html == snapshot


def test_render_html_inlines_images_with_artifact_dirs(tmp_path) -> None:
    # Given the run's artifact dirs, the transcript embeds plot images as base64
    # data URIs so it shows them wherever it's opened (a sidecar inside artifacts/,
    # a standalone download). Without the dirs it keeps the recorded relative path,
    # which still resolves beside the file in a --bundle zip.
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n fake png")
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "jutuldarcy"}),
        make_event(
            2,
            "artifact",
            {"path": "artifacts/plot.png", "mime": "image/png", "caption": "fig", "format": "png"},
        ),
    ]
    embedded = render_html(events, artifact_dirs=[tmp_path / "artifacts", tmp_path])
    assert "data:image/png;base64," in embedded
    assert 'src="artifacts/plot.png"' not in embedded
    assert 'src="artifacts/plot.png"' in render_html(events)  # fallback keeps the path


def test_render_html_does_not_inline_files_outside_the_artifact_dirs(tmp_path) -> None:
    # A recorded artifact path that escapes the run dirs (a traversal or absolute path)
    # must not be read and base64-embedded: the transcript only inlines files contained
    # in the session's own artifact dirs (mirrors the server's _resolve_artifact guard).
    import base64

    run = tmp_path / "run"
    (run / "artifacts").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"TOP SECRET")
    events = [
        make_event(
            1,
            "artifact",
            {"path": "artifacts/../../secret.txt", "mime": "image/png", "caption": "x"},
        ),
    ]
    html = render_html(events, artifact_dirs=[run / "artifacts", run])
    assert "data:image/png;base64," not in html  # nothing inlined
    assert base64.standard_b64encode(b"TOP SECRET").decode() not in html  # bytes not leaked


def test_render_full_turn(snapshot) -> None:
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "jutuldarcy"}),
        make_event(2, "message_user", {"content": "what is 2+2?"}),
        make_event(
            3,
            "tool_call",
            {"id": "call-1", "name": "run_julia", "args": {"code": "2+2"}},
        ),
        make_event(
            4,
            "tool_result",
            {"tool_call_id": "call-1", "name": "run_julia", "content": "4"},
        ),
        make_event(5, "message_assistant", {"content": "The answer is 4."}),
        make_event(6, "session_end", {"session_id": "abc"}),
    ]
    assert render_html(events) == snapshot


def test_render_escapes_html_in_payload(snapshot) -> None:
    events = [
        make_event(1, "message_user", {"content": "<script>alert(1)</script>"}),
    ]
    html = render_html(events)
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert html == snapshot


def test_render_escapes_raw_html_inside_markdown_message() -> None:
    """Markdown-looking prose with raw HTML smuggled in: the markdown renders,
    but the raw ``<script>`` is escaped to text, not passed through."""
    events = [
        make_event(
            1,
            "message_assistant",
            {"content": "## Findings\n\n<script>alert('x')</script>\n\nDone."},
        ),
    ]
    html = render_html(events)
    assert "<h2>Findings</h2>" in html  # markdown still works
    assert "<script>alert('x')</script>" not in html  # raw HTML did not pass through
    assert "&lt;script&gt;" in html


def test_filter_chip_counts_reflect_rendered_cards() -> None:
    """A tool call and its result render as one merged card, so the Tools chip
    should count 1, not 2 (call + result). Same for an approval round-trip."""
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "battmo"}),
        make_event(2, "tool_call", {"id": "c1", "name": "run_julia", "args": {"code": "1+1"}}),
        make_event(3, "tool_result", {"tool_call_id": "c1", "name": "run_julia", "content": "2"}),
        make_event(4, "hitl_request", {"interrupt_id": "i1", "value": {"action_requests": []}}),
        make_event(5, "hitl_response", {"interrupt_id": "i1", "payload": {"decisions": []}}),
    ]
    html = render_html(events)
    assert 'Tools <span class="count">1</span>' in html
    assert 'Approval <span class="count">1</span>' in html


def test_render_html_drops_internal_telemetry_events() -> None:
    """model_usage / eval_target are trace telemetry, not conversation: they must
    not render as cards or leak into the filter chips."""
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "battmo"}),
        make_event(2, "message_user", {"content": "hi"}),
        make_event(3, "model_usage", {"input_tokens": 10, "output_tokens": 5}),
        make_event(4, "eval_target", {"expected": "42"}),
        make_event(5, "message_assistant", {"content": "hello"}),
    ]
    html = render_html(events)
    assert "model_usage" not in html
    assert "eval_target" not in html
    assert "Raw payload" not in html  # no fall-through card for the telemetry kinds
    assert "hello" in html  # the real conversation still renders


def test_render_html_has_csp_with_script_hash() -> None:
    """The transcript is opened from disk with untrusted content in it, so it
    ships a CSP: only its own inline script runs (matched by hash), images are
    local/data-only, and there is no remote egress."""
    import base64
    import hashlib

    from makker.transcript.html import _SCRIPT

    head = render_html([make_event(1, "message_user", {"content": "hi"})]).split("</head>")[0]
    want = base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode()
    assert '<meta http-equiv="Content-Security-Policy"' in head
    assert f"script-src 'sha256-{want}'" in head  # the page's own script, by hash
    assert "img-src 'self' data: file:" in head  # local + inlined, never remote
    assert "default-src 'none'" in head
    assert "https:" not in head and "http:" not in head  # no remote origin allowed


def test_render_assistant_markdown(snapshot) -> None:
    events = [
        make_event(
            1,
            "message_assistant",
            {"content": "### Heading\n\n1. **Create wells**\n\n```julia\nProd = 1\n```"},
        ),
    ]
    html = render_html(events)
    assert 'class="highlight"' in html
    assert html == snapshot


def test_render_tool_result_markdown(snapshot) -> None:
    events = [
        make_event(
            1,
            "tool_call",
            {"id": "call-1", "name": "read_file", "args": {"file_path": "SKILL.md"}},
        ),
        make_event(
            2,
            "tool_result",
            {
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "# Setting up wells\n\n## When to use\n\nUse this skill.",
            },
        ),
    ]
    assert render_html(events) == snapshot


def test_render_read_file_numbered_markdown_with_julia(snapshot) -> None:
    content = (
        "     1\t# Setting up wells\n"
        "     2\t\n"
        "     3\t```julia\n"
        "     4\tProd = setup_vertical_well(domain, 1, 1, name = :Producer)\n"
        "     5\t```\n"
    )
    events = [
        make_event(
            1,
            "tool_call",
            {"id": "call-1", "name": "read_file", "args": {"file_path": "SKILL.md"}},
        ),
        make_event(
            2,
            "tool_result",
            {
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": content,
            },
        ),
    ]
    html = render_html(events)
    assert 'class="highlight"' in html
    assert html == snapshot


def test_render_todos_args(snapshot) -> None:
    events = [
        make_event(
            1,
            "tool_call",
            {
                "name": "write_todos",
                "args": {
                    "todos": [
                        {"content": "Inspect docs", "status": "completed"},
                        {"content": "Explain pattern", "status": "in_progress"},
                    ]
                },
            },
        ),
    ]
    html = render_html(events)
    assert "todo-list" in html
    assert html == snapshot


def test_filter_groups_merge_tools(snapshot) -> None:
    events = [
        make_event(1, "tool_call", {"name": "run_julia", "args": {}}),
        make_event(2, "tool_result", {"name": "run_julia", "content": "4"}),
    ]
    html = render_html(events)
    assert 'data-filter="tool_call"' not in html
    assert html == snapshot


def test_render_skips_empty_assistant(snapshot) -> None:
    events = [
        make_event(1, "message_assistant", {"content": ""}),
        make_event(2, "message_assistant", {"content": "hello"}),
    ]
    html = render_html(events)
    assert html.count('data-kind="message_assistant"') == 1
    assert html == snapshot


def test_render_reasoning_in_details(snapshot) -> None:
    events = [
        make_event(1, "message_reasoning", {"content": "thinking step by step"}),
    ]
    html = render_html(events)
    assert "<details>" in html
    assert html == snapshot


def test_render_hitl_events(snapshot) -> None:
    events = [
        make_event(
            1,
            "hitl_request",
            {
                "interrupt_id": "abc",
                "value": {"action_requests": [{"name": "execute", "args": {"command": "ls -la"}}]},
            },
        ),
        make_event(
            2,
            "hitl_response",
            {
                "interrupt_id": "abc",
                "payload": {"decisions": [{"type": "approve"}]},
            },
        ),
    ]
    assert render_html(events) == snapshot


def test_render_artifact_image(snapshot) -> None:
    events = [
        make_event(
            1,
            "artifact",
            {
                "path": "artifacts/plot-deadbeef.png",
                "mime": "image/png",
                "caption": "pressure field",
            },
        ),
    ]
    html = render_html(events)
    assert 'src="artifacts/plot-deadbeef.png"' in html
    assert html == snapshot


def test_render_artifact_with_provenance(snapshot) -> None:
    events = [
        make_event(
            1,
            "artifact",
            {
                "path": "artifacts/comparison.png",
                "mime": "image/png",
                "caption": "Well rates",
                "slot": "comparison",
                "format": "png",
                "size_px": [900, 500],
                "source_code": "well_rates_figure(wd)",
            },
        ),
    ]
    html = render_html(events)
    assert "slot: comparison" in html
    assert "artifact-source" in html
    assert html == snapshot


def test_render_unknown_kind_fallback(snapshot) -> None:
    events = [make_event(1, "custom_event", {"foo": "bar"})]
    html = render_html(events)
    assert "custom_event" in html
    assert html == snapshot


def test_lifecycle_kinds_render_as_markers_not_raw_dumps() -> None:
    """session_title/session_resume/context_compaction are session structure:
    title in the header, dividers in the timeline, never the raw-payload
    catch-all card (which stays the fallback for truly unknown kinds)."""
    from makker.transcript import render_markdown

    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": "battmo"}),
        make_event(2, "session_title", {"session_id": "abc", "title": "Discharge the chen cell"}),
        make_event(3, "message_user", {"content": "go"}),
        make_event(
            4,
            "context_compaction",
            {"summarized": 21, "kept": 9, "offloaded": True, "manual": True},
        ),
        make_event(5, "session_end", {"session_id": "abc"}),
        make_event(6, "session_resume", {"session_id": "abc", "simulator": "battmo"}),
        make_event(7, "message_assistant", {"content": "done"}),
        make_event(8, "session_end", {"session_id": "abc"}),
    ]

    html = render_html(events)
    assert "Discharge the chen cell" in html
    assert "Session resumed" in html
    assert "Context compacted (manual): 21 messages summarized, 9 kept" in html
    for kind in ("session_title", "session_resume", "context_compaction"):
        assert f"Event · {kind}" not in html

    markdown = render_markdown(events)
    assert "**Discharge the chen cell**" in markdown
    assert "_Session resumed:" in markdown
    assert "Context compacted (manual): 21 messages summarized, 9 kept" in markdown
    assert "### Event `session_resume`" not in markdown


def test_report_does_not_inline_files_outside_the_artifact_dirs(tmp_path) -> None:
    """The report's narrative and plot embedding honor the same containment the
    transcript enforces: an absolute or traversal path never pulls bytes in."""
    import base64

    from makker.transcript.report import render_report

    run = tmp_path / "run"
    (run / "artifacts").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"TOP SECRET")
    out = run / "report.html"
    render_report(
        [],
        out,
        narrative_markdown=f"![leak]({secret})\n\n![leak2](artifacts/../../secret.txt)",
        title="t",
        session_id="s",
        simulator="sim",
        artifact_dirs=[run / "artifacts", run],
    )
    html = out.read_text(encoding="utf-8")
    assert base64.standard_b64encode(b"TOP SECRET").decode() not in html


def test_steps_between_two_messages_fold_into_one_line() -> None:
    """The agent's work between two things it says is one line saying what it
    did, how many steps failed and how long it took, with each step inside."""
    events = [
        make_event(1, "message_user", {"content": "go"}),
        make_event(2, "message_reasoning", {"content": "Check the **example** first."}),
        make_event(
            3,
            "tool_call",
            {"id": "r", "name": "read_file", "args": {"file_path": "/a/b/c/wells.jl"}},
        ),
        make_event(4, "tool_result", {"tool_call_id": "r", "name": "read_file", "content": "x"}),
        make_event(5, "tool_call", {"id": "p", "name": "write_todos", "args": {"todos": []}}),
        make_event(
            6,
            "tool_call",
            {"id": "j", "name": "run_julia", "args": {"code": "boom", "description": "Run it"}},
        ),
        make_event(
            9,
            "tool_result",
            {"tool_call_id": "j", "name": "run_julia", "content": "ERROR: boom", "status": "error"},
        ),
        make_event(10, "message_assistant", {"content": "It failed."}),
        make_event(11, "tool_call", {"id": "l", "name": "ls", "args": {"path": "/w"}}),
        make_event(12, "tool_result", {"tool_call_id": "l", "name": "ls", "content": "[]"}),
    ]
    html = render_html(events)
    timeline = html.split('<div class="timeline">', 1)[1]
    group, rest = timeline.split("It failed.", 1)

    # One line for the stretch: the plan is not work, and the failure and the
    # time (first call to last result) show on it.
    assert '<span class="steps-what">Read a file, ran Julia</span>' in group
    assert '<span class="step-failed">1 failed</span><span class="step-took">6s</span>' in group
    # Inside, a row per step: the thought, the file by its last segments, the
    # run by what it says it does.
    assert '<span class="event-kind thought">Thought</span>' in group
    assert '<span class="tool-peek">Check the example first.</span>' in group
    assert '<span class="tool-peek">c/wells.jl</span>' in group
    assert '<span class="tool-peek">Run it</span><span class="step-failed">failed</span>' in group
    # A lone step after what the agent said is its own row, not a group.
    assert "steps-what" not in rest and '<span class="tool-peek">/w</span>' in rest


def test_a_stretch_of_one_call_and_its_thinking_names_the_call() -> None:
    events = [
        make_event(1, "message_reasoning", {"content": "Look first."}),
        make_event(2, "tool_call", {"id": "g", "name": "grep", "args": {"pattern": "setup_well"}}),
        make_event(3, "tool_result", {"tool_call_id": "g", "name": "grep", "content": "a.jl"}),
    ]
    assert '<span class="steps-what">Search setup_well</span>' in render_html(events)


def test_the_panel_size_reports_are_not_part_of_the_transcript() -> None:
    events = [make_event(1, "ui_event", {"payload": {"kind": "canvas_size", "width": 600}})]
    assert "canvas_size" not in render_html(events)


def test_a_call_that_was_stopped_is_not_counted_as_work_done() -> None:
    """A call cancelled mid-run has no result: the line counts what ran, as the
    terminal and the browser do."""
    events = [
        make_event(1, "tool_call", {"id": "a", "name": "run_julia", "args": {"code": "1"}}),
        make_event(2, "tool_result", {"tool_call_id": "a", "name": "run_julia", "content": "1"}),
        make_event(3, "tool_call", {"id": "b", "name": "run_julia", "args": {"code": "sim()"}}),
        make_event(4, "tool_call", {"id": "c", "name": "execute", "args": {"command": "ls"}}),
    ]
    assert '<span class="steps-what">Ran Julia</span>' in render_html(events)
    stopped = events[2:]
    assert '<span class="steps-what">Nothing ran</span>' in render_html(stopped)


def test_a_stretch_of_steps_hides_only_when_every_kind_in_it_is_filtered_out() -> None:
    """Unticking Tools leaves the thoughts in a stretch of work readable."""
    events = [
        make_event(1, "message_reasoning", {"content": "Look first."}),
        make_event(2, "tool_call", {"id": "g", "name": "grep", "args": {"pattern": "x"}}),
        make_event(3, "tool_call", {"id": "h", "name": "grep", "args": {"pattern": "y"}}),
    ]
    html = render_html(events)
    assert '<article class="event tool-event steps" data-filter="reasoning tools">' in html
    assert "kinds.every((kind) => hidden.has(kind))" in html
    thoughts = [events[0], make_event(4, "message_reasoning", {"content": "Then this."})]
    assert 'class="event tool-event steps" data-filter="reasoning"' in render_html(thoughts)
