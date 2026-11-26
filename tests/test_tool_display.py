"""Tests for what is behind a TUI step's line."""

from __future__ import annotations

from makker.agent.tool_output import strip_read_file_gutter
from makker.interfaces.tui.tool_display import display_path, display_tool_body, step_note


def test_display_path_keeps_filename_for_long_paths() -> None:
    # Short paths render whole.
    assert display_path("test_folder/alphabet.jl") == "test_folder/alphabet.jl"

    # Long paths drop leading parents (marked with …/) but keep the filename,
    # so the click target's name stays visible and the label stays on one line.
    out = display_path("testbed/jutuldarcy/deep/nested/dirs/here/advanced_simulation.jl")
    assert out.startswith("…/")
    assert out.endswith("/advanced_simulation.jl")
    assert len(out) <= 46

    # A single over-long filename is middle-truncated, keeping its extension.
    out2 = display_path("a_really_really_really_really_really_long_simulation_name.jl")
    assert out2.endswith(".jl")
    assert "…" in out2
    assert len(out2) <= 44


def test_strip_read_file_gutter() -> None:
    from deepagents.backends.utils import format_content_with_line_numbers

    # The legacy `cat -n` gutter, still present in stored transcripts.
    assert strip_read_file_gutter("     1\talpha\n     2\tbeta") == "alpha\nbeta"
    # The gutter `read_file` emits today, built with the framework's formatter
    # so a change to its separator fails here instead of showing up on a card.
    numbered = format_content_with_line_numbers(["alpha", "beta"])
    assert strip_read_file_gutter(numbered) == "alpha\nbeta"


def test_run_julia_body_shows_code_then_output() -> None:
    body = display_tool_body(
        "run_julia",
        {"code": "1 + 1"},
        output="2",
        expanded=False,
    )
    # Both the code the agent ran and the simulator output are visible; the
    # output isn't allowed to replace the code, and Jutul's iteration tables
    # would survive untouched if they were present.
    assert "**Code**" in body
    assert "1 + 1" in body
    assert "2" in body
    code_index = body.find("**Code**")
    output_index = body.find("output")
    assert code_index >= 0 and output_index > code_index


def test_run_julia_tail_truncation_keeps_return_value() -> None:
    lines = [f"progress line {i}" for i in range(60)]
    lines.extend(["→ 42", "[1.23s]"])
    output = "\n".join(lines)
    body = display_tool_body(
        "run_julia",
        {"code": "run()"},
        output=output,
        expanded=False,
    )
    assert "output truncated above" in body
    assert "→ 42" in body
    assert "[1.23s]" in body
    assert "progress line 0" not in body


def test_run_julia_preview_fits_jutul_timing_summary() -> None:
    """The simulation summary tables that follow a Jutul ``simulate(...)``
    call are ~30 lines. The collapsed preview must show all of them along
    with the trailing progress bar, return value, and elapsed marker so
    the user gets the real result without having to expand the card.
    """

    output = "\n".join(
        [
            "╭───────────────┬─────────┬───────────────┬─────────╮",
            "│ Iteration type │ Avg/step │ Avg/ministep │   Total │",
            "│                │ 18 steps │   21 ministeps │ (wasted)│",
            "├───────────────┼─────────┼───────────────┼─────────┤",
            "│ Newton         │  1.22222 │      1.04762 │ 22 (0)  │",
            "│ Linearization  │  2.38889 │      2.04762 │ 43 (0)  │",
            "│ Linear solver  │  2.66667 │      2.28571 │ 48 (0)  │",
            "│ Precond apply  │  5.33333 │      4.57143 │ 96 (0)  │",
            "╰───────────────┴─────────┴───────────────┴─────────╯",
            "",
            "╭──────────────┬─────────┬───────────┬────────╮",
            "│ Timing type  │    Each │  Relative │  Total │",
            "│              │      ms │         % │    s   │",
            "├──────────────┼─────────┼───────────┼────────┤",
            "│ Properties   │  0.3348 │   2.37 %  │ 0.1135 │",
            "│ Equations    │  2.8068 │  28.42 %  │ 1.3613 │",
            "│ Assembly     │  0.8855 │   8.97 %  │ 0.4295 │",
            "│ Linear solve │  0.4221 │   2.99 %  │ 0.1431 │",
            "│ Linear setup │  0.0000 │   0.00 %  │ 0.0000 │",
            "│ Precond apply│  0.0000 │   0.00 %  │ 0.0000 │",
            "│ Update       │  0.3636 │   2.57 %  │ 0.1233 │",
            "│ Convergence  │  2.7461 │  27.81 %  │ 1.3319 │",
            "│ Input/Output │  0.3804 │   1.16 %  │ 0.0555 │",
            "│ Other        │  3.6336 │  25.72 %  │ 1.2318 │",
            "│ Total        │ 14.1293 │ 100.00 %  │ 4.7898 │",
            "╰──────────────┴─────────┴───────────┴────────╯",
            "",
            "[stderr]",
            "Progress 100%|████████████████████████| Time: 0:00:05",
            "→ (145, (2.41708, 4.15396), (0.150268, 2.54521), 7048.44)",
            "[30.96s]",
        ]
    )
    body = display_tool_body(
        "run_julia",
        {"code": 'include("candidate.jl"); run_candidate()'},
        output=output,
        expanded=False,
    )
    assert "Iteration type" in body
    assert "Timing type" in body
    assert "Total" in body
    assert "Progress 100%" in body
    assert "→ (145" in body
    assert "[30.96s]" in body


def test_empty_ls_and_glob_report_no_entries() -> None:
    # `ls` and `glob` render a sentence rather than an empty list when nothing
    # matched; counting its lines would report an empty directory as "1 entry".
    from deepagents.middleware.filesystem import _format_file_paths

    empty = _format_file_paths([])
    assert step_note("ls", {"path": "/ws"}, empty) == "0 entries"
    assert step_note("glob", {"pattern": "*.jl"}, empty) == "0 files"


def test_shell_output_is_shown_as_typed() -> None:
    # Program output is not Markdown: `__init__.py` used to show as bold "init.py".
    from makker.preview import fenced_block

    output = "__init__.py\nsetup.py"
    body = display_tool_body("execute", {"command": "ls"}, output=output, expanded=False)
    assert fenced_block(output, language="sh") in body
