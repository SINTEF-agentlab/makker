"""Tests for the kernel's terminal-output emulation (render_terminal_output)."""

from __future__ import annotations

from makker.juliakernel.text import render_terminal_output


def test_render_terminal_output_passes_plain_text_through() -> None:
    assert render_terminal_output("a\nb\nc") == "a\nb\nc"


def test_render_terminal_output_collapses_carriage_return_overwrite() -> None:
    # Single-line progress bar that overwrites itself via \r.
    raw = "\rProgress  10%|##|\rProgress  50%|####|\rProgress 100%|######|"
    assert render_terminal_output(raw) == "Progress 100%|######|"


def test_render_terminal_output_handles_cursor_up_and_erase_line() -> None:
    # Three-line progress block, then cursor-up twice + erase-line to
    # overwrite the previous block with a fresh one — exactly what
    # ProgressMeter.jl emits.
    raw = (
        "Progress  10%|####      | ETA: 0:00:10\x1b[K\n"
        "   Solving step 17/100\x1b[K\n"
        "     Stats: 30 iterations\x1b[K"
        "\x1b[A\r\x1b[K\x1b[A\r\x1b[K"
        "Progress 100%|##########| Time: 0:00:01\x1b[K\n"
        "   Solving step 100/100\x1b[K\n"
        "     Stats: 200 iterations\x1b[K"
    )
    out = render_terminal_output(raw)
    assert out == (
        "Progress 100%|##########| Time: 0:00:01\n"
        "   Solving step 100/100\n"
        "     Stats: 200 iterations"
    )


def test_render_terminal_output_preserves_static_tables_before_progress() -> None:
    raw = "│ Newton │ 2.32 │\n│ Total  │ 14.18 │\n\n\rProgress  50%|####|\rProgress 100%|########|"
    out = render_terminal_output(raw)
    assert "│ Newton │ 2.32 │" in out
    assert "│ Total  │ 14.18 │" in out
    assert "Progress 100%|########|" in out
    assert "50%" not in out


def test_render_terminal_output_ignores_sgr_color_codes() -> None:
    raw = "\x1b[1;32mok\x1b[0m\n\x1b[31merror\x1b[0m"
    assert render_terminal_output(raw) == "ok\nerror"


def test_render_terminal_output_handles_backspace_and_tab() -> None:
    assert render_terminal_output("foox\bbar") == "foobar"
    # Tab moves to next 8-col stop.
    assert render_terminal_output("ab\tcd") == "ab      cd"


def test_render_terminal_output_fast_path_trims_like_the_screen_model() -> None:
    # Plain text (no cursor control) takes the fast path; it must trim trailing
    # whitespace per line and drop trailing blank lines, same as the screen model.
    assert render_terminal_output("a  \nb \n\n") == "a\nb"


def test_render_terminal_output_bounds_cursor_moves_with_huge_counts() -> None:
    # A real screen is finite. Without a bound, one short escape grows the buffer
    # by its count: `ESC[4000000B` used to produce four million rows.
    assert render_terminal_output("\x1b[4000000Bx").count("\n") < 1000
    assert len(render_terminal_output("\x1b[9999999999Cx")) <= 513
    assert render_terminal_output("\x1b[999999999;999999999Hx").count("\n") < 1000
    # Repeating a bounded move does not add up either.
    assert render_terminal_output("\x1b[1000B" * 5000 + "x").count("\n") < 1000
    # A count too long for `int()` is not a cursor move at all; it must not raise.
    render_terminal_output("\x1b[" + "9" * 5000 + "Hx")


def test_render_terminal_output_still_moves_within_the_screen() -> None:
    assert render_terminal_output("a\x1b[2Bb\x1b[2Ac") == "a c\n\n b"
    assert render_terminal_output("ab\x1b[5Cz") == "ab     z"
