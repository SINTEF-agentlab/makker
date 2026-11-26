"""The check for numbers the agent typed into a file without having been shown them."""

from __future__ import annotations

from makker.review.grounding import ungrounded_writes
from makker.trace import Event
from makker.trace.schema import MESSAGE_USER, TOOL_CALL, TOOL_RESULT


def _events(*steps: tuple[str, dict]) -> list[Event]:
    return [Event(i, "", kind, payload) for i, (kind, payload) in enumerate(steps)]


def _output(text: str) -> tuple[str, dict]:
    return TOOL_RESULT, {"name": "run_julia", "content": text}


def _write(path: str, content: str) -> tuple[str, dict]:
    return TOOL_CALL, {"name": "write_file", "args": {"file_path": path, "content": content}}


def test_values_shown_before_they_are_written_pass() -> None:
    events = _events(
        (MESSAGE_USER, {"content": "Use a porosity of 0.21375."}),
        _output("result = 0.7341927364518272\npressure: -1.23456e-5\nf=  2.30735D-02"),
        _write(
            "results.csv",
            # Full precision, rounded, signed, a percentage, from the user, and
            # Fortran-style output; short numbers are not checked.
            "a,b,c,d,e,f,g\n0.7341927364518272,0.734193,-1.23456e-5,73.4193%,0.21375,"
            "0.0230735,0.25\n",
        ),
        (
            TOOL_CALL,
            {"name": "edit_file", "args": {"file_path": "notes.md", "new_string": "x = 0.73419"}},
        ),
    )
    assert ungrounded_writes(events) == []


def test_a_value_no_output_showed_is_flagged() -> None:
    events = _events(
        _output("result = 0.7341927364518272"),
        _write("report.md", "Result 0.734193, and the second case gave 0.512345."),
        # Shown only after it was written: still not read from anywhere.
        _output("second = 0.512345"),
        # Code is not checked: its constants are knowledge, not results.
        _write("model.jl", "const R = 8.314462618"),
    )
    [write] = ungrounded_writes(events)
    assert write.path == "report.md"
    assert write.numbers == ("0.512345",)
    assert write.checked == 2
    finding = write.finding()
    assert finding.title == "Numbers written to report.md that no output showed"
    assert "0.512345" in finding.evidence and finding.severity == "high"
