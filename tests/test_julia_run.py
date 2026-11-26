"""Tests for the one-shot Julia runner and the cached version probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from makker.julia import requirements, run
from makker.julia.run import JuliaRunError, failure, run_julia


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    """Julia at /opt/julia that records each call and prints a version."""

    seen: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "julia version 1.12.6-beta1\n", "")

    monkeypatch.setattr(run.shutil, "which", lambda name: f"/opt/{name}")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run, "should_wrap_xvfb", lambda: True)
    return seen


def test_runs_the_resolved_julia_non_interactively(calls, tmp_path: Path) -> None:
    run_julia("-e", "1", project=tmp_path)
    [(argv, kwargs)] = calls
    assert argv == ["/opt/julia", f"--project={tmp_path}", "--startup-file=no", "-e", "1"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True
    # Julia's output is UTF-8 on every system, not the locale's code page.
    assert (kwargs["encoding"], kwargs["errors"]) == ("utf-8", "replace")


def test_only_a_display_run_is_wrapped_in_xvfb(calls) -> None:
    run_julia("-e", "1", display=True)
    assert calls[0][0][:2] == ["xvfb-run", "-a"]
    assert calls[0][0][4] == "/opt/julia"


def test_a_timeout_becomes_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def stalled(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(run.shutil, "which", lambda name: "/opt/julia")
    monkeypatch.setattr(run.subprocess, "run", stalled)
    with pytest.raises(JuliaRunError, match="Julia timed out after 30 seconds"):
        run_julia("--version", timeout=30)


def test_julia_missing_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    with pytest.raises(JuliaRunError, match="not on PATH"):
        run_julia("--version")


def test_failure_names_how_julia_ended_and_ends_with_its_output() -> None:
    lines = "".join(f"line {i}\n" for i in range(20))
    captured = subprocess.CompletedProcess([], 1, lines, "ERROR: boom\n")
    assert failure(captured, "Updating the env") == (
        "Updating the env failed (exit code 1):\n"
        + "".join(f"line {i}\n" for i in range(9, 20))
        + "ERROR: boom"
    )
    streamed = subprocess.CompletedProcess([], 1)
    assert failure(streamed) == "Julia failed (exit code 1); see output above"


def test_julia_version_is_asked_once_per_process(calls) -> None:
    first = requirements.check_julia()
    assert requirements.check_julia() is first
    assert first.version == (1, 12, 6)
    # The suffix stays: a system image is tied to the exact build.
    assert first.version_str == "1.12.6-beta1"
    assert len(calls) == 1

    requirements.check_julia.cache_clear()
    requirements.check_julia()
    assert len(calls) == 2
