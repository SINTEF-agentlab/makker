"""Tests for path helpers."""

from __future__ import annotations

import hashlib
import stat
import sys
from pathlib import Path

import pytest

from makker.paths import (
    resolve_agent_path,
    resolve_in_workspace,
    set_state_home,
    set_workspace_root,
    state_home,
    workspace_hash,
    workspace_state_dir,
)


def test_workspace_hash_is_deterministic(tmp_path: Path) -> None:
    ws = tmp_path / "project"
    ws.mkdir()
    expected = hashlib.sha256(str(ws.resolve()).encode("utf-8")).hexdigest()[:12]
    assert workspace_hash(ws) == expected


def test_workspace_hash_differs_for_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert workspace_hash(a) != workspace_hash(b)


def test_state_home_honours_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom-state"
    set_state_home(custom)
    assert state_home() == custom.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode")
def test_state_home_is_created_owner_only(tmp_path: Path) -> None:
    home = tmp_path / "data" / "makker"
    set_state_home(home)
    (state_home() / "sessions").mkdir(parents=True)
    assert stat.S_IMODE(home.stat().st_mode) == 0o700


def test_state_home_defaults_to_xdg(monkeypatch, tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    set_state_home(None)
    assert state_home() == (xdg / "makker").resolve()


def test_workspace_state_dir_under_state_home(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    state = tmp_path / "state"
    set_workspace_root(ws)
    set_state_home(state)
    assert workspace_state_dir() == state / "workspaces" / workspace_hash(ws)


def test_agent_paths_follow_one_rule(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    set_workspace_root(ws)

    # Relative paths are under the workspace.
    assert resolve_agent_path("experiments/report.html") == ws / "experiments/report.html"
    # An absolute path is itself, inside the workspace or not. A leading slash is
    # the filesystem root, as in the shell and the REPL, not the workspace.
    inside = ws / "x" / "y.html"
    assert resolve_agent_path(str(inside)) == inside
    elsewhere = tmp_path / "elsewhere.txt"
    assert resolve_agent_path(str(elsewhere)) == elsewhere
    assert resolve_agent_path("/model.jl") == Path(ws.anchor) / "model.jl"
    # Empty input resolves to nothing rather than the workspace root.
    assert resolve_agent_path("") is None


def test_resolve_in_workspace_keeps_only_workspace_files(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    set_workspace_root(ws)

    assert resolve_in_workspace("experiments/report.html") == ws / "experiments/report.html"
    inside = ws / "x" / "y.html"
    assert resolve_in_workspace(str(inside)) == inside

    # The old virtual forms are no longer workspace files.
    assert resolve_in_workspace("/experiments/report.html") is None
    assert resolve_in_workspace("/workspace/experiments/report.html") is None
    assert resolve_in_workspace(str(tmp_path / "elsewhere.txt")) is None
    # `..` escapes are rejected whatever their form.
    assert resolve_in_workspace("../outside/secret.txt") is None
    assert resolve_in_workspace("/../outside/secret.txt") is None
    assert resolve_in_workspace("") is None
