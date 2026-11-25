"""Canonical filesystem locations used across the package.

Three roots, cleanly separated:

- ``PACKAGE_ROOT``: computed from this file. Read-only at runtime; per-
  simulator assets (julia envs, skills, adapter modules) live under
  ``PACKAGE_ROOT / "simulators" / <name>``.
- ``workspace_root()``: the user's working directory at invocation time
  (or an explicit override). Read/write; default for shell and file tools.
- ``state_home()``: sessions, traces, per-workspace state. Defaults to
  ``$XDG_DATA_HOME/makker`` or ``~/.local/share/makker``.

The workspace and state-home anchors are runtime-mutable so the CLI can set
them once on startup and library code can read them via the helpers.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from datetime import date
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent

SHARED_SKILLS_DIR: Path = PACKAGE_ROOT / "simulators" / "shared_skills"

_workspace_root_override: Path | None = None
_state_home_override: Path | None = None


def set_workspace_root(path: Path | None) -> None:
    global _workspace_root_override
    _workspace_root_override = path.resolve() if path is not None else None


def workspace_root() -> Path:
    if _workspace_root_override is not None:
        return _workspace_root_override
    return Path.cwd().resolve()


def set_state_home(path: Path | None) -> None:
    global _state_home_override
    _state_home_override = path.resolve() if path is not None else None


def default_state_home() -> Path:
    """The standard data folder: where it is unless the user names another."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).resolve() / "makker"
    return Path.home() / ".local" / "share" / "makker"


def state_home() -> Path:
    if _state_home_override is not None:
        home = _state_home_override
    else:
        # Temporary: an install from before the rename keeps its data under the
        # old name until Makker has moved it.
        from makker import legacy

        home = legacy.resolve_state_home(default_state_home())
    # It holds keys, chats and memory, so create it owner-only (a no-op on
    # Windows before Python 3.13). A folder that cannot be made is left to the
    # write that needs it to report.
    with contextlib.suppress(OSError):
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
    return home


def user_config_path() -> Path:
    """User-global config file, at the root of the state home.

    Settings here apply across every workspace (e.g. the default model); the
    per-workspace ``.makker/config.toml`` overrides them.
    """
    return state_home() / "config.toml"


def resolve_agent_path(raw: str | Path, *, workspace: Path | None = None) -> Path | None:
    """The real file a path from the agent names, or ``None`` for an empty path.

    The one rule every tool follows (file tools, ``execute``, ``run_julia``): an
    absolute path is itself, and a relative one is under the workspace. So
    ``model.jl`` is ``<workspace>/model.jl`` and ``/model.jl`` is a file at the
    filesystem root.
    """

    text = str(raw)
    if not text:
        return None
    # Joining keeps an absolute path as it is, the same as the backend does.
    return (workspace or workspace_root()).resolve() / text


def resolve_in_workspace(raw: str | Path, *, workspace: Path | None = None) -> Path | None:
    """Like :func:`resolve_agent_path`, but ``None`` unless the file is in the workspace."""

    ws = (workspace or workspace_root()).resolve()
    path = resolve_agent_path(raw, workspace=ws)
    if path is None:
        return None
    try:
        path.resolve().relative_to(ws)
    except (ValueError, OSError):
        return None
    return path


def workspace_hash(workspace: Path | None = None) -> str:
    """Stable 12-char hash of the workspace's resolved path."""
    ws = (workspace or workspace_root()).resolve()
    return hashlib.sha256(str(ws).encode("utf-8")).hexdigest()[:12]


def workspace_state_dir(workspace: Path | None = None) -> Path:
    """Per-workspace state directory under ``state_home()``."""
    return state_home() / "workspaces" / workspace_hash(workspace)


WORKSPACE_OUTPUT_DIRNAME = "makker-output"


def workspace_output_dir(workspace: Path | None = None) -> Path:
    """Root of the visible per-workspace output tree (``makker-output/``).

    Named distinctly from the hidden ``.makker/`` config/env dir so the two
    don't read as the same folder.
    """
    return (workspace or workspace_root()) / WORKSPACE_OUTPUT_DIRNAME


# Session ids minted by ``session.default_session_id``: a sortable timestamp
# plus a short random suffix (e.g. ``2026-06-12-2315-3f2a``).
_DATED_SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}-[0-9a-f]{4}$")


def is_dated_session_id(session_id: str) -> bool:
    """Whether ``session_id`` is a sortable timestamped id (vs a legacy UUID)."""
    return bool(_DATED_SESSION_ID_RE.match(session_id))


def session_output_dir(session_id: str, workspace: Path | None = None) -> Path:
    """Visible output directory for one session under the workspace.

    Layout: ``<workspace>/makker-output/sessions/<session_id>/`` for
    timestamped ids (the id itself sorts chronologically), or the legacy
    ``<YYYY-MM-DD>-<short_id>/`` for UUID sessions.

    This directory holds user-facing outputs (plots, transcripts, reports).
    Internal state (SQLite trace, REPL log) stays in ``workspace_state_dir``.
    """
    if is_dated_session_id(session_id):
        name = session_id
    else:
        name = f"{date.today().isoformat()}-{session_id[:8]}"
    return workspace_output_dir(workspace) / "sessions" / name


def workspace_memory_dir(workspace: Path | None = None) -> Path:
    """Per-workspace memory directory (index + per-fact note files).

    Lives alongside ``sessions/`` under the workspace's state dir so memory
    is automatically scoped to the workspace path the user invokes
    Makker from. The agent maintains the contents with ``edit_file`` /
    ``write_file`` at this folder's real path.
    """
    return workspace_state_dir(workspace) / "memory"


def eval_sessions_state_root(simulator: str) -> Path:
    """A stable, discoverable home for an eval run's sessions.

    The eval harness otherwise drops each session in a temp dir that is cleaned
    up, so eval runs (where a golden answer is known and silent failures matter
    most) could never be reviewed. Pointing an eval session's ``state_root``
    here puts its trace under ``workspaces/eval-<sim>/sessions/``, which the
    review tooling discovers like any other session. A state-home layout fact,
    so it lives here rather than coupling the eval extra to the review package.
    """
    return state_home() / "workspaces" / f"eval-{simulator}"
