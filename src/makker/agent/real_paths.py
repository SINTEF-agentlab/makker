"""Let the deepagents file tools take real paths, the way the shell and REPL do.

The agent's filesystem backend runs in real-path mode (``virtual_mode=False``;
see :mod:`makker.agent.backend`): an absolute path is used as it is and a
relative one resolves against the workspace, the same as in ``execute`` and
``run_julia``. But deepagents' ``FilesystemMiddleware`` passes each path through
``validate_path`` before the backend sees it, and that helper is written for
virtual paths. It turns ``model.jl`` into ``/model.jl``, a file at the
filesystem root, and it rejects a Windows drive path (``C:\\...``) outright.

So we wrap ``validate_path``: a relative path stays relative and a drive path
passes through, both for the backend to resolve. Everything else, and the
traversal checks (``..``, ``~``), stays with the original. The middleware looks
``validate_path`` up as a module global at call time, so replacing the name on
each module that imported it reaches every file tool already built.

This is a shim over a deepagents internal; it can go once ``validate_path``
honours ``virtual_mode=False``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

# A leading drive letter (``C:``, ``D:\\``, ``c:/``) marks a real Windows absolute
# path. ``validate_path`` rejects exactly this; we let it through instead.
_WIN_DRIVE = re.compile(r"^[A-Za-z]:")
# Marks an already-wrapped callable so installation is idempotent.
_PATCH_FLAG = "_makker_real_paths"


def _wrap_validate_path(original: Callable[..., str]) -> Callable[..., str]:
    """Return a ``validate_path`` that leaves real paths for the backend to resolve.

    A drive-absolute path is handed back verbatim, and a relative path comes back
    normalized but still relative; ``original`` checks everything for traversal
    and normalizes the rest. Already-wrapped callables are returned as-is.
    """

    if getattr(original, _PATCH_FLAG, False):
        return original

    def validate_path(path: str, *, allowed_prefixes=None) -> str:
        if isinstance(path, str) and _WIN_DRIVE.match(path):
            return path
        normalized = original(path, allowed_prefixes=allowed_prefixes)
        if isinstance(path, str) and not path.startswith(("/", "\\")):
            return normalized.lstrip("/") or "."
        return normalized

    validate_path.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(validate_path, _PATCH_FLAG, True)
    return validate_path


def _install() -> None:
    """Replace ``validate_path`` with the wrapper everywhere it's bound.

    Patches the source module and each deepagents module that imported the name
    into its own namespace: the file tools, and the skill loader, which matches a
    ``read_file`` path the way the tool does. Idempotent.
    """

    from deepagents.backends import utils
    from deepagents.middleware import _skill_tools
    from deepagents.middleware import filesystem as fs_middleware

    for module in (utils, fs_middleware, _skill_tools):
        current = getattr(module, "validate_path", None)
        if current is not None:
            module.validate_path = _wrap_validate_path(current)


def enable_real_paths() -> None:
    """Let the deepagents file tools take real relative and Windows drive paths.

    Idempotent, so it is safe to call on every agent build. Call before the
    filesystem tools run.
    """

    _install()


# A glob metacharacter; the first path component carrying one starts the pattern.
_GLOB_META = re.compile(r"[*?\[]")


def split_windows_glob(pattern: str, path: str | None) -> tuple[str, str | None]:
    """Rewrite a Windows drive-absolute glob pattern into ``(pattern, base path)``.

    ``pathlib``'s ``rglob`` rejects non-relative patterns, so the deepagents glob
    backend silently matches nothing for ``glob("C:\\dir\\**\\*.jl")`` — the exact
    absolute form the skills tell the agent to use. We split off the longest
    metacharacter-free prefix as the search base and return the remainder as a
    relative pattern, which is the form the backend handles (``glob("**/*.jl",
    path="C:\\dir")``).

    A no-op off Windows, when an explicit ``path`` was already given, or when the
    pattern is not drive-absolute — so relative patterns and POSIX paths are
    untouched.
    """

    if os.name != "nt" or path is not None or not isinstance(pattern, str):
        return pattern, path
    if not _WIN_DRIVE.match(pattern):
        return pattern, path

    from pathlib import PureWindowsPath

    parts = PureWindowsPath(pattern).parts  # ("C:\\", "dir", "**", "*.jl")
    split_at = next(
        (i for i, part in enumerate(parts) if _GLOB_META.search(part)),
        len(parts),
    )
    # No metacharacter at all (an exact file path): treat the last component as the
    # pattern so the base is still a directory rglob can search.
    if split_at == len(parts):
        split_at -= 1
    if split_at <= 0:
        return pattern, path  # the drive anchor itself is the pattern; leave it

    base = str(PureWindowsPath(*parts[:split_at]))
    relative = "/".join(parts[split_at:]) or "*"
    return relative, base
