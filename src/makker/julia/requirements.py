"""Julia toolchain requirement checks, shared across the CLI.

One place that knows "is Julia usable?" so the runtime launch, ``init``,
and ``doctor`` all agree on the same answer and the same remediation text.
"""

from __future__ import annotations

import functools
import re
import shutil
from dataclasses import dataclass

from makker.julia.run import JuliaRunError, run_julia

# Floor set by the simulators, not the kernel (server.jl is stdlib-only): Mocca
# and JutulDarcy need 1.10. 1.10 is also the current Julia LTS.
MIN_JULIA_VERSION: tuple[int, int] = (1, 10)

_VERSION_RE = re.compile(r"julia version ((\d+)\.(\d+)\.(\d+)\S*)")


@dataclass(frozen=True)
class JuliaCheck:
    """Result of probing the ``julia`` executable on PATH."""

    found: bool
    path: str | None = None
    version: tuple[int, int, int] | None = None
    # As Julia prints it, so a prerelease keeps its suffix ("1.13.0-beta1"): a
    # system image is tied to the exact build.
    version_str: str | None = None
    error: str | None = None

    @property
    def version_ok(self) -> bool:
        return self.version is not None and self.version[:2] >= MIN_JULIA_VERSION

    @property
    def ok(self) -> bool:
        return self.found and self.version_ok


def _min_version_str() -> str:
    return ".".join(str(n) for n in MIN_JULIA_VERSION)


@functools.cache
def check_julia() -> JuliaCheck:
    """Probe for ``julia`` on PATH and parse its version, once per process.

    Never raises — failures are reported in the returned ``JuliaCheck``.
    """

    path = shutil.which("julia")
    if path is None:
        return JuliaCheck(found=False, error="`julia` is not on PATH")

    try:
        # A timeout too: the juliaup launcher can stall for minutes on its update check.
        proc = run_julia("--version", timeout=30)
    except JuliaRunError as exc:
        return JuliaCheck(found=True, path=path, error=str(exc))

    match = _VERSION_RE.search(proc.stdout)
    if match is None:
        return JuliaCheck(
            found=True,
            path=path,
            error=f"could not parse version from `julia --version`: {proc.stdout!r}",
        )

    version = (int(match[2]), int(match[3]), int(match[4]))
    return JuliaCheck(found=True, path=path, version=version, version_str=match[1])


def require_julia() -> JuliaCheck:
    """Like :func:`check_julia` but raises ``JuliaRequirementError`` if unusable."""

    check = check_julia()
    if not check.found:
        raise JuliaRequirementError(
            f"`julia` is not on PATH. Install Julia {_min_version_str()}+ via "
            "juliaup (https://github.com/JuliaLang/juliaup), then open a new terminal."
        )
    if not check.version_ok:
        raise JuliaRequirementError(
            f"Julia {_min_version_str()}+ is required, but `julia` is "
            f"{check.version_str or 'an unknown version'}. "
            f"Run `juliaup add {_min_version_str()} && juliaup default {_min_version_str()}`."
        )
    return check


class JuliaRequirementError(RuntimeError):
    pass
