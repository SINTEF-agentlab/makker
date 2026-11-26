"""Every ``julia`` we run to completion: the version probe, ``Pkg`` steps, the env
probe and the system-image build.

The live session has a supervised launch of its own (:mod:`makker.juliakernel`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from makker.display import should_wrap_xvfb
from makker.julia.precompile import julia_environment

# A GL context for headless Linux, where the precompile and image-build workloads
# save real GLMakie figures. The session starts its Xvfb directly instead.
_XVFB_RUN = ("xvfb-run", "-a", "-s", "-screen 0 1280x1024x24")


class JuliaRunError(RuntimeError):
    """Julia could not be started, or did not finish in time."""


def run_julia(
    *args: str,
    project: Path | None = None,
    capture: bool = True,
    timeout: float | None = None,
    display: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``julia [--project] --startup-file=no *args`` and wait for it.

    Never interactive (stdin is closed) and under the precompile ceiling, which
    any Julia it spawns inherits. ``capture=False`` streams the output instead, for
    steps long enough to need it. ``display`` wraps a GL workload in ``xvfb-run``
    on headless Linux. A nonzero exit is returned; :func:`failure` words it.
    """

    executable = shutil.which("julia")
    if executable is None:
        raise JuliaRunError("`julia` is not on PATH")
    project_flag = [f"--project={project}"] if project is not None else []
    argv = [executable, *project_flag, "--startup-file=no", *args]
    if display and should_wrap_xvfb():
        argv = [*_XVFB_RUN, *argv]
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=capture,
            # Julia writes UTF-8 whatever the system's code page is. Read as that,
            # so a path under a home folder such as C:/Users/Bjørn comes out intact.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=julia_environment(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JuliaRunError(f"Julia timed out after {timeout:g} seconds") from exc
    except OSError as exc:
        raise JuliaRunError(f"could not run Julia: {exc}") from exc


def failure(result: subprocess.CompletedProcess[str], what: str = "Julia") -> str:
    """``<what> failed (<how it ended>)``, then the end of the output if it was captured."""

    from makker.juliakernel.kernel import describe_exit

    message = f"{what} failed ({describe_exit(result.returncode)})"
    if result.stdout is None:
        return f"{message}; see output above"
    return f"{message}:\n{output_tail(result)}"


def output_tail(result: subprocess.CompletedProcess[str], lines: int = 12) -> str:
    """The last ``lines`` of a captured run's output, stdout then stderr."""

    output = (result.stdout or "") + (result.stderr or "")
    return "\n".join(output.strip().splitlines()[-lines:])
