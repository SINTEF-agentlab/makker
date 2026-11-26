"""``makker mcp``: serve this folder's session to an MCP client over stdin/stdout.

The client starts this command itself, from an entry in its MCP configuration;
see docs/mcp.md. ``makker.__main__``
dispatches here before the CLI's commands load, because they import LangChain
and a client gives a server seconds to answer.
"""

from __future__ import annotations

import argparse


def _seconds(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    from makker.interfaces.cli._helpers import add_julia_flags, add_workspace_flags
    from makker.interfaces.mcp.runtime import DEFAULT_WAIT

    parser = argparse.ArgumentParser(
        prog="makker mcp",
        description=(
            "Serve this folder's Makker session to another agent over the Model "
            "Context Protocol, on stdin/stdout. An MCP client starts it; it is not "
            "run by hand."
        ),
    )
    parser.add_argument(
        "--sim",
        default=None,
        metavar="NAME",
        help="The simulator. Default: the folder's saved choice, else what its packages say.",
    )
    parser.add_argument(
        "--max-wait",
        type=_seconds,
        default=DEFAULT_WAIT,
        metavar="SECONDS",
        help=(
            "How long a call may run before it answers 'still running' "
            f"(default {DEFAULT_WAIT:.0f}). Keep it under your client's tool time limit."
        ),
    )
    add_workspace_flags(parser)
    add_julia_flags(parser)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    import anyio

    from makker.interfaces.cli._helpers import apply_workspace_flags
    from makker.interfaces.mcp.runtime import Launch
    from makker.interfaces.mcp.server import serve_stdio
    from makker.paths import workspace_root

    apply_workspace_flags(args)
    launch = Launch(
        workspace=workspace_root(),
        simulator=args.sim,
        julia_project=args.julia_project,
        threads=args.threads,
        sysimage=args.sysimage,
        wait=args.max_wait,
    )
    anyio.run(serve_stdio, launch)
    return 0
