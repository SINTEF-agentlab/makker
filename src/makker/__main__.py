"""Entry point for the Makker CLI.

Delegates to ``interfaces.cli.main`` so the CLI plumbing has one home.
"""

from __future__ import annotations

import sys
import warnings

# The LangChain packages import pydantic.v1 so they can accept V1 models, and on
# Python 3.14+ that import alone warns at the start of every command. Makker
# never hands them a V1 model, so the warning says nothing about this program.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    ``mcp`` is dispatched here, before the CLI package is imported: an MCP client
    gives a server seconds to answer, and the other commands import LangChain.
    """
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["mcp"]:
        from makker.interfaces.mcp.cli import main as mcp

        return mcp(args[1:])
    from makker.interfaces.cli.main import main as cli

    return cli(argv)


if __name__ == "__main__":
    sys.exit(main())
