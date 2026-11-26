"""The tools an MCP client is shown: the session's own, and the three the server adds.

The session's tools are listed as Makker's own agent sees them (name, description
and arguments), read from the tools themselves.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any

WAIT = "julia_wait"
INTERRUPT = "julia_interrupt"
GUIDE = "read_guide"

_NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}}

_OWN: list[dict[str, Any]] = [
    {
        "name": WAIT,
        "title": "Wait for Julia",
        "description": (
            "Wait for a call that answered 'still running'. Returns its result once it "
            "has finished, or the output so far if it is still running; call again to "
            "keep waiting. Long simulations are normal: keep waiting rather than "
            "starting the work again."
        ),
        "inputSchema": _NO_ARGUMENTS,
    },
    {
        "name": INTERRUPT,
        "title": "Interrupt Julia",
        "description": (
            "Stop the Julia code that is running now, like Ctrl-C at a REPL. Variables "
            "and loaded packages are kept unless Julia had to be restarted to stop it, "
            "which the result says."
        ),
        "inputSchema": _NO_ARGUMENTS,
    },
    {
        "name": GUIDE,
        "title": "Read a guide",
        "description": (
            "Read one of Makker's guides for this folder's simulator: how to set up, run "
            "and plot with its packages, written against the installed versions. Call it "
            "with no name for the list. Read the matching guide before the first use of "
            "an API, rather than writing it from memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The guide to read, from the list."}
            },
        },
    },
]

OWN_TOOLS = frozenset(spec["name"] for spec in _OWN)


@functools.cache
def tool_specs() -> tuple[dict[str, Any], ...]:
    """Every tool the client is shown.

    Built once, on the server's own thread: it imports the tools (and LangChain),
    which must not happen while the session's start imports the rest of the stack
    in another thread (see ``runtime._import_agent_stack``). The tools only bind
    their session, so a stand-in with nothing behind it is enough to read them.
    """
    from types import SimpleNamespace

    from langchain_core.utils.function_calling import convert_to_openai_tool

    from makker.agent.tools import session_tools
    from makker.tool_labels import tool_label

    stand_in = SimpleNamespace(output_dir=Path("."), simulator=None, surface="mcp")
    specs = []
    for tool in session_tools(stand_in):  # type: ignore[arg-type]
        function = convert_to_openai_tool(tool)["function"]
        specs.append(
            {
                "name": tool.name,
                "title": tool_label(tool.name),
                "description": inspect.cleandoc(function.get("description") or ""),
                "inputSchema": function.get("parameters") or _NO_ARGUMENTS,
            }
        )
    return (*specs, *_OWN)


def session_tool_names() -> frozenset[str]:
    return frozenset(spec["name"] for spec in tool_specs()) - OWN_TOOLS
