"""The MCP server: the session's tools, served over stdio.

Protocol glue only. The runtime owns the session; this maps it to what an MCP
client sees: the shipped tool list, results (text, images and links), Julia's
live output as progress notifications, and the instructions that orient the
client's model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server

from makker.interfaces.mcp import specs
from makker.interfaces.mcp.runtime import Launch, McpRuntime, Outcome, UnknownTool
from makker.juliakernel.text import render_terminal_output

# How often live output is forwarded: often enough to watch a solve, not so often
# that a chatty one floods the client.
PROGRESS_INTERVAL = 1.0
_LINE_LIMIT = 200
# The longest a shutdown waits for the session to close.
_CLOSE_TIMEOUT = 60.0


def build_server(runtime: McpRuntime) -> Server[McpRuntime]:
    """The MCP server for ``runtime``'s session, which it stops when the client goes."""
    from makker import __version__
    from makker.agent.prompts import assemble_mcp_instructions

    instructions = assemble_mcp_instructions(
        runtime.simulator(), workspace=runtime.launch.workspace
    )
    # Some clients never show their model the instructions, so the first result
    # carries them too: by then the model is using Makker and reads it.
    first = True

    @contextlib.asynccontextmanager
    async def lifespan(_server: Server[McpRuntime]) -> AsyncIterator[McpRuntime]:
        try:
            yield runtime
        finally:
            # Shielded, so a shutdown that arrives as a cancellation still stops
            # Julia rather than orphaning it; bounded, as the kernel's close is.
            with anyio.move_on_after(_CLOSE_TIMEOUT, shield=True):
                await runtime.aclose()

    async def list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec["name"],
                    title=spec["title"],
                    description=spec["description"],
                    input_schema=spec["inputSchema"],
                )
                for spec in specs.tool_specs()
            ]
        )

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        nonlocal first
        if (client := ctx.session.client_params) is not None:
            runtime.note_client(client.client_info.title or client.client_info.name)
        relay = ProgressRelay(ctx.session.report_progress)
        try:
            outcome = await runtime.call(
                params.name, dict(params.arguments or {}), on_output=relay.write
            )
        except UnknownTool:
            outcome = Outcome(f"Unknown tool: {params.name}", failed=True)
        finally:
            relay.close()
        if first:
            first = False
            outcome.notes.append(f"How to work with Makker (sent once):\n\n{instructions}")
        return call_result(outcome)

    return Server(
        "makker",
        version=__version__,
        title="Makker",
        instructions=instructions,
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve_stdio(launch: Launch) -> None:
    """Serve one session over stdin/stdout until the client disconnects."""
    from mcp.server.stdio import stdio_server

    runtime = McpRuntime(launch)
    server = build_server(runtime)
    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):  # Windows has no SIGTERM to catch
        loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(_terminate(runtime)))
    # While serving, the transport points fd 1 at stderr, so nothing else in the
    # process (Julia's output included) can write into the protocol stream.
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _terminate(runtime: McpRuntime) -> None:
    """Stop on SIGTERM as on a closed connection: interrupt running Julia code and
    close the session, then exit. Python's default would leave a busy Julia running
    until its code finishes, and cancelling the server would wait on its blocked
    stdin read, which only the client can end."""
    with anyio.move_on_after(_CLOSE_TIMEOUT, shield=True):
        await runtime.aclose()
    os._exit(0)


def call_result(outcome: Outcome) -> types.CallToolResult:
    """A call's outcome as MCP content: the tool's reply, then where to see its work."""
    blocks = content_blocks(outcome.content)
    if outcome.heading:
        blocks.insert(0, types.TextContent(text=outcome.heading))
    blocks += [types.TextContent(text=note) for note in outcome.notes]
    return types.CallToolResult(content=blocks, is_error=outcome.failed)


def content_blocks(content: str | list[Any]) -> list[types.ContentBlock]:
    """A tool's reply (a string, or text and image blocks) as MCP content blocks."""
    if isinstance(content, str):
        return [types.TextContent(text=content)]
    blocks: list[types.ContentBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(types.TextContent(text=item))
        elif isinstance(item, dict) and item.get("type") == "text":
            blocks.append(types.TextContent(text=str(item.get("text", ""))))
        elif isinstance(item, dict) and item.get("type") == "image" and item.get("base64"):
            blocks.append(
                types.ImageContent(
                    data=str(item["base64"]), mime_type=str(item.get("mime_type") or "image/png")
                )
            )
        else:
            blocks.append(types.TextContent(text=json.dumps(item, default=str)))
    return blocks or [types.TextContent(text="")]


class ProgressRelay:
    """Show the client the last line of a running call's output, at most once a second.

    ``write`` is called from the kernel's output pump. A client that asked for no
    progress gets none (``report`` is a no-op then), and losing a notification
    never fails the call.
    """

    def __init__(self, report: Callable[[float, float | None, str | None], Awaitable[None]]):
        self._report = report
        self._text = ""
        self._fresh = False
        self._task = asyncio.get_running_loop().create_task(self._pump())

    def write(self, text: str) -> None:
        self._text = (self._text + text)[-4 * _LINE_LIMIT :]
        self._fresh = True

    async def _pump(self) -> None:
        sent = 0
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL)
            if self._fresh and (line := last_line(self._text)):
                self._fresh, sent = False, sent + 1
                with contextlib.suppress(Exception):
                    await self._report(sent, None, line)

    def close(self) -> None:
        """Stop forwarding; the result carries the output."""
        self._task.cancel()


def last_line(text: str) -> str:
    """The last line of ``text`` that says something, as a terminal would show it."""
    lines = [line.strip() for line in render_terminal_output(text).splitlines()]
    return next((line[:_LINE_LIMIT] for line in reversed(lines) if line), "")
