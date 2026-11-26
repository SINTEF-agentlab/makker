"""The MCP interface: one session's tools served to another agent.

The server runs in process against a fake kernel (no Julia), through the SDK's
own client, so these exercise the protocol path a real client takes: the tool
list, calls that outlast the wait limit, cancels, and the figure links.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import mcp_types as types
import pytest
from mcp import Client

from makker.interfaces.mcp.runtime import (
    TEXT_LIMIT,
    Launch,
    McpRuntime,
    shorten,
)
from makker.interfaces.mcp.server import ProgressRelay, build_server, last_line
from makker.juliakernel import EvalResult
from makker.lab.fakes import FakeJulia, make_fake_adapter
from makker.session import Session
from makker.session_host import SessionHost
from makker.trace.schema import ARTIFACT, TOOL_CALL, TOOL_RESULT


def _runtime(tmp_path: Path, kernel: FakeJulia | None = None, **launch: Any) -> McpRuntime:
    """A runtime whose session wraps a fake kernel."""
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)

    async def start_host() -> SessionHost:
        session = Session.create(
            julia=kernel or FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
        )
        return SessionHost(session=session, agent=None)

    return McpRuntime(Launch(workspace=workspace, **launch), start_host=start_host)


def _text(result: types.CallToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, types.TextContent))


class _Gate:
    """A kernel whose eval prints two lines, then blocks until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()  # the eval has printed and is blocked
        self.release = asyncio.Event()
        self.kernel = FakeJulia(eval_handler=self._eval, stream_chunks=["step 1\n", "step 2\n"])

    async def _eval(self, code: str) -> EvalResult:
        self.started.set()
        await self.release.wait()
        return EvalResult(output=f"done: {code}")


def test_the_server_answers_before_loading_the_agent_stack() -> None:
    # A client gives a server seconds to connect and list its tools. Starting
    # loads no LangChain, and listing loads the tools but not the agent stack.
    code = (
        "import sys, makker.__main__, makker.interfaces.mcp.cli, makker.interfaces.mcp.server;"
        "loaded=lambda: [m for m in ('langchain_core', 'deepagents') if m in sys.modules];"
        "started=loaded();"
        "from makker.interfaces.mcp import specs; specs.tool_specs();"
        "print(started, loaded())"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "[] ['langchain_core']", result.stdout + result.stderr


async def test_connecting_and_listing_tools_start_no_session(tmp_path) -> None:
    # A client starts the server for every session of its own, in any folder, so
    # one that never calls Makker must not cost a Julia process.
    starts = 0

    async def start_host() -> SessionHost:
        nonlocal starts
        starts += 1
        raise RuntimeError("no Julia here")

    runtime = McpRuntime(Launch(workspace=tmp_path), start_host=start_host)
    async with Client(build_server(runtime)) as client:
        listed = (await client.list_tools()).tools
        assert client.instructions and "no simulator" in client.instructions
        # Nor do guides, in a folder that has no simulator to have guides for.
        await client.call_tool("read_guide", {})
        await asyncio.sleep(0.05)
        idle = starts
        await client.call_tool("run_julia", {"code": "1"})

    assert idle == 0 and starts == 1
    names = [tool.name for tool in listed]
    assert names[:3] == ["run_julia", "reset_julia", "plot_julia"]
    assert {"julia_wait", "julia_interrupt", "read_guide"} <= set(names)
    by_name = {tool.name: tool for tool in listed}
    # The id a plot is filed under is the framework's to inject, not the caller's.
    assert "tool_call_id" not in by_name["plot_julia"].input_schema.get("properties", {})


async def test_a_call_runs_in_the_session_and_is_traced(tmp_path) -> None:
    kernel = FakeJulia(answers={"1 + 1": "2"})
    runtime = _runtime(tmp_path, kernel)
    async with Client(build_server(runtime)) as client:
        result = await client.call_tool("run_julia", {"code": "1 + 1"})
        events = [(e.kind, e.payload) for e in runtime.host.session.trace.events_after(0)]

    assert not result.is_error and "2" in _text(result)
    calls = [p for kind, p in events if kind == TOOL_CALL]
    results = [p for kind, p in events if kind == TOOL_RESULT]
    assert calls[-1]["name"] == "run_julia" and calls[-1]["args"] == {"code": "1 + 1"}
    assert results[-1]["tool_call_id"] == calls[-1]["id"]
    assert results[-1]["status"] == "success"


async def test_the_first_result_carries_the_instructions(tmp_path) -> None:
    # Some clients never show their model the server's instructions.
    async with Client(build_server(_runtime(tmp_path))) as client:
        instructions = client.instructions
        first = _text(await client.call_tool("run_julia", {"code": "1"}))
        second = _text(await client.call_tool("run_julia", {"code": "1"}))

    assert instructions and instructions in first
    assert instructions not in second


async def test_failures_come_back_as_error_results(tmp_path) -> None:
    async with Client(build_server(_runtime(tmp_path))) as client:
        unknown = await client.call_tool("no_such_tool", {})
        missing_argument = await client.call_tool("run_julia", {})

    assert unknown.is_error and "no_such_tool" in _text(unknown)
    assert missing_argument.is_error and "run_julia" in _text(missing_argument)


async def test_a_long_call_answers_still_running_and_is_collected_later(tmp_path) -> None:
    gate = _Gate()
    async with Client(build_server(_runtime(tmp_path, gate.kernel, wait=0.2))) as client:
        first = await client.call_tool("run_julia", {"code": "solve()"})
        await gate.started.wait()
        again = await client.call_tool("julia_wait", {})
        gate.release.set()
        done = await client.call_tool("julia_wait", {})
        nothing = await client.call_tool("julia_wait", {})

    assert not first.is_error
    assert "Still running" in _text(first) and "job-1: run_julia" in _text(first)
    assert "Still running" in _text(again)
    # What the code has printed so far comes with the news.
    assert "step 2" in _text(again)
    assert _text(done).startswith("Result of job-1 (run_julia):") and "done: solve()" in _text(done)
    assert "No call is running" in _text(nothing)


async def test_a_cancelled_call_keeps_running(tmp_path) -> None:
    # A client that times out sends the same cancel a user's stop would, so a
    # cancel must not cost the user a long simulation.
    gate = _Gate()
    async with Client(build_server(_runtime(tmp_path, gate.kernel, wait=30))) as client:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.call_tool("run_julia", {"code": "solve()"}), 0.3)
        gate.release.set()
        done = await client.call_tool("julia_wait", {})

    assert "done: solve()" in _text(done)


async def test_an_interrupt_stops_the_running_call(tmp_path) -> None:
    gate = _Gate()
    async with Client(build_server(_runtime(tmp_path, gate.kernel, wait=0.2))) as client:
        await client.call_tool("run_julia", {"code": "forever()"})
        await gate.started.wait()
        stopped = await client.call_tool("julia_interrupt", {})
        idle = await client.call_tool("julia_interrupt", {})
        nothing = await client.call_tool("julia_wait", {})

    assert "Interrupted job-1 (run_julia)" in _text(stopped)
    assert "intact" in _text(stopped) and "step 2" in _text(stopped)
    assert "No Julia code is running" in _text(idle)
    assert "No call is running" in _text(nothing)


async def test_an_interrupt_while_the_session_starts_says_so(tmp_path) -> None:
    started = asyncio.Event()

    async def start_host() -> SessionHost:
        await started.wait()
        raise AssertionError("unreachable")

    runtime = McpRuntime(Launch(workspace=tmp_path, wait=0.2), start_host=start_host)
    async with Client(build_server(runtime)) as client:
        await client.call_tool("run_julia", {"code": "1"})
        stopped = _text(await client.call_tool("julia_interrupt", {}))

    assert "still starting" in stopped and "cannot be interrupted" in stopped


async def test_an_interrupt_passes_over_a_finished_call_nobody_collected(tmp_path) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def evaluate(code: str) -> EvalResult:
        if code == "slow()":
            started.set()
            await release.wait()
        return EvalResult(output=f"done: {code}")

    kernel = FakeJulia(eval_handler=evaluate)
    async with Client(build_server(_runtime(tmp_path, kernel, wait=0.2))) as client:
        with pytest.raises(TimeoutError):  # finishes, but its caller has gone
            await asyncio.wait_for(client.call_tool("run_julia", {"code": "quick()"}), 0.001)
        await client.call_tool("run_julia", {"code": "slow()"})
        await started.wait()
        stopped = await client.call_tool("julia_interrupt", {})

    assert "Interrupted job-2 (run_julia)" in _text(stopped)


def test_a_wait_limit_must_be_positive() -> None:
    from makker.interfaces.mcp.cli import build_parser

    assert build_parser().parse_args(["--max-wait", "30"]).max_wait == 30
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-wait", "0"])


async def test_calls_take_turns(tmp_path) -> None:
    # A client may send its calls at once; the session runs them one at a time.
    gate = _Gate()
    runtime = _runtime(tmp_path, gate.kernel, wait=0.2)
    await runtime.start()  # up first, so the later call waits its turn, not for the start
    async with Client(build_server(runtime)) as client:
        first, second = await asyncio.gather(
            client.call_tool("run_julia", {"code": "a()"}),
            client.call_tool("run_julia", {"code": "b()"}),
        )
        gate.release.set()
        results = [_text(await client.call_tool("julia_wait", {})) for _ in range(2)]

    waiting = _text(second) if "job-2" in _text(second) else _text(first)
    assert "Waiting for an earlier call to finish (job-1: run_julia" in waiting
    assert "done: a()" in results[0] and "done: b()" in results[1]


async def test_a_failed_start_is_reported_then_tried_again(tmp_path) -> None:
    attempts = 0

    async def start_host() -> SessionHost:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Julia is not installed")
        session = Session.create(
            julia=FakeJulia(answers={"1": "1"}),
            simulator=make_fake_adapter(tmp_path),
            state_root=tmp_path,
        )
        return SessionHost(session=session, agent=None)

    runtime = McpRuntime(Launch(workspace=tmp_path), start_host=start_host)
    async with Client(build_server(runtime)) as client:
        failed = await client.call_tool("run_julia", {"code": "1"})
        retried = await client.call_tool("run_julia", {"code": "1"})

    assert failed.is_error and "Julia is not installed" in _text(failed)
    assert not retried.is_error and attempts == 2


async def test_guides_are_listed_and_read(tmp_path) -> None:
    runtime = _runtime(tmp_path, simulator="jutuldarcy")
    async with Client(build_server(runtime)) as client:
        assert "jutuldarcy-overview" in (client.instructions or "")
        listing = _text(await client.call_tool("read_guide", {}))
        # An agent reads a guide before it writes code, so that starts Julia.
        async with asyncio.timeout(30):
            while runtime.host is None:
                await asyncio.sleep(0.01)
        guide = _text(await client.call_tool("read_guide", {"name": "jutuldarcy-overview"}))
        unknown = _text(await client.call_tool("read_guide", {"name": "nope"}))
        # Once the session is up, a guide read is in the trace like any call, so
        # a transcript shows whether the other agent consulted the guides.
        await client.call_tool("run_julia", {"code": "1"})
        await client.call_tool("read_guide", {"name": "plotting-basics"})
        traced = [e.payload for e in runtime.host.session.trace.events_of(TOOL_CALL)]

    assert traced[-1]["name"] == "read_guide"
    assert traced[-1]["args"] == {"name": "plotting-basics"}
    # Including the read that started the session, made before there was a trace.
    assert [(call["name"], call["args"]) for call in traced[:2]] == [
        ("read_guide", {}),
        ("read_guide", {"name": "jutuldarcy-overview"}),
    ]
    assert "- jutuldarcy-overview:" in listing and "- plotting-basics:" in listing
    assert "JutulDarcy" in guide and "skills" in guide  # the text, then where its files are
    assert "no guide named 'nope'" in unknown and "- jutuldarcy-overview:" in unknown


def test_long_text_reaches_the_client_as_its_start_and_end(tmp_path) -> None:
    path = tmp_path / "tool-results" / "call.txt"
    text = "a" * TEXT_LIMIT + "MIDDLE" + "z" * TEXT_LIMIT

    short = shorten(text, path)

    assert len(short) < TEXT_LIMIT and short.startswith("aaa") and short.endswith("zzz")
    assert "MIDDLE" not in short and str(path) in short
    assert path.read_text(encoding="utf-8") == text
    assert shorten("brief", path) == "brief"
    blocks = shorten([{"type": "text", "text": text}, {"type": "image", "base64": "x"}], path)
    assert len(blocks[0]["text"]) < TEXT_LIMIT and blocks[1] == {"type": "image", "base64": "x"}


async def test_progress_carries_the_last_line_of_output(monkeypatch) -> None:
    import makker.interfaces.mcp.server as server

    monkeypatch.setattr(server, "PROGRESS_INTERVAL", 0.01)
    seen: list[str] = []

    async def report(_progress: float, _total: float | None, message: str | None) -> None:
        seen.append(message or "")

    relay = ProgressRelay(report)
    relay.write("Step 1\nStep ")
    relay.write("2 of 10\r\x1b[2KStep 3 of 10\n\n")
    async with asyncio.timeout(5):
        while not seen:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # a few more ticks, with nothing new to send
    relay.close()

    # Only what is new is sent: the last line that says something.
    assert seen == ["Step 3 of 10"]
    assert last_line("") == ""


async def test_figure_links_open_only_with_their_key(tmp_path) -> None:
    from makker.interfaces.mcp.pages import Pages

    session = Session.create(
        julia=FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )
    session.web_plot_secret, session.web_plot_port = "s3cret", 1
    host = SessionHost(session=session, agent=None)
    pages = Pages(lambda: host)
    try:
        base = await pages.base_url()
        page = f"/popout/{session.session_id}?route=fig&w=600&h=400"
        async with httpx.AsyncClient() as http:
            opened = await http.get(base + page)
            followed = await http.get(base + page + "&follow=1")
            drawn = f"/popout/{session.session_id}/drawn?route=fig"
            before = (await http.get(base + drawn)).json()
            session.trace.append(ARTIFACT, {"slot": "fig", "size_px": [640, 480]})
            after = (await http.get(base + drawn)).json()
            keyless = await http.get(base.rsplit("/", 1)[0] + page)
            other_session = await http.get(f"{base}/popout/2020-01-01-0000-0000?route=fig&w=1&h=1")
            other_host = await http.get(base + page, headers={"Host": "evil.example"})
    finally:
        await pages.aclose()
        session.finalize()

    assert opened.status_code == 200
    # The page frames the live figure under the same key, at the plot's own secret.
    assert f"{pages.prefix}/live/{session.session_id}/s3cret/viz/fig" in opened.text
    # Nothing else refreshes a page opened from a link, so with `follow` it asks
    # which drawing of its plot is current, and loads the new one when that changes.
    assert pages.prefix + drawn in followed.text and drawn not in opened.text
    assert before == {"drawn": None, "size": None}
    assert after["drawn"] > 0 and after["size"] == [640, 480]
    assert keyless.status_code == 404
    assert other_session.status_code == 404
    assert other_host.status_code == 403


async def test_a_page_server_that_cannot_start_is_tried_again(monkeypatch) -> None:
    import socket

    from makker.interfaces.mcp.pages import Pages

    bind = socket.socket.bind
    failures = iter([OSError("no ports left")])

    def bind_once_failing(sock: socket.socket, address: Any) -> None:
        if (error := next(failures, None)) is not None:
            raise error
        bind(sock, address)

    monkeypatch.setattr(socket.socket, "bind", bind_once_failing)
    pages = Pages(lambda: None)
    try:
        with pytest.raises(OSError):
            await pages.base_url()
        assert (await pages.base_url()).endswith(pages.prefix)
    finally:
        await pages.aclose()
