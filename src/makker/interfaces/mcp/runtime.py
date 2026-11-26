"""The session behind the MCP server, and how its tools are called.

One server process is one session: one Julia kernel, one trace, one workspace.
A client starts a server for every one of its own sessions, in whatever folder
that is, so nothing is started until Makker is first used: a session that never
uses it costs no Julia process.

A client also cuts a call off after a while (some after a minute), and tells the
server with the same cancel a user's stop would send. So no call holds the client
past ``wait`` seconds: one that needs longer answers "still running", keeps going
as a job, and ``julia_wait`` collects it. A cancel therefore only stops the
waiting, never the work; ``julia_interrupt`` stops the work.

A call goes through the session's own tools (``session_tools``, the list the
bundled agent is given) and is recorded in the trace as the agent's calls are,
so a transcript or report of an MCP session reads like any other.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from makker.agent.skills import skill_index
from makker.interfaces.cli._helpers import folder_simulator, remember_simulator
from makker.interfaces.mcp import specs
from makker.interfaces.mcp.pages import Pages
from makker.interfaces.mcp.specs import GUIDE, INTERRUPT, WAIT
from makker.juliakernel.text import render_terminal_output
from makker.simulators import registry

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from makker.session_host import SessionHost
    from makker.simulators.base import SimulatorAdapter

SURFACE = "mcp"

# How long a call may hold the client before it answers "still running": under
# the strictest tool time limit seen in a client (a minute).
DEFAULT_WAIT = 45.0

# Text longer than this reaches the client as its start and end, with all of it
# saved to a file the result names: a printed array or solver log can run to
# hundreds of thousands of characters, which would fill the other agent's context.
TEXT_LIMIT = 16_000
_TEXT_HEAD = 10_000
_TEXT_TAIL = 4_000

# How much of a running job's live output is kept, and how much of it a "still
# running" answer shows.
_OUTPUT_KEPT = 16_000
_OUTPUT_SHOWN = 2_000

# How many finished calls wait for a `julia_wait` that may never come.
_RESULTS_KEPT = 20

# How long an interrupt waits for Julia to unwind. The kernel's own recovery is
# bounded (it restarts Julia when an interrupt is ignored).
_INTERRUPT_GRACE = 60.0


@dataclass(frozen=True)
class Launch:
    """What the server runs: a folder, and how its session is set up."""

    workspace: Path
    simulator: str | None = None  # as asked for; the folder's own when None
    julia_project: Path | None = None
    threads: str | None = None
    sysimage: bool | None = None
    wait: float = DEFAULT_WAIT


class StartupFailed(Exception):
    """The session could not start; the message says why, for the user."""


class UnknownTool(Exception):
    """A call named a tool this server does not have."""


@dataclass
class Outcome:
    """A finished call, or the news that it is still running."""

    content: str | list[Any]  # the tool's reply: text, or text and image blocks
    failed: bool = False
    heading: str | None = None  # said before the reply
    # Where the user can see what the call produced, and anything else to add.
    notes: list[str] = field(default_factory=list)


class Job:
    """One call to a session tool, from when it is made until its result is collected."""

    def __init__(self, number: int, tool: str) -> None:
        self.name = f"job-{number}"
        self.tool = tool
        self.task: asyncio.Task[Outcome]
        self.started = time.monotonic()
        self.running = False  # False while it waits its turn
        self.listeners: list[Callable[[str], None]] = []  # one per call waiting on it
        self._output = ""

    def write(self, text: str) -> None:
        """Take a fragment of live output: keep its tail, and pass it to whoever waits."""
        self._output = (self._output + text)[-_OUTPUT_KEPT:]
        for listener in self.listeners:
            listener(text)

    def output(self) -> str:
        """The end of the output so far, as a terminal would show it."""
        return render_terminal_output(self._output).strip()[-_OUTPUT_SHOWN:]

    def seconds(self) -> int:
        return round(time.monotonic() - self.started)


def _import_agent_stack() -> None:
    """Import what a session is built from, off the event loop.

    It takes seconds, and the server must keep answering the client meanwhile.
    Nothing else may import any of it until this returns: two threads importing
    LangChain at once deadlock on each other's module locks. So the code that
    calls the tools imports them only once the session is up.
    """
    importlib.import_module("makker.session_host")
    importlib.import_module("makker.agent.builder")


class McpRuntime:
    """The one session a server process runs, started when Makker is first used."""

    def __init__(
        self,
        launch: Launch,
        *,
        start_host: Callable[[], Awaitable[SessionHost]] | None = None,
    ) -> None:
        """``start_host`` stands in for the real session start, so tests can wrap fakes."""
        self.launch = launch
        self._start_host = start_host
        self.host: SessionHost | None = None
        self._tools: dict[str, BaseTool] = {}  # the session's, once it is up
        self._startup: asyncio.Task[SessionHost] | None = None
        self._lock = asyncio.Lock()  # one call in the session at a time
        self._jobs: dict[str, Job] = {}  # unfinished or uncollected, oldest first
        self._count = 0
        self._client: str | None = None  # what the client calls itself
        # Calls the server answered itself before there was a trace to record them in.
        self._unrecorded: list[tuple[datetime, str, dict[str, Any], Outcome]] = []
        self._pages = Pages(lambda: self.host)

    def simulator(self) -> SimulatorAdapter | None:
        """The folder's simulator, asked afresh: the user may set it while connected."""
        name = folder_simulator(self.launch.simulator, self.launch.workspace)
        return registry.get(name) if name in registry.names() else None

    def note_client(self, name: str | None) -> None:
        """Name the session after the client: there is no first prompt to take a title from."""
        self._client = name
        self._title()

    def _title(self) -> None:
        if self.host is not None and self._client and self.host.session.title is None:
            self.host.session.retitle(f"{self._client} session")

    def start(self) -> asyncio.Task[SessionHost]:
        """Begin starting the session, unless that has begun, without waiting for it."""
        if self._startup is None:
            specs.tool_specs()  # imported here first, not in two threads at once
            self._startup = asyncio.create_task(self._start_session())
        return self._startup

    async def _start_session(self) -> SessionHost:
        await asyncio.to_thread(_import_agent_stack)
        from makker.agent.tools import session_tools

        host = await (self._start_host or self._start_real_host)()
        host.session.base_path = self._pages.prefix
        self._tools = {tool.name: tool for tool in session_tools(host.session, surface=SURFACE)}
        self.host = host
        for at, name, arguments, outcome in self._unrecorded:
            self._trace_answered(name, arguments, outcome, at)
        self._unrecorded.clear()
        self._title()
        _note("Julia is up")
        return host

    async def _start_real_host(self) -> SessionHost:
        from makker.session_host import SessionHost

        launch, adapter = self.launch, self.simulator()
        if adapter is None:
            raise StartupFailed(
                f"no simulator is set for {launch.workspace}. Run `makker init --sim <name>` "
                f"there, with one of: {', '.join(registry.names())}. If that is not the "
                "folder to work in, the server's arguments need `--workspace <folder>`."
            )
        remember_simulator(adapter.name, launch.workspace)
        _note(f"starting Julia for {adapter.display_name} in {launch.workspace}")
        return await SessionHost.start(
            simulator=adapter,
            surface=SURFACE,
            workspace=launch.workspace,
            julia_project=launch.julia_project,
            threads=launch.threads,
            open_windows=False,
            with_agent=False,
            sysimage=launch.sysimage,
        )

    async def _session(self) -> SessionHost:
        """The running session, once it is up; ``StartupFailed`` if it cannot start.

        A failed start is reported to the call that finds it and then forgotten, so
        the next call tries again: the user may have fixed what was wrong.
        """
        startup = self.start()
        try:
            # Shielded: one caller giving up must not cancel the start others share.
            return await asyncio.shield(startup)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._startup is startup:
                self._startup = None
            if isinstance(exc, StartupFailed):
                raise
            raise StartupFailed(f"Makker could not start its Julia session: {exc}") from exc

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_output: Callable[[str], None] | None = None,
    ) -> Outcome:
        """Make one call, waiting at most ``wait`` seconds; ``on_output`` gets live output."""
        if name == GUIDE:
            return self._record(name, arguments, self._guide(arguments.get("name")))
        if name == INTERRUPT:
            return self._record(name, arguments, await self._interrupt())
        if name == WAIT:
            # The oldest call whose result has not been collected.
            job = next(iter(self._jobs.values()), None)
            if job is None:
                return Outcome("No call is running, and every result has been collected.")
        elif name in specs.session_tool_names():
            # A result nobody collected goes once enough newer ones have come.
            finished = [job.name for job in self._jobs.values() if job.task.done()]
            for stale in finished[:-_RESULTS_KEPT]:
                del self._jobs[stale]
            self._count += 1
            job = Job(self._count, name)
            job.task = asyncio.create_task(self._run(job, arguments))
            self._jobs[job.name] = job
        else:
            raise UnknownTool(name)
        outcome = await self._attend(job, on_output)
        if name == WAIT and job.name not in self._jobs:
            # Say whose result this is: it need not be the call made last.
            outcome.heading = f"Result of {job.name} ({job.tool}):"
        return outcome

    def _trace_call(
        self, call_id: str, name: str, arguments: dict[str, Any], at: datetime | None = None
    ) -> None:
        # Imported only once the session is up: see ``_import_agent_stack``.
        from makker.trace.schema import TOOL_CALL, tool_call_payload

        assert self.host is not None
        payload = tool_call_payload(id=call_id, name=name, args=arguments)
        self.host.session.trace.append(TOOL_CALL, payload, at=at)

    def _trace_result(
        self, call_id: str, name: str, content: Any, failed: bool, at: datetime | None = None
    ) -> None:
        from makker.trace.messages import content_to_str
        from makker.trace.schema import TOOL_RESULT, tool_result_payload

        assert self.host is not None
        status = "error" if failed else "success"
        payload = tool_result_payload(
            tool_call_id=call_id, name=name, content=content_to_str(content), status=status
        )
        self.host.session.trace.append(TOOL_RESULT, payload, at=at)

    def _trace_answered(
        self, name: str, arguments: dict[str, Any], outcome: Outcome, at: datetime | None = None
    ) -> None:
        """Trace a call the server answered itself, with its result."""
        call_id = _call_id()
        self._trace_call(call_id, name, arguments, at)
        self._trace_result(call_id, name, outcome.content, outcome.failed, at)

    def _record(self, name: str, arguments: dict[str, Any], outcome: Outcome) -> Outcome:
        """Trace a call the server answered itself; one made before the session is
        up (such as the guide read that starts it) is traced once it is."""
        if self.host is None:
            self._unrecorded.append((datetime.now(UTC), name, arguments, outcome))
        else:
            self._trace_answered(name, arguments, outcome)
        return outcome

    async def _attend(self, job: Job, on_output: Callable[[str], None] | None) -> Outcome:
        """Wait on ``job`` for at most ``wait`` seconds; its result, or how far it is.

        ``asyncio.wait`` leaves the job running when the wait ends, or when the
        caller is cancelled: the job is then collected by a later ``julia_wait``.
        """
        if on_output is not None:
            job.listeners.append(on_output)
        try:
            done, _ = await asyncio.wait({job.task}, timeout=self.launch.wait)
        finally:
            if on_output is not None:
                job.listeners.remove(on_output)
        if not done:
            return Outcome(self._still_running(job))
        if self._jobs.pop(job.name, None) is None:
            # Two calls waited on it, and the result goes to one.
            return Outcome(f"{job.name} ({job.tool}) finished; its result went to the other call.")
        try:
            outcome = job.task.result()
        except asyncio.CancelledError:
            outcome = Outcome(f"{job.name} ({job.tool}) was interrupted.", failed=True)
        except Exception as exc:
            outcome = Outcome(f"ERROR: {type(exc).__name__}: {exc}", failed=True)
        if waiting := ", ".join(f"{other.name} ({other.tool})" for other in self._jobs.values()):
            outcome.notes.append(f"Not yet collected: {waiting}. `{WAIT}` returns the next.")
        return outcome

    def _still_running(self, job: Job) -> str:
        if self.host is None:
            state = (
                "Makker is still starting its Julia session. The first time in a folder "
                "it installs the Julia environment, which takes several minutes."
            )
        elif not job.running:
            ahead = next((j for j in self._jobs.values() if j.running), None)
            state = "Waiting for an earlier call to finish" + (
                f" ({ahead.name}: {ahead.tool}, {ahead.seconds()} s so far)." if ahead else "."
            )
        else:
            state = ""
        text = f"Still running after {job.seconds()} s ({job.name}: {job.tool}). {state}".strip()
        if output := job.output():
            text += f"\n\nOutput so far:\n{output}"
        return (
            f"{text}\n\nCall `{WAIT}` to keep waiting for the result, "
            f"or `{INTERRUPT}` to stop the running code."
        )

    async def _run(self, job: Job, arguments: dict[str, Any]) -> Outcome:
        """Run ``job`` when its turn comes, recorded in the trace."""
        async with self._lock:
            try:
                host = await self._session()
            except StartupFailed as exc:
                return Outcome(f"ERROR: {exc}", failed=True)
            # Only now: see ``_import_agent_stack``.
            from makker.agent.tools import tool_output_writer
            from makker.trace.messages import content_to_str
            from makker.trace.schema import ARTIFACT

            tool = self._tools[job.tool]
            job.running = True
            trace, call_id = host.session.trace, _call_id()
            mark = trace.max_id()
            self._trace_call(call_id, job.tool, arguments)
            token = tool_output_writer.set(job.write)
            try:
                content, failed = await _invoke(tool, call_id, arguments)
            except asyncio.CancelledError:
                self._trace_result(call_id, job.tool, "Interrupted.", True)
                raise
            finally:
                tool_output_writer.reset(token)
            self._trace_result(call_id, job.tool, content, failed)  # the status, as the agent's
            # Calls take turns, so whatever was saved meanwhile is this call's.
            artifacts = [event.payload for event in trace.events_after(mark, kinds=(ARTIFACT,))]
            saved = host.session.output_dir / "tool-results" / f"{call_id}.txt"
            return Outcome(
                content=shorten(content, saved),
                # The tools also report a failure as text that starts with "ERROR".
                failed=failed or content_to_str(content).startswith("ERROR"),
                notes=await self._links(host, artifacts),
            )

    async def _links(self, host: SessionHost, artifacts: list[dict[str, Any]]) -> list[str]:
        """Where the user can see what a call produced: live figures and saved files."""
        from makker.agent.plot_julia import _plot_id_of

        session, links = host.session, []
        for payload in artifacts:
            if payload.get("live_url"):
                width, height = payload.get("authored_px") or payload.get("size_px") or (900, 600)
                try:
                    base = await self._pages.base_url()
                except Exception as exc:  # the plot still stands without its live view
                    links.append(f"The live figure could not be served ({exc}).")
                else:
                    # `follow`: nothing else refreshes this page, so it keeps up
                    # with the plot itself when the plot is drawn again.
                    links.append(
                        f"Live figure: {base}/popout/{session.session_id}"
                        f"?route={quote(_plot_id_of(payload))}&w={int(width)}&h={int(height)}"
                        "&follow=1"
                    )
            for key in ("poster", "path"):
                if payload.get(key):
                    links.append(f"Saved: {session.output_dir / payload[key]}")
                    break
        return links

    async def _interrupt(self) -> Outcome:
        """Stop the call that is in Julia now.

        Cancelling its task is enough: the kernel answers a cancelled eval by
        interrupting Julia and waiting for the code to unwind, and restarts Julia
        only if the interrupt is ignored.
        """
        job = next((j for j in self._jobs.values() if j.running and not j.task.done()), None)
        if job is None:
            if self._startup is not None and not self._startup.done():
                return Outcome(
                    "Makker is still starting its Julia session (the first time in a folder "
                    "it installs the Julia environment), which cannot be interrupted. "
                    f"`{WAIT}` says when it is up."
                )
            return Outcome("No Julia code is running.")
        output = job.output()
        job.task.cancel()
        done, _ = await asyncio.wait({job.task}, timeout=_INTERRUPT_GRACE)
        if not done:
            # Still the job that holds the session, so it stays where it can be collected.
            return Outcome(
                f"{job.name} ({job.tool}) has not stopped after {_INTERRUPT_GRACE:.0f} s. Julia "
                f"is being restarted to stop it; `{WAIT}` says when that is done.",
                failed=True,
            )
        self._jobs.pop(job.name, None)
        julia = self.host.session.julia if self.host is not None else None
        kept = getattr(julia, "cancel_preserved_state", True)
        text = f"Interrupted {job.name} ({job.tool}) after {job.seconds()} s. " + (
            "Variables and loaded packages are intact."
            if kept
            else "Julia had to be restarted to stop it, so the session is new: "
            "load the packages and rebuild the state again."
        )
        return Outcome(text + (f"\n\nOutput before the interrupt:\n{output}" if output else ""))

    def _guide(self, name: Any) -> Outcome:
        """The guides for this folder's simulator: their list, or one of them in full."""
        adapter = self.simulator()
        if adapter is None:
            return Outcome("This folder has no simulator set, so there are no guides yet.")
        # An agent reads a guide before it writes code, so Julia starts now and is
        # up, or nearly, when the code arrives.
        self.start()
        skills = skill_index(adapter)
        skill = next((skill for skill in skills if skill.name == name), None)
        if skill is None:
            listing = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
            unknown = f"There is no guide named {name!r}. " if name else ""
            return Outcome(f"{unknown}Guides for {adapter.display_name}:\n{listing}")
        text = skill.path.read_text(encoding="utf-8")
        return Outcome(f"{text}\n\n(Files this guide refers to are under {skill.path.parent})")

    async def aclose(self) -> None:
        startup, self._startup = self._startup, None
        jobs, self._jobs = list(self._jobs.values()), {}
        pending = [job.task for job in jobs if not job.task.done()]
        if startup is not None and not startup.done():
            pending.append(startup)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)
        for task in (*[job.task for job in jobs], startup):
            if task is not None and not task.cancelled():
                task.exception()  # retrieved; whoever waited was told
        host, self.host = self.host, None
        if host is not None:
            await host.aclose()
        await self._pages.aclose()


def _call_id() -> str:
    return f"mcp-{uuid.uuid4().hex[:12]}"


async def _invoke(tool: BaseTool, call_id: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
    """Call a tool as the agent would (as a tool call, so the id it files a figure
    under is the call's): its reply, and whether it failed."""
    call = {"type": "tool_call", "id": call_id, "name": tool.name, "args": arguments}
    try:
        message = await tool.ainvoke(call)
    except Exception as exc:  # bad arguments, or a tool that raised
        return f"ERROR: `{tool.name}` failed: {type(exc).__name__}: {exc}", True
    return message.content, getattr(message, "status", None) == "error"


def shorten(content: Any, path: Path) -> Any:
    """Cut a long text reply to its start and end, saving all of it to ``path``."""
    if isinstance(content, list):
        return [
            {**block, "text": shorten(block["text"], path)}
            if isinstance(block, dict) and isinstance(block.get("text"), str)
            else block
            for block in content
        ]
    if not isinstance(content, str) or len(content) <= TEXT_LIMIT:
        return content
    saved = f"all {len(content):,} characters are in {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        saved = f"the rest could not be saved ({exc})"
    cut = len(content) - _TEXT_HEAD - _TEXT_TAIL
    return (
        f"{content[:_TEXT_HEAD]}\n\n[... {cut:,} characters left out; {saved} ...]\n\n"
        f"{content[-_TEXT_TAIL:]}"
    )


def _note(message: str) -> None:
    """A line for the client's server log (stderr; stdout is the protocol)."""
    print(f"makker mcp: {message}", file=sys.stderr, flush=True)
