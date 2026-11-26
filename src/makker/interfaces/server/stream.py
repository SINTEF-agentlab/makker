"""The per-session turn stream: the WebSocket at ``/sessions/{id}/stream``.

A turn belongs to the session (``_SessionStream``), not to the socket that
started it. A socket that attaches gets a snapshot of the session first, then
follows the running turn live.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from makker.agent.approval import SUPPORTED_APPROVAL_DECISIONS, build_resume_payload
from makker.agent.turns import TurnUsage
from makker.context_panel import context_meter
from makker.interfaces.server import protocol
from makker.interfaces.server.deps import ManagerDep
from makker.interfaces.server.live import _authored_of, _fit_target, _sane_size
from makker.interfaces.server.manager import SessionManager
from makker.interfaces.server.replay import artifact_wire_events, replay_events
from makker.interfaces.server.urls import public_url
from makker.preview import TOOL_STREAM_RENDER_INTERVAL, TOOL_STREAM_TAIL_CAP
from makker.session_host import SessionHost
from makker.trace import schema

router = APIRouter()


@router.websocket("/sessions/{session_id}/stream")
async def stream(websocket: WebSocket, session_id: str, manager: ManagerDep) -> None:
    await _serve_stream(websocket, manager, session_id)


async def _serve_stream(websocket: WebSocket, manager: SessionManager, session_id: str) -> None:
    await websocket.accept()
    # Claim the session atomically: acquire promotes + attaches it under the manager
    # lock, so it can't be evicted in the window between lookup and attach (which would
    # leave us running turns against a torn-down kernel). One connection per session: a
    # second (e.g. a duplicate tab) is refused cleanly.
    host = await manager.acquire(session_id)
    if host is None:
        existing = manager.get(session_id)
        message = (
            "this session is already open in another window"
            if existing is not None and existing.attached
            else "no such session"
        )
        await _safe_send(websocket, protocol.error_to_wire(message))
        await websocket.close()
        return

    stream = _stream_of(host)
    try:
        await stream.attach(websocket)
        while True:
            try:
                message = await websocket.receive_json()
            except (ValueError, KeyError, TypeError):  # not JSON, or a binary frame
                message = None
            if not isinstance(message, dict):
                await _safe_send(
                    websocket, protocol.error_to_wire("invalid message (expected a JSON object)")
                )
                continue
            await stream.handle(message)
    except WebSocketDisconnect:
        pass
    finally:
        # The turn, if one is running, carries on without this socket. detach()
        # must run even so, or no later connection could acquire the session.
        try:
            stream.detach(websocket)
        finally:
            host.detach()


def _stream_of(host: SessionHost) -> _SessionStream:
    """The session's stream, made on its first connection and kept with the host."""
    if host.turn_stream is None:
        host.turn_stream = _SessionStream(host)
    return host.turn_stream


# What a turn sends that the conversation shows, kept for a socket that attaches
# before the turn ends. The rest answers the socket that asked, or ends the turn.
_TURN_CONTENT = frozenset({"text", "reasoning", "tool", "viz", "artifact", "ui"})


class _SessionStream:
    """One session's turns and what they send, whichever socket is watching.

    A turn belongs to the session, not to the connection that started it:
    switching chats or a dropped connection leaves it running (a simulation
    minutes in once died because the user looked at another chat), and so does
    closing the tab; the stop button ends it. The running turn also keeps the
    manager from evicting the kernel under it. It sends to whichever socket is
    attached now, if any. A socket that attaches first gets a snapshot of the
    session: the recorded conversation and, while a turn runs, everything that
    turn has sent so far. So a reconnect shows the turn exactly as a socket that
    never left would, and then follows it live.
    """

    def __init__(self, host: SessionHost) -> None:
        self._host = host
        self._ws: WebSocket | None = None
        self._pending: list[Any] = []
        self._turn: asyncio.Task[None] | None = None
        # The running turn, for a socket that attaches before it ends: the last
        # trace event before it, its prompt (the client shows its own), and the
        # content it has sent, merged as the client would merge it.
        self._turn_start_id = 0
        self._turn_prompt: str | None = None
        self._turn_log: list[dict[str, Any]] = []
        # Whether the running turn has already told the client it ended, so the
        # backstop in ``_run_turn`` never sends a second end.
        self._turn_ended = True
        # Whether this turn has already nudged the client to re-read its session
        # list; see ``_announce_history_once``.
        self._announced = True
        # Held so the fire-and-forget titling task isn't garbage-collected mid-run.
        self._title_task: asyncio.Task[None] | None = None
        # High-water mark over trace event ids for side outputs (artifacts, ui),
        # so each is forwarded exactly once whether flushed mid-turn or at the end.
        self._side_output_id = 0
        # Per-tool-call streaming state, for rendering tool output the way the TUI
        # does (terminal-rendered, throttled): the accumulated raw output, the last
        # render time, the last delta's wire (a send template), and any pending
        # trailing-flush task.
        self._tool_streams: dict[str, str] = {}
        self._tool_render_at: dict[str, float] = {}
        self._tool_delta_wire: dict[str, dict[str, Any]] = {}
        self._tool_flush: dict[str, asyncio.Task[None]] = {}
        # This connection's popout views: live URL -> route id, so a close can
        # only release a route this connection's own popouts created.
        self._popouts: dict[str, str] = {}
        self._popout_seq = 0
        # A host-app selection that arrived mid-turn, held as a 1-tuple so a
        # deferred ``None`` (the host cleared its selection) is distinguishable
        # from "nothing is waiting". Applied when the turn settles.
        self._deferred_host_context: tuple[dict[str, Any] | None] | None = None
        # In-flight refit evals (see ``_refit``): held so they aren't collected.
        self._refits: set[asyncio.Task[None]] = set()

    async def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        # A selection that arrived mid-turn is applied at the first idle moment.
        # Doing it here as well as at turn end covers the turns that never reach
        # a clean end (cancelled, or paused on an approval that is then
        # abandoned): the next thing the user does gets the current selection.
        if self._deferred_host_context is not None and not self._busy():
            await self._flush_host_context()
        if kind == "prompt":
            text = str(message.get("text") or "")
            _log_prompt(self._host.session_id, text)
            await self._start_prompt(text)
        elif kind == "decision":
            await self._start_decision(message)
        elif kind == "cancel":
            await self.cancel_turn()
        elif kind == "replot":
            await self._start_replot(message)
        elif kind == "refit":
            await self._refit(message)
        elif kind == "popout_closed":
            await self._close_popout(str(message.get("url") or ""))
        elif kind == "ui_event":
            payload = message.get("payload")
            # The canvas-size hint doubles as live session state: plot_julia
            # shapes a new figure toward the panel it will land in.
            if isinstance(payload, dict) and payload.get("kind") == "canvas_size":
                size = _sane_size(payload.get("width"), payload.get("height"))
                if size is not None:
                    self._host.session.web_canvas_hint = (size[0], size[1])
            self._host.session.trace.append(schema.UI_EVENT, {"payload": payload})
        elif kind == "host_context":
            await self.set_host_context(message.get("context"))
        elif kind == "command":
            await self._handle_command(message)
        else:
            await self._send(protocol.error_to_wire(f"unknown message {kind!r}"))

    async def set_host_context(self, context: Any) -> None:
        """Adopt a selection the host application changed while the session was open.

        The selection lives in the system prompt, so adopting one rebuilds the
        agent, which must not happen under a running turn (it would swap the
        runner mid-flight). A change that arrives during a turn is therefore held
        and applied when the turn settles, which is also the friendlier
        behaviour: the turn the user is watching finishes against the objects it
        started on, instead of the ground moving underneath it.
        """
        if context is not None and not isinstance(context, dict):
            await self._send(protocol.error_to_wire("host_context must be a JSON object"))
            return
        if self._busy():
            self._deferred_host_context = (context,)
            return
        self._deferred_host_context = None
        await self._apply_host_context(context)

    async def _flush_host_context(self) -> None:
        """Apply a selection that arrived mid-turn, now that the turn is settling.

        Deliberately does not re-check ``_busy``: the caller at the end of a turn
        is still running inside that turn's own task, so the check it already
        made ("nothing else is in flight") is the meaningful one.
        """
        held, self._deferred_host_context = self._deferred_host_context, None
        if held is not None:
            await self._apply_host_context(held[0])

    async def _apply_host_context(self, context: dict[str, Any] | None) -> None:
        """Adopt the selection and tell the user, if it is not what we already had.

        A failed rebuild is reported and the session kept, like a rejected model
        or approval mode: this runs from the message loop, so letting it raise
        would take the connection down over a selection change. The session has
        already recorded the new selection and will state it at the next
        successful rebuild, so the miss is transient rather than lost.
        """
        try:
            changed = self._host.set_host_context(context)
        except Exception as exc:
            await self._send(
                protocol.error_to_wire(f"could not apply the application's selection: {exc}"),
            )
            return
        if changed:
            await self._send(
                protocol.notice_to_wire("The application's selection changed; the agent was told."),
            )

    async def _handle_command(self, message: dict[str, Any]) -> None:
        """Apply a session setting (model, approval policy) mid-conversation.

        Rebuilds the agent in place — the kernel, the conversation history, and the
        live Julia state all survive — so a front end can offer these as commands.
        """
        if self._busy():
            await self._send(
                protocol.error_to_wire("finish the current turn before changing settings"),
            )
            return
        command = message.get("command")
        arg = str(message.get("arg") or "")
        try:
            if command == "set_model":
                # A model switch under a paused approval would rebuild the agent
                # out from under the pending interrupt (the TUI refuses this too).
                if self._pending or await self._host.pending_interrupts():
                    await self._send(
                        protocol.error_to_wire(
                            "answer the pending approval before switching models"
                        ),
                    )
                    return
                # Mirror the TUI: a model whose key is missing prompts for it
                # instead of failing the switch. The UI saves the key, then retries.
                from makker.credentials import missing_credential
                from makker.models import (
                    local_model_error,
                    missing_provider_error,
                    provider_info,
                )

                # A prefix-less free-text spec would skip every check below and
                # fail opaquely on the next turn instead.
                problem = missing_provider_error(arg)
                if problem is not None:
                    await self._send(protocol.error_to_wire(problem))
                    return
                env_var = missing_credential(arg)
                if env_var is not None:
                    info = provider_info(arg)
                    await self._send(
                        protocol.credential_required_to_wire(
                            provider=info.name if info else "",
                            label=info.label if info else arg,
                            env_var=env_var,
                        ),
                    )
                    return
                # A local model that isn't reachable/pulled/tool-capable fails
                # here with a reason, not opaquely on the next turn.
                problem = await local_model_error(arg)
                if problem is not None:
                    await self._send(protocol.error_to_wire(problem))
                    return
                self._host.reconfigure(model=arg)
                from makker.model_choices import remember_model
                from makker.paths import workspace_root

                # Saved like the terminal's switch: the next session here starts on it.
                remember_model(arg, workspace=self._host.workspace or workspace_root())
                # The page shows the switch only once it has taken effect.
                await self._send(protocol.model_to_wire(self._host.model))
                await self._send_context()
            elif command == "set_approval":
                self._host.set_approval_mode(arg)
                # A pending approval the new policy already allows resumes now
                # instead of waiting for a decision that is a foregone conclusion.
                await self._auto_resolve_pending()
            elif command == "add_dir":
                await self._send(protocol.notice_to_wire(self._host.add_dir(arg)))
            elif command == "compact":
                # Compacting rewrites the conversation, which drops the paused
                # call along with the decision about to be given for it.
                if self._pending or await self._host.pending_interrupts():
                    await self._send(
                        protocol.error_to_wire("answer the pending approval before compacting"),
                    )
                    return
                note, _result = await self._host.compact()
                await self._send(protocol.notice_to_wire(note))
                await self._send_context()
            else:
                await self._send(protocol.error_to_wire(f"unknown command {command!r}"))
                return
        except Exception as exc:  # surface a bad model/mode, keep the session alive
            await self._send(protocol.error_to_wire(f"could not apply {command}: {exc}"))

    async def _start_prompt(self, text: str) -> None:
        if self._busy():
            await self._send(protocol.error_to_wire("a turn is already running"))
            return
        # Name the session from its first prompt, like the CLI/TUI do, so it reads
        # well in the history list. Idempotent (only the first prompt sets it).
        with contextlib.suppress(Exception):
            self._host.session.adopt_title(text)
        runner = self._host.runner
        self._spawn(lambda: runner.run_prompt(text, on_message=self._on_message), prompt=text)

    async def _start_decision(self, message: dict[str, Any]) -> None:
        if self._busy():
            await self._send(protocol.error_to_wire("a turn is already running"))
            return
        if not self._pending:
            await self._send(protocol.error_to_wire("no approval is pending"))
            return
        kind = str(message.get("decision") or "")
        if kind not in (*SUPPORTED_APPROVAL_DECISIONS, "always_allow"):
            # Never read a missing or misspelt decision as an approval.
            await self._send(protocol.error_to_wire(f"unknown decision {kind!r}"))
            return
        # "always_allow" is approve plus a session policy: remember what these cards
        # offered (file edits in the workspace), so later ones go through unasked.
        if kind == "always_allow":
            approval = self._host.approval
            offers = [approval.always_allow_offer(i.value) for i in self._pending]
            for category in frozenset.intersection(*offers):
                approval.allow(category)
            kind = "approve"
        decision: dict[str, str] = {"type": kind}
        text = message.get("message")
        if text:
            decision["message"] = str(text)
        elif kind == "respond":
            # langchain's HITL reads decision["message"] by subscript for a respond
            # (unlike reject, which uses .get), so it must always be present; an empty
            # reply is valid.
            decision["message"] = ""
        payload = build_resume_payload(self._pending, decision)
        self._pending = []
        runner = self._host.runner
        self._spawn(lambda: runner.resume(payload, on_message=self._on_message))

    async def _start_replot(self, message: dict[str, Any]) -> None:
        """Replay a recorded plot (the canvas regenerate and popout actions).

        Runs as the turn task so it shares the busy guard and the cancel path
        with agent turns: neither can run while the other holds the kernel.
        """
        record = str(message.get("record") or "")
        target = str(message.get("target") or "revive")
        if not record:
            return
        if self._busy():
            reason = "finish the current turn before regenerating a plot"
            if target == "popout":
                await self._send(protocol.popout_ready_to_wire(record, None, error=reason))
            else:
                await self._send(protocol.error_to_wire(reason))
            return
        # A regenerate adds a note to the conversation, which would land between
        # the paused call and its result and drop the decision about to be given.
        # A popout adds nothing, so it may go ahead.
        if target != "popout" and (self._pending or await self._host.pending_interrupts()):
            await self._send(
                protocol.error_to_wire("answer the pending approval before regenerating a plot"),
            )
            return
        # An optional target size the client measured (the stage on a regenerate,
        # the popup window on a popout): the replay re-fits the figure to it.
        size = _sane_size(message.get("width"), message.get("height"))
        self._spawn_task(self._replot_turn(record, target, size))

    async def _replot_turn(self, record: str, target: str, size: list[int] | None) -> None:
        """Run a replay so that it always ends its turn, whatever happens.

        A replay holds the same turn guard a prompt does, so anything escaping
        without a closing message leaves the UI spinning with nothing to click.
        ``_replot`` reports the failures it anticipates; the backstop below
        covers the rest (a trace read that throws, a flush that fails, a cancel
        arriving outside the one await that expects it).
        """
        end = self._replay_end(record, target)
        self._turn_ended = False
        try:
            await self._replot(record, target, size)
        except asyncio.CancelledError:
            await self._end_turn(protocol.turn_cancelled_to_wire())
            raise
        except Exception as exc:
            await self._end_turn(end, error=f"could not regenerate the plot: {exc}")
        finally:
            await self._end_turn(end)

    def _replay_end(self, record: str, target: str) -> dict[str, Any]:
        """What ends a replay's turn: a popout answers its request, the rest end
        the turn the client is showing as busy."""

        if target == "popout":
            return protocol.popout_ready_to_wire(record, None, error="the replay failed")
        return protocol.turn_end_to_wire([])

    async def _replot(self, record: str, target: str, size: list[int] | None = None) -> None:
        """Re-run a recorded plot's code and deliver the resulting view.

        ``record`` names the plot's artifact in the trace; the stored source code
        is what re-runs (never code a client sent). A revive re-serves on the
        plot's own route and re-finalizes the artifact, so the side-output flush
        delivers a fresh ``viz`` and the browser view revives in place. A popout
        serves an independent figure on its own route and answers with the URL of
        a scale-to-fit wrapper page hosting it, sized from the figure's echo.
        The code replays in the current kernel state, where variables may have
        changed or be gone after a restart, so a failure is reported, not hidden.
        """
        from makker.agent.plot_julia import replot_web

        popout = target == "popout"

        async def fail(reason: str) -> None:
            if popout:
                await self._end_turn(protocol.popout_ready_to_wire(record, None, error=reason))
            else:
                await self._end_turn(
                    protocol.turn_end_to_wire([]),
                    error=f"could not regenerate the plot: {reason}",
                )

        payload: dict[str, Any] | None = None
        for event in reversed(self._host.session.trace.events_of(schema.ARTIFACT)):
            if event.payload.get("path") == record and event.payload.get("source_code"):
                payload = event.payload
                break
        if payload is None:
            await fail("this plot has no recorded code to re-run")
            return
        self._side_output_id = self._latest_event_id()
        suffix = ""
        if popout:
            self._popout_seq += 1
            suffix = f"--pop{self._popout_seq}"
        try:
            err, url, size_px = await replot_web(
                self._host.session, payload, route_suffix=suffix, record=not popout, fit_to=size
            )
        except Exception as exc:  # surface the failure, keep the session alive
            # Cancellation is deliberately not caught: ``_replot_turn`` reports
            # it, in one place, for every point a cancel can arrive at.
            err, url, size_px = (
                f"the Julia session is unavailable ({type(exc).__name__}: {exc})",
                None,
                None,
            )
        if err is not None:
            await fail(err)
            return
        if popout:
            if url:
                # The popup gets a wrapper that stages the figure at its echoed
                # size and scales it to the window, the same guarantee as the
                # inline canvas, so a later resize or fullscreen letterboxes
                # instead of leaving the figure's layout behind the window's.
                route_id = url.rsplit("/viz/", 1)[-1]
                w, h = size_px or size or [1200, 800]
                url = public_url(
                    self._host.session.base_path,
                    f"/popout/{self._host.session.session_id}"
                    f"?route={quote(route_id)}&w={int(w)}&h={int(h)}",
                )
                self._popouts[url] = route_id
            await self._end_turn(protocol.popout_ready_to_wire(record, url))
            return
        await self._flush_side_outputs()
        await self._end_turn(protocol.turn_end_to_wire([]))
        # Make the regeneration part of the shared conversation, so the next turn
        # knows the plot's code re-ran in the shared kernel. Best-effort.
        with contextlib.suppress(Exception):
            await self._host.runner.add_user_action(
                "I regenerated a plot in the canvas by re-running its code:\n\n"
                f"```julia\n{payload.get('source_code')}\n```"
            )

    async def _refit(self, message: dict[str, Any]) -> None:
        """Re-fit a live figure to its panel's new size, in place.

        The canvas sends this when its stage and a live view disagree on size.
        The target comes from the same rule that fits a fresh plot, anchored to
        the recorded *authored* size, and the kernel ``resize!``-es the routed
        figure, camera and widget state untouched, with no code re-run. The answer
        is a ``refit_done`` with the echoed size: the client stages what the
        figure became, not what it asked for. The eval is spawned, not awaited:
        during a running turn it queues on the kernel's eval lock and lands
        when the kernel frees, while the WebSocket loop stays responsive.
        """
        record = str(message.get("record") or "")
        stage = _sane_size(message.get("width"), message.get("height"))
        if not record or stage is None:
            return
        from makker.agent.plot_julia import _plot_id_of, refit_web

        payload = next(
            (
                event.payload
                for event in reversed(self._host.session.trace.events_of(schema.ARTIFACT))
                if event.payload.get("path") == record
            ),
            None,
        )
        if payload is None:
            return
        target = _fit_target(stage, payload.get("authored_px") or payload.get("size_px"))

        async def run() -> None:
            with contextlib.suppress(Exception):
                err, echoed = await refit_web(
                    self._host.session, _plot_id_of(payload), target, _authored_of(payload)
                )
                if err is None and echoed is not None:
                    await self._send(protocol.refit_done_to_wire(record, echoed[0], echoed[1]))

        task = asyncio.create_task(run())
        # Keep a reference so the task isn't garbage-collected mid-flight.
        self._refits.add(task)
        task.add_done_callback(self._refits.discard)

    async def _close_popout(self, url: str) -> None:
        """Release the figure behind a closed popout window. Best-effort: only a
        route this connection's own popout created, and only when the kernel is
        free (a busy kernel just leaves it to the route cap).

        Deadlined for the same reason a re-fit is (see ``REFIT_TIMEOUT_S``):
        nobody waits on this, it holds the one kernel lock while it runs, and
        the route cap already releases the figure if this never lands, so it
        must not be able to leave the next turn waiting behind it."""
        route_id = self._popouts.pop(url, None)
        if route_id is None or self._busy():
            return
        from makker.agent import plot_julia_src as jl
        from makker.agent.plot_julia import REFIT_TIMEOUT_S, note_live_routes

        with contextlib.suppress(Exception):
            result = await asyncio.wait_for(
                self._host.session.julia.eval(
                    jl.close_web_plots_call(route_id, self._host.session.web_plot_secret or "")
                ),
                REFIT_TIMEOUT_S,
            )
            note_live_routes(self._host.session, result.output)

    def _busy(self) -> bool:
        return self._host.busy or (self._turn is not None and not self._turn.done())

    async def attach(self, websocket: WebSocket) -> None:
        """Send this session's messages to ``websocket`` from now on.

        It first gets the session as it stands (see ``_snapshot``), then any
        approval the session is paused on. The snapshot is taken and the socket
        adopted with no await in between, so every later message follows the
        snapshot and none is in both.
        """
        snapshot = self._snapshot()
        self._ws = websocket
        await _safe_send(websocket, snapshot)
        # Read back from the graph state when this stream has none in hand: the
        # session was resumed from disk, or paused before this server started.
        if not self._pending and not self._busy():
            with contextlib.suppress(Exception):
                self._pending = list(await self._host.runner.pending_interrupts())
        for interrupt in self._pending:
            await self._send_interrupt(interrupt)

    def detach(self, websocket: WebSocket) -> None:
        """Stop sending to ``websocket``; a running turn carries on regardless."""
        if self._ws is websocket:
            self._ws = None

    def _snapshot(self) -> dict[str, Any]:
        """The session as a newly attached socket should show it.

        The recorded conversation, replayed like ``GET /sessions/{id}/messages``.
        While a turn runs, the record stops where the turn began, and the turn
        follows as its prompt and the messages it has sent so far: the trace only
        gets a model's text once the model has finished, and a tool's output once
        the tool has, so it can't show a turn in progress.
        """
        session = self._host.session
        busy = self._busy()
        events = session.trace.iter_events()
        if busy:
            events = [event for event in events if event.id <= self._turn_start_id]
        live = session.web_plot_port is not None
        messages = replay_events(
            events,
            session.session_id,
            live=live,
            live_routes=session.live_routes if live else None,
            base_path=session.base_path,
        )
        if busy and self._turn_prompt:
            messages.append(protocol.replay_message("user", self._turn_prompt))
        return protocol.snapshot_to_wire(
            messages,
            self._turn_log if busy else [],
            busy=busy,
            model=self._host.model,
            context=context_meter(session.trace),
        )

    async def _send_context(self) -> None:
        """Send the context meter as the trace has it now."""
        with contextlib.suppress(Exception):
            await self._send(protocol.context_to_wire(context_meter(self._host.session.trace)))

    async def _send(self, message: dict[str, Any]) -> None:
        """Send to the attached socket, if any, keeping the running turn's content."""
        if message.get("type") in _TURN_CONTENT and self._busy():
            self._keep(message)
        if self._ws is not None:
            await _safe_send(self._ws, message)

    def _keep(self, message: dict[str, Any]) -> None:
        """Add a turn message to the turn's log, merged as the client merges it:
        text and reasoning deltas into one, a tool's output to its latest state
        (each sent delta is the whole rendered output so far)."""
        log = self._turn_log
        kind = message.get("type")
        if kind in ("text", "reasoning") and log and log[-1].get("type") == kind:
            log[-1]["text"] += message.get("text") or ""
            return
        if kind == "tool" and message.get("event") == "delta":
            cid = message.get("tool_call_id")
            for i, entry in enumerate(log):
                if entry.get("event") == "delta" and entry.get("tool_call_id") == cid:
                    log[i] = dict(message)
                    return
        log.append(dict(message))

    async def _send_interrupt(self, interrupt: Any) -> None:
        offer = self._host.approval.always_allow_offer(interrupt.value)
        await self._send(protocol.interrupt_to_wire(interrupt, always_allow=offer))

    async def _auto_resolve_pending(self) -> None:
        """Resume a paused approval the (changed) policy now auto-allows."""
        if self._busy() or not self._host.approval.approves(self._pending):
            return
        payload = build_resume_payload(self._pending, {"type": "approve"})
        self._pending = []
        runner = self._host.runner
        self._spawn(lambda: runner.resume(payload, on_message=self._on_message))

    def _spawn(self, factory, *, prompt: str | None = None) -> None:
        self._spawn_task(self._run_turn(factory), prompt=prompt)

    def _spawn_task(self, coro, *, prompt: str | None = None) -> None:
        task = asyncio.create_task(coro)
        if not self._host.claim_turn(task):
            task.cancel()
            return
        self._turn = task
        # Before the task first runs, so a socket attaching at any point of the
        # turn finds its start.
        self._turn_start_id = self._side_output_id = self._latest_event_id()
        self._turn_prompt = prompt
        self._turn_log = []

    async def _run_turn(self, factory) -> None:
        """Drive one turn and, whatever happens, tell the client it ended.

        The client locks its composer for the duration and unlocks it only on an
        ``interrupt``, an ``error`` or a ``turn_end``. So a turn that raises
        outside the guarded call below (flushing artifacts, summarising usage,
        delivering the end itself) would leave the browser waiting on a turn no
        one is running, with the stop button equally inert: the task is done, so
        there is nothing left to cancel, and only a reload frees it. The end is
        therefore delivered from a ``finally``, once per turn.
        """
        self._turn_ended = False
        self._announced = False
        try:
            await self._drive(factory)
        except asyncio.CancelledError:
            # Surface whatever the turn produced before it was stopped.
            with contextlib.suppress(Exception):
                await self._flush_side_outputs()
            # However the turn ended, its model calls may have changed the context.
            await self._send_context()
            await self._end_turn(protocol.turn_cancelled_to_wire())
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._flush_side_outputs()
            await self._send_context()
            await self._end_turn(protocol.turn_end_to_wire([]), error=str(exc))
        finally:
            # A turn that reached here without ending raised somewhere the client
            # cannot see; end it anyway rather than wedge the composer.
            await self._end_turn(protocol.turn_end_to_wire([]), error="the turn ended unexpectedly")

    async def _end_turn(self, wire: dict[str, Any], *, error: str | None = None) -> None:
        """Tell the client the turn is over, at most once per turn.

        Every later call is a no-op, so the backstop in ``_run_turn`` is free to
        fire unconditionally: it only speaks for a turn that ended no other way.
        A turn paused on an approval counts as ended (the ``interrupt`` frees the
        composer by itself), so no spurious end follows it.
        """
        if self._turn_ended:
            return
        self._turn_ended = True
        if error is not None:
            await self._send(protocol.error_to_wire(error))
        await self._send(wire)

    async def _drive(self, factory) -> None:
        """The turn itself. Reports its own outcome through ``_end_turn``; a path
        that raises before doing so is caught by ``_run_turn``'s backstop."""
        try:
            # The host owns the policy loop (auto-resume past interrupts the
            # mode or the session allowlist already allows) and the settle
            # hooks, shared with the TUI.
            result = await self._host.drive_turn(factory, on_message=self._on_message)
        finally:
            # A cancelled/errored turn can leave a tool mid-stream; clear streaming
            # state so a stale trailing flush can't fire on a later turn.
            self._end_all_tool_streams()
        # Best-effort from here on: the artifacts and the usage tick are worth
        # having, but never at the cost of the turn's end (see ``_run_turn``).
        with contextlib.suppress(Exception):
            await self._flush_side_outputs()
        self._pending = list(result.interrupts)
        if self._pending:
            # The turn paused for approval. Send the requests and wait for a
            # decision; the turn ends only once it runs to completion.
            self._turn_ended = True
            await self._send_context()
            for interrupt in self._pending:
                await self._send_interrupt(interrupt)
            return
        await self._send_context()
        await self._end_turn(protocol.turn_end_to_wire(result.messages))
        with contextlib.suppress(Exception):
            await self._flush_host_context()
            task = self._host.maybe_title(self._on_titled)
            if task is not None:
                self._title_task = task

    async def _on_titled(self, title: str) -> None:
        """Nudge the front end to refresh its history list with the new title."""
        await self._send(protocol.ui_command("history_changed", {"title": title}))

    def _latest_event_id(self) -> int:
        return self._host.session.trace.max_id()

    async def _flush_side_outputs(self) -> None:
        """Forward side outputs produced since the last flush: artifacts (plots,
        reports) and UI commands a tool emitted. Tracks a high-water mark over trace
        event ids, so a plot or report appears inline the moment its tool finishes
        (flushed from ``_on_message``) rather than all at once at turn end. Only the
        events since the last flush are read, so a long turn does not re-scan the
        whole trace on every tool completion."""
        for event in self._host.session.trace.events_after(self._side_output_id):
            self._side_output_id = event.id
            if event.kind == schema.ARTIFACT:
                for wire in artifact_wire_events(
                    [event.payload],
                    self._host.session_id,
                    base_path=self._host.session.base_path,
                ):
                    await self._send(wire)
            elif event.kind == schema.UI_COMMAND:
                action = str(event.payload.get("action") or "")
                payload = event.payload.get("payload")
                await self._send(protocol.ui_command(action, payload))

    async def _announce_history_once(self) -> None:
        """Tell the client the session list changed, at most once per turn.

        A session only becomes listable when its first prompt is on the trace, and
        that append is the first thing ``run_prompt`` does, so by the time any
        message of the turn reaches here the session has a title to show. Waiting
        for the titling hook instead meant a new chat did not appear in the sidebar
        until the whole turn had finished, which is minutes when the turn runs a
        simulation.
        """
        if self._announced:
            return
        self._announced = True
        with contextlib.suppress(Exception):
            await self._send(protocol.ui_command("history_changed", {}))

    async def _on_message(self, event: Any) -> None:
        await self._announce_history_once()
        if isinstance(event, TurnUsage):
            await self._send_context()
            return
        wire = protocol.to_wire(event)
        if wire is None:
            return
        if wire.get("type") == "tool":
            cid = wire.get("tool_call_id")
            kind = wire.get("event")
            if kind == "delta" and cid:
                await self._on_tool_delta(cid, wire)
                return  # _on_tool_delta sends (now or on its trailing flush)
            if kind in ("finished", "error") and cid:
                # The final result (terminal-rendered by the kernel) replaces the
                # live stream; stop any pending flush and drop the per-call buffers.
                self._end_tool_stream(cid)
        await self._send(wire)
        # A tool just finished: surface any artifacts/ui it produced right away, so a
        # plot or report appears inline as it happens instead of all at turn end.
        if wire.get("type") == "tool" and wire.get("event") in ("finished", "error"):
            await self._flush_side_outputs()

    async def _on_tool_delta(self, cid: str, wire: dict[str, Any]) -> None:
        """Accumulate a tool's raw output delta and send the terminal-rendered state.

        Mirrors the TUI: cursor moves and carriage returns are replayed through the
        screen emulator so progress output reads as one updating block, not a gap,
        and re-rendering is throttled. Throttling is leading+trailing — a delta that
        arrives within the interval schedules a trailing flush — so the last partial
        line never lingers unshown until the next event.
        """
        import time

        raw = self._tool_streams.get(cid, "") + (wire.get("content") or "")
        buf = raw[-TOOL_STREAM_TAIL_CAP:]
        self._tool_streams[cid] = buf
        self._tool_delta_wire[cid] = wire  # send template (name/label) for the flush
        if time.monotonic() - self._tool_render_at.get(cid, 0.0) >= TOOL_STREAM_RENDER_INTERVAL:
            await self._flush_tool_stream(cid)
        elif cid not in self._tool_flush:
            self._tool_flush[cid] = asyncio.create_task(self._delayed_flush(cid))

    async def _delayed_flush(self, cid: str) -> None:
        try:
            await asyncio.sleep(TOOL_STREAM_RENDER_INTERVAL)
            await self._flush_tool_stream(cid)
        except asyncio.CancelledError:
            pass
        finally:
            self._tool_flush.pop(cid, None)

    async def _flush_tool_stream(self, cid: str) -> None:
        import time

        from makker.juliakernel.text import render_terminal_output

        buf = self._tool_streams.get(cid)
        if buf is None:
            return
        self._tool_render_at[cid] = time.monotonic()
        wire = {
            **self._tool_delta_wire[cid],
            "content": render_terminal_output(buf),
            "replace": True,
        }
        await self._send(wire)

    def _end_tool_stream(self, cid: str) -> None:
        task = self._tool_flush.pop(cid, None)
        if task is not None:
            task.cancel()
        self._tool_streams.pop(cid, None)
        self._tool_render_at.pop(cid, None)
        self._tool_delta_wire.pop(cid, None)

    def _end_all_tool_streams(self) -> None:
        """Drop all per-tool streaming state at turn end.

        A normal turn ends each stream via its tool's ``finished`` event; a
        cancelled or errored turn leaves a tool mid-stream (no such event), so
        without this its pending ``_delayed_flush`` would fire a stale frame on a
        later turn and the per-call dicts would grow across cancellations."""
        for task in self._tool_flush.values():
            task.cancel()
        self._tool_flush.clear()
        self._tool_streams.clear()
        self._tool_render_at.clear()
        self._tool_delta_wire.clear()

    async def cancel_turn(self) -> None:
        """Stop the running turn, and always answer, even when none is running.

        The client's composer is locked until the server says a turn ended, so a
        cancel that answers nothing leaves a page reload as the only escape. If
        the client believes a turn is in flight and the server does not, the
        client is the one stuck.
        """
        task = self._turn
        if task is not None and not task.done():
            task.cancel()  # the turn answers for itself as it unwinds
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return
        await self._send(protocol.turn_cancelled_to_wire())


# How much of a prompt the server log shows: enough to tell which log lines a
# message belongs to, short enough to keep most of a prompt out of the log.
PROMPT_LOG_CHARS = 80


def _log_prompt(session_id: str, text: str) -> None:
    """Log the start of a received prompt, in uvicorn's "INFO:" console format."""

    preview = " ".join(text.split())
    if len(preview) > PROMPT_LOG_CHARS:
        preview = preview[:PROMPT_LOG_CHARS] + "…"
    print(f"INFO:     received message [{session_id}]: {preview!r}", file=sys.stderr, flush=True)


async def _safe_send(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Send a JSON message, ignoring a socket that is already closing."""
    with contextlib.suppress(Exception):
        await websocket.send_json(message)
