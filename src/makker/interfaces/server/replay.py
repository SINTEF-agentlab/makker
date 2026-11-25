"""A recorded session, read back for the browser.

``replay_events`` turns a session's trace into the messages the turn stream sends
live, so a reopened chat renders as it did; ``artifact_wire_events`` is the
artifact part of that, which the live stream uses too. The rest finds a session's
record on disk and summarizes it for the history list.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from makker.agent.summarization import COMPACTION_NOTE
from makker.interfaces.server import protocol
from makker.interfaces.server.urls import public_url
from makker.trace import schema


def _artifact_url(session_id: str, rel: str, *, base_path: str = "") -> str:
    """The fetch URL for a session artifact given its workspace-relative path."""
    rel = rel[len("artifacts/") :] if rel.startswith("artifacts/") else rel
    return public_url(base_path, f"/sessions/{session_id}/artifacts/{rel}")


def artifact_wire_events(
    payloads: list[dict[str, Any]],
    session_id: str,
    *,
    live: bool = True,
    live_routes: Collection[str] | None = None,
    base_path: str = "",
) -> list[dict[str, Any]]:
    """Wire events for produced artifacts: interactive HTML as ``viz``, the rest as ``artifact``.

    An HTML artifact (an interactive plot, or a written report) becomes a ``viz``
    the front end pins to its canvas, carrying the artifact's ``kind``, ``slot``,
    and a ``poster`` image URL when one was saved alongside.

    ``live=False`` is for replaying a resumed session: the Julia process (and with
    it any Bonito server that backed a live plot) has restarted, so a recorded
    ``live_url`` is dead. The figure then falls back to its saved PNG poster, shown
    inline as a static image, instead of an embed pointing at a gone server.

    ``live_routes``, when given, are the routes the live server still serves: a
    figure closed or released since it was recorded falls back to its poster too.

    ``base_path`` prefixes site-relative URLs when the UI is served under a public
    path (e.g. ``/restricted``); durable artifact records stay unprefixed.
    """
    events: list[dict[str, Any]] = []
    live_prefix = f"/live/{session_id}"
    for payload in payloads:
        url = _artifact_url(session_id, str(payload.get("path") or ""), base_path=base_path)
        live_url = payload.get("live_url") if live else None
        if live_url and live_routes is not None:
            route = str(live_url).removeprefix(live_prefix)
            live_url = live_url if route in live_routes else None
        poster = payload.get("poster")
        poster_url = _artifact_url(session_id, str(poster), base_path=base_path) if poster else None
        kind = payload.get("kind")
        # A plot or report is a canvas view; a bare image or file is a plain artifact.
        if live_url or payload.get("mime") == "text/html" or kind in ("plot", "report"):
            # What to embed: the live Bonito server while it's up, else the static
            # HTML record, else the saved poster image. On resume the live server is
            # gone, so a live plot falls back to its still-on-disk PNG poster instead
            # of a dead live URL — the figure stays viewable, just not interactive.
            if live_url:
                view_url = public_url(base_path, str(live_url))
            elif payload.get("mime") == "text/html":
                view_url = url
            else:
                view_url = poster_url or url
            size_px = payload.get("size_px") or [None, None]
            events.append(
                protocol.viz_to_wire(
                    view_url,
                    title=payload.get("caption"),
                    kind=str(kind or "plot"),
                    poster=poster_url,
                    slot=payload.get("slot"),
                    live=bool(live_url),
                    # A plot with recorded code can be replayed: name its trace
                    # record so the front end can ask for a re-run (regenerate a
                    # dead view, or serve an independent popout figure).
                    record=(str(payload.get("path")) if payload.get("source_code") else None),
                    width=size_px[0],
                    height=size_px[1],
                )
            )
        else:
            events.append(protocol.artifact_to_wire(payload, url=url))
    return events


def replay_events(
    events: list[Any],
    session_id: str,
    *,
    live: bool = False,
    live_routes: Collection[str] | None = None,
    base_path: str = "",
) -> list[dict[str, Any]]:
    """Wire messages that reconstruct a recorded conversation for a resumed session.

    The trace-event analogue of the live ``protocol.to_wire`` path: it maps each
    persisted event (user/assistant/reasoning text, tool calls paired with their
    results, artifacts) to the same wire messages the WebSocket streams during a
    turn, so a reopened chat renders identically, tool cards and all. Kept as one
    function (not inlined in the endpoint) so the replay mapping lives in a single,
    testable place. ``live`` says whether the session's live plot server is still
    running (its figures alive); without it a recorded live URL is dead and the
    figure falls back to its poster. ``live_routes`` narrows that to the figures
    the server still serves.
    """

    items: list[dict[str, Any]] = []
    # Tool calls whose result was recorded. Some never record one (e.g. a
    # write_todos that ends a turn); without a terminal event their replayed
    # card would spin forever, so we synthesize a finished event for them.
    result_ids = {e.payload.get("tool_call_id") for e in events if e.kind == schema.TOOL_RESULT}
    replayed_kinds = {
        schema.MESSAGE_USER: "user",
        schema.MESSAGE_ASSISTANT: "assistant",
        schema.MESSAGE_REASONING: "reasoning",
    }
    for ev in events:
        if ev.kind in replayed_kinds:
            text = str(ev.payload.get("content", "")).strip()
            if text:
                items.append(protocol.replay_message(replayed_kinds[ev.kind], text))
        elif ev.kind == schema.TOOL_CALL:
            name = ev.payload.get("name")
            cid = ev.payload.get("id")
            items.append(
                protocol.replay_tool_event(
                    event="requested", name=name, tool_call_id=cid, args=ev.payload.get("args")
                )
            )
            if cid not in result_ids:
                items.append(
                    protocol.replay_tool_event(event="finished", name=name, tool_call_id=cid)
                )
        elif ev.kind == schema.TOOL_RESULT:
            finished = "error" if ev.payload.get("status") == "error" else "finished"
            items.append(
                protocol.replay_tool_event(
                    event=finished,
                    name=ev.payload.get("name"),
                    tool_call_id=ev.payload.get("tool_call_id"),
                    content=ev.payload.get("content"),
                )
            )
        elif ev.kind == schema.CONTEXT_COMPACTION:
            items.append(protocol.compaction_to_wire(COMPACTION_NOTE, ev.payload))
        elif ev.kind == schema.ARTIFACT:
            items.extend(
                artifact_wire_events(
                    [ev.payload],
                    session_id,
                    live=live,
                    live_routes=live_routes,
                    base_path=base_path,
                )
            )
    return items


# What counts as working on a session, for ordering the history list.
_ACTIVITY_KINDS = (schema.MESSAGE_USER, schema.MESSAGE_ASSISTANT)


def _session_overview(state_dir: Path) -> tuple[str | None, str | None, str | None]:
    """A persisted session's simulator, first user prompt, and last-activity time.

    Used to label the history list, give an untitled session a fallback title, and
    order the list by when each was last used. Three indexed point-queries, so it
    stays cheap even on a long trace. Returns ``(None, None, None)`` on a
    missing/unreadable trace.
    """
    from makker.trace import TraceLog

    try:
        with TraceLog.open_readonly(state_dir / "trace.sqlite") as log:
            start = log.first_payload(schema.SESSION_START) or {}
            user = log.first_payload(schema.MESSAGE_USER) or {}
            sim = start.get("simulator")
            content = user.get("content")
            first_prompt = content if isinstance(content, str) else None
            # What was last *said*, not the last event of any kind: resuming a
            # session writes lifecycle events, and counting those sent a chat to
            # the top of the list just for being opened.
            last = log.last_timestamp_of(_ACTIVITY_KINDS) or log.last_timestamp()
            return sim, first_prompt, last
    except Exception:
        return None, None, None


# A session id is server-generated and shaped like ``2026-06-21-2315-3f2a`` (plus
# an optional title slug). Validate the shape so a client-supplied id can never be
# a path traversal (``..``, separators, encoded slashes) into ``mkdir`` or a read.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_RE.match(session_id)) and ".." not in session_id


def _session_state_dir(session_id: str) -> Path | None:
    """The on-disk state dir for a (possibly not-loaded) session, if it exists."""
    from makker.session import sessions_root

    if not _is_valid_session_id(session_id):
        return None
    root = sessions_root().resolve()
    candidate = (root / session_id).resolve()
    if not candidate.is_relative_to(root):  # belt-and-braces against traversal
        return None
    return candidate if (candidate / "trace.sqlite").exists() else None
