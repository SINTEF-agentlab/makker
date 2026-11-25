"""Session lifecycle: create, resume, list and close sessions, the history of
sessions on disk, and a session's recorded conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from makker.agent.capabilities import (
    HttpToolSpec,
    host_context_capability,
    http_tool_capability,
)
from makker.interfaces.server import protocol
from makker.interfaces.server.deps import BasePathDep, DefaultsDep, ManagerDep, UpdaterDep
from makker.interfaces.server.manager import SessionBusyError
from makker.interfaces.server.replay import (
    _is_valid_session_id,
    _session_overview,
    _session_state_dir,
    replay_events,
)
from makker.interfaces.server.stream import _stream_of
from makker.interfaces.server.updates import Updater
from makker.simulators.env_setup import EnvOutdatedError, EnvSetupError
from makker.sysimage import SysimageUnavailable

router = APIRouter()


class CreateSessionRequest(BaseModel):
    # Optional: a server bound to a simulator (the web case) uses its own; a
    # request may still name one, which must match the bound simulator.
    sim: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    workspace: str | None = None
    # The host app's declarative HTTP tools, validated straight into the domain
    # ``HttpToolSpec`` (one schema, no parallel request model to keep in sync).
    tools: list[HttpToolSpec] | None = None
    # What the host application currently has selected: an opaque JSON object of
    # its own identifiers, which the agent is told about and its tools can read.
    host_context: dict[str, Any] | None = None
    # Where the host application's own HTTP API listens, for this launch. Never
    # stored: it describes the application as it is running now, so a remembered
    # one could send the agent somewhere that has since moved or closed.
    host_api: str | None = None


class ResumeSessionRequest(BaseModel):
    sim: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    workspace: str | None = None
    # The host's selection *now*, which supersedes what this session was last
    # told. Omitted (or null) means "unchanged": a front end opened outside the
    # host application has nothing to say about the selection, and must not be
    # able to erase it by resuming a session the host started.
    host_context: dict[str, Any] | None = None
    # Re-declared on resume for the same reason they are declared on create: the
    # tools belong to the running application, not to the stored session, and a
    # session resumed without them would come back with the host app's routines
    # missing from an otherwise intact conversation.
    tools: list[HttpToolSpec] | None = None
    host_api: str | None = None


def _host_api_url(value: str | None) -> str | None:
    """Validate the host application's API address, or raise a 400.

    This is the trust boundary: the bundled UI checks the value too, but a
    request can reach here without going through it. The address ends up in URLs
    the agent's tools call, and a capability may interpolate it into code it
    evaluates, so only a plain http(s) origin (with an optional path prefix) is
    accepted: no credentials, no query, no fragment, and none of the characters
    that let a string escape the literal it is placed in.

    Rejection is an error rather than a silent ``None`` because a malformed
    address is always a caller bug: any real URL passes, and a session whose
    tools quietly cannot reach the application is harder to diagnose than a 400.
    """
    if value is None:
        return None
    candidate = value.strip().rstrip("/")
    if not candidate:
        return None
    parsed = urlparse(candidate)
    unsafe = set("\"'\\`$\n\r\t<>")
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or unsafe & set(candidate)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"host_api must be a plain http(s) address, got {value!r}",
        )
    return candidate


def _request_extensions(
    tools: list[HttpToolSpec] | None, host_context: dict[str, Any] | None = None
) -> list:
    """The capability layers a session-create request brings with it.

    Two things an embedding application declares per session: the routines it
    exposes to the agent (HTTP tool specs) and what it currently has selected.
    Both are request-scoped rather than installed, because they describe this
    launch, which is also why they travel as capabilities instead of as
    arguments threaded through the session bootstrap.
    """
    layers = []
    if tools:
        layers.append(http_tool_capability("host-app", tools))
    selection = host_context_capability(host_context)
    if selection is not None:
        layers.append(selection)
    return layers


def _bound_sim(requested: str | None, default_sim: str | None) -> str:
    """The simulator a new/resumed session must use.

    The server is bound to one folder, and a folder is bound to one simulator
    (chosen when ``makker web`` starts), so every session here uses that
    one — the web UI does not switch simulators in place. A request for a
    different simulator is refused. Without a bound simulator (tests, or a future
    multi-folder launcher) the caller's choice is honoured.
    """
    if default_sim is None:
        if not requested:
            raise HTTPException(status_code=400, detail="no simulator specified")
        return requested
    if requested and requested != default_sim:
        raise HTTPException(
            status_code=409,
            detail=(
                f"this server is bound to simulator '{default_sim}'; run "
                "`makker` from another folder to use a different simulator"
            ),
        )
    return default_sim


def _require_credential(model: str | None) -> None:
    """Raise a structured 400 if ``model``'s provider needs a key we don't have.

    Resolves the same effective model a new session would use (the request's,
    else the server default, else the catalog default), so the guard matches
    what the kernel would actually load.
    """
    from makker.credentials import missing_credential
    from makker.models import DEFAULT_MODEL, provider_info

    effective = model or DEFAULT_MODEL
    env_var = missing_credential(effective)
    if env_var is None:
        return
    info = provider_info(effective)
    raise HTTPException(
        status_code=400,
        detail=protocol.credential_required_to_wire(
            provider=info.name if info else "",
            label=info.label if info else effective,
            env_var=env_var,
        ),
    )


def _workspace_for(requested: str | None) -> Path | None:
    """The folder a session runs in: an explicit request, else the server's folder.

    The server runs in one folder (its launch directory, where the bound
    simulator's Julia environment lives), so a normal session runs there
    (``None`` lets ``SessionHost.start`` fall back to ``workspace_root()``).
    The ``requested`` override is retained for a future launcher that opens a
    session in a chosen folder.
    """
    return Path(requested) if requested else None


@router.get("/sessions/history")
def session_history(manager: ManagerDep, defaults: DefaultsDep, limit: int = 40) -> dict[str, Any]:
    """Resumable sessions on disk, newest first, with a title and simulator.

    A session's title comes from its stored ``title`` file (the first-prompt or
    LLM name). When that is missing (its titling never persisted) we fall back to
    deriving a title from the first user prompt, so a real conversation still
    shows in history instead of vanishing. Only a session with no prompt at all
    (an abandoned new-chat) is omitted.

    Ordered by last activity (the most recently used first), from each trace's
    last event time, since that is what a user looks for — not when the session
    was first created.

    Each entry also says whether the session is still ``live``: its host is in
    memory, so its Julia kernel and REPL state survive a resume, where a session
    that has been evicted replays from disk against a fresh Julia. Read from the
    in-memory registry, so it costs no disk access and is only ever true for this
    server process.

    Deliberately not reported: whether a connection is attached. It is true of
    the session you just left until its disconnect teardown finishes, so a
    history refresh right after "New chat" would mark it as in use by someone
    else, and one browser client is the normal case anyway.
    """
    from makker.session import derive_session_title, list_sessions

    live_ids = set(manager.list_ids())

    # Consider every session, then sort by last activity and cap last — slicing
    # before the sort would order by creation and cut an old-but-recently-used
    # session even though it belongs near the top.
    sessions: list[dict[str, Any]] = []
    for info in list_sessions():
        sim, first_prompt, last_active = _session_overview(info.state_dir)
        title = info.title or (derive_session_title(first_prompt) if first_prompt else None)
        if not title:
            continue  # no stored title and no prompt: an empty/abandoned new-chat
        sessions.append(
            {
                "id": info.session_id,
                "title": title,
                "started": info.started.isoformat(),
                "last_active": last_active or info.started.isoformat(),
                "sim": sim or defaults.sim,
                "live": info.session_id in live_ids,
            }
        )
    sessions.sort(key=lambda s: s["last_active"], reverse=True)
    return {"sessions": sessions[: max(0, limit)]}


@router.get("/sessions/{session_id}/messages")
def session_messages(
    session_id: str, manager: ManagerDep, base_path: BasePathDep
) -> dict[str, Any]:
    """The full conversation for replay on resume, in the live wire shape.

    Emits the same message types the WebSocket streams during a turn — user
    and assistant text, reasoning, tool calls paired with their results, and
    views — so a reopened chat reconstructs inline exactly as it looked when
    the user left it, tool cards and all. Whether replayed plots keep their
    live URLs depends on the session: a still-running host with a live plot
    server has the figures it still serves alive, while a from-disk resume
    restarted Julia, so its recorded live URLs are dead and the embeds fall
    back to their saved posters. A page that still holds a figure's frame (an
    in-page switch back) keeps it; one that doesn't (after a reload) opens a
    new view, which takes the figure over (see ``MakkerWebViews``).
    """
    host = manager.get(session_id)
    state_dir = host.session.state_dir if host else _session_state_dir(session_id)
    if state_dir is None:
        raise HTTPException(status_code=404, detail="no such session")
    from makker.trace import TraceLog

    with TraceLog(state_dir / "trace.sqlite") as log:
        events = list(log.iter_events())
    live = host is not None and host.session.web_plot_port is not None
    bp = host.session.base_path if host is not None else base_path
    routes = host.session.live_routes if live else None
    messages = replay_events(events, session_id, live=live, live_routes=routes, base_path=bp)
    return {"messages": messages}


def _refuse_while_updating(updater: Updater) -> None:
    if updater.running:
        raise HTTPException(
            status_code=409,
            detail="Makker is updating. Start a session when it has finished.",
        )


def _env_conflict(exc: EnvSetupError | SysimageUnavailable, updater: Updater) -> HTTPException:
    """A 409 for an environment a session can't start in, saying how to fix it."""
    if isinstance(exc, EnvOutdatedError) and updater.local:
        found = ", ".join(f"{pkg} {ver}" for pkg, ver, _ in exc.outdated)
        needed = ", ".join(f"{pkg} {spec}" for pkg, _, spec in exc.outdated)
        return HTTPException(
            status_code=409,
            detail=(
                f"this folder's Julia environment has {found}, but this version of "
                f"Makker requires {needed}. Update it with “Update environment” "
                "at the foot of the sidebar."
            ),
        )
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/sessions")
async def create_session(
    req: CreateSessionRequest, manager: ManagerDep, updater: UpdaterDep, defaults: DefaultsDep
) -> dict[str, str]:
    _refuse_while_updating(updater)
    sim = _bound_sim(req.sim, defaults.sim)
    # Refuse before standing up the kernel if the model has no key: the UI
    # shows a key prompt on this structured error, then retries the create.
    _require_credential(req.model or defaults.model)
    # Likewise validate the host address up here, so a rejected one cannot
    # leave a live session behind that no caller knows the id of.
    host_api = _host_api_url(req.host_api)
    try:
        host = await manager.create(
            sim=sim,
            model=req.model or defaults.model,
            approval_mode=req.approval_mode or defaults.approval_mode,
            workspace=_workspace_for(req.workspace),
            extensions=_request_extensions(req.tools, req.host_context),
        )
    except KeyError as exc:  # unknown simulator
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (SysimageUnavailable, EnvSetupError) as exc:
        # The CLI checked the prepared environment before starting the server,
        # so reaching here normally means it moved while the server was up (a
        # package was installed or edited, or Makker itself was upgraded).
        # The message explains itself.
        raise _env_conflict(exc, updater) from exc
    # The agent already carries the selection (it was built with the layer
    # above); this records it on the session, so it is persisted for a later
    # resume, visible in the trace, and readable by capability tools. The API
    # URL is only ever read from the session, so assigning it is all it needs.
    host.set_host_context(req.host_context, rebuild=False)
    host.session.host_api = host_api
    return {"session_id": host.session_id}


@router.get("/sessions")
def list_sessions(manager: ManagerDep) -> dict[str, list[str]]:
    return {"sessions": manager.list_ids()}


@router.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: str,
    req: ResumeSessionRequest,
    manager: ManagerDep,
    updater: UpdaterDep,
    defaults: DefaultsDep,
) -> dict[str, Any]:
    if not _is_valid_session_id(session_id):
        raise HTTPException(status_code=404, detail="no such session")
    _refuse_while_updating(updater)
    # Validated before anything is reattached or rebuilt, so a bad address
    # cannot leave a session half-resumed.
    host_api = _host_api_url(req.host_api)
    existing = manager.get(session_id)
    if existing is not None:
        # Another connection is using it: re-resuming would build a fresh kernel
        # and tear the live one down under it, so refuse.
        if existing.attached:
            raise HTTPException(
                status_code=409, detail="session is already open in another connection"
            )
        # Still live (the user navigated away and came back, or the socket dropped
        # and reconnected): reattach as-is. The live host is authoritative
        # because an in-session model or approval change already updated it via the
        # set_model/set_approval command. Re-applying the resume request's values
        # here would let a stale UI value clobber the user's in-session choice on
        # every reconnect: the UI always reports the default model by name (it does
        # not know the server's --model), and it never sends the approval mode, so
        # the request's defaults do not reflect what the session is actually using.
        # The kernel, history, and live REPL are untouched.
        manager.promote(session_id)
        # The host's selection is the exception to "the live host is
        # authoritative": it describes the application right now, not this
        # session's earlier settings, so a reattach carries it over. A turn may
        # still be running (it outlives its socket), and the rebuild a change
        # triggers must not happen under it, so the stream applies it now or
        # when that turn ends.
        if req.host_context is not None:
            await _stream_of(existing).set_host_context(req.host_context)
        # Likewise the API URL, which moves whenever the application restarts:
        # a session reattached against a stale port would only find a closed
        # door. Sent on every resume, so its absence genuinely means "no host".
        existing.session.host_api = host_api
        return {"session_id": existing.session_id, "kernel_restarted": False}
    # Not live anymore (evicted, or a new server): resume from disk, which starts
    # a fresh kernel — the conversation is restored but in-memory REPL state is not.
    #
    # The selection the rebuilt agent is told about is the request's if the
    # front end is running inside the host application, and otherwise the one
    # stored with the session. It has to be resolved here, before the build,
    # because it is a system-prompt layer: reading it after start would mean
    # rebuilding the agent we just built.
    from makker.session import read_host_context

    state_dir = _session_state_dir(session_id)
    if state_dir is None:  # before anything starts a kernel for it
        raise HTTPException(status_code=404, detail="no such session")
    host_context = req.host_context
    if host_context is None:
        host_context = read_host_context(state_dir)
    try:
        host = await manager.resume(
            session_id,
            sim=_bound_sim(req.sim, defaults.sim),
            model=req.model or defaults.model,
            approval_mode=req.approval_mode or defaults.approval_mode,
            workspace=_workspace_for(req.workspace),
            extensions=_request_extensions(req.tools, host_context),
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SysimageUnavailable, EnvSetupError) as exc:
        # Match fresh-session creation: environment preparation can reveal
        # that an image or the environment went stale while the server was
        # running, and that is an actionable launch conflict rather than an
        # internal server error the page would wait on.
        raise _env_conflict(exc, updater) from exc
    # The resumed session read its own stored selection on the way up. Defer to
    # that when the request carried none: a front end running outside the host
    # application has nothing to say about the selection, and adopting its
    # silence as "nothing selected" would erase what the session was working on.
    host.set_host_context(
        req.host_context if req.host_context is not None else host.session.host_context,
        rebuild=False,
    )
    host.session.host_api = host_api
    return {"session_id": host.session_id, "kernel_restarted": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, manager: ManagerDep) -> dict[str, bool]:
    # Refuse to tear down a session a connection is still driving — closing its
    # kernel/checkpointer under a running turn would crash that turn. The client
    # closes its WebSocket first (which detaches), so this only blocks a stray
    # delete of an in-use session. The check-and-close is one atomic step in the
    # manager, so a connection can't attach in a gap and lose its kernel mid-turn.
    # Shutdown still force-closes via manager.aclose.
    try:
        closed = await manager.close(session_id, require_idle=True)
    except SessionBusyError as exc:
        raise HTTPException(
            status_code=409, detail="session is open in a connection; close it first"
        ) from exc
    if not closed:
        raise HTTPException(status_code=404, detail="no such session")
    return {"ok": True}
