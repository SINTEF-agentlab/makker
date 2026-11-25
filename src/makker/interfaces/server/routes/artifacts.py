"""What a session holds: its artifacts, the transcript download, the memory and
context panels, and files uploaded into its workspace.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from makker.interfaces.server.deps import ManagerDep
from makker.interfaces.server.replay import _is_valid_session_id, _session_state_dir
from makker.trace import schema

router = APIRouter()


# Inert page policy for the canvas iframe: no scripts (the body is markdown the
# agent may have read), only inline styles and images. Defense in depth alongside
# the markdown renderer's html=False escaping.
_DOC_CSP = (
    "default-src 'none'; img-src 'self' data: http: https:; style-src 'unsafe-inline'; "
    "font-src data:; base-uri 'none'; form-action 'none'"
)


def _doc_page(title: str, body_html: str) -> str:
    """Wrap rendered HTML in a minimal, self-contained page for the canvas iframe."""
    import html as _html

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='Content-Security-Policy' content=\"{_DOC_CSP}\">"
        "<title>" + _html.escape(title) + "</title><style>"
        "body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "color:#1f2328;background:#fff;line-height:1.6}"
        ".page{max-width:760px;margin:0 auto;padding:2rem 1.6rem}"
        "h1,h2,h3{line-height:1.3;letter-spacing:-0.01em}h1{font-size:1.5rem}"
        "code{font-family:ui-monospace,Consolas,monospace;background:#f0f1ee;padding:.1em .35em;"
        "border-radius:5px;font-size:.88em}"
        "pre{background:#f0f1ee;border:1px solid #e3e3df;border-radius:10px;padding:.8rem;"
        "overflow:auto}"
        "pre code{background:none;padding:0}a{color:#0e7490}"
        "</style></head><body><div class='page'>" + body_html + "</div></body></html>"
    )


def _resolve_artifact(output_dir: Path, path: str):
    """The artifact file for ``path``, or ``None`` if it escapes the artifacts dir."""
    base = (output_dir / "artifacts").resolve()
    target = (base / path).resolve()
    if not target.is_file() or not target.is_relative_to(base):
        return None
    return target


# HEAD too: the canvas probes a plot's URL with it before mounting the frame.
@router.api_route("/sessions/{session_id}/artifacts/{path:path}", methods=["GET", "HEAD"])
def get_artifact(session_id: str, path: str, manager: ManagerDep) -> FileResponse:
    # Resolve against the live session when it's loaded, else its on-disk output
    # dir — the same fallback get_transcript uses — so a plot or report stays
    # fetchable after the session is evicted from the live registry or the server
    # restarts, instead of going dead (404) while the file is still on disk.
    from makker.session import existing_output_dir

    host = manager.get(session_id)
    if host is not None:
        out_dir: Path | None = host.session.output_dir
    elif _is_valid_session_id(session_id):
        out_dir = existing_output_dir(session_id)
    else:
        out_dir = None
    if out_dir is None:
        raise HTTPException(status_code=404, detail="no such session")
    target = _resolve_artifact(out_dir, path)
    if target is None:
        raise HTTPException(status_code=404, detail="no such artifact")
    return FileResponse(target)


@router.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str, manager: ManagerDep, format: str = "html") -> Response:
    """Download the session transcript to share (html or md)."""
    host = manager.get(session_id)
    state_dir = host.session.state_dir if host else _session_state_dir(session_id)
    if state_dir is None:
        raise HTTPException(status_code=404, detail="no such session")
    from makker.session import existing_output_dir
    from makker.trace import TraceLog
    from makker.transcript import render_html, render_markdown

    with TraceLog(state_dir / "trace.sqlite") as log:
        events = list(log.iter_events())
    md = format in ("md", "markdown")
    # Inline images so the downloaded transcript shows its plots on its own,
    # off the server (the artifacts live in the session's output folder).
    out_dir = host.session.output_dir if host else existing_output_dir(session_id)
    artifact_dirs = [out_dir / "artifacts", out_dir] if out_dir else []
    body = render_markdown(events) if md else render_html(events, artifact_dirs=artifact_dirs)
    ext = "md" if md else "html"
    return PlainTextResponse(
        body,
        media_type="text/markdown" if md else "text/html",
        headers={"Content-Disposition": f"attachment; filename=transcript.{ext}"},
    )


@router.get("/sessions/{session_id}/memory")
def get_memory(session_id: str, manager: ManagerDep) -> Response:
    """The session's workspace memory, rendered as a page for the canvas."""
    host = manager.get(session_id)
    if host is None:
        raise HTTPException(status_code=404, detail="no such session")
    from makker.agent.memory import render_memory_overview
    from makker.transcript.markdown_html import render_markdown_html

    body = render_markdown_html(render_memory_overview(host.memory_dir))
    return HTMLResponse(_doc_page("Memory", body))


@router.get("/sessions/{session_id}/context")
def get_context(session_id: str, manager: ManagerDep) -> dict[str, Any]:
    """The full context-usage panel (same render as the TUI), as markdown.

    Usage figures come from the session's ``model_usage`` trace events (the
    first/last call and the count); the system-prompt and memory-index sizes
    are approximated the same way the TUI does. Rendered server-side so the web
    UI and the terminal show identical detail.
    """
    host = manager.get(session_id)
    if host is None:
        raise HTTPException(status_code=404, detail="no such session")
    from makker.agent.context_editing import clear_tool_uses_trigger_tokens
    from makker.agent.memory import list_memory_notes
    from makker.agent.summarization import auto_compact_trigger_tokens
    from makker.context_panel import context_component_estimates, render_context_panel
    from makker.models import DEFAULT_MODEL, context_window
    from makker.trace import TraceLog

    # Read usage from a fresh connection on the trace file: this endpoint runs in
    # a threadpool, and the session's own SQLite connection is bound to the thread
    # it was created on (the event loop), so reusing it here raises.
    with TraceLog(host.session.state_dir / "trace.sqlite") as log:
        usages = [e.payload for e in log.events_of(schema.MODEL_USAGE)]
    model = host.model or DEFAULT_MODEL
    window = context_window(model)

    memory_dir = host.memory_dir
    system_tokens, memory_tokens = context_component_estimates(host.session, memory_dir)

    body = render_context_panel(
        model_label=model,
        usage=usages[-1] if usages else None,
        window=window,
        first_usage=usages[0] if usages else None,
        model_calls=len(usages),
        system_prompt_tokens=system_tokens,
        memory_index_tokens=memory_tokens,
        memory_notes=len(list_memory_notes(memory_dir)),
        compact_trigger_tokens=auto_compact_trigger_tokens(window),
        clear_trigger_tokens=clear_tool_uses_trigger_tokens(window),
    )
    return {"markdown": body}


@router.post("/sessions/{session_id}/upload")
async def upload_file(
    session_id: str, file: Annotated[UploadFile, File()], manager: ManagerDep
) -> dict[str, str]:
    """Save an uploaded file into the session workspace so the agent can use it.

    Files land under ``uploads/`` in the workspace the agent runs in, so the
    user can refer to ``uploads/<name>`` and the file tools / REPL read it.
    """
    from makker.paths import workspace_root

    host = manager.get(session_id)
    if host is None:
        raise HTTPException(status_code=404, detail="no such session")
    ws = host.workspace or workspace_root()
    # Basename only, then a conservative safe name (no path separators escape).
    name = Path(file.filename or "upload").name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "upload"
    dest = ws / "uploads" / safe
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stream to disk with a size cap so a large upload can't exhaust memory
    # (the whole server runs on one event loop).
    max_bytes = 100 * 1024 * 1024
    written = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="upload too large (max 100 MB)")
            fh.write(chunk)
    rel = f"uploads/{safe}"
    host.session.trace.append(schema.UPLOAD, {"path": rel})
    return {"path": rel}
