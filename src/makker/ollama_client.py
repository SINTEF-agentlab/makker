"""Local Ollama integration: reachability, installed models, and streaming pull.

Thin async wrapper over the ``ollama`` package. The default host is
``http://localhost:11434`` (override with ``$OLLAMA_HOST``). If the server is
down, ``is_reachable`` is ``False`` and the checks return empty rather than
raising.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
from collections.abc import AsyncIterator
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"

# The smallest context a local model is loaded with, even when part of the model
# then runs on the CPU: the agent's own prompt and tools take 9K to 15K tokens,
# so a slower model is better than one that cannot hold a task.
MIN_NUM_CTX = 32_768

# A load can take minutes for a large model read from a cold disk.
_LOAD_TIMEOUT_S = 300.0
# Loads to spend narrowing the search after the first two (each takes seconds);
# six narrow a 256K-token range to within an eighth.
_MAX_PROBES = 6


def host() -> str:
    """Ollama server URL the client will talk to."""
    return os.environ.get("OLLAMA_HOST") or DEFAULT_HOST


def model_name(model_id: str) -> str:
    """The Ollama model tag from an ``ollama:<name>`` id (``<name>``)."""
    _, sep, rest = model_id.partition(":")
    return rest if sep else model_id


@dataclass(frozen=True)
class PullProgress:
    """One progress update from a streaming pull."""

    status: str
    fraction: float | None  # 0..1 when the layer reports total + completed


def _client():
    from ollama import AsyncClient

    return AsyncClient()


async def is_reachable() -> bool:
    """True if the Ollama server answers (and the package is importable)."""
    try:
        await _client().list()
        return True
    except Exception:
        return False


async def installed_models() -> list[str]:
    """Names (tags) of the models pulled locally; empty if the server is down."""
    names: list[str] = []
    with contextlib.suppress(Exception):
        resp = await _client().list()
        for model in getattr(resp, "models", None) or []:
            name = getattr(model, "model", None)
            if name is None and isinstance(model, dict):
                name = model.get("model")
            if name:
                names.append(name)
    return names


async def capabilities(name: str) -> list[str]:
    """Capability tags the daemon reports for a pulled model (e.g. ``tools``,
    ``vision``); empty on any error."""
    with contextlib.suppress(Exception):
        info = await _client().show(name)
        caps = getattr(info, "capabilities", None)
        if caps is None and isinstance(info, dict):
            caps = info.get("capabilities")
        if caps:
            return list(caps)
    return []


async def supports_tools(name: str) -> bool:
    """Whether the daemon exposes tool calling for ``name``."""
    return "tools" in await capabilities(name)


def thinks(name: str) -> bool:
    """Whether the daemon reports ``name`` as a thinking-capable model.

    Sync because it runs at model-build time; best-effort (False when the
    daemon can't answer).
    """
    with contextlib.suppress(Exception):
        from ollama import Client

        info = Client(timeout=2.0).show(name)
        caps = getattr(info, "capabilities", None)
        if caps is None and isinstance(info, dict):
            caps = info.get("capabilities")
        return "thinking" in (caps or [])
    return False


def context_window(name: str) -> int | None:
    """The model's max context length (tokens) the daemon reports, or None.

    Lets callers size ``num_ctx`` to the model rather than guess. Sync because it
    runs at model-build time; best-effort (None when the daemon can't answer).
    """
    with contextlib.suppress(Exception):
        from ollama import Client

        info = Client(timeout=2.0).show(name)
        modelinfo = getattr(info, "modelinfo", None)
        if modelinfo is None and isinstance(info, dict):
            modelinfo = info.get("modelinfo") or info.get("model_info")
        lengths = [
            value
            for key, value in (modelinfo or {}).items()
            if isinstance(value, int) and key.endswith("context_length")
        ]
        if lengths:
            return max(lengths)
    return None


def num_ctx(name: str) -> int | None:
    """Context window (tokens) to load the local model ``name`` with.

    The ``num_ctx`` its Modelfile sets, when it sets one. Otherwise the largest
    context, up to the model's maximum, that Ollama keeps entirely on this
    machine's GPU: measured by loading the model, once per model version and
    machine, and remembered in the state home. ``None`` when the daemon cannot
    report the model.

    This is the *loaded* window, the single figure the rest of the app sizes
    against: the model is built with it, and the context budget (compaction,
    ``/context``) reads it, so compaction is never scheduled past a window the
    model was not loaded with.
    """
    maximum = context_window(name)
    if maximum is None:
        return None
    configured = _modelfile_num_ctx(name)
    if configured:
        return min(configured, maximum)
    digest = _digest(name)
    key = f"{platform.node()} {host()} {digest}"
    remembered = _measured().get(key)
    if isinstance(remembered, int):
        return remembered
    try:
        measured = _largest_on_gpu(name, maximum)
    except Exception:
        # The daemon stopped answering mid-way: use the floor for now and
        # measure again next time.
        return min(maximum, MIN_NUM_CTX)
    if digest:
        _remember(key, measured)
    return measured


def _modelfile_num_ctx(name: str) -> int | None:
    """The ``num_ctx`` the model's Modelfile sets (e.g. a ``-64k`` variant), or None."""
    with contextlib.suppress(Exception):
        from ollama import Client

        parameters = getattr(Client(timeout=2.0).show(name), "parameters", None) or ""
        for line in parameters.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == "num_ctx":
                return int(fields[1])
    return None


def _digest(name: str) -> str | None:
    """The installed model's digest, which changes when the model does."""
    with contextlib.suppress(Exception):
        from ollama import Client

        for model in Client(timeout=2.0).list().models:
            if _matches(model.model or "", name):
                return model.digest
    return None


@dataclass(frozen=True)
class _Load:
    """What Ollama allocated for one load of a model, in bytes."""

    num_ctx: int
    size: int
    vram: int  # the part on the GPU

    @property
    def fits(self) -> bool:
        return 0 < self.size <= self.vram


def _load(name: str, num_ctx: int) -> _Load | None:
    """Load ``name`` with a ``num_ctx`` context and read back what it took.

    ``None`` when Ollama refuses the load (too large for this machine's memory);
    a connection failure or timeout raises.
    """
    from ollama import Client, ResponseError

    client = Client(timeout=_LOAD_TIMEOUT_S)
    try:
        # An empty prompt only loads the model.
        client.generate(name, prompt="", options={"num_ctx": num_ctx})
    except ResponseError:
        return None
    for model in client.ps().models:
        if _matches(model.model or "", name):
            return _Load(model.context_length or num_ctx, model.size or 0, model.size_vram or 0)
    return None


def _largest_on_gpu(name: str, maximum: int) -> int:
    """The largest context, up to ``maximum``, that Ollama keeps entirely on the GPU.

    Ollama places each load itself and reports where it went, so it decides,
    not an estimate (memory per token differs between models and between small
    and large contexts, and a load that spills reports sizes unlike one that
    fits). One load at the maximum settles it when the GPU has room. Otherwise
    the answer lies between a load that fits and one that spills, and halving
    the gap narrows it to within an eighth. Below ``MIN_NUM_CTX`` it does not
    look.
    """
    top = _load(name, maximum)
    if top is not None and top.fits:
        return top.num_ctx
    low = _load(name, MIN_NUM_CTX) if maximum > MIN_NUM_CTX else None
    if low is None or not low.fits:
        # Too large for the GPU at any useful size: part of it runs on the CPU.
        return min(maximum, MIN_NUM_CTX)
    fits, spills = low.num_ctx, maximum
    for _ in range(_MAX_PROBES):
        if spills - fits <= fits // 8:
            break
        guess = (fits + spills) // 2 // 1024 * 1024
        probe = _load(name, guess)
        if probe is not None and probe.fits:
            fits = probe.num_ctx
        else:
            spills = guess
    return fits


def _measured_path():
    from makker.paths import state_home

    return state_home() / "ollama-context.json"


def _measured() -> dict:
    """Measured contexts, keyed by machine, Ollama host and model digest."""
    with contextlib.suppress(OSError, ValueError):
        data = json.loads(_measured_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def _remember(key: str, value: int) -> None:
    path = _measured_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**_measured(), key: value}, indent=1), encoding="utf-8")


def _matches(installed: str, name: str) -> bool:
    # Installed tags carry a version (e.g. "llama3.1:latest"); match a bare name
    # against the tag's base so "llama3.1" finds "llama3.1:latest".
    if installed == name:
        return True
    return ":" not in name and installed.split(":", 1)[0] == name


async def is_installed(name: str) -> bool:
    """True if ``name`` (a bare model name or full tag) is already pulled."""
    return any(_matches(installed, name) for installed in await installed_models())


async def pull(name: str) -> AsyncIterator[PullProgress]:
    """Stream a model pull, yielding ``PullProgress`` until it completes.

    Raises whatever the ``ollama`` client raises (e.g. a connection error).
    """
    stream = await _client().pull(name, stream=True)
    async for chunk in stream:
        status = getattr(chunk, "status", None) or ""
        total = getattr(chunk, "total", None)
        completed = getattr(chunk, "completed", None)
        fraction = completed / total if total and completed is not None else None
        yield PullProgress(status=status, fraction=fraction)
