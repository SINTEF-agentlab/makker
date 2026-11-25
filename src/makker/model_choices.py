"""What the model menus offer first, and the choices they remember.

The short list is built by rule, so nothing hand-kept goes stale: the models the
user pinned, the recent ones, the newest few general-purpose models of each
provider with a key, and the local models installed. Everything else is one
search away in the full catalog. Both front ends show this list.

A switch is remembered twice: as the most recent model, and as the model for
this workspace (or for every workspace).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from makker.models import PROVIDERS, ModelInfo, ProviderInfo
from makker.recent_models import record_recent_model

NEWEST_PER_PROVIDER = 3

# The newest-per-provider rule skips special-purpose models, dated snapshots and
# moving aliases: they are in the full catalog for whoever wants them.
_SPECIAL = re.compile(r"realtime|image|audio|tts|transcribe|search|embed|live|codex|computer|-chat")
_SNAPSHOT = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")
_ALIAS = re.compile(r"latest|preview|-exp")

Section = tuple[str, list[ModelInfo]]


def _general(model: ModelInfo) -> bool:
    name = model.id.partition(":")[2]
    return not (
        model.deprecated
        or model.note
        or _SPECIAL.search(name)
        or _SNAPSHOT.search(name)
        or _ALIAS.search(name)
    )


def has_key(provider: ProviderInfo) -> bool:
    return provider.local or bool(os.environ.get(provider.key_env_var or ""))


def short_list(
    models: Iterable[ModelInfo],
    *,
    current: str | None = None,
    pinned: Iterable[str] = (),
    recent: Iterable[str] = (),
    keyed: Callable[[ProviderInfo], bool] = has_key,
) -> list[Section]:
    """The menu's sections in order, each model listed once, at its first section.

    The current model is always on it. OpenRouter's hundreds of models are
    reached through pins, recents and search, never listed by the newest rule.
    """
    models = list(models)
    by_id = {model.id: model for model in models}
    seen: set[str] = set()
    sections: list[Section] = []

    def add(title: str, ids: Iterable[str]) -> None:
        rows = []
        for model_id in ids:
            if model_id and model_id not in seen:
                seen.add(model_id)
                rows.append(by_id.get(model_id) or ModelInfo(model_id, model_id))
        if rows:
            sections.append((title, rows))

    add("Pinned", pinned)
    add("Recent", [current or "", *recent])
    for name, provider in PROVIDERS.items():
        if provider.local or name == "openrouter" or not keyed(provider):
            continue
        newest = [m for m in models if m.provider == name and _general(m)]
        newest.sort(key=lambda m: (m.released or "", m.id), reverse=True)
        add(provider.label, [m.id for m in newest[:NEWEST_PER_PROVIDER]])
    add("Local", [m.id for m in models if m.local and not m.note])
    return sections


def missing_keys(keyed: Callable[[ProviderInfo], bool] = has_key) -> list[ProviderInfo]:
    """The cloud providers without a key, offered as "Add key" rows."""
    return [p for p in PROVIDERS.values() if not p.local and not keyed(p)]


def load_pinned() -> list[str]:
    from makker.user_config import load_user_config

    return list(load_user_config().pinned_models)


def set_pinned(model_id: str, pinned: bool) -> list[str]:
    """Pin or unpin ``model_id`` in the user config; returns the pinned list."""
    from makker.user_config import load_user_config, write_user_config

    config = load_user_config()
    current = [m for m in config.pinned_models if m != model_id]
    if pinned:
        current.append(model_id)
    write_user_config(replace(config, pinned_models=tuple(current)))
    return current


def remember_model(model_id: str, *, scope: str = "workspace", workspace: Path) -> str:
    """Record ``model_id`` as the most recent and save it for ``scope``
    (``workspace`` or ``global``). Returns where it was saved, for a note."""
    record_recent_model(model_id)
    try:
        if scope == "global":
            from makker.user_config import load_user_config, write_user_config

            write_user_config(replace(load_user_config(), model=model_id))
            return "user config"
        from makker.workspace import load_workspace_config, write_workspace_config

        config = load_workspace_config(workspace)
        write_workspace_config(replace(config, model=model_id), workspace=workspace)
        return "this workspace"
    except OSError:
        return "this session only (could not write config)"
