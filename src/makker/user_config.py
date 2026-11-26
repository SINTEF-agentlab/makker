"""User-global configuration shared across every workspace.

Stored at ``state_home()/config.toml`` (see ``paths.user_config_path``). It
shares the top-level ``model`` key with the per-workspace config, and the
workspace value wins (see ``resolve_model`` in ``agent.builder``). Secrets are
not stored here — API keys go in the global ``.env`` instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from makker.paths import user_config_path


@dataclass(frozen=True)
class UserConfig:
    """Contents of the user-global ``config.toml``. Empty is fine."""

    model: str | None = None
    # Models pinned to the top of the model menu, in the order pinned.
    pinned_models: tuple[str, ...] = ()
    # Per-model settings (effort, output limit, context size), keyed by model id;
    # see ``model_settings``.
    model_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Where automatic compaction runs, in percent of the window; see
    # ``context_budget.compact_percent``.
    compact_at: int | None = None


def load_user_config() -> UserConfig:
    """Read the user-global ``config.toml`` if present; else an empty config."""

    path = user_config_path()
    if not path.exists():
        return UserConfig()

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    model = data.get("model")
    pinned = data.get("pinned_models")
    settings = data.get("model_settings")
    compact_at = data.get("compact_at")
    return UserConfig(
        model=model if isinstance(model, str) else None,
        pinned_models=tuple(m for m in pinned if isinstance(m, str))
        if isinstance(pinned, list)
        else (),
        model_settings={k: dict(v) for k, v in settings.items() if isinstance(v, dict)}
        if isinstance(settings, dict)
        else {},
        compact_at=compact_at
        if isinstance(compact_at, int) and not isinstance(compact_at, bool)
        else None,
    )


def write_user_config(config: UserConfig) -> Path:
    """Persist ``config`` to the user-global config file. Returns the path."""

    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    from makker.workspace import toml_basic_string

    lines: list[str] = []
    if config.model:
        lines.append(f"model = {toml_basic_string(config.model)}")
    if config.pinned_models:
        pinned = ", ".join(toml_basic_string(m) for m in config.pinned_models)
        lines.append(f"pinned_models = [{pinned}]")
    if config.compact_at is not None:
        lines.append(f"compact_at = {int(config.compact_at)}")
    for model_id, values in sorted(config.model_settings.items()):
        if not values:
            continue
        lines.append(f"\n[model_settings.{toml_basic_string(model_id)}]")
        for key, value in values.items():
            literal = toml_basic_string(value) if isinstance(value, str) else str(int(value))
            lines.append(f"{key} = {literal}")

    body = "\n".join(lines)
    path.write_text(body + "\n" if body else "", encoding="utf-8")
    return path
