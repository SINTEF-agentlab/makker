"""How each model runs: its reasoning effort, its output limit, and a local model's
context size.

Saved per model in the user config, so a model runs the same way in every
workspace. The model build reads ``load_settings`` (``agent.builder``,
``models.local_context``); the front ends show ``options``: only the settings a
model supports, each with the choices it offers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

# Output limits offered, up to the model's own maximum.
_OUTPUT_CHOICES = (8_192, 16_384, 32_768, 65_536, 131_072)
# The output limit a provider runs with when none is chosen. Anthropic requires
# one; thinking spends from it, so it clears the thinking with room for a long
# reply. Elsewhere the model's own maximum applies.
DEFAULT_OUTPUT = {"anthropic": 24_000}
# The effort asked for when a model lists levels but no default.
_DEFAULT_EFFORT = "medium"


@dataclass(frozen=True)
class ModelSettings:
    """A model's chosen settings; ``None`` runs it at its default."""

    effort: str | None = None
    output: int | None = None
    context: int | None = None


def load_settings(model_id: str) -> ModelSettings:
    """The settings saved for ``model_id`` (all defaults when none are)."""
    from makker.user_config import load_user_config

    return settings_from(load_user_config().model_settings.get(model_id, {}))


def save_settings(model_id: str, settings: ModelSettings) -> None:
    """Save ``settings`` for ``model_id`` in the user config; defaults are dropped."""
    from makker.user_config import load_user_config, write_user_config

    config = load_user_config()
    stored = dict(config.model_settings)
    values = {key: value for key, value in asdict(settings).items() if value is not None}
    if values:
        stored[model_id] = values
    else:
        stored.pop(model_id, None)
    write_user_config(replace(config, model_settings=stored))


def effort_levels(model_id: str) -> tuple[list[str], str | None]:
    """The effort levels ``model_id`` accepts, lowest first, and its default.

    Cloud models list theirs in the provider package's data; a local model's come
    from the Ollama daemon. No levels means the model offers no choice.
    """
    from makker.models import bundled_profile, provider_of

    if provider_of(model_id) == "ollama":
        from makker import ollama_client

        return ollama_client.thinking_levels(ollama_client.model_name(model_id))
    profile = bundled_profile(model_id)
    listed = profile.get("reasoning_effort_levels") or []
    levels = [level for level in listed if isinstance(level, str)]
    if not levels:
        return [], None
    default = profile.get("reasoning_effort_default")
    if default not in levels:
        default = _DEFAULT_EFFORT if _DEFAULT_EFFORT in levels else None
    return levels, default


def chosen_effort(model_id: str) -> str | None:
    """The effort level chosen for ``model_id``, if the model still offers it.

    Asks for the model's levels only when a level was chosen, so building a model
    left at its default never queries the Ollama daemon for them.
    """
    chosen = load_settings(model_id).effort
    if not chosen:
        return None
    levels, _ = effort_levels(model_id)
    return chosen if chosen in levels else None


def effort(model_id: str) -> str | None:
    """The effort ``model_id`` runs at: the chosen level, else the model's default
    (``None`` when it offers no levels)."""
    return chosen_effort(model_id) or effort_levels(model_id)[1]


def options(model_id: str, *, measure: bool = False) -> dict[str, Any]:
    """The settings ``model_id`` supports, with their choices and defaults, as data
    for the front ends. A setting the model does not support is ``None``.

    A local model's context sizes carry the memory each takes on this machine
    when it has been measured; ``measure`` measures it (loading the model a few
    times, so seconds to a minute, once per model version).
    """
    from makker.models import bundled_profile, is_ollama_cloud, provider_of

    settings = load_settings(model_id)
    levels, default = effort_levels(model_id)
    result: dict[str, Any] = {
        "settings": asdict(settings),
        "effort": {"levels": levels, "default": default, "current": effort(model_id)}
        if levels
        else None,
        "output": None,
        "context": None,
    }
    provider = provider_of(model_id)
    maximum = bundled_profile(model_id).get("max_output_tokens")
    if isinstance(maximum, int) and maximum > 0:
        result["output"] = {
            "choices": [c for c in _OUTPUT_CHOICES if c < maximum] + [maximum],
            "default": DEFAULT_OUTPUT.get(provider, maximum),
        }
    if provider == "ollama" and not is_ollama_cloud(model_id):
        result["context"] = _context_options(model_id, measure=measure)
    return result


def _context_options(model_id: str, *, measure: bool) -> dict[str, Any] | None:
    """A local model's context sizes: powers of two from the smallest the agent
    works with up to the model's maximum, each with its memory and whether it
    stays on the GPU, plus what "auto" picks."""
    from makker import ollama_client

    name = ollama_client.model_name(model_id)
    maximum = ollama_client.context_window(name)
    if not maximum:
        return None
    sizes = []
    size = ollama_client.MIN_NUM_CTX
    while size < maximum:
        sizes.append(size)
        size *= 2
    sizes.append(maximum)
    found = ollama_client.calibration(name, maximum=maximum, measure=measure)
    choices = [
        {
            "tokens": tokens,
            "memory": found.memory(tokens) if found else None,
            "on_gpu": tokens <= found.fits if found and found.points else None,
        }
        for tokens in sizes
    ]
    return {
        "choices": choices,
        "auto": ollama_client.num_ctx(name, measure=False),
        "measured": bool(found and len(found.points) > 1),
    }


def settings_from(values: dict[str, Any]) -> ModelSettings:
    """``ModelSettings`` from saved or requested values, ignoring anything malformed."""

    def count(key: str) -> int | None:
        value = values.get(key)
        whole = isinstance(value, int) and not isinstance(value, bool)
        return value if whole and value > 0 else None

    chosen = values.get("effort")
    return ModelSettings(
        effort=chosen if isinstance(chosen, str) and chosen else None,
        output=count("output"),
        context=count("context"),
    )
