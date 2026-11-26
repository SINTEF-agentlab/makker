"""How much context a model has, where that figure came from, and how it is kept in it.

One ``ContextBudget`` per running agent, resolved when the agent is built and on
every model switch. The summarizer, the tool-result offloading, the status bar and
``/context`` all read it, so they agree on one window and one set of marks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# The window a model is assumed to have when no source reports one: a model newer
# than the provider package's data, from a provider that does not list windows.
# Nine in ten tool-calling models released in the year to 2026-11 have at least
# this. A guess that is too high fails visibly: the provider rejects the request,
# and the summarizer then compacts and retries. One that is too low would quietly
# waste the window.
ASSUMED_WINDOW = 200_000

# Compaction runs at this share of the window (in percent; the user config's
# ``compact_at`` sets another within the range), and earlier when the conversation
# would not leave room for the model's output allowance within 95% of it (the
# summarizer refuses to send a request past that point anyway).
DEFAULT_COMPACT_PERCENT = 85
COMPACT_PERCENT_RANGE = (50, 95)
_INPUT_FRACTION = 0.95
_OUTPUT_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
# Older tool results move to files once the context reaches this share of the
# compaction point, until it is back under the lower share. Offloading in a chunk
# rather than one result per call keeps the prompt stable between calls, which
# provider prompt caching depends on.
_OFFLOAD_AT = 0.7
_OFFLOAD_TO = 0.45
# The newest tool results always stay: the agent is working with them. Fewer on a
# small window, so that the kept set cannot fill it.
_SMALL_WINDOW = 100_000

WindowSource = Literal["provider", "ollama", "assumed"]


@dataclass(frozen=True)
class ContextBudget:
    """A model's input window, where it came from, and its compaction point.

    ``source`` is ``provider`` (the provider's model data or API), ``ollama`` (what
    the Ollama daemon reports or loads) or ``assumed`` (no source knew).
    ``output_reserve`` is the model's configured output allowance, which the
    summarizer keeps free. ``compact_percent`` is the share of the window at which
    compaction runs.
    """

    window: int
    source: WindowSource
    output_reserve: int = 0
    compact_percent: int = DEFAULT_COMPACT_PERCENT

    @property
    def compact_at(self) -> int:
        """The context size at which automatic compaction runs."""
        at = self.window * self.compact_percent // 100
        if self.output_reserve:
            at = min(at, int(self.window * _INPUT_FRACTION) - self.output_reserve)
        return max(0, at)

    @property
    def offload_at(self) -> int:
        """The context size at which older tool results start moving to files."""
        return round(self.compact_at * _OFFLOAD_AT)

    @property
    def offload_to(self) -> int:
        """The context size offloading brings the conversation back down to."""
        return round(self.compact_at * _OFFLOAD_TO)

    @property
    def keep_tool_results(self) -> int:
        """How many of the newest tool results always stay in the conversation."""
        return 3 if self.window < _SMALL_WINDOW else 6


def least_window(output_reserve: int, least: int) -> int:
    """The smallest window for a model that keeps ``output_reserve`` free for its
    reply: one that leaves it as much room to work in as a window of ``least``
    leaves a model keeping nothing free. Anthropic models keep 24K free, and in a
    window smaller than that plus the agent's own prompt no turn can run."""
    if not output_reserve:
        return least
    return math.ceil((least * DEFAULT_COMPACT_PERCENT / 100 + output_reserve) / _INPUT_FRACTION)


def context_budget(model_id: str | None, model: Any = None) -> ContextBudget:
    """The budget for ``model_id``, or for a model instance passed in directly.

    A model id is looked up (see ``models.context_window``). An instance passed
    without an id (tests, the lab) answers from its own profile. Either way an
    instance contributes its output allowance.
    """
    from makker.models import context_window, is_ollama_cloud, provider_of

    window: int | None = None
    source: WindowSource = "provider"
    reserve = _output_reserve(model)
    if model_id:
        from makker.model_settings import MIN_CONTEXT, load_settings

        window = context_window(model_id)
        if provider_of(model_id) == "ollama":
            source = "ollama"
        # A context size the user chose caps the window: a local model is loaded
        # with it, and any model is compacted against it.
        chosen = load_settings(model_id).context
        if chosen:
            if source != "ollama" or is_ollama_cloud(model_id):
                # A hosted model's cap is ours alone, so one saved before the
                # output allowance grew is raised to the least that still works.
                chosen = max(chosen, least_window(reserve, MIN_CONTEXT))
            window = min(window, chosen) if window else chosen
    elif model is not None:
        profile = getattr(model, "profile", None)
        value = profile.get("max_input_tokens") if isinstance(profile, dict) else None
        window = value if isinstance(value, int) and value > 0 else None
    if not window:
        window, source = ASSUMED_WINDOW, "assumed"
    return ContextBudget(
        window=window,
        source=source,
        output_reserve=reserve,
        compact_percent=compact_percent(),
    )


def compact_percent() -> int:
    """The share of the window, in percent, at which compaction runs: the user
    config's ``compact_at`` when it is in range, else the default."""
    from makker.user_config import load_user_config

    chosen = load_user_config().compact_at
    low, high = COMPACT_PERCENT_RANGE
    return chosen if chosen is not None and low <= chosen <= high else DEFAULT_COMPACT_PERCENT


def save_compact_percent(percent: int) -> None:
    """Save the compaction point in the user config; the default is not stored.

    A running agent keeps its budget until it is rebuilt (a model switch, or the
    web UI's ``set_model`` with the same model)."""
    from dataclasses import replace

    from makker.user_config import load_user_config, write_user_config

    low, high = COMPACT_PERCENT_RANGE
    if not low <= percent <= high:
        raise ValueError(f"The compaction point must be between {low}% and {high}%.")
    chosen = None if percent == DEFAULT_COMPACT_PERCENT else percent
    write_user_config(replace(load_user_config(), compact_at=chosen))


def _output_reserve(model: Any) -> int:
    """The largest output limit configured on ``model`` (0 when none is)."""
    limits = [getattr(model, key, None) for key in _OUTPUT_LIMIT_KEYS] if model else []
    return max(
        (v for v in limits if isinstance(v, int) and not isinstance(v, bool)),
        default=0,
    )
