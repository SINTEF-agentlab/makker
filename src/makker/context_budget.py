"""How much context a model has, where that figure came from, and how it is kept in it.

One ``ContextBudget`` per running agent, resolved when the agent is built and on
every model switch. The summarizer, the tool-result offloading, the status bar and
``/context`` all read it, so they agree on one window and one set of marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# The window a model is assumed to have when no source reports one: a model newer
# than the provider package's data, from a provider that does not list windows.
# Nine in ten tool-calling models released in the year to 2026-11 have at least
# this. A guess that is too high fails visibly: the provider rejects the request,
# and the summarizer then compacts and retries. One that is too low would quietly
# waste the window.
ASSUMED_WINDOW = 200_000

# Compaction runs at this share of the window, and earlier when the conversation
# would not leave room for the model's output allowance within 95% of it (the
# summarizer refuses to send a request past that point anyway).
_COMPACT_FRACTION = 0.85
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
    summarizer keeps free.
    """

    window: int
    source: WindowSource
    output_reserve: int = 0

    @property
    def compact_at(self) -> int:
        """The context size at which automatic compaction runs."""
        at = int(self.window * _COMPACT_FRACTION)
        if self.output_reserve:
            at = min(at, int(self.window * _INPUT_FRACTION) - self.output_reserve)
        return max(0, at)

    @property
    def offload_at(self) -> int:
        """The context size at which older tool results start moving to files."""
        return int(self.compact_at * _OFFLOAD_AT)

    @property
    def offload_to(self) -> int:
        """The context size offloading brings the conversation back down to."""
        return int(self.compact_at * _OFFLOAD_TO)

    @property
    def keep_tool_results(self) -> int:
        """How many of the newest tool results always stay in the conversation."""
        return 3 if self.window < _SMALL_WINDOW else 6


def context_budget(model_id: str | None, model: Any = None) -> ContextBudget:
    """The budget for ``model_id``, or for a model instance passed in directly.

    A model id is looked up (see ``models.context_window``). An instance passed
    without an id (tests, the lab) answers from its own profile. Either way an
    instance contributes its output allowance.
    """
    from makker.models import context_window, provider_of

    window: int | None = None
    source: WindowSource = "provider"
    if model_id:
        window = context_window(model_id)
        if provider_of(model_id) == "ollama":
            source = "ollama"
    elif model is not None:
        profile = getattr(model, "profile", None)
        value = profile.get("max_input_tokens") if isinstance(profile, dict) else None
        window = value if isinstance(value, int) and value > 0 else None
    if not window:
        window, source = ASSUMED_WINDOW, "assumed"
    return ContextBudget(window=window, source=source, output_reserve=_output_reserve(model))


def _output_reserve(model: Any) -> int:
    """The largest output limit configured on ``model`` (0 when none is)."""
    limits = [getattr(model, key, None) for key in _OUTPUT_LIMIT_KEYS] if model else []
    return max(
        (v for v in limits if isinstance(v, int) and not isinstance(v, bool)),
        default=0,
    )
