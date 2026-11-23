"""Human-in-the-loop approval for the workspace file and shell tools.

Every gated tool sits in the graph's ``interrupt_on`` map with a ``when`` check
that asks the session's :class:`ApprovalPolicy` whether this one call needs a
person. The policy is live, so a change of mode or of "always allow" applies to
the next tool call, mid-turn included, with no rebuild; and an approval card
only ever holds the calls the policy could not allow.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from langchain.agents.middleware import InterruptOnConfig

from makker.paths import resolve_in_workspace

# Tools that require human approval. Action name → human-readable description
# shown in the approval card. All entries use approve/reject only.
_APPROVAL_TOOLS: dict[str, str] = {
    "execute": "Run a shell command in the workspace.",
    "write_file": "Write a file.",
    "edit_file": "Edit a file.",
    "delete": "Delete a file or directory (directories go recursively).",
}

# The only calls "accept edits" and "always allow" wave through, and only when
# the file is inside the workspace. A recursive delete and a shell command always
# get their own prompt outside auto mode.
_EDIT_TOOLS = frozenset({"write_file", "edit_file"})

# The one category a session can "always allow".
ALLOWLIST_FILE_EDITS = "file_edits"


class ApprovalMode(StrEnum):
    """How side-effecting Deep Agents tools are gated before execution."""

    ASK = "ask"
    """Prompt before every gated tool (default)."""

    WORKSPACE = "workspace"
    """Auto-allow write_file and edit_file inside the workspace; prompt for the rest."""

    AUTO = "auto"
    """Auto-allow all gated tools (non-interactive trust mode)."""

    def display_label(self) -> str:
        return {
            ApprovalMode.ASK: "default",
            ApprovalMode.WORKSPACE: "accept edits",
            ApprovalMode.AUTO: "auto",
        }[self]

    def cycle_next(self) -> ApprovalMode:
        order = (ApprovalMode.ASK, ApprovalMode.WORKSPACE, ApprovalMode.AUTO)
        index = order.index(self)
        return order[(index + 1) % len(order)]


def parse_approval_mode(value: str | None) -> ApprovalMode:
    if not value:
        return ApprovalMode.ASK
    normalized = value.strip().lower()
    for mode in ApprovalMode:
        if mode == normalized:
            return mode
    raise ValueError(
        f"unknown approval mode {value!r}; expected one of: "
        + ", ".join(m.value for m in ApprovalMode)
    )


def _actions(value: Any) -> list[dict[str, Any]]:
    """The ``action_requests`` of an interrupt payload (empty when malformed)."""

    actions = value.get("action_requests") if isinstance(value, dict) else None
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


class ApprovalPolicy:
    """Decides, for each gated tool call, whether a person must approve it.

    One per session, held by the session host and read by the agent graph, so
    front ends change the mode and "always allow" here and nowhere else.
    """

    def __init__(
        self, mode: ApprovalMode | str | None = None, *, workspace: Path | None = None
    ) -> None:
        self.mode = mode if isinstance(mode, ApprovalMode) else parse_approval_mode(mode)
        self._workspace = workspace
        self._allow_edits = False
        self._frozen: dict[str, bool] = {}

    @property
    def allowlist(self) -> frozenset[str]:
        """The categories this session always allows."""
        return frozenset({ALLOWLIST_FILE_EDITS}) if self._allow_edits else frozenset()

    def allow(self, category: str) -> None:
        """Always allow ``category`` for the rest of the session (file edits only)."""
        if category == ALLOWLIST_FILE_EDITS:
            self._allow_edits = True

    def needs_approval(self, name: str, args: Any) -> bool:
        """Whether a call to tool ``name`` with ``args`` must wait for a person."""
        if name not in _APPROVAL_TOOLS or self.mode is ApprovalMode.AUTO:
            return False
        if name in _EDIT_TOOLS and (self.mode is ApprovalMode.WORKSPACE or self._allow_edits):
            return not self._in_workspace(args)
        return True

    def _in_workspace(self, args: Any) -> bool:
        path = args.get("file_path") if isinstance(args, dict) else None
        return isinstance(path, str) and (
            resolve_in_workspace(path, workspace=self._workspace) is not None
        )

    def gate(self, request: Any) -> bool:
        """The ``when`` check on every gated tool: interrupt for this call?

        The answer is frozen per tool call. Resuming a paused turn runs the check
        again for the same model message, and the calls it picks must match the
        decisions sent, or deepagents raises; worse, a call it now lets through
        would run whatever the decision was.
        """
        call = request.tool_call
        call_id = call.get("id")
        if not call_id:
            return self.needs_approval(call["name"], call.get("args"))
        if call_id not in self._frozen:
            self._frozen[call_id] = self.needs_approval(call["name"], call.get("args"))
        return self._frozen[call_id]

    def settled(self) -> None:
        """Forget the frozen answers once a turn ends with nothing pending."""
        self._frozen.clear()

    def approves(self, interrupts: Iterable[SupportsInterrupt]) -> bool:
        """Whether every action of these pending interrupts is now allowed.

        True after a change such as switching to auto, so the front end can
        resume a card that no longer needs anyone.
        """
        actions = [action for interrupt in interrupts for action in _actions(interrupt.value)]
        return bool(actions) and not any(
            self.needs_approval(str(action.get("name") or ""), action.get("args"))
            for action in actions
        )

    def always_allow_offer(self, value: Any) -> frozenset[str]:
        """What a card may offer to "always allow": file edits, when that is all it holds.

        Offered only when every action is an edit inside the workspace, so the
        choice never covers a delete, a shell command, or a file elsewhere.
        """
        actions = _actions(value)
        if actions and all(
            action.get("name") in _EDIT_TOOLS and self._in_workspace(action.get("args"))
            for action in actions
        ):
            return frozenset({ALLOWLIST_FILE_EDITS})
        return frozenset()


def interrupt_on_config(policy: ApprovalPolicy | None = None) -> dict[str, InterruptOnConfig]:
    """``interrupt_on`` map keyed by tool name (used by `create_deep_agent`).

    With a ``policy``, each tool interrupts only for the calls the policy says
    need a person; without one, every call interrupts.
    """

    config: dict[str, InterruptOnConfig] = {}
    for name, description in _APPROVAL_TOOLS.items():
        config[name] = {"allowed_decisions": ["approve", "reject"], "description": description}
        if policy is not None:
            config[name]["when"] = policy.gate
    return config


# ---------------------------------------------------------------------------
# Interrupt decisions and resume payloads.
#
# Shared across interfaces (TUI, server) so every front end resolves an
# approval interrupt the same way. An interrupt payload follows deepagents'
# contract: a dict with an ``action_requests`` list and a parallel
# ``review_configs`` list of allowed-decision policies.

# The decisions a front end can resolve an interrupt with.
SUPPORTED_APPROVAL_DECISIONS = frozenset({"approve", "reject", "respond"})


class SupportsInterrupt(Protocol):
    """A pending interrupt: an id and its raw payload value."""

    interrupt_id: str
    value: Any


def review_config_map(review_configs: Any) -> dict[str, frozenset[str]]:
    """Map each action name to its allowed decisions from a ``review_configs`` list."""

    config_map: dict[str, frozenset[str]] = {}
    if not isinstance(review_configs, list):
        return config_map
    for review in review_configs:
        if not isinstance(review, dict):
            continue
        action_name = review.get("action_name")
        allowed = review.get("allowed_decisions")
        if isinstance(action_name, str) and isinstance(allowed, list):
            config_map[action_name] = frozenset(str(item) for item in allowed)
    return config_map


def allowed_decisions_for_interrupt(value: Any) -> frozenset[str]:
    """Return the intersection of decisions allowed across an interrupt's actions.

    An interrupt may bundle multiple ``action_requests``; each can declare its
    own ``allowed_decisions`` in a sibling ``review_configs`` block. A front end
    can only resume an interrupt with a decision every action accepts, so we
    intersect. Empty / malformed payloads fall back to all supported decisions,
    matching deepagents' default.
    """

    if not isinstance(value, dict):
        return SUPPORTED_APPROVAL_DECISIONS

    config_map = review_config_map(value.get("review_configs"))
    action_requests = value.get("action_requests")
    if not isinstance(action_requests, list):
        return SUPPORTED_APPROVAL_DECISIONS

    shared: frozenset[str] | None = None
    for action in action_requests:
        if not isinstance(action, dict):
            continue
        action_name = str(action.get("name") or "")
        allowed = config_map.get(action_name, SUPPORTED_APPROVAL_DECISIONS)
        shared = allowed if shared is None else shared & allowed
    return shared if shared is not None else SUPPORTED_APPROVAL_DECISIONS


def pending_allowed_decisions(interrupts: Iterable[SupportsInterrupt]) -> frozenset[str]:
    """Decisions every pending interrupt accepts, scoped to the supported set."""

    shared: frozenset[str] | None = None
    for interrupt in interrupts:
        allowed = allowed_decisions_for_interrupt(interrupt.value)
        shared = allowed if shared is None else shared & allowed
    if shared is None:
        return frozenset()
    return shared & SUPPORTED_APPROVAL_DECISIONS


def build_resume_payload(
    interrupts: Iterable[SupportsInterrupt],
    decision: dict[str, str],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Apply one decision to every action of every pending interrupt, for ``resume``.

    The result is the ``Command(resume=...)`` payload deepagents expects: keyed
    by interrupt id, each carrying one decision per ``action_request``.
    """

    payload: dict[str, dict[str, list[dict[str, str]]]] = {}
    for interrupt in interrupts:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        action_requests = value.get("action_requests")
        count = len(action_requests) if isinstance(action_requests, list) else 1
        payload[interrupt.interrupt_id] = {"decisions": [deepcopy(decision) for _ in range(count)]}
    return payload
