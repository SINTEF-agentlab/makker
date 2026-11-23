"""Tests for the approval selection menu."""

from __future__ import annotations

from makker.interfaces.tui.approval_menu import build_approval_options


def test_build_approval_options_without_always_allow() -> None:
    options = build_approval_options(allowed_decisions=frozenset({"approve", "reject"}))
    labels = [option.label for option in options]
    assert labels == ["Yes", "No, and tell the agent what to do differently"]


def test_build_approval_options_for_file_edits() -> None:
    options = build_approval_options(
        allowed_decisions=frozenset({"approve", "reject"}),
        always_allow=frozenset({"file_edits"}),
    )
    assert "file edits" in options[1].label
    assert options[1].allowlist_categories == {"file_edits"}
