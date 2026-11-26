"""Tests for the ``makker key`` subcommand."""

from __future__ import annotations

import pytest

from makker.interfaces.cli.main import main

# The autouse ``_reset_workspace_overrides`` fixture points the workspace and
# state_home at fresh tmp dirs, so the .env files these tests read are empty
# unless a test writes them.


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_key_show_lists_providers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["key", "--show"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out
    assert "ANTHROPIC_API_KEY" in out
    assert "OPENROUTER_API_KEY" in out
    assert "not set" in out
    assert ".env" in out  # tells the user where keys live


def test_key_show_reports_a_saved_key(capsys: pytest.CaptureFixture[str]) -> None:
    from makker.credentials import store_credential_for_provider

    store_credential_for_provider("openai", "sk-saved-key-abcdef123456")
    assert main(["key", "--show"]) == 0
    out = capsys.readouterr().out
    assert "saved" in out
    assert "sk-saved-key-abcdef123456" not in out  # masked, never the raw value


def test_key_set_unknown_provider_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["key", "bogus"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_key_set_requires_a_tty(capsys: pytest.CaptureFixture[str]) -> None:
    # Non-interactive (pytest has no TTY): setting a key is refused with guidance.
    assert main(["key", "openai"]) == 2
    assert "interactive terminal" in capsys.readouterr().err
