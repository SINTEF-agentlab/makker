"""Tests for the model selector and API-key modals."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, OptionList, Static

from makker.interfaces.tui.model_menu import ApiKeyModal, ModelMenu, OllamaPullModal
from makker.ollama_client import PullProgress


@pytest.fixture(autouse=True)
def _offline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker import models
    from makker.interfaces.tui import model_menu

    monkeypatch.setattr(model_menu, "discover_available_models", models.discover_models)


class _Host(App[None]):
    def compose(self):
        yield Static("host")


def _option_ids(menu: ModelMenu) -> list[str | None]:
    options = menu.query_one("#model-options", OptionList)
    return [options.get_option_at_index(i).id for i in range(options.option_count)]


async def test_model_menu_lists_discovered_models() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        ids = _option_ids(app.screen)
        # A discovered id is present; provider headers are non-selectable (id=None).
        assert "anthropic:claude-sonnet-4-6" in ids
        assert any(id is not None and id.startswith("openrouter:") for id in ids)
        assert None in ids


async def test_model_menu_adds_api_listed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker import models
    from makker.interfaces.tui import model_menu

    def catalog():
        groups = {name: list(items) for name, items in models.discover_models().items()}
        groups["openai"].insert(
            0, models.ModelInfo("openai:test-api-only-model", "test-api-only-model")
        )
        return groups

    monkeypatch.setattr(model_menu, "discover_available_models", catalog)
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "openai:test-api-only-model" in _option_ids(app.screen)


async def test_model_menu_shows_recommended_and_cloud_ollama(monkeypatch) -> None:
    # No installed local models in tests; recommended-to-pull local models and
    # hosted cloud models still appear so you can pick without typing a tag.
    from makker import ollama_client

    async def _none() -> list[str]:
        return []

    monkeypatch.setattr(ollama_client, "installed_models", _none)
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        ids = _option_ids(app.screen)
        assert "ollama:qwen3.6:27b" in ids  # recommended local, pullable
        assert "ollama:glm-5.1:cloud" in ids  # hosted cloud


async def test_model_menu_shows_recent_section(monkeypatch) -> None:
    # A free-text id that isn't in any provider catalog can only appear via the
    # Recent section, so its presence proves that section is rendered.
    from makker.interfaces.tui import model_menu as mm

    monkeypatch.setattr(mm, "load_recent_models", lambda: ["openrouter:custom/model"])
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        model_ids = [mm._model_id(i) for i in _option_ids(app.screen)]
        assert "openrouter:custom/model" in model_ids


async def test_model_menu_filters_by_text() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        app.screen.query_one("#model-filter", Input).value = "haiku"
        await pilot.pause()

        ids = _option_ids(app.screen)
        assert "anthropic:claude-haiku-4-5" in ids
        assert "openai:gpt-5.4" not in ids


async def test_model_menu_free_text_enter_saves_to_workspace() -> None:
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None), results.append)
        await pilot.pause()
        app.screen.query_one("#model-filter", Input).value = "openrouter:some/model"
        await pilot.press("enter")
        await pilot.pause()

    assert results == [("openrouter:some/model", "workspace")]


async def test_model_menu_ctrl_a_saves_to_all_workspaces() -> None:
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None), results.append)
        await pilot.pause()
        app.screen.query_one("#model-filter", Input).value = "openrouter:some/model"
        await pilot.press("ctrl+a")
        await pilot.pause()

    assert results == [("openrouter:some/model", "global")]


async def test_model_menu_escape_cancels() -> None:
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None), results.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


async def test_model_menu_typed_id_that_a_section_lists_is_one_row(monkeypatch) -> None:
    # The free-text row checked only the provider catalog, so a typed Ollama id
    # was a second row with the same id, and Textual raised DuplicateID.
    from makker import ollama_client

    async def _none() -> list[str]:
        return []

    monkeypatch.setattr(ollama_client, "installed_models", _none)
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        app.screen.query_one("#model-filter", Input).value = "ollama:glm-5.1:cloud"
        await pilot.pause()
        assert _option_ids(app.screen).count("ollama:glm-5.1:cloud") == 1


async def test_model_menu_lists_a_used_cloud_model_once(monkeypatch) -> None:
    # Ollama lists a cloud model it has run among the installed ones.
    from makker import ollama_client

    async def _installed() -> list[str]:
        return ["glm-5.1:cloud"]

    monkeypatch.setattr(ollama_client, "installed_models", _installed)
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert _option_ids(app.screen).count("ollama:glm-5.1:cloud") == 1


async def test_model_menu_keeps_the_highlight_when_discovery_returns(monkeypatch) -> None:
    # The list is rebuilt when Ollama answers; the highlight used to jump back
    # to the first row, so Enter picked a model the user had moved away from.
    import asyncio

    from makker import ollama_client

    answer = asyncio.Event()

    async def _installed() -> list[str]:
        await answer.wait()
        return ["llama3.1:8b"]

    monkeypatch.setattr(ollama_client, "installed_models", _installed)
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(ModelMenu(current=None), results.append)
        await pilot.pause()
        await pilot.press("down", "down")
        options = app.screen.query_one("#model-options", OptionList)
        chosen = options.highlighted_option.id

        answer.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "ollama:llama3.1:8b" in _option_ids(app.screen)
        assert options.highlighted_option.id == chosen
        await pilot.press("enter")
        await pilot.pause()

    assert results == [(chosen, "workspace")]


async def test_api_key_modal_returns_value() -> None:
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(
            ApiKeyModal(env_var="OPENAI_API_KEY", provider_label="OpenAI"), results.append
        )
        await pilot.pause()
        key_input = app.screen.query_one("#api-key-input", Input)
        assert key_input.password  # the key is not echoed on screen
        key_input.value = "sk-xyz"
        await pilot.press("enter")
        await pilot.pause()

    assert results == ["sk-xyz"]


async def test_api_key_modal_escape_returns_none() -> None:
    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(
            ApiKeyModal(env_var="OPENAI_API_KEY", provider_label="OpenAI"), results.append
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


async def test_ollama_pull_modal_succeeds() -> None:
    async def puller(name):
        yield PullProgress(status="pulling", fraction=0.4)
        yield PullProgress(status="success", fraction=1.0)

    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(OllamaPullModal(model_name="llama3.1", puller=puller), results.append)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert results == [True]


async def test_ollama_pull_modal_reports_failure_then_closes() -> None:
    async def puller(name):
        raise ConnectionError("connection refused")
        yield  # make this an async generator

    results: list = []
    app = _Host()
    async with app.run_test() as pilot:
        app.push_screen(OllamaPullModal(model_name="llama3.1", puller=puller), results.append)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Failure leaves the dialog open (no auto-dismiss); Esc closes it as False.
        assert results == []
        assert isinstance(app.screen, OllamaPullModal)

        await pilot.press("escape")
        await pilot.pause()

    assert results == [False]
