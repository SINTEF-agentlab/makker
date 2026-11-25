"""Modal dialogs for selecting a model, entering an API key, and pulling a model.

``ModelMenu`` opens on the short list the web's model menu shows too (pinned,
recent, the newest per provider, installed local models); typing searches the
whole catalog, and any ``provider:model`` typed in is accepted. It returns
``(model_id, scope)``. ``ApiKeyModal`` collects a provider key when one is
missing. ``OllamaPullModal`` pulls a local model. The caller (``app.TUIApp``)
applies the resulting switch.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, ProgressBar, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from makker.model_choices import load_pinned, short_list
from makker.models import (
    PROVIDERS,
    ModelInfo,
    discover_available_models,
    discover_models,
    discover_ollama_models,
    provider_of,
)
from makker.ollama_client import PullProgress
from makker.recent_models import load_recent_models

ModelChoice = tuple[str, str]


def _matches(model: ModelInfo, needle: str) -> bool:
    if not needle:
        return True
    return (
        needle in model.id.lower() or needle in model.label.lower() or needle in model.note.lower()
    )


def _render_model(model: ModelInfo, *, current: str | None) -> Text:
    """One row: the model's name, and its note if it has one (``not pulled``,
    ``deprecated``)."""
    text = Text()
    marker = "● " if model.id == current else "  "
    text.append(marker, style="cyan" if model.id == current else "")
    text.append(model.label)
    if model.note:
        text.append(f"  {model.note}", style="dim")
    return text


class ModelMenu(ModalScreen[ModelChoice | None]):
    """Filterable model picker. Dismisses with ``(model_id, scope)`` or ``None``."""

    DEFAULT_CSS = """
    ModelMenu {
        align: center middle;
    }
    #model-menu {
        width: 84;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #model-menu-title {
        text-style: bold;
        padding-bottom: 1;
    }
    /* A clean round border; the Input default is a `tall` half-block border
       that leaves stray edge marks inside the modal. */
    #model-filter {
        border: round $surface-lighten-2;
    }
    #model-filter:focus {
        border: round $primary;
    }
    #model-options {
        height: auto;
        max-height: 16;
        margin: 1 0;
        background: $surface;
    }
    #model-menu-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        # Confirm + set as the global default; plain Enter saves to the workspace.
        Binding("ctrl+a", "select_global", "All workspaces", show=False, priority=True),
    ]

    def __init__(self, *, current: str | None = None) -> None:
        super().__init__()
        self._current = current
        self._recent: list[str] = load_recent_models()
        self._pinned: list[str] = load_pinned()
        self._catalog = discover_models()
        # Ollama ships no static profiles, so its models are discovered from the
        # daemon by a worker after mount; empty until (and unless) it answers.
        self._ollama: list[ModelInfo] = []

    def compose(self):
        with Vertical(id="model-menu"):
            yield Static("Select a model", id="model-menu-title")
            yield Input(
                placeholder="Search all models, or type provider:model",
                id="model-filter",
            )
            yield OptionList(id="model-options")
            yield Static(
                "Enter · this workspace\nCtrl+A · all workspaces · Esc cancel",
                id="model-menu-hint",
            )

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#model-filter", Input).focus()
        self.run_worker(self._discover_ollama())
        self.run_worker(self._discover_cloud())

    async def _discover_cloud(self) -> None:
        """Refresh API-listed models without holding up the picker."""
        import asyncio

        self._catalog = await asyncio.to_thread(discover_available_models)
        if self.is_mounted:
            self._populate(self.query_one("#model-filter", Input).value, keep_highlight=True)

    async def _discover_ollama(self) -> None:
        """Merge Ollama's models into the catalog, best-effort."""
        try:
            self._ollama = await discover_ollama_models()
        except Exception:  # daemon down or unreachable; leave Ollama out
            return
        if self._ollama and self.is_mounted:
            self._populate(self.query_one("#model-filter", Input).value, keep_highlight=True)

    def _sections(self, needle: str) -> list[tuple[str, list[ModelInfo]]]:
        """``(header, models)`` in display order: the short list, or with a search,
        every matching model by provider."""
        if not needle:
            models = [m for group in self._catalog.values() for m in group] + self._ollama
            return short_list(
                models, current=self._current, pinned=self._pinned, recent=self._recent
            )
        sections: list[tuple[str, list[ModelInfo]]] = []
        for provider, info in PROVIDERS.items():
            models = self._ollama if info.local else self._catalog.get(provider, [])
            if visible := [m for m in models if _matches(m, needle)]:
                sections.append(("Ollama" if info.local else info.label, visible))
        return sections

    def _populate(self, filter_text: str, *, keep_highlight: bool = False) -> None:
        """List the models matching ``filter_text``, highlighting the first.

        ``keep_highlight`` (a refresh when discovery returns) keeps the row the
        user had moved to, so Enter still picks it.
        """
        options = self.query_one("#model-options", OptionList)
        previous = options.highlighted_option.id if options.highlighted_option else None
        options.clear_options()
        raw = filter_text.strip()
        sections = self._sections(raw.lower())

        # Free-text row: any provider:model not already listed (ids must be unique).
        listed = {model.id for _, models in sections for model in models}
        first_enabled: int | None = None
        if ":" in raw and raw not in listed:
            options.add_option(Option(Text.assemble(("Use ", "bold"), (raw, "bold cyan")), id=raw))
            first_enabled = 0

        for header, models in sections:
            options.add_option(Option(Text(header, style="bold"), disabled=True))
            if first_enabled is None:
                first_enabled = options.option_count
            options.add_options(
                Option(_render_model(model, current=self._current), id=model.id) for model in models
            )

        highlight = first_enabled
        if keep_highlight and previous is not None:
            with contextlib.suppress(OptionDoesNotExist):
                highlight = options.get_option_index(previous)
        if highlight is not None:
            options.highlighted = highlight

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-filter":
            self._populate(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._confirm("workspace")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self._select(event.option.id, "workspace")

    def on_key(self, event) -> None:
        # While the filter input holds focus, ↑/↓ still drive the option list.
        if event.key in ("down", "up") and isinstance(self.focused, Input):
            options = self.query_one("#model-options", OptionList)
            if event.key == "down":
                options.action_cursor_down()
            else:
                options.action_cursor_up()
            event.stop()

    def action_select_global(self) -> None:
        self._confirm("global")

    def _confirm(self, scope: str) -> None:
        target = self._current_target()
        if target is not None:
            self._select(target, scope)

    def _current_target(self) -> str | None:
        """What Enter / Ctrl+A would pick: a typed provider:model, else the highlight."""
        raw = self.query_one("#model-filter", Input).value.strip()
        if ":" in raw:
            return raw
        options = self.query_one("#model-options", OptionList)
        if options.highlighted is None:
            return None
        return options.get_option_at_index(options.highlighted).id

    def _select(self, model_id: str, scope: str) -> None:
        self.dismiss((model_id, scope))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApiKeyModal(ModalScreen[str | None]):
    """Masked prompt for a provider API key. Dismisses with the key or ``None``."""

    DEFAULT_CSS = """
    ApiKeyModal {
        align: center middle;
    }
    #api-key-modal {
        width: 72;
        max-width: 90%;
        height: auto;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #api-key-title {
        text-style: bold;
    }
    #api-key-desc {
        color: $text-muted;
        padding: 1 0;
    }
    #api-key-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, *, env_var: str, provider_label: str) -> None:
        super().__init__()
        self._env_var = env_var
        self._provider_label = provider_label

    def compose(self):
        with Vertical(id="api-key-modal"):
            yield Static(f"{self._provider_label} needs an API key", id="api-key-title")
            yield Static(
                f"Enter {self._env_var}. It's saved for future runs and used right away.",
                id="api-key-desc",
            )
            yield Input(placeholder=self._env_var, password=True, id="api-key-input")
            yield Static("Enter to save · Esc to cancel", id="api-key-hint")

    def on_mount(self) -> None:
        self.query_one("#api-key-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


Puller = Callable[[str], AsyncIterator[PullProgress]]


class OllamaPullModal(ModalScreen[bool]):
    """Pull a local model. Dismisses ``True`` on success; Esc cancels or closes."""

    DEFAULT_CSS = """
    OllamaPullModal {
        align: center middle;
    }
    #ollama-pull {
        width: 72;
        max-width: 90%;
        height: auto;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #ollama-pull-title {
        text-style: bold;
        padding-bottom: 1;
    }
    #ollama-pull-status {
        color: $text-muted;
        padding-top: 1;
    }
    #ollama-pull-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, *, model_name: str, puller: Puller | None = None) -> None:
        super().__init__()
        self._model_name = model_name
        self._puller = puller
        self._worker = None

    def compose(self):
        with Vertical(id="ollama-pull"):
            yield Static(
                f"Pulling {self._model_name} from Ollama…", id="ollama-pull-title", markup=False
            )
            yield ProgressBar(total=100, show_eta=False, id="ollama-pull-bar")
            yield Static("Starting…", id="ollama-pull-status", markup=False)
            yield Static(
                "This can take a while for large models · Esc to cancel", id="ollama-pull-hint"
            )

    def on_mount(self) -> None:
        self._worker = self.run_worker(self._run(), exclusive=True)

    async def _run(self) -> None:
        from makker import ollama_client

        puller = self._puller or ollama_client.pull
        try:
            async for progress in puller(self._model_name):
                self._render_progress(progress)
            self.dismiss(True)
        except Exception as exc:  # connection refused, model not found, ...
            self.query_one("#ollama-pull-status", Static).update(f"Pull failed: {exc}")
            self.query_one("#ollama-pull-hint", Static).update("Esc to close")

    def _render_progress(self, progress: PullProgress) -> None:
        bar = self.query_one("#ollama-pull-bar", ProgressBar)
        status = self.query_one("#ollama-pull-status", Static)
        label = progress.status or "working"
        if progress.fraction is not None:
            bar.update(total=100, progress=progress.fraction * 100)
            label = f"{label} · {progress.fraction * 100:.0f}%"
        status.update(label)

    def action_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.dismiss(False)


__all__ = [
    "ApiKeyModal",
    "ModelChoice",
    "ModelMenu",
    "OllamaPullModal",
    "provider_of",
]
