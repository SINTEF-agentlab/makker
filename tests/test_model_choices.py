"""The model menu's short list, pins, and the saved choice."""

from __future__ import annotations

from pathlib import Path

from makker.model_choices import (
    NEWEST_PER_PROVIDER,
    load_pinned,
    missing_keys,
    remember_model,
    set_pinned,
    short_list,
)
from makker.models import ModelInfo
from makker.recent_models import load_recent_models


def _model(model_id: str, released: str | None = None, **kw) -> ModelInfo:
    return ModelInfo(model_id, model_id.partition(":")[2], released=released, **kw)


CATALOG = [
    _model("openai:gpt-7", "2026-12-01"),
    _model("openai:gpt-7-realtime", "2026-12-02"),
    _model("openai:gpt-6.5-2026-11-20", "2026-11-20"),
    _model("openai:gpt-6.5", "2026-11-19"),
    _model("openai:gpt-6", "2026-10-01"),
    _model("openai:gpt-5", "2026-01-01"),
    _model("openai:gpt-4", "2026-11-30", deprecated=True, note="deprecated"),
    _model("anthropic:claude-x", "2026-11-01"),
    _model("openrouter:vendor/model", "2026-12-03"),
    _model("ollama:qwen3.8:27b", local=True),
    _model("ollama:qwen3.6:27b", local=True, note="not pulled"),
    _model("ollama:glm-5.1:cloud"),
]


def _keyed(*providers: str):
    return lambda info: info.local or info.name in providers


def _ids(sections) -> list[tuple[str, list[str]]]:
    return [(title, [m.id for m in rows]) for title, rows in sections]


def test_short_list_takes_the_newest_general_models_of_each_keyed_provider() -> None:
    sections = _ids(short_list(CATALOG, keyed=_keyed("openai")))
    # Realtime, a dated snapshot and a deprecated model are skipped; OpenRouter is
    # never listed by this rule, and a provider without a key is left out.
    assert sections == [
        ("OpenAI", ["openai:gpt-7", "openai:gpt-6.5", "openai:gpt-6"][:NEWEST_PER_PROVIDER]),
        ("Local", ["ollama:qwen3.8:27b"]),
    ]


def test_pins_recents_and_the_current_model_come_first_and_only_once() -> None:
    sections = _ids(
        short_list(
            CATALOG,
            current="openai:gpt-6",
            pinned=["openrouter:vendor/model"],
            recent=["openai:gpt-6", "openai:gpt-7", "custom:typed-in"],
            keyed=_keyed("openai"),
        )
    )
    assert sections[0] == ("Pinned", ["openrouter:vendor/model"])
    assert sections[1] == ("Recent", ["openai:gpt-6", "openai:gpt-7", "custom:typed-in"])
    # The newest three are each listed once: two already are, under Recent.
    assert sections[2] == ("OpenAI", ["openai:gpt-6.5"])


def test_a_recent_model_missing_from_the_catalog_keeps_its_id() -> None:
    (_, rows), *_ = short_list([], recent=["custom:typed-in"], keyed=_keyed())
    assert rows[0].id == rows[0].label == "custom:typed-in"


def test_providers_without_a_key_are_offered_as_add_key_rows() -> None:
    names = [p.name for p in missing_keys(_keyed("openai"))]
    assert names == ["anthropic", "google_genai", "openrouter"]


def test_pins_are_saved_in_the_user_config_in_order() -> None:
    assert load_pinned() == []
    set_pinned("openai:gpt-7", True)
    set_pinned("anthropic:claude-x", True)
    assert load_pinned() == ["openai:gpt-7", "anthropic:claude-x"]
    assert set_pinned("openai:gpt-7", False) == ["anthropic:claude-x"]
    assert load_pinned() == ["anthropic:claude-x"]


def test_remember_model_records_it_and_saves_it_where_asked(tmp_path: Path) -> None:
    from makker.user_config import load_user_config
    from makker.workspace import load_workspace_config

    assert remember_model("openai:gpt-7", workspace=tmp_path) == "this workspace"
    assert load_workspace_config(tmp_path).model == "openai:gpt-7"
    assert remember_model("openai:gpt-6", scope="global", workspace=tmp_path) == "user config"
    assert load_user_config().model == "openai:gpt-6"
    assert load_workspace_config(tmp_path).model == "openai:gpt-7"
    assert load_recent_models()[:2] == ["openai:gpt-6", "openai:gpt-7"]
