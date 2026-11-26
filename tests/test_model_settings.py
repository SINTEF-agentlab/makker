"""Per-model settings: saved choices, what each model offers, and the model build."""

from __future__ import annotations

import pytest

from makker import model_settings, ollama_client
from makker.model_settings import ModelSettings, load_settings, options, save_settings

PROFILES = {
    "openai:gpt-test": {
        "reasoning_effort_levels": ["none", "low", "medium", "high"],
        "max_output_tokens": 128_000,
    },
    "anthropic:claude-test": {
        "reasoning_effort_levels": ["low", "medium", "high"],
        "reasoning_effort_default": "high",
        "max_output_tokens": 64_000,
    },
}


@pytest.fixture(autouse=True)
def _profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker import models

    monkeypatch.setattr(models, "bundled_profile", lambda model_id: PROFILES.get(model_id, {}))


def test_settings_are_saved_per_model_and_defaults_are_dropped() -> None:
    from makker.user_config import load_user_config

    assert load_settings("openai:gpt-test") == ModelSettings()
    save_settings("openai:gpt-test", ModelSettings(effort="high", output=32_768))
    save_settings("ollama:qwen3.8:27b", ModelSettings(context=131_072))
    assert load_settings("openai:gpt-test") == ModelSettings(effort="high", output=32_768)
    assert load_settings("ollama:qwen3.8:27b").context == 131_072
    # Back to all defaults: the model's entry goes, the others stay.
    save_settings("openai:gpt-test", ModelSettings())
    assert set(load_user_config().model_settings) == {"ollama:qwen3.8:27b"}


def test_effort_offers_the_models_levels_and_its_default() -> None:
    openai = options("openai:gpt-test")["effort"]
    # No default in the data: medium, which the build asks for.
    assert openai == {
        "levels": ["none", "low", "medium", "high"],
        "default": "medium",
        "current": "medium",
    }
    save_settings("anthropic:claude-test", ModelSettings(effort="low"))
    assert options("anthropic:claude-test")["effort"]["current"] == "low"
    # A level the model no longer offers falls back to its default.
    save_settings("anthropic:claude-test", ModelSettings(effort="max"))
    assert options("anthropic:claude-test")["effort"]["current"] == "high"
    assert options("openrouter:vendor/model")["effort"] is None


def test_a_hosted_model_offers_only_context_sizes_its_reply_fits_in(monkeypatch) -> None:
    from makker import models
    from makker.model_settings import MIN_CONTEXT

    monkeypatch.setattr(models, "context_window", lambda model_id: 1_000_000)
    # Anthropic models keep 24K free for the reply: 32K would leave no room to work in.
    anthropic = options("anthropic:claude-test")["context"]
    assert anthropic["min"] == 39_923
    assert [c["tokens"] for c in anthropic["choices"]][:2] == [65_536, 131_072]
    openai = options("openai:gpt-test")["context"]
    assert openai["min"] == MIN_CONTEXT
    assert openai["choices"][0]["tokens"] == 32_768
    # An output length chosen for a model is room it keeps free as well.
    save_settings("openai:gpt-test", ModelSettings(output=32_768))
    assert options("openai:gpt-test")["context"]["choices"][0]["tokens"] == 65_536


def test_output_limits_stop_at_the_models_maximum() -> None:
    output = options("anthropic:claude-test")["output"]
    assert output == {"choices": [8_192, 16_384, 32_768, 64_000], "default": 24_000}
    assert options("openai:gpt-test")["output"]["default"] == 128_000
    assert options("openrouter:vendor/model")["output"] is None


def test_a_local_model_offers_context_sizes_with_their_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama_client, "thinking_levels", lambda name: ([], None))
    monkeypatch.setattr(ollama_client, "context_window", lambda name: 262_144)
    monkeypatch.setattr(ollama_client, "num_ctx", lambda name, **kw: 131_072)
    found = ollama_client.Calibration(
        131_072,
        {
            32_768: (16 * 10**9, 16 * 10**9),
            65_536: (16 * 10**9 + 3 * 10**8, 16 * 10**9 + 3 * 10**8),
            131_072: (17 * 10**9, 17 * 10**9),
            262_144: (30 * 10**9, 9 * 10**9),
        },
    )
    monkeypatch.setattr(ollama_client, "calibration", lambda name, **kw: found)
    context = options("ollama:qwen3.8:27b")["context"]
    assert [c["tokens"] for c in context["choices"]] == [32_768, 65_536, 131_072, 262_144]
    # Each size shows what its own load took, nothing in between estimated.
    assert context["choices"][1]["memory"] == 16 * 10**9 + 3 * 10**8
    # Past what stays on the GPU, part of the model runs on the CPU, and the
    # figure is what that load took in all.
    assert [c["on_gpu"] for c in context["choices"]] == [True, True, True, False]
    assert context["choices"][3]["memory"] == 30 * 10**9
    assert context["auto"] == 131_072 and context["measured"]
    # A hosted model's sizes carry no memory: they only cap the window.
    hosted = options("ollama:glm-5.1:cloud")["context"]
    assert hosted["local"] is False and hosted["auto"] == 262_144


def test_a_chosen_context_is_what_the_local_model_loads_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makker.models import local_context

    monkeypatch.setattr(ollama_client, "context_window", lambda name: 262_144)
    monkeypatch.setattr(ollama_client, "num_ctx", lambda name, **kw: 131_072)
    assert local_context("ollama:qwen3.8:27b") == 131_072
    save_settings("ollama:qwen3.8:27b", ModelSettings(context=65_536))
    assert local_context("ollama:qwen3.8:27b") == 65_536
    # Never past the model's maximum.
    save_settings("ollama:qwen3.8:27b", ModelSettings(context=10**6))
    assert local_context("ollama:qwen3.8:27b") == 262_144


@pytest.mark.parametrize(
    ("model_id", "chosen", "expected"),
    [
        ("openai:gpt-test", "high", {"reasoning": {"effort": "high", "summary": "auto"}}),
        ("openai:gpt-test", "none", {"reasoning": {"effort": "none"}}),
        ("anthropic:claude-test", "low", {"output_config": {"effort": "low"}}),
        ("google_genai:gemini-test", "low", {"reasoning_effort": "low"}),
    ],
)
def test_the_build_asks_for_the_chosen_effort(
    monkeypatch: pytest.MonkeyPatch, model_id: str, chosen: str, expected: dict
) -> None:
    from makker.agent import builder

    monkeypatch.setattr(
        builder, "model_profile", lambda m: {"reasoning_output": True, "temperature": False}
    )
    monkeypatch.setattr(model_settings, "effort_levels", lambda m: (["none", "low", "high"], "low"))
    save_settings(model_id, ModelSettings(effort=chosen))
    settings = builder._MODEL_SETTINGS[model_id.partition(":")[0]](model_id)
    assert expected.items() <= settings.items()


def test_the_build_asks_for_the_chosen_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker.agent import builder

    monkeypatch.setattr(
        builder, "model_profile", lambda m: {"reasoning_output": True, "temperature": True}
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    save_settings("anthropic:claude-test", ModelSettings(output=16_384))
    model = builder._resolve_model_for_agent("anthropic:claude-test")
    assert model.max_tokens == 16_384
    # The thinking budget stays under the limit it spends from.
    assert model.thinking == {"type": "enabled", "budget_tokens": 8_192}


def test_a_local_model_thinks_at_the_chosen_level(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker.agent.builder import _ollama_settings

    monkeypatch.setattr(ollama_client, "num_ctx", lambda name, **kw: 65_536)
    monkeypatch.setattr(ollama_client, "thinks", lambda name: True)
    levels = (["none", "low", "medium", "xhigh"], "medium")
    monkeypatch.setattr(ollama_client, "thinking_levels", lambda name: levels)
    assert _ollama_settings("ollama:qwen3.8:27b")["reasoning"] is True
    save_settings("ollama:qwen3.8:27b", ModelSettings(effort="xhigh"))
    assert _ollama_settings("ollama:qwen3.8:27b")["reasoning"] == "xhigh"
    save_settings("ollama:qwen3.8:27b", ModelSettings(effort="none"))
    assert _ollama_settings("ollama:qwen3.8:27b")["reasoning"] is False


def test_a_chosen_context_caps_a_cloud_models_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from makker import models
    from makker.context_budget import context_budget

    monkeypatch.setattr(models, "context_window", lambda model_id: 1_050_000)
    assert context_budget("openai:gpt-test").window == 1_050_000
    save_settings("openai:gpt-test", ModelSettings(context=65_536))
    budget = context_budget("openai:gpt-test")
    # Compaction runs against the chosen size, well before the model's own window.
    assert budget.window == 65_536 and budget.compact_at < 65_536
    context = options("openai:gpt-test")["context"]
    assert context["local"] is False and context["max"] == 1_050_000
    assert [c["tokens"] for c in context["choices"]][-2:] == [524_288, 1_050_000]


def test_a_context_below_what_the_agent_needs_is_not_saved() -> None:
    from makker.model_settings import MIN_CONTEXT, settings_from

    assert settings_from({"context": MIN_CONTEXT - 1}).context is None
    assert settings_from({"context": 100_000}).context == 100_000
