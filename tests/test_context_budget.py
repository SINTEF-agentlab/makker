"""The context budget: one window per model, where it came from, and when compaction runs."""

from __future__ import annotations

import pytest

from makker import models, ollama_client
from makker.context_budget import ASSUMED_WINDOW, ContextBudget, context_budget


def test_window_comes_from_the_provider_data_without_building_the_model(monkeypatch) -> None:
    def no_build(model_id: str) -> dict:
        raise AssertionError(f"built {model_id} to read its window")

    monkeypatch.setattr(models, "model_profile", no_build)
    name, profile = next(
        (n, p)
        for n, p in models._provider_profiles("anthropic").items()
        if models._is_chat_capable(p)
    )
    budget = context_budget(f"anthropic:{name}")
    assert budget == ContextBudget(window=profile["max_input_tokens"], source="provider")


def test_an_unknown_model_gets_the_assumed_window(monkeypatch) -> None:
    monkeypatch.setattr(models, "model_profile", lambda model_id: {})
    monkeypatch.setattr(models, "_google_context_window", lambda name: None)
    for model_id in ("openai:gpt-from-the-future", "google_genai:gemini-from-the-future"):
        assert context_budget(model_id) == ContextBudget(window=ASSUMED_WINDOW, source="assumed")


def test_ollama_cloud_runs_at_its_full_window(monkeypatch) -> None:
    """A hosted model has no local memory limit: no 64k cap, no context size sent."""
    from makker.agent.builder import _ollama_settings

    monkeypatch.setattr(ollama_client, "context_window", lambda name: 262_144)
    monkeypatch.setattr(ollama_client, "num_ctx", lambda name, **kw: 65_536)
    monkeypatch.setattr(ollama_client, "thinks", lambda name: False)
    assert context_budget("ollama:kimi-k2.6:cloud") == ContextBudget(262_144, "ollama")
    assert "num_ctx" not in _ollama_settings("ollama:kimi-k2.6:cloud")
    # A local model is loaded at the size measured for this machine, and the budget agrees.
    assert context_budget("ollama:qwen3.8:27b").window == 65_536
    assert _ollama_settings("ollama:qwen3.8:27b")["num_ctx"] == 65_536


def test_a_model_instance_answers_from_its_own_profile() -> None:
    class _Model:
        def __init__(self) -> None:
            self.profile = {"max_input_tokens": 50_000}
            self.max_tokens = 4_000

    assert context_budget(None, _Model()) == ContextBudget(50_000, "provider", output_reserve=4_000)


@pytest.mark.parametrize(
    ("budget", "compact_at"),
    [
        # 85% of the window...
        (ContextBudget(200_000, "provider"), 170_000),
        # ...or earlier, when the model's output allowance would not fit in 95% of it.
        (ContextBudget(200_000, "provider", output_reserve=24_000), 166_000),
        (ContextBudget(1_000_000, "provider", output_reserve=128_000), 822_000),
    ],
)
def test_compact_at(budget: ContextBudget, compact_at: int) -> None:
    assert budget.compact_at == compact_at


async def test_building_an_agent_records_the_model_and_sizes_the_summarizer(tmp_path) -> None:
    """A model with no known window runs on the assumed one, everywhere at once."""
    from makker.agent.builder import build_agent
    from makker.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
    from makker.session import Session

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    model = make_scripted_model([scripted_final("hi")])
    build_agent(session, model=model)

    assert session.context_budget == ContextBudget(ASSUMED_WINDOW, "assumed")
    assert model.profile["max_input_tokens"] == ASSUMED_WINDOW  # what the summarizer reads
    (event,) = [e.payload for e in session.trace.iter_events() if e.kind == "model"]
    assert event["window"] == ASSUMED_WINDOW and event["window_source"] == "assumed"
    assert event["compact_at"] == int(ASSUMED_WINDOW * 0.85)
    session.finalize()


async def test_an_assumed_window_that_is_too_large_recovers_by_compacting(tmp_path) -> None:
    """The provider rejects a request longer than its real window; the summarizer
    then compacts and retries, so the turn carries on and the user is told."""
    from langchain_core.exceptions import ContextOverflowError
    from langchain_core.messages.utils import count_tokens_approximately
    from langgraph.checkpoint.memory import InMemorySaver

    from makker.agent.builder import build_agent
    from makker.agent.turns import TurnCompaction, TurnRunner
    from makker.lab.fakes import FakeJulia, ScriptedChatModel, make_fake_adapter, scripted_final
    from makker.session import Session

    real_window = 60_000  # tokens; the session assumes 200k

    class SmallWindowModel(ScriptedChatModel):
        rejected: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if count_tokens_approximately(messages) > real_window:
                self.rejected += 1
                raise ContextOverflowError("prompt is too long")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    model = SmallWindowModel(
        responses=[scripted_final(f"reply {i}: " + "detail " * 4_000) for i in range(40)]
    )
    agent, _ = build_agent(session, model=model, checkpointer=InMemorySaver())
    assert session.context_budget.source == "assumed"
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)

    notes: list[TurnCompaction] = []
    for i in range(20):
        await runner.run_prompt(
            f"question {i}",
            on_message=lambda e: notes.append(e) if isinstance(e, TurnCompaction) else None,
        )
        if notes:
            break
    assert model.rejected >= 1, "the conversation never outgrew the real window"
    assert len(notes) == 1 and notes[0].payload["manual"] is False
    # The turn that hit the limit still produced its reply, and the next one works too.
    replies = [e for e in session.trace.iter_events() if e.kind == "message_assistant"]
    assert len(replies) == i + 1
    await runner.run_prompt("one more")
    session.finalize()
