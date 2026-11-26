"""Tests for the Ollama client wrapper (no real server required)."""

from __future__ import annotations

import pytest

from makker import ollama_client
from makker.ollama_client import _matches, host, model_name


class _Model:
    def __init__(self, name: str) -> None:
        self.model = name


class _ListResp:
    def __init__(self, names: list[str]) -> None:
        self.models = [_Model(n) for n in names]


class _ShowResp:
    def __init__(self, capabilities: list[str]) -> None:
        self.capabilities = capabilities


def _fake_client(*, names=None, list_error=None, pull_chunks=None, caps=None, show_error=None):
    class _Chunk:
        def __init__(self, status, total=None, completed=None):
            self.status, self.total, self.completed = status, total, completed

    class _Client:
        async def list(self):
            if list_error is not None:
                raise list_error
            return _ListResp(names or [])

        async def pull(self, name, stream=False):
            async def _gen():
                for status, total, completed in pull_chunks or []:
                    yield _Chunk(status, total, completed)

            return _gen()

        async def show(self, name):
            if show_error is not None:
                raise show_error
            return _ShowResp(caps or [])

    return _Client()


def test_model_name_strips_provider_prefix() -> None:
    assert model_name("ollama:llama3.1") == "llama3.1"
    assert model_name("llama3.1") == "llama3.1"


def test_matches_handles_latest_tag() -> None:
    assert _matches("llama3.1:latest", "llama3.1")
    assert _matches("llama3.1", "llama3.1")
    assert not _matches("qwen2.5:latest", "llama3.1")


def test_host_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert host() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234")
    assert host() == "http://box:1234"


async def test_is_reachable_true_and_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_client, "_client", lambda: _fake_client(names=[]))
    assert await ollama_client.is_reachable() is True

    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(list_error=ConnectionError("down"))
    )
    assert await ollama_client.is_reachable() is False


async def test_installed_models_and_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client,
        "_client",
        lambda: _fake_client(names=["llama3.1:latest", "qwen2.5:latest"]),
    )
    assert set(await ollama_client.installed_models()) == {"llama3.1:latest", "qwen2.5:latest"}
    assert await ollama_client.is_installed("llama3.1") is True
    assert await ollama_client.is_installed("mistral") is False


async def test_supports_tools_from_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(caps=["completion", "tools", "thinking"])
    )
    assert await ollama_client.capabilities("qwen3:14b") == ["completion", "tools", "thinking"]
    assert await ollama_client.supports_tools("qwen3:14b") is True


async def test_supports_tools_false_when_capability_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # A daemon too old for the model's template reports a reduced set (no tools).
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(caps=["completion", "vision"])
    )
    assert await ollama_client.supports_tools("qwen3.6:27b") is False


async def test_capabilities_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(show_error=ConnectionError("down"))
    )
    assert await ollama_client.capabilities("x") == []
    assert await ollama_client.supports_tools("x") is False


def test_context_window_reads_max_context_length(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Info:
        def __init__(self) -> None:
            self.modelinfo = {"qwen35.context_length": 262144, "general.parameter_count": 27}

    class _SyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        def show(self, name):
            return _Info()

    monkeypatch.setattr("ollama.Client", _SyncClient)
    assert ollama_client.context_window("qwen3.6:27b") == 262144


def test_context_window_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        def show(self, name):
            raise ConnectionError("down")

    monkeypatch.setattr("ollama.Client", _SyncClient)
    assert ollama_client.context_window("x") is None


def _local_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    maximum=262_144,
    configured=None,
    digest="d1",
    version="0.40.18",
):
    """A pulled model the daemon reports, with ``_load`` left for the test to set."""
    monkeypatch.setattr(ollama_client, "context_window", lambda name: maximum)
    monkeypatch.setattr(ollama_client, "_modelfile_num_ctx", lambda name: configured)
    monkeypatch.setattr(ollama_client, "_digest", lambda name: digest)
    monkeypatch.setattr(ollama_client, "version", lambda: version)


def _gpu_holding(largest: int, loads: list[int]):
    """A fake ``_load`` on a GPU that holds the model up to ``largest`` tokens of context."""

    def load(name: str, num_ctx: int) -> ollama_client._Load:
        loads.append(num_ctx)
        size = 16 * 2**30 + num_ctx * 9_000
        return ollama_client._Load(num_ctx, size, size if num_ctx <= largest else size // 2)

    return load


def test_num_ctx_uses_the_models_own_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_model(monkeypatch, configured=65_536)
    monkeypatch.setattr(ollama_client, "_load", lambda *a: pytest.fail("loaded a model"))
    assert ollama_client.num_ctx("qwen3.8:27b-64k") == 65_536
    # A setting beyond what the model supports is loaded at the model's maximum.
    _local_model(monkeypatch, maximum=40_960, configured=65_536)
    assert ollama_client.num_ctx("qwen3:0.6b") == 40_960


def test_num_ctx_is_the_maximum_when_it_fits_and_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_model(monkeypatch)
    loads: list[int] = []
    monkeypatch.setattr(ollama_client, "_load", _gpu_holding(10**9, loads))
    assert ollama_client.num_ctx("qwen3.8:27b") == 262_144
    assert ollama_client.num_ctx("qwen3.8:27b") == 262_144
    assert loads == [262_144]  # one load, then the remembered answer
    # A new version of the model is measured again, and so is a model after an
    # Ollama upgrade, which can change what a load takes.
    _local_model(monkeypatch, digest="d2")
    ollama_client.num_ctx("qwen3.8:27b")
    _local_model(monkeypatch, digest="d2", version="0.41.0")
    ollama_client.num_ctx("qwen3.8:27b")
    assert loads == [262_144] * 3


def test_every_context_size_is_measured_not_estimated(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_model(monkeypatch)
    loads: list[int] = []
    monkeypatch.setattr(ollama_client, "_load", _gpu_holding(111_000, loads))
    sizes = (32_768, 65_536, 131_072, 262_144)
    found = ollama_client.calibration("qwen3.8:27b", sizes=sizes)
    assert found is not None and found.fits <= 111_000
    # Each size has what its own load took, the part on the GPU included.
    total = 16 * 2**30 + 65_536 * 9_000
    assert found.loads[65_536] == (total, total)
    spilled = 16 * 2**30 + 131_072 * 9_000
    assert found.loads[131_072] == (spilled, spilled // 2)
    assert all(size in loads for size in sizes)
    # Remembered, so asking again (even without measuring) loads nothing.
    monkeypatch.setattr(ollama_client, "_load", lambda *a: pytest.fail("loaded a model"))
    assert ollama_client.calibration("qwen3.8:27b", sizes=sizes) == found
    assert ollama_client.calibration("qwen3.8:27b", measure=False) == found


def test_a_load_ollama_refuses_is_noted_as_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_model(monkeypatch, maximum=1_048_576)
    fits = _gpu_holding(200_000, [])
    monkeypatch.setattr(
        ollama_client, "_load", lambda name, n: None if n > 500_000 else fits(name, n)
    )
    found = ollama_client.calibration("long-model", sizes=(65_536, 1_048_576))
    assert found is not None and found.loads[1_048_576] is None
    assert found.loads[65_536] is not None
    # Not measured at all: nothing to report, and nothing loaded.
    _local_model(monkeypatch, digest="d2")
    assert ollama_client.num_ctx("long-model", measure=False) is None


def test_thinking_levels_read_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    replies = {
        "qwen3.8:27b": {"values": [False, "low", "medium", "xhigh"], "default": "medium"},
        "qwen3:0.6b": {"values": [False, True], "default": True},
        "qwen3-coder:30b": {"values": [False], "default": False},
    }

    def post(url, json, timeout):
        return httpx.Response(200, json={"thinking": replies[json["model"]]})

    monkeypatch.setattr(httpx, "post", post)
    assert ollama_client.thinking_levels("qwen3.8:27b") == (
        ["none", "low", "medium", "xhigh"],
        "medium",
    )
    # Thinking that can only be on or off offers no levels.
    assert ollama_client.thinking_levels("qwen3:0.6b") == ([], None)
    assert ollama_client.thinking_levels("qwen3-coder:30b") == ([], None)


def test_num_ctx_finds_the_largest_context_that_stays_on_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_model(monkeypatch)
    loads: list[int] = []
    monkeypatch.setattr(ollama_client, "_load", _gpu_holding(111_000, loads))
    chosen = ollama_client.num_ctx("qwen3.8:27b")
    # Verified on the GPU, and within an eighth of the largest that is.
    assert chosen <= 111_000 < chosen * 9 / 8
    assert loads[:2] == [262_144, ollama_client.MIN_NUM_CTX]
    assert len(loads) <= 2 + ollama_client._MAX_PROBES


def test_num_ctx_floor_when_the_model_spills_even_when_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_model(monkeypatch)
    loads: list[int] = []
    monkeypatch.setattr(ollama_client, "_load", _gpu_holding(8_192, loads))
    assert ollama_client.num_ctx("big-model") == ollama_client.MIN_NUM_CTX
    assert loads == [262_144, ollama_client.MIN_NUM_CTX]


def test_num_ctx_treats_a_refused_load_as_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama refuses a load larger than the machine's memory; the search goes lower."""
    _local_model(monkeypatch, maximum=1_048_576)
    loads: list[int] = []
    fits = _gpu_holding(200_000, loads)
    monkeypatch.setattr(
        ollama_client, "_load", lambda name, n: None if n > 500_000 else fits(name, n)
    )
    assert ollama_client.num_ctx("long-model") <= 200_000


def test_num_ctx_without_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_model(monkeypatch, maximum=None)
    assert ollama_client.num_ctx("x") is None
    # A daemon that stops answering mid-way gives the floor, not remembered.
    _local_model(monkeypatch)

    def gone(name: str, num_ctx: int):
        raise ConnectionError("down")

    monkeypatch.setattr(ollama_client, "_load", gone)
    assert ollama_client.num_ctx("x") == ollama_client.MIN_NUM_CTX
    assert ollama_client._measured() == {}


def test_a_model_is_not_measured_again_for_every_agent_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every agent build asks for the size, and measuring loads the model. A
    daemon that gives no version leaves nothing to store the answer under, and
    one that stops answering leaves no answer: neither is asked again and again."""
    _local_model(monkeypatch, version=None)
    loads: list[int] = []
    monkeypatch.setattr(ollama_client, "_load", _gpu_holding(10**9, loads))
    assert ollama_client.num_ctx("qwen3.8:27b") == 262_144
    assert ollama_client.num_ctx("qwen3.8:27b") == 262_144
    assert loads == [262_144]
    assert ollama_client._measured() == {}  # known to this process only

    _local_model(monkeypatch)
    tries: list[int] = []

    def gone(name: str, num_ctx: int):
        tries.append(num_ctx)
        raise ConnectionError("down")

    monkeypatch.setattr(ollama_client, "_load", gone)
    assert ollama_client.num_ctx("other") == ollama_client.MIN_NUM_CTX
    assert ollama_client.num_ctx("other") == ollama_client.MIN_NUM_CTX
    assert len(tries) == 1
    # Asking for the sizes (the settings menu) is a reason to try again.
    ollama_client.calibration("other", sizes=(32_768,))
    assert len(tries) == 2


def test_load_and_modelfile_read_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    class _SyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        def show(self, name):
            return SimpleNamespace(parameters='stop "<|im_end|>"\nnum_ctx                65536')

        def generate(self, name, prompt, options):
            self.num_ctx = options["num_ctx"]

        def ps(self):
            loaded = SimpleNamespace(
                model="qwen3:0.6b", context_length=self.num_ctx, size=5, size_vram=5
            )
            return SimpleNamespace(models=[loaded])

    monkeypatch.setattr("ollama.Client", _SyncClient)
    assert ollama_client._modelfile_num_ctx("qwen3.8:27b-64k") == 65_536
    assert ollama_client._load("qwen3:0.6b", 40_960) == ollama_client._Load(40_960, 5, 5)


async def test_installed_models_empty_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client, "_client", lambda: _fake_client(list_error=ConnectionError("down"))
    )
    assert await ollama_client.installed_models() == []
    assert await ollama_client.is_installed("llama3.1") is False


async def test_pull_yields_progress_with_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama_client,
        "_client",
        lambda: _fake_client(pull_chunks=[("pulling", 100, 50), ("success", None, None)]),
    )
    progress = [p async for p in ollama_client.pull("llama3.1")]
    assert progress[0].status == "pulling"
    assert progress[0].fraction == 0.5
    assert progress[1].status == "success"
    assert progress[1].fraction is None
