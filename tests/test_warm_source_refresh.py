"""The in-env Makker runtime auto-refreshes when the install's source changes.

Warm packages are dev-pathed copies laid down at bootstrap, so without this an
existing managed env keeps a stale Makker after a Makker update.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makker import workspace
from makker.simulators import env_setup
from makker.simulators.base import SimulatorAdapter
from makker.workspace import (
    PRECOMPILE_MARKER,
    WARM_SOURCE_MARKER,
    mark_warm_source,
    recopy_warm_sources,
    warm_source_fingerprint,
    warm_source_is_current,
)


@pytest.fixture
def shared_pkg(tmp_path: Path, monkeypatch) -> Path:
    """A fake shared Makker source the fingerprint reads from."""
    pkg = tmp_path / "shared" / "Makker"
    (pkg / "src").mkdir(parents=True)
    (pkg / "Project.toml").write_text('name = "Makker"\n', encoding="utf-8")
    (pkg / "src" / "ensemble.jl").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "shared_julia_package_path", lambda: pkg)
    return pkg


@pytest.fixture
def template(tmp_path: Path) -> Path:
    """A template env with a per-simulator warm package and the [sources] table."""
    tpl = tmp_path / "template"
    (tpl / "MakkerSim" / "src").mkdir(parents=True)
    (tpl / "MakkerSim" / "Project.toml").write_text('name = "MakkerSim"\n', encoding="utf-8")
    (tpl / "MakkerSim" / "src" / "warm.jl").write_text("# warm v1\n", encoding="utf-8")
    (tpl / "Project.toml").write_text(
        '[deps]\n\n[sources]\nMakker = {path = "Makker"}\nMakkerSim = {path = "MakkerSim"}\n',
        encoding="utf-8",
    )
    return tpl


def _make_env(tmp_path: Path, template: Path, shared_pkg: Path) -> Path:
    """An env bootstrapped from the template (warm sources copied + fingerprinted)."""
    env = tmp_path / "env"
    env.mkdir()
    (env / "Project.toml").write_text(
        template.joinpath("Project.toml").read_text(), encoding="utf-8"
    )
    workspace._copy_source_packages(template, env, {"MakkerSim": {"path": "MakkerSim"}})
    workspace.sync_shared_julia_package(env)
    workspace.mark_warm_source(env, template)
    (env / PRECOMPILE_MARKER).touch()
    return env


def test_fresh_env_is_current(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    assert warm_source_is_current(env, template) is True


def test_fingerprint_tracks_shared_and_template_sources(tmp_path, template, shared_pkg):
    before = warm_source_fingerprint(template)
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    assert warm_source_fingerprint(template) != before
    after_shared = warm_source_fingerprint(template)
    (template / "MakkerSim" / "src" / "warm.jl").write_text("# warm v2\n", encoding="utf-8")
    assert warm_source_fingerprint(template) != after_shared


def test_recopy_then_mark_is_the_refresh_cycle(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    # The install's shared source changes (a Makker update).
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    assert warm_source_is_current(env, template) is False

    assert recopy_warm_sources(env, template) is True
    # New source copied into the env and the precompile marker dropped (re-bake).
    assert (env / "Makker" / "src" / "ensemble.jl").read_text() == "# v2 changed\n"
    assert not (env / PRECOMPILE_MARKER).exists()
    # recopy does NOT mark current — the caller marks only after a successful resolve,
    # so a failed resolve leaves it stale and the next launch retries.
    assert warm_source_is_current(env, template) is False

    mark_warm_source(env, template)
    assert warm_source_is_current(env, template) is True


def test_never_fingerprinted_env_is_stale(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    (env / WARM_SOURCE_MARKER).unlink()  # an env from before this feature existed
    assert warm_source_is_current(env, template) is False


def test_recopy_skips_user_owned_env(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    (env / "Project.toml").write_text("[deps]\n", encoding="utf-8")  # no [sources]
    assert recopy_warm_sources(env, template) is False


def test_recopy_ignores_paths_the_env_names_itself(tmp_path, template, shared_pkg):
    # The env's Project.toml sits in the workspace, so its paths are not trusted:
    # "../.." once resolved to the workspace, which the re-copy then removed.
    ws = tmp_path / "ws"
    env = ws / ".makker" / "julia-env"
    env.mkdir(parents=True)
    (ws / "precious.txt").write_text("the user's work\n", encoding="utf-8")
    (env / "Project.toml").write_text(
        '[sources]\nMakkerSim = {path = "../.."}\nOther = {path = "../.."}\n', encoding="utf-8"
    )

    assert recopy_warm_sources(env, template) is True

    assert (ws / "precious.txt").read_text() == "the user's work\n"
    # The package the env declares is still refreshed, at the template's own path.
    assert (env / "MakkerSim" / "src" / "warm.jl").read_text() == "# warm v1\n"


def _adapter(template: Path, monkeypatch: pytest.MonkeyPatch) -> SimulatorAdapter:
    monkeypatch.setattr(
        SimulatorAdapter, "julia_env_template_path", property(lambda _self: template)
    )
    return SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=template.parent,
        package_imports=("MakkerSim",),
        primary_package="MakkerSim",
        domain_hints="",
    )


def _declare_and_resolve(env: Path) -> None:
    """Declare the warm packages, resolve them, then mark the bake done."""
    project = (env / "Project.toml").read_text(encoding="utf-8")
    (env / "Project.toml").write_text(
        project.replace("[deps]\n", '[deps]\nMakker = "u1"\nMakkerSim = "u2"\n'),
        encoding="utf-8",
    )
    (env / "Manifest.toml").write_text(
        '[[deps.Makker]]\npath = "Makker"\n\n[[deps.MakkerSim]]\npath = "MakkerSim"\n',
        encoding="utf-8",
    )
    (env / PRECOMPILE_MARKER).touch()


def test_pending_preparation_names_what_a_launch_would_still_do(
    tmp_path: Path, template: Path, shared_pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An update asks this to do the launch's work itself; "up to date" used to leave
    # a new Makker runtime and its minutes-long bake to the first session.
    adapter = _adapter(template, monkeypatch)
    env = _make_env(tmp_path, template, shared_pkg)
    _declare_and_resolve(env)
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) is None

    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) == "packages"

    recopy_warm_sources(env, template)
    mark_warm_source(env, template)
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) == "bake"
    (env / PRECOMPILE_MARKER).touch()
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) is None

    # A warm package on record as failing is tried again once what it failed
    # against has changed (Julia, here), as a launch would.
    (env / workspace.WARM_FAILURE_MARKER).write_text("older attempt", encoding="utf-8")
    monkeypatch.setattr(env_setup, "warm_failure_is_current", lambda _env: False)
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) == "bake"
    monkeypatch.setattr(env_setup, "warm_failure_is_current", lambda _env: True)
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) is None

    (env / "Manifest.toml").write_text("", encoding="utf-8")  # the simulator unresolved
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) == "packages"


def test_pending_preparation_leaves_an_env_without_warm_packages_alone(
    tmp_path: Path, template: Path, shared_pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(template, monkeypatch)
    env = _make_env(tmp_path, template, shared_pkg)
    (env / WARM_SOURCE_MARKER).unlink()
    assert env_setup.pending_env_preparation(adapter, tmp_path, env) is None


def test_an_update_copies_the_new_runtime_before_its_one_bake(
    tmp_path: Path, template: Path, shared_pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Left to the first launch, the copy would make it bake the whole env again.
    adapter = _adapter(template, monkeypatch)
    ws = tmp_path / "ws"
    env = ws / ".makker" / "julia-env"
    env.parent.mkdir(parents=True)
    _make_env(tmp_path, template, shared_pkg).rename(env)
    _declare_and_resolve(env)
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    baked_with: list[str] = []
    monkeypatch.setattr(
        env_setup,
        "update_env",
        lambda project: baked_with.append(
            (project / "Makker" / "src" / "ensemble.jl").read_text(encoding="utf-8")
        ),
    )

    env_setup.upgrade_env(adapter, ws, env, "test")

    assert baked_with == ["# v2 changed\n"]
    assert warm_source_is_current(env, template) is True


def test_failed_refresh_restores_previous_sources_and_markers(
    tmp_path: Path, template: Path, shared_pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _make_env(tmp_path, template, shared_pkg)
    (env / "Manifest.toml").write_text("old manifest", encoding="utf-8")
    old_marker = (env / WARM_SOURCE_MARKER).read_text(encoding="utf-8")
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")

    def failed_resolve(project: Path, **_kwargs) -> None:
        (project / "Manifest.toml").write_text("partial manifest", encoding="utf-8")
        raise env_setup.EnvSetupError("resolve failed")

    monkeypatch.setattr(env_setup, "resolve_and_instantiate", failed_resolve)
    adapter = SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=template.parent,
        package_imports=("MakkerSim",),
        primary_package="MakkerSim",
        domain_hints="",
    )
    monkeypatch.setattr(
        SimulatorAdapter, "julia_env_template_path", property(lambda _self: template)
    )

    env_setup._refresh_warm_sources(adapter, tmp_path, env, "test")

    assert (env / "Makker" / "src" / "ensemble.jl").read_text() == "# v1\n"
    assert (env / "Manifest.toml").read_text() == "old manifest"
    assert (env / WARM_SOURCE_MARKER).read_text(encoding="utf-8") == old_marker
    assert (env / PRECOMPILE_MARKER).exists()
