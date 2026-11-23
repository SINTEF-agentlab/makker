"""The real-path shim over deepagents' ``validate_path``.

The wrapper logic is tested directly, then the real install is exercised against
deepagents, and last the real file tools are driven with relative paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makker.agent import real_paths


def _fake_validate_path(path: str, *, allowed_prefixes=None) -> str:
    """Stand-in matching deepagents: reject Windows-absolute, normalize the rest."""
    import re

    if re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"Windows absolute paths are not supported: {path}")
    if ".." in path.split("/"):
        raise ValueError("Path traversal not allowed")
    return path if path.startswith("/") else f"/{path}"


def test_windows_absolute_path_passes_through_unchanged():
    wrapped = real_paths._wrap_validate_path(_fake_validate_path)
    assert wrapped(r"C:\Users\jakobt\.julia\packages\JutulDarcy\x\src\foo.jl") == (
        r"C:\Users\jakobt\.julia\packages\JutulDarcy\x\src\foo.jl"
    )
    assert wrapped("C:/Users/jakobt/file.txt") == "C:/Users/jakobt/file.txt"


def test_relative_paths_stay_relative_and_absolute_ones_defer_to_the_original():
    wrapped = real_paths._wrap_validate_path(_fake_validate_path)
    # POSIX absolute: original normalizes (leading slash preserved).
    assert wrapped("/home/u/file.txt") == "/home/u/file.txt"
    # Relative: no leading slash added, so the backend resolves it in the workspace.
    assert wrapped("foo/bar.jl") == "foo/bar.jl"
    assert wrapped(".") == "."
    # Traversal still rejected by the original.
    with pytest.raises(ValueError, match="traversal"):
        wrapped("../etc/passwd")


def test_wrapping_is_idempotent():
    once = real_paths._wrap_validate_path(_fake_validate_path)
    twice = real_paths._wrap_validate_path(once)
    assert twice is once


def _unwrap(fn):
    """The true (unpatched) ``validate_path`` under any of our wrappers."""
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_install_patches_every_module_that_binds_validate_path():
    """``_install`` reaches each deepagents module that imported ``validate_path``.

    Independent of prior global state: an earlier ``build_agent`` may have already
    installed the patch, so we pin the genuine original first.
    """
    from deepagents.backends import utils
    from deepagents.middleware import _skill_tools
    from deepagents.middleware import filesystem as fs_middleware

    modules = (utils, fs_middleware, _skill_tools)
    before = [module.validate_path for module in modules]
    original = _unwrap(before[0])

    # The genuine deepagents helper roots a relative path and rejects a drive path.
    assert original("foo/bar.jl") == "/foo/bar.jl"
    with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
        original(r"C:\Users\x\file.txt")

    for module in modules:
        module.validate_path = original  # start from a known-unpatched state
    try:
        real_paths._install()
        for module in modules:
            assert module.validate_path(r"C:\Users\x\file.txt") == r"C:\Users\x\file.txt"
            assert module.validate_path("foo/bar.jl") == "foo/bar.jl"
            assert module.validate_path("/abs/./file.txt") == "/abs/file.txt"
    finally:
        for module, previous in zip(modules, before, strict=True):
            module.validate_path = previous


def test_file_tools_resolve_relative_paths_in_the_workspace(tmp_path: Path) -> None:
    """Through the real middleware, a relative path names a workspace file.

    Without the shim deepagents turns ``model.jl`` into ``/model.jl``, so the
    write is refused at the filesystem root and the read finds nothing.
    """
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.tools import ToolRuntime

    from makker.agent.builder import build_backend

    real_paths.enable_real_paths()
    tools = {
        tool.name: tool
        for tool in FilesystemMiddleware(backend=build_backend(workspace=tmp_path)).tools
    }

    def call(name: str, **args) -> str:
        runtime = ToolRuntime(
            state={"messages": []},
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )
        call = {
            "type": "tool_call",
            "name": name,
            "args": {**args, "runtime": runtime},
            "id": "call-1",
        }
        return str(tools[name].invoke(call).content)

    call("write_file", file_path="model.jl", content="x = 1\n")
    assert (tmp_path / "model.jl").read_text() == "x = 1\n"
    assert "x = 1" in call("read_file", file_path="model.jl")
    call("edit_file", file_path="./model.jl", old_string="x = 1", new_string="x = 2")
    assert (tmp_path / "model.jl").read_text() == "x = 2\n"

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.jl").write_text("y = 3\n")
    assert "a.jl" in call("ls", path="sub")
    assert "a.jl" in call("grep", pattern="y = 3", path="sub")
    assert "model.jl" in call("ls", path=".")

    assert "traversal" in call("read_file", file_path="../outside.txt")
    call("delete", file_path="sub")
    assert not (tmp_path / "sub").exists()


def test_split_windows_glob_drive_absolute_pattern(monkeypatch):
    monkeypatch.setattr(real_paths.os, "name", "nt")
    # An absolute drive pattern is split into a relative pattern + base directory.
    assert real_paths.split_windows_glob(r"C:\proj\src\**\*.py", None) == (
        "**/*.py",
        r"C:\proj\src",
    )
    # Forward slashes (how a model often writes a Windows path) work the same.
    assert real_paths.split_windows_glob("C:/proj/src/**/*.py", None) == (
        "**/*.py",
        r"C:\proj\src",
    )
    # An exact file path (no metacharacter): the last component becomes the pattern.
    assert real_paths.split_windows_glob(r"C:\proj\src\model.jl", None) == (
        "model.jl",
        r"C:\proj\src",
    )


def test_split_windows_glob_leaves_other_inputs_untouched(monkeypatch):
    monkeypatch.setattr(real_paths.os, "name", "nt")
    # Relative pattern: unchanged.
    assert real_paths.split_windows_glob("**/*.py", None) == ("**/*.py", None)
    # An explicit path already given: don't second-guess it.
    assert real_paths.split_windows_glob(r"C:\proj\**\*.py", "C:\\base") == (
        r"C:\proj\**\*.py",
        "C:\\base",
    )


def test_split_windows_glob_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(real_paths.os, "name", "posix")
    # Off Windows a drive-looking pattern is left for the backend to handle.
    assert real_paths.split_windows_glob(r"C:\proj\**\*.py", None) == (
        r"C:\proj\**\*.py",
        None,
    )
