"""The approval policy: which tool calls wait for a person."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from makker.agent.approval import (
    ALLOWLIST_FILE_EDITS,
    ApprovalMode,
    ApprovalPolicy,
    build_resume_payload,
    parse_approval_mode,
)
from makker.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from makker.session import Session
from makker.session_host import SessionHost


def _call(name: str, call_id: str = "call-1", **args) -> SimpleNamespace:
    """A stand-in for the ``ToolCallRequest`` the gate receives."""
    return SimpleNamespace(tool_call={"id": call_id, "name": name, "args": args})


def test_parse_approval_mode_defaults_to_ask() -> None:
    assert parse_approval_mode(None) is ApprovalMode.ASK


def test_approval_mode_cycles() -> None:
    assert ApprovalMode.ASK.cycle_next() is ApprovalMode.WORKSPACE
    assert ApprovalMode.WORKSPACE.cycle_next() is ApprovalMode.AUTO
    assert ApprovalMode.AUTO.cycle_next() is ApprovalMode.ASK


def test_each_mode_asks_for_the_right_calls(tmp_path: Path) -> None:
    inside = {"file_path": "model.jl"}
    outside = {"file_path": str(tmp_path.parent / "elsewhere" / ".zshrc")}
    policy = ApprovalPolicy("ask", workspace=tmp_path)
    for name in ("execute", "write_file", "edit_file", "delete"):
        assert policy.needs_approval(name, inside)
    assert not policy.needs_approval("read_file", inside)

    # Accept edits: edits inside the workspace only. A file elsewhere, a
    # recursive delete and a shell command still ask.
    policy.mode = ApprovalMode.WORKSPACE
    assert not policy.needs_approval("write_file", inside)
    assert not policy.needs_approval("edit_file", {"file_path": str(tmp_path / "a.jl")})
    assert policy.needs_approval("write_file", outside)
    assert policy.needs_approval("write_file", {"file_path": "../escape.jl"})
    assert policy.needs_approval("delete", inside)
    assert policy.needs_approval("execute", {"command": "ls"})

    policy.mode = ApprovalMode.AUTO
    for name in ("execute", "write_file", "edit_file", "delete"):
        assert not policy.needs_approval(name, outside)


def test_always_allow_covers_workspace_edits_and_nothing_else(tmp_path: Path) -> None:
    policy = ApprovalPolicy("ask", workspace=tmp_path)
    policy.allow("shell")  # not a category a session can allow
    assert policy.allowlist == frozenset()
    assert policy.needs_approval("execute", {"command": "ls"})

    policy.allow(ALLOWLIST_FILE_EDITS)
    assert not policy.needs_approval("edit_file", {"file_path": "model.jl"})
    assert policy.needs_approval("edit_file", {"file_path": str(tmp_path.parent / "x.jl")})
    assert policy.needs_approval("delete", {"file_path": "scratch"})


def test_always_allow_is_offered_only_for_workspace_edits(tmp_path: Path) -> None:
    policy = ApprovalPolicy("ask", workspace=tmp_path)

    def offer(*actions: dict) -> frozenset[str]:
        return policy.always_allow_offer({"action_requests": list(actions)})

    edit = {"name": "edit_file", "args": {"file_path": "model.jl"}}
    assert offer(edit) == {ALLOWLIST_FILE_EDITS}
    assert offer(edit, {"name": "delete", "args": {"file_path": "scratch"}}) == frozenset()
    assert offer(edit, {"name": "execute", "args": {"command": "ls"}}) == frozenset()
    assert offer({"name": "write_file", "args": {"file_path": "/etc/hosts"}}) == frozenset()
    assert offer() == frozenset()


def test_the_gate_answers_once_per_tool_call(tmp_path: Path) -> None:
    """A resumed turn asks again for the same calls and must get the same answer."""
    policy = ApprovalPolicy("ask", workspace=tmp_path)
    assert policy.gate(_call("write_file", file_path="model.jl"))

    policy.mode = ApprovalMode.AUTO
    assert policy.gate(_call("write_file", file_path="model.jl"))  # same call: still asks
    assert not policy.gate(_call("write_file", "call-2", file_path="model.jl"))

    policy.settled()
    assert not policy.gate(_call("write_file", file_path="model.jl"))


# --- Through a real agent graph ----------------------------------------------


def _host(tmp_path: Path, responses: list, mode: str = "ask") -> SessionHost:
    from langgraph.checkpoint.memory import InMemorySaver

    from makker.agent.builder import build_agent

    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path / "state", simulator=make_fake_adapter(tmp_path)
    )
    policy = ApprovalPolicy(mode)
    agent, backend = build_agent(
        session,
        model=make_scripted_model([*responses, scripted_final("done")]),
        checkpointer=InMemorySaver(),
        approval_mode=policy,
    )
    return SessionHost(
        session=session, agent=agent, backend=backend, approval_mode=policy, surface="tui"
    )


def _calls(*calls: tuple[str, str, dict]) -> AIMessage:
    """One model message asking for several tool calls at once."""
    return AIMessage(
        content="",
        tool_calls=[{"id": cid, "name": name, "args": args} for cid, name, args in calls],
    )


def _asked(result) -> list[str]:
    return [a["name"] for i in result.interrupts for a in i.value["action_requests"]]


async def test_an_allowed_edit_never_carries_a_delete_with_it(tmp_path: Path) -> None:
    (tmp_path / "scratch").mkdir()
    host = _host(
        tmp_path,
        [
            _calls(
                ("w1", "write_file", {"file_path": "notes.txt", "content": "hi"}),
                ("d1", "delete", {"file_path": "scratch"}),
            )
        ],
    )
    host.approval.allow(ALLOWLIST_FILE_EDITS)

    result = await host.drive_turn(lambda: host.runner.run_prompt("tidy up"))
    assert _asked(result) == ["delete"]  # the edit is allowed; the delete asks alone

    payload = build_resume_payload(result.interrupts, {"type": "reject"})
    result = await host.drive_turn(lambda: host.runner.resume(payload))
    assert result.interrupts == []
    assert (tmp_path / "notes.txt").read_text() == "hi"
    assert (tmp_path / "scratch").is_dir()


async def test_a_change_while_a_card_waits_does_not_break_the_resume(tmp_path: Path) -> None:
    """The user picks "always allow edits" on a card that also holds a delete.

    Resuming asks the gate again for the same message. Were the edit now let
    through, two decisions would meet one paused call and the turn would fail.
    """
    (tmp_path / "scratch").mkdir()
    host = _host(
        tmp_path,
        [
            _calls(
                ("w1", "write_file", {"file_path": "notes.txt", "content": "hi"}),
                ("d1", "delete", {"file_path": "scratch"}),
            )
        ],
    )
    result = await host.drive_turn(lambda: host.runner.run_prompt("tidy up"))
    assert _asked(result) == ["write_file", "delete"]

    host.approval.allow(ALLOWLIST_FILE_EDITS)
    payload = build_resume_payload(result.interrupts, {"type": "reject"})
    result = await host.drive_turn(lambda: host.runner.resume(payload))

    assert result.interrupts == []
    assert not (tmp_path / "notes.txt").exists()  # both stay rejected
    assert (tmp_path / "scratch").is_dir()


async def test_accept_edits_asks_for_a_file_outside_the_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere.txt"
    host = _host(
        tmp_path,
        [
            _calls(
                ("w1", "write_file", {"file_path": "inside.txt", "content": "a"}),
                ("w2", "write_file", {"file_path": str(outside), "content": "b"}),
            )
        ],
        mode="workspace",
    )

    result = await host.drive_turn(lambda: host.runner.run_prompt("write both"))

    assert [a["args"]["file_path"] for a in result.interrupts[0].value["action_requests"]] == [
        str(outside)
    ]


async def test_accept_edits_asks_for_an_outside_file_spelt_with_backslashes(
    tmp_path: Path,
) -> None:
    # The file tools turn backslashes into separators on every system, so this
    # names the same file as the test above and must be asked about the same way.
    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere.txt"
    spelt = str(outside).replace("/", "\\")
    host = _host(
        tmp_path,
        [_calls(("w1", "write_file", {"file_path": spelt, "content": "b"}))],
        mode="workspace",
    )

    result = await host.drive_turn(lambda: host.runner.run_prompt("write it"))

    assert [a["args"]["file_path"] for a in result.interrupts[0].value["action_requests"]] == [
        spelt
    ]
    assert not outside.exists()


async def test_a_mode_change_applies_without_a_rebuild(tmp_path: Path) -> None:
    host = _host(
        tmp_path,
        [_calls(("e1", "execute", {"command": "echo hi"}))],
        mode="auto",
    )
    agent = host.agent

    host.set_approval_mode("ask")
    result = await host.drive_turn(lambda: host.runner.run_prompt("run it"))

    assert host.agent is agent
    assert _asked(result) == ["execute"]
