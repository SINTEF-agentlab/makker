"""Tests for streamed tool-output normalization."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from makker.agent.tool_output import is_interrupt_payload, normalize_tool_output


def test_normalize_tool_message_object() -> None:
    message = ToolMessage(content="hello", tool_call_id="call-1", name="read_file")
    assert normalize_tool_output(message) == "hello"


def test_normalize_tool_message_repr() -> None:
    raw = (
        "[ToolMessage(content='     1\\tline one\\n     2\\tline two', "
        "name='read_file', tool_call_id='call-1', additional_kwargs={})]"
    )
    assert normalize_tool_output(raw) == "     1\tline one\n     2\tline two"


def test_text_that_mentions_a_tool_message_is_left_alone() -> None:
    # Reading a source file that builds a ToolMessage used to show only the
    # captured content string in place of the file.
    source = 'def reply():\n    return ToolMessage(content="done", tool_call_id="c1")\n'
    assert normalize_tool_output(source) == source


def test_interrupt_payload_detection() -> None:
    text = "Interrupt(value={'action_requests': [{'name': 'write_file'}]})"
    assert is_interrupt_payload(text) is True


def test_normalize_multimodal_content_keeps_text_and_drops_the_image() -> None:
    """A tool that hands the model an image must not paste it into the transcript.

    The plot tools reply with a summary plus the rendered PNG as base64, so
    serializing the whole list puts a screenful of encoded bytes where the
    one-line summary belongs. The image reaches the user as the artifact card.
    """

    blocks = [
        {"type": "text", "text": "recaptured view to artifacts/recapture-ab12.png"},
        {"type": "image", "mime_type": "image/png", "base64": "iVBORw0KGgo" + "A" * 4000},
    ]
    assert normalize_tool_output(blocks) == "recaptured view to artifacts/recapture-ab12.png"


def test_normalize_plain_list_data_is_still_json() -> None:
    """Only tagged content blocks are flattened; list data keeps its structure."""

    assert normalize_tool_output([{"a": 1}, {"b": 2}]) == '[{"a": 1}, {"b": 2}]'
    assert normalize_tool_output(["x", {"y": 1}]) == '["x", {"y": 1}]'


def test_normalize_a_command_gives_the_reply_it_carries() -> None:
    """A tool that updates state answers with a command; shown as-is it is a
    screenful of repr where the tool's one-line reply belongs."""
    from langgraph.types import Command

    reply = ToolMessage(content="Updated todo list", tool_call_id="t1", name="write_todos")
    command = Command(update={"todos": [{"content": "a"}], "messages": [reply]})
    assert normalize_tool_output(command) == "Updated todo list"
