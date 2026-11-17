"""Tests for Ollama tool-call parsing and bounded Valence orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from io import BytesIO

import pytest

from proton_i.valence.backend import (
    BackendError,
    ChatMessage,
    ChatTurn,
    ModelToolCall,
    OllamaChatBackend,
)
from proton_i.valence.engine import ValenceEngine, ValenceError
from proton_i.valence.tools import (
    CapabilityRegistry,
    ExecutionPolicy,
    ToolDefinition,
    ToolParameter,
)


class FakeResponse(BytesIO):
    """Minimal context-managed HTTP response for Ollama transport tests."""

    def __enter__(self) -> BytesIO:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_ollama_tool_request_uses_function_schema_and_parses_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": {"query": "CSS Grid"},
                        },
                    }
                ],
            }
        }
        return FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr("proton_i.valence.backend.urlopen", fake_urlopen)
    backend = OllamaChatBackend()
    turn = backend.complete_turn(
        [ChatMessage("user", "Search for CSS Grid")],
        tools=[
            {
                "name": "search_web",
                "description": "Search public websites.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "requires_approval": False,
            }
        ],
    )

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
    assert payload["stream"] is False
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search public websites.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert turn.content == ""
    assert turn.tool_calls == (ModelToolCall("search_web", {"query": "CSS Grid"}),)


def test_ollama_tool_result_messages_are_serialized_for_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, _timeout: float) -> FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return FakeResponse(
            b'{"message":{"role":"assistant","content":"The result is ready."}}'
        )

    monkeypatch.setattr("proton_i.valence.backend.urlopen", fake_urlopen)
    messages = [
        ChatMessage("user", "Look this up."),
        ChatMessage(
            "assistant",
            "",
            tool_calls=(ModelToolCall("search_web", {"query": "HTML"}),),
        ),
        ChatMessage(
            "tool",
            '{"status":"success","data":{"result_count":1}}',
            tool_name="search_web",
        ),
    ]

    turn = OllamaChatBackend().complete_turn(messages, tools=())

    payload = captured["payload"]
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "search_web", "arguments": {"query": "HTML"}},
            }
        ],
    }
    assert payload["messages"][2] == {
        "role": "tool",
        "content": '{"status":"success","data":{"result_count":1}}',
        "tool_name": "search_web",
    }
    assert "tools" not in payload
    assert turn.content == "The result is ready."


@pytest.mark.parametrize(
    "tool_calls",
    [
        "not-a-list",
        [{"function": {"name": "search_web", "arguments": "not-json-object"}}] * 9,
        [{"function": {"name": "", "arguments": {}}}],
        [{"function": {"name": "search_web", "arguments": {"huge": "x" * 20_000}}}],
    ],
)
def test_ollama_backend_rejects_malformed_or_oversized_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    tool_calls: object,
) -> None:
    body = {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}
    monkeypatch.setattr(
        "proton_i.valence.backend.urlopen",
        lambda *_args, **_kwargs: FakeResponse(json.dumps(body).encode("utf-8")),
    )

    with pytest.raises(BackendError):
        OllamaChatBackend().complete_turn([ChatMessage("user", "Run a tool")])


def test_ollama_backend_rejects_tool_schema_with_duplicate_names() -> None:
    backend = OllamaChatBackend()
    schema = {
        "name": "same_tool",
        "description": "A tool.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    with pytest.raises(ValueError, match="unique"):
        backend.complete_turn(
            [ChatMessage("user", "Hi")],
            tools=[schema, schema],
        )


def test_chat_message_validates_tool_result_and_assistant_tool_call_roles() -> None:
    with pytest.raises(ValueError, match="tool_name"):
        ChatMessage("tool", '{"ok":true}')
    with pytest.raises(ValueError, match="only assistant"):
        ChatMessage(
            "user",
            "Hi",
            tool_calls=(ModelToolCall("search_web", {}),),
        )
    with pytest.raises(ValueError, match="tool calls"):
        ChatMessage("assistant", "")


class FakeToolBackend:
    """Fake provider that records each tool-enabled chat turn."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[tuple[list[ChatMessage], list[Mapping[str, object]]]] = []
        self.plain_calls: list[list[ChatMessage]] = []

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.plain_calls.append(list(messages))
        return "Plain response."

    def complete_turn(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, object]],
    ) -> ChatTurn:
        self.calls.append((list(messages), list(tools)))
        if not self.turns:
            raise BackendError("No scripted model turn remains.")
        return self.turns.pop(0)


def make_registry(handler=None, requires_approval: bool = False) -> CapabilityRegistry:
    """Create a registry with one deterministic double-number tool."""

    def double(arguments: dict[str, object]) -> object:
        if handler is not None:
            return handler(arguments)
        return {"value": arguments["value"] * 2}  # type: ignore[operator]

    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="double_number",
            description="Double a number.",
            parameters=(
                ToolParameter(
                    "value",
                    "integer",
                    "The integer to double.",
                    minimum=-1_000_000,
                    maximum=1_000_000,
                ),
            ),
            handler=double,
            requires_approval=requires_approval,
        )
    )
    return registry


def test_engine_executes_allowed_tool_and_feeds_result_to_model() -> None:
    backend = FakeToolBackend(
        [
            ChatTurn(
                "",
                (ModelToolCall("double_number", {"value": 6}),),
            ),
            ChatTurn("The doubled value is 12."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
    )

    answer = engine.ask("Double 6 for me.")

    assert answer == "The doubled value is 12."
    assert len(backend.calls) == 2
    first_messages, first_tools = backend.calls[0]
    assert [message.role for message in first_messages] == ["system", "user"]
    assert first_tools[0]["name"] == "double_number"
    assert "requires_approval" not in first_tools[0]

    second_messages, second_tools = backend.calls[1]
    assert second_tools == first_tools
    assert second_messages[2] == ChatMessage(
        "assistant",
        "",
        tool_calls=(ModelToolCall("double_number", {"value": 6}),),
    )
    tool_result = json.loads(second_messages[3].content)
    assert second_messages[3].role == "tool"
    assert second_messages[3].tool_name == "double_number"
    assert tool_result == {
        "tool_name": "double_number",
        "status": "success",
        "data": {"value": 12},
    }
    assert engine.history == (
        ChatMessage("user", "Double 6 for me."),
        ChatMessage("assistant", "The doubled value is 12."),
    )


def test_engine_does_not_offer_or_execute_tools_under_default_deny_policy() -> None:
    backend = FakeToolBackend([])
    engine = ValenceEngine(backend, registry=make_registry())

    answer = engine.ask("Say hello.")

    assert answer == "Plain response."
    assert backend.calls == []
    assert len(backend.plain_calls) == 1


def test_engine_sends_denied_tool_result_for_unregistered_tool() -> None:
    backend = FakeToolBackend(
        [
            ChatTurn("", (ModelToolCall("delete_everything", {}),)),
            ChatTurn("I could not use that tool."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
    )

    answer = engine.ask("Delete everything.")

    assert answer == "I could not use that tool."
    result = json.loads(backend.calls[1][0][-1].content)
    assert result["status"] == "failed"
    assert result["error"] == "Tool is not registered."


def test_engine_never_executes_tool_that_requires_user_approval() -> None:
    calls: list[bool] = []

    def sensitive(_arguments: dict[str, object]) -> object:
        calls.append(True)
        return {"changed": True}

    backend = FakeToolBackend(
        [
            ChatTurn("", (ModelToolCall("double_number", {"value": 2}),)),
            ChatTurn("That action needs approval."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(handler=sensitive, requires_approval=True),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
    )

    answer = engine.ask("Do the action.")

    assert answer == "That action needs approval."
    assert calls == []
    result = json.loads(backend.calls[1][0][-1].content)
    assert result["status"] == "denied"
    assert "approval" in result["error"]


def test_engine_applies_total_tool_call_budget_and_returns_final_answer() -> None:
    calls: list[int] = []

    def double(arguments: dict[str, object]) -> object:
        calls.append(int(arguments["value"]))
        return {"value": int(arguments["value"]) * 2}

    backend = FakeToolBackend(
        [
            ChatTurn(
                "",
                (
                    ModelToolCall("double_number", {"value": 2}),
                    ModelToolCall("double_number", {"value": 3}),
                ),
            ),
            ChatTurn("Only the first call was executed."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(handler=double),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
        max_tool_calls=1,
    )

    answer = engine.ask("Double both numbers.")

    assert answer == "Only the first call was executed."
    assert calls == [2]
    tool_messages = [
        message for message in backend.calls[1][0] if message.role == "tool"
    ]
    results = [json.loads(message.content) for message in tool_messages]
    assert [result["status"] for result in results] == ["success", "denied"]
    assert "tool-call limit" in results[1]["error"]


def test_engine_caps_tool_rounds_and_disables_tools_for_final_turn() -> None:
    backend = FakeToolBackend(
        [
            ChatTurn("", (ModelToolCall("double_number", {"value": 2}),)),
            ChatTurn("A final bounded response."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
        max_tool_rounds=1,
    )

    answer = engine.ask("Double the number.")

    assert answer == "A final bounded response."
    assert len(backend.calls) == 2
    assert backend.calls[0][1]
    assert backend.calls[1][1] == ()


def test_engine_rejects_backend_that_cannot_perform_tool_calls() -> None:
    class PlainOnlyBackend:
        def complete(self, _messages: Sequence[ChatMessage]) -> str:
            return "No tool protocol."

    engine = ValenceEngine(
        PlainOnlyBackend(),
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
    )

    with pytest.raises(ValenceError, match="does not support tool calling"):
        engine.ask("Use a tool.")
    assert engine.history == ()


def test_engine_rejects_model_tool_calls_after_final_tool_free_turn() -> None:
    backend = FakeToolBackend(
        [
            ChatTurn("", (ModelToolCall("double_number", {"value": 2}),)),
            ChatTurn("", (ModelToolCall("double_number", {"value": 3}),)),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
        max_tool_rounds=1,
    )

    with pytest.raises(ValenceError, match="continued requesting tools"):
        engine.ask("Keep requesting tools.")
    assert engine.history == ()


def test_engine_rejects_invalid_model_arguments_and_does_not_store_failed_turn() -> (
    None
):
    backend = FakeToolBackend(
        [
            ChatTurn("", (ModelToolCall("double_number", "not-an-object"),)),
            ChatTurn("The input was invalid."),
        ]
    )
    engine = ValenceEngine(
        backend,
        registry=make_registry(),
        tool_policy=ExecutionPolicy(frozenset({"double_number"})),
    )

    answer = engine.ask("Try a malformed call.")

    assert answer == "The input was invalid."
    result = json.loads(backend.calls[1][0][-1].content)
    assert result["status"] == "invalid"
    assert "JSON object" in result["error"]


def test_engine_validates_tool_loop_limits() -> None:
    with pytest.raises(ValueError, match="max_tool_rounds"):
        ValenceEngine(FakeToolBackend([]), max_tool_rounds=0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        ValenceEngine(FakeToolBackend([]), max_tool_calls=17)
