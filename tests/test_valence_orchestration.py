"""Tests for bounded, policy-controlled model tool orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from io import BytesIO

import pytest

from proton_i.valence.api import create_server
from proton_i.valence.backend import (
    BackendError,
    ChatMessage,
    OllamaChatBackend,
    ToolCallRequest,
)
from proton_i.valence.engine import ValenceEngine, ValenceError
from proton_i.valence.tools import (
    CapabilityRegistry,
    ExecutionPolicy,
    ToolDefinition,
    ToolParameter,
)


class FakeResponse(BytesIO):
    """Minimal context-managed HTTP response used by transport tests."""

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_tool_call_message_serializes_to_ollama_message_shape() -> None:
    message = ChatMessage(
        "assistant",
        "",
        tool_calls=(ToolCallRequest("search_web", {"query": "CSS"}),),
    )

    assert message.as_payload() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "search_web",
                    "arguments": {"query": "CSS"},
                }
            }
        ],
    }


def test_tool_result_message_includes_tool_name() -> None:
    message = ChatMessage("tool", '{"status":"success"}', tool_name="search_web")

    assert message.as_payload() == {
        "role": "tool",
        "content": '{"status":"success"}',
        "tool_name": "search_web",
    }


def test_chat_message_rejects_tool_calls_on_non_assistant_roles() -> None:
    with pytest.raises(ValueError, match="assistant"):
        ChatMessage(
            "user",
            "hello",
            tool_calls=(ToolCallRequest("search_web", {"query": "CSS"}),),
        )


def test_ollama_backend_sends_tool_schemas_and_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {"query": "CSS Flexbox"},
                                }
                            }
                        ],
                    }
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("proton_i.valence.backend.urlopen", fake_urlopen)
    backend = OllamaChatBackend()
    result = backend.complete_with_tools(
        [ChatMessage("user", "Find CSS Flexbox documentation.")],
        [
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
    assert request.full_url == "http://127.0.0.1:11434/api/chat"  # type: ignore[attr-defined]
    assert captured["timeout"] == 120.0
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
    assert result.role == "assistant"
    assert result.content == ""
    assert result.tool_calls[0].name == "search_web"
    assert result.tool_calls[0].arguments == {"query": "CSS Flexbox"}


def test_ollama_backend_rejects_malformed_tool_call_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "proton_i.valence.backend.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b'{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"x","arguments":[]}}]}}'
        ),
    )

    with pytest.raises(BackendError, match="tool call"):
        OllamaChatBackend().complete_with_tools(
            [ChatMessage("user", "Use a tool.")],
            [
                {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
            ],
        )


class ScriptedToolBackend:
    """Return predetermined model tool-call and final-answer messages."""

    def __init__(self, responses: Sequence[ChatMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], tuple[dict[str, object], ...]]] = []

    def complete(self, _messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("complete() must not be used when approved tools exist")

    def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, object]],
    ) -> ChatMessage:
        self.calls.append((list(messages), tuple(dict(tool) for tool in tools)))
        if not self.responses:
            raise AssertionError("scripted response queue is empty")
        return self.responses.pop(0)


def make_lookup_registry(calls: list[str]) -> CapabilityRegistry:
    """Create a registry with one deterministic, side-effect-free lookup tool."""

    def lookup(arguments: dict[str, object]) -> object:
        query = arguments["query"]
        assert isinstance(query, str)
        calls.append(query)
        return {"summary": f"Found documentation for {query}."}

    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            description="Look up a short query.",
            parameters=(
                ToolParameter(
                    "query",
                    "string",
                    "The query to look up.",
                    max_length=80,
                ),
            ),
            handler=lookup,
        )
    )
    return registry


def test_engine_executes_allowed_tool_and_returns_result_to_model() -> None:
    executed: list[str] = []
    registry = make_lookup_registry(executed)
    backend = ScriptedToolBackend(
        [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCallRequest("lookup", {"query": "CSS Grid"}),),
            ),
            ChatMessage("assistant", "I found the CSS Grid documentation."),
        ]
    )
    assistant = ValenceEngine(
        backend,
        registry=registry,
        policy=ExecutionPolicy(frozenset({"lookup"})),
    )

    answer = assistant.ask("Find CSS Grid documentation.")

    assert answer == "I found the CSS Grid documentation."
    assert executed == ["CSS Grid"]
    assert len(backend.calls) == 2
    second_request = backend.calls[1][0]
    assert second_request[-2].role == "assistant"
    assert second_request[-2].tool_calls[0].name == "lookup"
    assert second_request[-1].role == "tool"
    assert second_request[-1].tool_name == "lookup"
    tool_payload = json.loads(second_request[-1].content)
    assert tool_payload["status"] == "success"
    assert tool_payload["data"] == {"summary": "Found documentation for CSS Grid."}
    assert assistant.history[-1] == ChatMessage(
        "assistant", "I found the CSS Grid documentation."
    )


def test_engine_does_not_execute_a_tool_denied_by_policy() -> None:
    executed: list[str] = []
    registry = make_lookup_registry(executed)
    backend = ScriptedToolBackend(
        [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCallRequest("lookup", {"query": "CSS"}),),
            ),
            ChatMessage("assistant", "I cannot access that tool in this session."),
        ]
    )
    assistant = ValenceEngine(backend, registry=registry, policy=ExecutionPolicy())

    answer = assistant.ask("Look up CSS.")

    assert answer == "I cannot access that tool in this session."
    assert executed == []
    result_payload = json.loads(backend.calls[1][0][-1].content)
    assert result_payload["status"] == "denied"
    assert "not allowed" in result_payload["error"]


def test_engine_returns_unknown_tool_error_without_running_code() -> None:
    registry = make_lookup_registry([])
    backend = ScriptedToolBackend(
        [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCallRequest("unregistered", {"command": "whoami"}),),
            ),
            ChatMessage("assistant", "That capability is not available."),
        ]
    )
    assistant = ValenceEngine(
        backend,
        registry=registry,
        policy=ExecutionPolicy(frozenset({"lookup"})),
    )

    assert assistant.ask("Run an unknown tool.") == "That capability is not available."
    result_payload = json.loads(backend.calls[1][0][-1].content)
    assert result_payload["status"] == "failed"
    assert "not registered" in result_payload["error"]


def test_engine_enforces_maximum_tool_call_rounds() -> None:
    registry = make_lookup_registry([])
    call = ChatMessage(
        "assistant",
        "",
        tool_calls=(ToolCallRequest("lookup", {"query": "repeat"}),),
    )
    backend = ScriptedToolBackend([call, call])
    assistant = ValenceEngine(
        backend,
        registry=registry,
        policy=ExecutionPolicy(frozenset({"lookup"})),
        max_tool_rounds=1,
    )

    with pytest.raises(ValenceError, match="tool-call round limit"):
        assistant.ask("Keep calling the lookup tool.")

    assert assistant.history == ()


def test_default_engine_does_not_disclose_or_execute_tools() -> None:
    class TextBackend:
        def __init__(self) -> None:
            self.calls: list[list[ChatMessage]] = []

        def complete(self, messages: Sequence[ChatMessage]) -> str:
            self.calls.append(list(messages))
            return "Plain text answer."

    backend = TextBackend()
    assistant = ValenceEngine(backend)

    assert assistant.ask("Hello.") == "Plain text answer."
    assert len(backend.calls) == 1


def test_api_session_receives_explicit_tool_registry_and_policy() -> None:
    executed: list[str] = []
    registry = make_lookup_registry(executed)
    backend = ScriptedToolBackend(
        [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCallRequest("lookup", {"query": "API"}),),
            ),
            ChatMessage("assistant", "The API documentation is available."),
        ]
    )
    server = create_server(
        backend,
        port=0,
        tool_registry=registry,
        tool_policy=ExecutionPolicy(frozenset({"lookup"})),
    )
    try:
        session_id, session = server.create_session()  # type: ignore[misc]
        assert session.engine.ask("Find API documentation.") == (
            "The API documentation is available."
        )
        assert executed == ["API"]
    finally:
        server.server_close()
