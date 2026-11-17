"""Loopback HTTP API tests for Valence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from http.client import HTTPConnection
from threading import Thread

import pytest

from proton_i.valence.api import create_server
from proton_i.gluons.search import SearchResult
from proton_i.valence.backend import ChatMessage, ChatTurn, ModelToolCall
from proton_i.valence.integrations.gluons import register_gluons_tool
from proton_i.valence.tools import CapabilityRegistry, ExecutionPolicy, ToolDefinition


class RecordingBackend:
    """Small deterministic backend for local API integration tests."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(["Reply"] if replies is None else replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise RuntimeError("no test reply configured")
        return self.replies.pop(0)


@pytest.fixture
def running_api():
    backend = RecordingBackend(["First reply", "Second reply", "After reset"])
    server = create_server(backend, port=0, max_sessions=4)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, backend
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def request_json(
    server,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Send a JSON request to the ephemeral test server."""
    connection = HTTPConnection(server.server_address[0], server.server_address[1])
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    body = json.dumps(payload).encode("utf-8") if payload is not None else b"{}"
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    status = response.status
    response_body = response.read()
    connection.close()
    return status, json.loads(response_body.decode("utf-8"))


def test_health_and_capabilities_endpoints(running_api) -> None:
    server, _backend = running_api
    status, health = request_json(server, "GET", "/health")
    assert status == 200
    assert health["status"] == "ok"
    assert health["active_sessions"] == 0

    status, capabilities = request_json(server, "GET", "/v1/capabilities")
    assert status == 200
    assert capabilities["local_chat"] == "supported"
    assert capabilities["web_search"] == "not_enabled"
    assert capabilities["model_tool_calling"] == "text_only"


def test_chat_creates_a_random_session_and_returns_reply(running_api) -> None:
    server, _backend = running_api
    status, response = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Hello Valence"},
    )

    assert status == 200
    assert response["message"] == "First reply"
    assert isinstance(response["session_id"], str)
    assert len(response["session_id"]) > 20


def test_chat_keeps_context_and_reset_clears_it(running_api) -> None:
    server, backend = running_api
    first_status, first = request_json(
        server, "POST", "/v1/chat", {"message": "First question"}
    )
    session_id = first["session_id"]
    second_status, _second = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Follow-up", "session_id": session_id},
    )

    assert first_status == second_status == 200
    assert backend.calls[1][1:] == [
        ChatMessage("user", "First question"),
        ChatMessage("assistant", "First reply"),
        ChatMessage("user", "Follow-up"),
    ]

    reset_status, reset = request_json(
        server, "POST", "/v1/reset", {"session_id": session_id}
    )
    after_status, after = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Fresh start", "session_id": session_id},
    )

    assert reset_status == after_status == 200
    assert reset["status"] == "reset"
    assert after["message"] == "After reset"
    assert len(backend.calls[2]) == 2
    assert backend.calls[2][0].role == "system"
    assert backend.calls[2][1] == ChatMessage("user", "Fresh start")


def test_chat_rejects_unknown_session(running_api) -> None:
    server, _backend = running_api
    status, response = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Hello", "session_id": "not-a-session"},
    )
    assert status == 404
    assert "not found" in response["error"]


def test_reset_requires_an_existing_session(running_api) -> None:
    server, _backend = running_api
    status, response = request_json(
        server, "POST", "/v1/reset", {"session_id": "unknown"}
    )
    assert status == 404
    assert "not found" in response["error"]


def test_api_rejects_non_json_content_type(running_api) -> None:
    server, _backend = running_api
    connection = HTTPConnection(server.server_address[0], server.server_address[1])
    connection.request(
        "POST",
        "/v1/chat",
        body=b"message=hello",
        headers={"Content-Type": "text/plain"},
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 415
    assert "application/json" in body["error"]


def test_api_rejects_invalid_json(running_api) -> None:
    server, _backend = running_api
    connection = HTTPConnection(server.server_address[0], server.server_address[1])
    connection.request(
        "POST",
        "/v1/chat",
        body=b"{invalid",
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 400
    assert "valid UTF-8 JSON" in body["error"]


def test_api_rejects_oversized_requests(running_api) -> None:
    server, _backend = running_api
    connection = HTTPConnection(server.server_address[0], server.server_address[1])
    connection.request(
        "POST",
        "/v1/chat",
        body=b"x" * (16 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 413
    assert "16 KB" in body["error"]


def test_api_rejects_remote_browser_origins(running_api) -> None:
    server, _backend = running_api
    connection = HTTPConnection(server.server_address[0], server.server_address[1])
    connection.request(
        "POST",
        "/v1/chat",
        body=b'{"message":"hello"}',
        headers={
            "Content-Type": "application/json",
            "Origin": "https://malicious.example",
        },
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 403
    assert "loopback browser origins" in body["error"]


def test_api_enforces_session_limit() -> None:
    backend = RecordingBackend(["one", "two"])
    server = create_server(backend, port=0, max_sessions=1)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first_status, _first = request_json(
            server, "POST", "/v1/chat", {"message": "first"}
        )
        second_status, second = request_json(
            server, "POST", "/v1/chat", {"message": "second"}
        )
        assert first_status == 200
        assert second_status == 429
        assert "session limit" in second["error"]
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_api_refuses_non_loopback_bind_address() -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server(RecordingBackend(), host="0.0.0.0", port=8765)


def test_api_accepts_same_origin_loopback_browser_request(running_api) -> None:
    server, _backend = running_api
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    status, response = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Hello from same origin"},
        headers={"Origin": origin},
    )
    assert status == 200
    assert response["message"] == "First reply"


def test_api_rejects_loopback_origin_from_different_port(running_api) -> None:
    server, _backend = running_api
    status, response = request_json(
        server,
        "POST",
        "/v1/chat",
        {"message": "Cross-origin request"},
        headers={"Origin": "http://127.0.0.1:5173"},
    )
    assert status == 403
    assert "same-origin" in response["error"]


def test_api_sessions_keep_conversation_history_isolated() -> None:
    backend = RecordingBackend(["A reply", "B reply", "A follow-up"])
    server = create_server(backend, port=0, max_sessions=3)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first_status, first = request_json(
            server, "POST", "/v1/chat", {"message": "First session question"}
        )
        second_status, second = request_json(
            server, "POST", "/v1/chat", {"message": "Second session question"}
        )
        followup_status, _followup = request_json(
            server,
            "POST",
            "/v1/chat",
            {
                "message": "First session follow-up",
                "session_id": first["session_id"],
            },
        )

        assert first_status == second_status == followup_status == 200
        assert first["session_id"] != second["session_id"]
        assert backend.calls[2][1:] == [
            ChatMessage("user", "First session question"),
            ChatMessage("assistant", "A reply"),
            ChatMessage("user", "First session follow-up"),
        ]
        assert all(
            message.content != "Second session question" for message in backend.calls[2]
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_api_backend_failure_returns_sanitized_error() -> None:
    backend = RecordingBackend([])
    server = create_server(backend, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, response = request_json(
            server, "POST", "/v1/chat", {"message": "Trigger a failure"}
        )
        assert status == 502
        assert response["error"] == "Valence could not complete this message."
        assert "no test reply configured" not in response["error"]
        assert isinstance(response["session_id"], str)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

def test_api_passes_explicit_tool_registry_and_policy_to_new_sessions() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="local_lookup",
            description="A test lookup tool.",
            parameters=(),
            handler=lambda _arguments: {"ok": True},
        )
    )
    policy = ExecutionPolicy(frozenset({"local_lookup"}))
    server = create_server(
        RecordingBackend(),
        port=0,
        registry=registry,
        tool_policy=policy,
    )
    try:
        _session_id, session = server.create_session()  # type: ignore[misc]
        assert session.engine.registry is registry
        assert session.engine.tool_policy == policy
    finally:
        server.server_close()


def test_api_rejects_tool_policy_without_a_registry() -> None:
    with pytest.raises(ValueError, match="without a registry"):
        create_server(
            RecordingBackend(),
            port=0,
            tool_policy=ExecutionPolicy(frozenset({"search_web"})),
        )


class SearchTestProvider:
    """Deterministic search provider to verify the full API-to-Gluons path."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.queries.append((query, max_results))
        return [
            SearchResult(
                title="Example CSS reference",
                url="https://example.com/css",
                snippet="CSS controls the presentation of a web document.",
            )
        ][:max_results]


class ToolCallingBackend:
    """Model double that asks for one search and then summarizes its result."""

    def __init__(self) -> None:
        self.turns = [
            ChatTurn("", (ModelToolCall("search_web", {"query": "CSS"}),)),
            ChatTurn("CSS controls presentation. Source: https://example.com/css"),
        ]
        self.calls: list[tuple[list[ChatMessage], list[dict[str, object]]]] = []

    def complete(self, _messages: Sequence[ChatMessage]) -> str:
        return "Plain response."

    def complete_turn(self, messages, tools) -> ChatTurn:
        self.calls.append((list(messages), list(tools)))
        return self.turns.pop(0)


def test_api_opt_in_runs_the_gluons_tool_and_returns_source_attribution() -> None:
    backend = ToolCallingBackend()
    searcher = SearchTestProvider()
    registry = CapabilityRegistry()
    register_gluons_tool(registry, searcher=searcher)
    policy = ExecutionPolicy(frozenset({"search_web"}))
    server = create_server(
        backend,
        port=0,
        registry=registry,
        tool_policy=policy,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, capabilities = request_json(server, "GET", "/v1/capabilities")
        assert status == 200
        assert capabilities["web_search"] == "enabled"
        assert capabilities["model_tool_calling"] == "supported"

        chat_status, response = request_json(
            server,
            "POST",
            "/v1/chat",
            {"message": "Find a source about CSS."},
        )

        assert chat_status == 200
        assert "example.com/css" in response["message"]
        assert searcher.queries == [("CSS", 5)]
        assert len(backend.calls) == 2
        tool_messages = [
            message for message in backend.calls[1][0] if message.role == "tool"
        ]
        assert len(tool_messages) == 1
        assert "untrusted_web_content" in tool_messages[0].content
        assert "https://example.com/css" in tool_messages[0].content
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_api_reports_registered_search_as_disabled_without_opt_in() -> None:
    backend = ToolCallingBackend()
    searcher = SearchTestProvider()
    registry = CapabilityRegistry()
    register_gluons_tool(registry, searcher=searcher)
    server = create_server(
        backend,
        port=0,
        registry=registry,
        tool_policy=ExecutionPolicy(),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, capabilities = request_json(server, "GET", "/v1/capabilities")
        assert status == 200
        assert capabilities["web_search"] == "disabled_by_policy"
        chat_status, response = request_json(
            server, "POST", "/v1/chat", {"message": "Answer without searching."}
        )
        assert chat_status == 200
        assert response["message"] == "Plain response."
        assert searcher.queries == []
        assert backend.calls == []
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
