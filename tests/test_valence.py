"""Tests for Valence conversation handling and local model transport."""

from __future__ import annotations

import json
from io import BytesIO
from urllib.error import URLError

import pytest

from proton_i.valence import (
    BackendError,
    ChatMessage,
    OllamaChatBackend,
    ValenceEngine,
    ValenceError,
)


class FakeBackend:
    """Deterministic backend used to test conversation orchestration."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["Hello from Valence."])
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise BackendError("simulated backend failure")
        return self.replies.pop(0)


def test_engine_sends_system_prompt_and_user_message() -> None:
    backend = FakeBackend(["Hello."])
    engine = ValenceEngine(backend)

    assert engine.ask("  Hi Valence  ") == "Hello."
    assert [message.role for message in backend.calls[0]] == ["system", "user"]
    assert backend.calls[0][-1].content == "Hi Valence"
    assert engine.history == (
        ChatMessage("user", "Hi Valence"),
        ChatMessage("assistant", "Hello."),
    )


def test_engine_preserves_context_between_turns() -> None:
    backend = FakeBackend(["The capital is Paris.", "It is in France."])
    engine = ValenceEngine(backend)

    engine.ask("What is the capital of France?")
    answer = engine.ask("Which country is that in?")

    assert answer == "It is in France."
    second_call = backend.calls[1]
    assert second_call[1:] == [
        ChatMessage("user", "What is the capital of France?"),
        ChatMessage("assistant", "The capital is Paris."),
        ChatMessage("user", "Which country is that in?"),
    ]


def test_engine_trims_history_to_configured_limit() -> None:
    backend = FakeBackend(["A1", "A2", "A3"])
    engine = ValenceEngine(backend, max_history_messages=2)

    engine.ask("Q1")
    engine.ask("Q2")
    engine.ask("Q3")

    assert len(engine.history) == 2
    assert engine.history == (ChatMessage("user", "Q3"), ChatMessage("assistant", "A3"))
    assert backend.calls[2][1:] == [
        ChatMessage("user", "Q2"),
        ChatMessage("assistant", "A2"),
        ChatMessage("user", "Q3"),
    ]


def test_engine_reset_clears_history_but_keeps_backend() -> None:
    backend = FakeBackend(["First answer.", "Second answer."])
    engine = ValenceEngine(backend)

    engine.ask("First question")
    engine.reset()
    assert engine.history == ()
    engine.ask("Second question")

    assert [message.role for message in backend.calls[1]] == ["system", "user"]


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_engine_rejects_blank_prompts(prompt: str) -> None:
    engine = ValenceEngine(FakeBackend())
    with pytest.raises(ValueError, match="must not be empty"):
        engine.ask(prompt)


def test_engine_rejects_oversized_prompts() -> None:
    engine = ValenceEngine(FakeBackend())
    with pytest.raises(ValueError, match="12,000"):
        engine.ask("x" * 12_001)


def test_failed_generation_does_not_mutate_conversation() -> None:
    backend = FakeBackend([])
    engine = ValenceEngine(backend)

    with pytest.raises(ValenceError, match="simulated backend failure"):
        engine.ask("This should not be stored.")

    assert engine.history == ()


def test_engine_rejects_empty_model_response_without_saving_turn() -> None:
    engine = ValenceEngine(FakeBackend(["   "]))
    with pytest.raises(ValenceError, match="empty response"):
        engine.ask("Hello")
    assert engine.history == ()


def test_engine_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        ValenceEngine(FakeBackend(), system_prompt=" ")
    with pytest.raises(ValueError, match="at least 2"):
        ValenceEngine(FakeBackend(), max_history_messages=1)


class FakeResponse(BytesIO):
    """Minimal context-managed HTTP response."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_ollama_backend_sends_non_streaming_chat_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            json.dumps(
                {"message": {"role": "assistant", "content": "Hello locally."}}
            ).encode("utf-8")
        )

    monkeypatch.setattr("proton_i.valence.backend.urlopen", fake_urlopen)
    backend = OllamaChatBackend(model="qwen2.5:1.5b-instruct", timeout=8.0)
    answer = backend.complete([ChatMessage("user", "Hello")])

    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"  # type: ignore[attr-defined]
    payload = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
    assert payload["model"] == "qwen2.5:1.5b-instruct"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["timeout"] == 8.0
    assert answer == "Hello locally."


def test_ollama_backend_rejects_non_loopback_hosts() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaChatBackend(host="http://example.com:11434")
    with pytest.raises(ValueError, match="loopback"):
        OllamaChatBackend(host="http://192.168.1.10:11434")


@pytest.mark.parametrize(
    "host",
    [
        "https://localhost:11434",
        "http://localhost:11434/custom",
        "http://user@localhost:11434",
        "http://localhost:99999",
    ],
)
def test_ollama_backend_rejects_malformed_hosts(host: str) -> None:
    with pytest.raises(ValueError):
        OllamaChatBackend(host=host)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_ollama_backend_rejects_invalid_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        OllamaChatBackend(timeout=timeout)


def test_ollama_backend_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr("proton_i.valence.backend.urlopen", fail_urlopen)
    backend = OllamaChatBackend()

    with pytest.raises(BackendError, match="Could not reach"):
        backend.complete([ChatMessage("user", "Hello")])


def test_ollama_backend_rejects_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "proton_i.valence.backend.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"{invalid"),
    )
    with pytest.raises(BackendError, match="invalid JSON"):
        OllamaChatBackend().complete([ChatMessage("user", "Hello")])


def test_ollama_backend_rejects_empty_response_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "proton_i.valence.backend.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b'{"message":{"role":"assistant","content":"  "}}'
        ),
    )
    with pytest.raises(BackendError, match="empty assistant response"):
        OllamaChatBackend().complete([ChatMessage("user", "Hello")])


def test_ollama_backend_rejects_invalid_message_count() -> None:
    backend = OllamaChatBackend()
    with pytest.raises(ValueError, match="between 1"):
        backend.complete([])
