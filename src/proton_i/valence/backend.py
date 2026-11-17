"""Provider contracts and local Ollama transport for Valence."""

from __future__ import annotations

import ipaddress
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

MessageRole = Literal["system", "user", "assistant", "tool"]
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARS = 50_000
_MAX_TOOLS = 32
_MAX_TOOL_CALLS = 8
_MAX_TOOL_NAME_CHARS = 64


class BackendError(RuntimeError):
    """Raised when a model backend cannot return a valid assistant message."""


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """One untrusted tool request emitted by the language model."""

    name: str
    arguments: object

    def as_payload(self) -> dict[str, object]:
        """Return the Ollama-compatible assistant tool-call representation."""
        arguments = dict(self.arguments) if isinstance(self.arguments, Mapping) else {}
        return {"function": {"name": self.name, "arguments": arguments}}


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One assistant response, optionally containing structured tool requests."""

    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message, including provider-neutral tool-call/result metadata."""

    role: MessageRole
    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_name: str | None = None

    def __post_init__(self) -> None:
        """Validate message roles and the shape of tool-call messages."""
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("message role is not supported")
        if not isinstance(self.content, str) or len(self.content) > _MAX_MESSAGE_CHARS:
            raise ValueError("message content must be text within the size limit")
        if self.role == "assistant":
            if not self.content.strip() and not self.tool_calls:
                raise ValueError("assistant message must contain text or tool calls")
            if self.tool_name is not None:
                raise ValueError("assistant messages cannot contain a tool_name")
            if not isinstance(self.tool_calls, tuple) or any(
                not isinstance(call, ModelToolCall) for call in self.tool_calls
            ):
                raise TypeError("assistant tool_calls must be a tuple of ModelToolCall")
            if len(self.tool_calls) > _MAX_TOOL_CALLS:
                raise ValueError("assistant message contains too many tool calls")
        elif self.tool_calls:
            raise ValueError("only assistant messages can contain tool_calls")

        if self.role == "tool":
            if not isinstance(self.tool_name, str) or not self.tool_name:
                raise ValueError("tool result messages must identify their tool_name")
            if len(self.tool_name) > _MAX_TOOL_NAME_CHARS:
                raise ValueError("tool_name exceeds the size limit")
            if not self.content.strip():
                raise ValueError("tool result content must not be empty")
        elif self.tool_name is not None:
            raise ValueError("tool_name is only valid for tool result messages")

        if self.role in ("system", "user") and not self.content.strip():
            raise ValueError("system and user message content must not be empty")

    def as_payload(self) -> dict[str, object]:
        """Return the provider-neutral JSON representation of this message."""
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.as_payload() for call in self.tool_calls]
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        return payload


class ChatBackend(Protocol):
    """Protocol implemented by every Valence language-model backend."""

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return one assistant response for an ordered conversation."""


class ToolCallingChatBackend(Protocol):
    """Optional extension for providers that support structured function calls."""

    def complete_turn(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, object]],
    ) -> ChatTurn:
        """Return assistant text and/or structured tool-call requests."""


class OllamaChatBackend:
    """Call a local Ollama server through its non-streaming chat endpoint.

    The endpoint is restricted to localhost/loopback. Tool-call requests are
    parsed as untrusted data and are never executed by the backend itself.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 120.0,
    ) -> None:
        """Configure a local model, loopback host, and positive timeout."""
        if not isinstance(model, str) or not model.strip() or len(model) > 120:
            raise ValueError(
                "model must be a non-empty string of at most 120 characters"
            )
        if isinstance(timeout, bool):
            raise TypeError("timeout must be a positive finite number")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a positive finite number") from error
        if not math.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError("timeout must be a positive finite number")

        self.model = model.strip()
        self.host = self._validate_host(host)
        self.timeout = timeout_value

    @staticmethod
    def _validate_host(host: str) -> str:
        """Accept only a plain HTTP loopback origin with no custom path."""
        if not isinstance(host, str):
            raise TypeError("Ollama host must be a localhost HTTP URL")
        parsed = urlsplit(host.strip())
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama host must be a plain localhost HTTP URL")

        hostname = parsed.hostname.lower()
        is_loopback = hostname == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("Valence only permits a localhost or loopback Ollama host")

        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("Ollama host contains an invalid port") from error
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Ollama port must be between 1 and 65535")

        return urlunsplit((parsed.scheme, parsed.netloc, "/api/chat", "", ""))

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Send a plain chat request and return one assistant text response."""
        turn = self.complete_turn(messages, tools=())
        if turn.tool_calls:
            raise BackendError(
                "The model requested tools, but no tool executor was configured."
            )
        return turn.content

    def complete_turn(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, object]] = (),
    ) -> ChatTurn:
        """Send a chat request and parse assistant text/tool calls without executing."""
        if not messages or len(messages) > _MAX_MESSAGES:
            raise ValueError(
                f"messages must contain between 1 and {_MAX_MESSAGES} items"
            )
        if len(tools) > _MAX_TOOLS:
            raise ValueError(f"tools must contain no more than {_MAX_TOOLS} items")

        payload_messages: list[dict[str, object]] = []
        for message in messages:
            if not isinstance(message, ChatMessage):
                raise TypeError("messages must contain ChatMessage instances")
            payload_messages.append(message.as_payload())

        payload: dict[str, object] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = self._normalize_tools(tools)

        try:
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("chat request must contain valid JSON data") from error
        if len(request_body) > _MAX_REQUEST_BYTES:
            raise ValueError("chat request exceeds the 1 MB safety limit")

        request = Request(
            self.host,
            data=request_body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        response_body = self._fetch_response(request)
        result = self._decode_response(response_body)
        message_data = result.get("message")
        if not isinstance(message_data, Mapping):
            raise BackendError("The local model response did not contain a message.")
        if message_data.get("role") not in (None, "assistant"):
            raise BackendError("The local model returned an unexpected message role.")

        content = message_data.get("content", "")
        if not isinstance(content, str):
            raise BackendError("The local model returned invalid assistant content.")
        tool_calls = self._parse_tool_calls(message_data.get("tool_calls", []))
        if not content.strip() and not tool_calls:
            raise BackendError("The local model returned an empty assistant response.")
        if len(content) > _MAX_MESSAGE_CHARS:
            raise BackendError("The local model response exceeded the message size limit.")

        return ChatTurn(content=content.strip(), tool_calls=tool_calls)

    @staticmethod
    def _normalize_tools(
        tools: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Validate provider-neutral function schemas and remove app-only metadata."""
        normalized: list[dict[str, object]] = []
        names: set[str] = set()
        for schema in tools:
            if not isinstance(schema, Mapping):
                raise ValueError("tool schemas must be JSON objects")
            name = schema.get("name")
            description = schema.get("description")
            parameters = schema.get("parameters")
            if (
                not isinstance(name, str)
                or not name
                or len(name) > _MAX_TOOL_NAME_CHARS
                or not isinstance(description, str)
                or not description.strip()
                or not isinstance(parameters, Mapping)
            ):
                raise ValueError("tool schema contains invalid fields")
            if name in names:
                raise ValueError("tool schema names must be unique")
            names.add(name)
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description[:1_000],
                        "parameters": dict(parameters),
                    },
                }
            )
        return normalized

    @staticmethod
    def _parse_tool_calls(raw_calls: object) -> tuple[ModelToolCall, ...]:
        """Validate model-produced tool-call envelopes without trusting arguments."""
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, list):
            raise BackendError("The local model returned an invalid tool-call list.")
        if len(raw_calls) > _MAX_TOOL_CALLS:
            raise BackendError("The local model returned too many tool calls.")

        parsed: list[ModelToolCall] = []
        for item in raw_calls:
            if not isinstance(item, Mapping):
                raise BackendError("The local model returned an invalid tool call.")
            function = item.get("function")
            if not isinstance(function, Mapping):
                raise BackendError("The local model returned an invalid tool function.")
            name = function.get("name")
            if not isinstance(name, str) or not name or len(name) > _MAX_TOOL_NAME_CHARS:
                raise BackendError("The local model returned an invalid tool name.")
            arguments = function.get("arguments", {})
            try:
                encoded = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as error:
                raise BackendError(
                    "The local model returned invalid tool arguments."
                ) from error
            if len(encoded) > 16_384:
                raise BackendError("The local model tool arguments exceeded 16 KB.")
            parsed.append(ModelToolCall(name=name, arguments=arguments))
        return tuple(parsed)

    def _fetch_response(self, request: Request) -> bytes:
        """Fetch one bounded response from the loopback Ollama server."""
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            raise BackendError(
                f"Local model server returned HTTP {error.code}."
            ) from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise BackendError(
                "Could not reach the local model server. Check that Ollama is running."
            ) from error

        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise BackendError(
                "The local model response exceeded the 4 MB safety limit."
            )
        return response_body

    @staticmethod
    def _decode_response(response_body: bytes) -> Mapping[str, object]:
        """Decode and validate the top-level non-streaming chat response."""
        try:
            result = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError(
                "The local model server returned invalid JSON."
            ) from error
        if not isinstance(result, Mapping):
            raise BackendError(
                "The local model server returned an unexpected response."
            )
        return result
