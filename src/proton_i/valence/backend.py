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

MessageRole = Literal["system", "user", "assistant"]
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARS = 50_000


class BackendError(RuntimeError):
    """Raised when a model backend cannot return a valid assistant message."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One validated chat message with a role and plain-text content."""

    role: MessageRole
    content: str

    def as_payload(self) -> dict[str, str]:
        """Return the provider-neutral JSON representation of this message."""
        return {"role": self.role, "content": self.content}


class ChatBackend(Protocol):
    """Protocol implemented by every Valence language-model backend."""

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return one assistant response for an ordered conversation."""


class OllamaChatBackend:
    """Call a local Ollama server through its non-streaming chat endpoint.

    The endpoint is restricted to localhost/loopback so the first Valence
    release cannot send prompts to an arbitrary remote host by configuration.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 120.0,
    ) -> None:
        """Configure a local model, loopback host, and positive timeout."""
        if not isinstance(model, str) or not model.strip() or len(model) > 120:
            raise ValueError("model must be a non-empty string of at most 120 characters")
        if isinstance(timeout, bool):
            raise ValueError("timeout must be a positive finite number")
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
            raise ValueError("Ollama host must be a localhost HTTP URL")
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
        """Send bounded conversation messages and return assistant text."""
        if not messages or len(messages) > _MAX_MESSAGES:
            raise ValueError(f"messages must contain between 1 and {_MAX_MESSAGES} items")

        payload_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, ChatMessage):
                raise TypeError("messages must contain ChatMessage instances")
            if message.role not in ("system", "user", "assistant"):
                raise ValueError("message role is not supported")
            if not message.content or len(message.content) > _MAX_MESSAGE_CHARS:
                raise ValueError("message content is empty or exceeds the size limit")
            payload_messages.append(message.as_payload())

        request_body = json.dumps(
            {
                "model": self.model,
                "messages": payload_messages,
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            self.host,
            data=request_body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            raise BackendError(f"Local model server returned HTTP {error.code}.") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise BackendError(
                "Could not reach the local model server. Check that Ollama is running."
            ) from error

        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise BackendError("The local model response exceeded the 4 MB safety limit.")

        try:
            result = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError("The local model server returned invalid JSON.") from error

        if not isinstance(result, Mapping):
            raise BackendError("The local model server returned an unexpected response.")
        message_data = result.get("message")
        if not isinstance(message_data, Mapping):
            raise BackendError("The local model response did not contain a message.")
        content = message_data.get("content")
        if not isinstance(content, str) or not content.strip():
            raise BackendError("The local model returned an empty assistant response.")
        return content.strip()
