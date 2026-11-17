"""Conversation orchestration for Proton-I's Valence stage."""

from __future__ import annotations

from collections.abc import Sequence

from proton_i.valence.backend import BackendError, ChatBackend, ChatMessage

DEFAULT_SYSTEM_PROMPT = (
    "You are Valence, a helpful general-purpose AI assistant. "
    "Be accurate, clear, and honest about uncertainty. "
    "Ask a concise follow-up question when essential information is missing. "
    "Do not claim to have browsed websites, created files, run code, or used tools "
    "unless the application actually supplied that result."
)
DEFAULT_MAX_HISTORY_MESSAGES = 24
MAX_INPUT_CHARS = 12_000
MAX_OUTPUT_CHARS = 50_000


class ValenceError(RuntimeError):
    """Raised when Valence cannot safely process a conversation turn."""


class ValenceEngine:
    """Manage one conversation and delegate generation to a model backend.

    History is held in memory for this engine instance only. Persistence,
    retrieval, tool use, and multi-user sessions are separate future layers.
    """

    def __init__(
        self,
        backend: ChatBackend,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        """Create a conversation with a backend and bounded message history."""
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide a complete(messages) method")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if len(system_prompt) > 4_000:
            raise ValueError("system_prompt must be 4,000 characters or fewer")
        if (
            not isinstance(max_history_messages, int)
            or isinstance(max_history_messages, bool)
            or max_history_messages < 2
        ):
            raise ValueError("max_history_messages must be an integer of at least 2")

        self.backend = backend
        self.system_message = ChatMessage("system", system_prompt.strip())
        self.max_history_messages = max_history_messages
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return a read-only snapshot of recent user and assistant messages."""
        return tuple(self._history)

    def ask(self, prompt: str) -> str:
        """Generate a reply and commit the turn only after success."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt must not be empty")
        if len(cleaned_prompt) > MAX_INPUT_CHARS:
            raise ValueError(f"prompt must be {MAX_INPUT_CHARS:,} characters or fewer")

        previous_messages = self._trim_history(self._history)
        user_message = ChatMessage("user", cleaned_prompt)
        request_messages = [self.system_message, *previous_messages, user_message]

        try:
            response = self.backend.complete(request_messages)
        except BackendError as error:
            raise ValenceError(str(error)) from error
        except Exception as error:
            raise ValenceError("Valence could not complete this message.") from error

        if not isinstance(response, str) or not response.strip():
            raise ValenceError("The model backend returned an empty response.")
        cleaned_response = response.strip()
        if len(cleaned_response) > MAX_OUTPUT_CHARS:
            raise ValenceError("The model response exceeded the output size limit.")

        assistant_message = ChatMessage("assistant", cleaned_response)
        self._history = self._trim_history(
            [*previous_messages, user_message, assistant_message]
        )
        return cleaned_response

    def reset(self) -> None:
        """Clear the current conversation without changing the backend."""
        self._history.clear()

    def _trim_history(self, messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        """Keep the latest complete user/assistant turn pairs within the limit."""
        recent = list(messages[-self.max_history_messages :])
        if recent and recent[0].role == "assistant":
            recent.pop(0)
        return recent
