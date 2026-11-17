"""Valence: Proton-I's general-purpose conversational assistant."""

from proton_i.valence.backend import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    BackendError,
    ChatMessage,
    OllamaChatBackend,
)
from proton_i.valence.engine import ValenceEngine, ValenceError

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "BackendError",
    "ChatMessage",
    "OllamaChatBackend",
    "ValenceEngine",
    "ValenceError",
]
