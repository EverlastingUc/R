"""Interactive command-line interface for Valence."""

from __future__ import annotations

import argparse
import os
import sys

from proton_i.valence.backend import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    OllamaChatBackend,
)
from proton_i.valence.engine import ValenceEngine, ValenceError


def main() -> int:
    """Start a local Valence conversation until the user exits."""
    parser = argparse.ArgumentParser(
        prog="valence",
        description="Valence: Proton-I's local conversational assistant.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VALENCE_MODEL", DEFAULT_MODEL),
        help="installed Ollama model name",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("VALENCE_OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help="local Ollama server URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="model response timeout in seconds",
    )
    arguments = parser.parse_args()

    try:
        backend = OllamaChatBackend(
            model=arguments.model,
            host=arguments.host,
            timeout=arguments.timeout,
        )
    except ValueError as error:
        parser.error(str(error))

    assistant = ValenceEngine(backend)
    print("Valence is ready. Type /reset to clear the conversation or /exit to quit.")

    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nValence session ended.")
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            print("Valence session ended.")
            return 0
        if prompt.lower() == "/reset":
            assistant.reset()
            print("Valence > Conversation cleared.")
            continue

        try:
            answer = assistant.ask(prompt)
        except ValenceError as error:
            print(f"Valence error: {error}", file=sys.stderr)
            continue
        print(f"Valence > {answer}")


if __name__ == "__main__":
    raise SystemExit(main())
