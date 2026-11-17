"""Command-line entry point for Gluons."""

from __future__ import annotations

import argparse
import sys

from proton_i.gluons.assistant import GluonsAssistant


def main() -> int:
    """Read one question and print one plain-text answer."""
    parser = argparse.ArgumentParser(
        prog="gluons",
        description="Gluons: Proton-I's basic web-assisted assistant.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="a short question to answer (omit to read one line from stdin)",
    )
    arguments = parser.parse_args()
    question = " ".join(arguments.question).strip()
    if not question and not sys.stdin.isatty():
        question = sys.stdin.readline().strip()
    if not question:
        parser.error("provide a question")

    try:
        print(GluonsAssistant().answer(question))
    except (TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
