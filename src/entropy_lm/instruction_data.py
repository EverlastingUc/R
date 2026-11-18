"""Normalize common instruction-tuning JSONL formats into chat conversations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from entropy_lm.chat_data import _validate_messages


def _nonempty_string(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value


def normalize_instruction_record(record: Any, *, location: str) -> list[dict[str, str]]:
    """Convert one supported instruction record without guessing its schema."""
    if not isinstance(record, dict):
        raise TypeError(f"{location}: record must be a JSON object")

    if "messages" in record:
        messages = record["messages"]
    elif "prompt" in record or "completion" in record:
        if "prompt" not in record or "completion" not in record:
            raise ValueError(f"{location}: prompt and completion must appear together")
        messages = [
            {"role": "user", "content": _nonempty_string(record["prompt"], "prompt", location)},
            {"role": "assistant", "content": _nonempty_string(record["completion"], "completion", location)},
        ]
    elif "instruction" in record or "output" in record:
        if "instruction" not in record or "output" not in record:
            raise ValueError(f"{location}: instruction and output must appear together")
        instruction = _nonempty_string(record["instruction"], "instruction", location)
        user_input = record.get("input", "")
        if not isinstance(user_input, str):
            raise TypeError(f"{location}: input must be a string when supplied")
        user_content = instruction if not user_input.strip() else f"{instruction}\n\n{user_input}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _nonempty_string(record["output"], "output", location)},
        ]
    else:
        raise ValueError(
            f"{location}: expected messages, prompt/completion, or instruction/output fields"
        )

    normalized: list[dict[str, str]] = []
    system = record.get("system")
    if system is not None:
        normalized.append(
            {"role": "system", "content": _nonempty_string(system, "system", location)}
        )
    validated = _validate_messages(messages, source=location, line_number=1)
    if system is not None and validated[0]["role"] == "system":
        raise ValueError(f"{location}: system is supplied both separately and in messages")
    normalized.extend(validated)
    return normalized


def normalize_instruction_dataset(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Write normalized conversations atomically; malformed records fail closed."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("input and output paths must be different")
    if not source.is_file() or source.suffix.lower() != ".jsonl":
        raise ValueError("input must be an existing .jsonl file")
    try:
        decoded = source.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("instruction dataset must be valid UTF-8") from error

    output_lines: list[str] = []
    blank_lines = 0
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            blank_lines += 1
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {source.name}:{line_number}") from error
        messages = normalize_instruction_record(
            record, location=f"{source.name}:{line_number}"
        )
        output_lines.append(
            json.dumps({"messages": messages}, ensure_ascii=False, sort_keys=True)
        )

    if not output_lines:
        raise ValueError("instruction dataset contains no usable records")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write("\n".join(output_lines) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "records_written": len(output_lines),
        "blank_lines_skipped": blank_lines,
        "output": str(destination),
        "content_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize instruction JSONL into Entropy chat JSONL."
    )
    parser.add_argument("input", type=Path, help="source instruction .jsonl file")
    parser.add_argument("output", type=Path, help="normalized conversation .jsonl file")
    args = parser.parse_args()
    report = normalize_instruction_dataset(args.input, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
