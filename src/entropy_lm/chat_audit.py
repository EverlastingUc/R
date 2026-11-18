"""Non-destructive quality audit for raw Entropy conversation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from entropy_lm.chat_data import _discover_files, _validate_messages


def audit_chat_dataset(
    input_path: str | Path,
    *,
    report_path: str | Path | None = None,
    short_response_chars: int = 20,
    long_response_chars: int = 8000,
) -> dict[str, Any]:
    """Audit JSONL conversations without changing or printing their contents.

    Invalid records are counted and described by source/line and reason. The
    audit intentionally reports heuristics rather than assigning a subjective
    quality score or automatically deleting examples.
    """
    if (
        isinstance(short_response_chars, bool)
        or not isinstance(short_response_chars, int)
        or short_response_chars < 0
    ):
        raise ValueError("short_response_chars must be a non-negative integer")
    if (
        isinstance(long_response_chars, bool)
        or not isinstance(long_response_chars, int)
        or long_response_chars <= short_response_chars
    ):
        raise ValueError(
            "long_response_chars must be greater than short_response_chars"
        )

    source = Path(input_path).expanduser().resolve()
    files = _discover_files(source, Path.cwd() / "__entropy_audit_output__")
    if not files:
        raise ValueError("no UTF-8 .jsonl conversation files were found")

    role_counts: Counter[str] = Counter()
    seen: dict[str, str] = {}
    duplicate_count = 0
    valid_count = 0
    blank_count = 0
    invalid_count = 0
    short_responses = 0
    long_responses = 0
    repeated_assistant_messages = 0
    conversations_without_final_assistant = 0
    consecutive_same_role = 0
    message_counts: list[int] = []
    response_lengths: list[int] = []
    issues: list[dict[str, str]] = []
    file_count = len(files)
    line_count = 0

    for path in files:
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            invalid_count += 1
            issues.append(
                {
                    "source": (
                        path.name
                        if source.is_file()
                        else path.relative_to(source).as_posix()
                    ),
                    "line": "0",
                    "reason": "file is not valid UTF-8",
                }
            )
            continue
        relative = (
            path.name if source.is_file() else path.relative_to(source).as_posix()
        )
        for line_number, line in enumerate(decoded.splitlines(), 1):
            line_count += 1
            if not line.strip():
                blank_count += 1
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be a JSON object")
                messages = _validate_messages(
                    record.get("messages"),
                    source=relative,
                    line_number=line_number,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                invalid_count += 1
                issues.append(
                    {
                        "source": relative,
                        "line": str(line_number),
                        "reason": str(error).split(": ", 1)[-1][:180],
                    }
                )
                continue

            canonical = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            if digest in seen:
                duplicate_count += 1
                issues.append(
                    {
                        "source": relative,
                        "line": str(line_number),
                        "reason": f"exact duplicate of {seen[digest]}",
                    }
                )
                continue
            seen[digest] = f"{relative}:{line_number}"
            valid_count += 1
            message_counts.append(len(messages))
            role_counts.update(message["role"] for message in messages)

            assistant_texts: set[str] = set()
            previous_role: str | None = None
            for message in messages:
                role = message["role"]
                content = message["content"].strip()
                if role == "assistant":
                    length = len(content)
                    response_lengths.append(length)
                    if length < short_response_chars:
                        short_responses += 1
                    if length > long_response_chars:
                        long_responses += 1
                    normalized = " ".join(content.casefold().split())
                    if normalized in assistant_texts:
                        repeated_assistant_messages += 1
                    assistant_texts.add(normalized)
                if previous_role == role:
                    consecutive_same_role += 1
                previous_role = role
            if messages[-1]["role"] != "assistant":
                conversations_without_final_assistant += 1

    if valid_count == 0:
        issues.append(
            {
                "source": str(source),
                "line": "0",
                "reason": "no valid unique conversations were found",
            }
        )

    warnings: list[str] = []
    if duplicate_count:
        warnings.append("Exact duplicate conversations were found.")
    if invalid_count:
        warnings.append("Malformed or schema-invalid records need review.")
    if short_responses:
        warnings.append("Some assistant responses are unusually short.")
    if long_responses:
        warnings.append("Some assistant responses are unusually long.")
    if repeated_assistant_messages:
        warnings.append("Some conversations repeat identical assistant messages.")
    if conversations_without_final_assistant:
        warnings.append("Some conversations do not end with an assistant response.")
    if consecutive_same_role:
        warnings.append(
            "Some conversations contain consecutive messages with the same role."
        )

    report: dict[str, Any] = {
        "format_version": 1,
        "source": str(source),
        "files_scanned": file_count,
        "lines_scanned": line_count,
        "valid_unique_conversations": valid_count,
        "exact_duplicate_conversations": duplicate_count,
        "blank_lines": blank_count,
        "invalid_records_or_files": invalid_count,
        "message_counts_by_role": dict(sorted(role_counts.items())),
        "conversation_message_count": {
            "minimum": min(message_counts) if message_counts else 0,
            "maximum": max(message_counts) if message_counts else 0,
            "mean": round(mean(message_counts), 2) if message_counts else 0,
        },
        "assistant_response_chars": {
            "count": len(response_lengths),
            "minimum": min(response_lengths) if response_lengths else 0,
            "maximum": max(response_lengths) if response_lengths else 0,
            "mean": round(mean(response_lengths), 2) if response_lengths else 0,
            "below_short_threshold": short_responses,
            "above_long_threshold": long_responses,
            "short_threshold_chars": short_response_chars,
            "long_threshold_chars": long_response_chars,
        },
        "repeated_assistant_messages": repeated_assistant_messages,
        "conversations_without_final_assistant": conversations_without_final_assistant,
        "consecutive_same_role_transitions": consecutive_same_role,
        "warnings": warnings,
        "issues": issues,
        "content_included": False,
    }
    if report_path is not None:
        destination = Path(report_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw Entropy conversation JSONL without modifying it."
    )
    parser.add_argument("input", type=Path, help="JSONL file or directory")
    parser.add_argument("--report", type=Path, help="optional JSON report destination")
    parser.add_argument("--short-response-chars", type=int, default=20)
    parser.add_argument("--long-response-chars", type=int, default=8000)
    args = parser.parse_args()
    report = audit_chat_dataset(
        args.input,
        report_path=args.report,
        short_response_chars=args.short_response_chars,
        long_response_chars=args.long_response_chars,
    )
    summary = {
        key: report[key]
        for key in (
            "files_scanned",
            "lines_scanned",
            "valid_unique_conversations",
            "exact_duplicate_conversations",
            "blank_lines",
            "invalid_records_or_files",
            "message_counts_by_role",
            "assistant_response_chars",
            "warnings",
        )
    }
    if args.report is not None:
        summary["report"] = str(args.report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
