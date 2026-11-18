"""Conversation dataset preparation with assistant-only supervised labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.tokenizer import ByteTokenizer

IGNORE_INDEX = -100
TOKENIZER_TYPES = ("byte", "bpe")
ALLOWED_ROLES = ("system", "user", "assistant")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _discover_files(source: Path, output: Path) -> list[Path]:
    source = source.resolve()
    output = output.resolve()
    if source.is_file():
        if source.suffix.lower() != ".jsonl":
            raise ValueError("conversation input file must have a .jsonl extension")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"conversation input does not exist: {source}")
    files: list[Path] = []
    for path in sorted(source.rglob("*.jsonl"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"symbolic links are not accepted: {path}")
        resolved = path.resolve()
        if output == resolved or output in resolved.parents:
            continue
        files.append(path)
    return files


def _validate_messages(
    value: Any, *, source: str, line_number: int
) -> list[dict[str, str]]:
    location = f"{source}:{line_number}"
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{location}: messages must contain at least two entries")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise TypeError(f"{location}: message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{location}: unsupported role in message {index}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{location}: message {index} content must be non-empty")
        messages.append({"role": role, "content": content})
    roles = {message["role"] for message in messages}
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(
            f"{location}: each conversation needs user and assistant messages"
        )
    return messages


def _read_conversations(
    source: Path, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    conversations: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    files = _discover_files(source, output)
    if not files:
        raise ValueError("no UTF-8 .jsonl conversation files were found")

    for path in files:
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"conversation file is not valid UTF-8: {path}") from error
        relative = (
            path.name if source.is_file() else path.relative_to(source).as_posix()
        )
        for line_number, line in enumerate(decoded.splitlines(), 1):
            if not line.strip():
                skipped.append(
                    {"source": relative, "line": str(line_number), "reason": "blank"}
                )
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid conversation JSON at {relative}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"conversation record must be an object at {relative}:{line_number}"
                )
            messages = _validate_messages(
                record.get("messages"), source=relative, line_number=line_number
            )
            canonical = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest = _sha256(canonical)
            source_ref = f"{relative}:{line_number}"
            if digest in by_hash:
                by_hash[digest]["duplicate_sources"].append(source_ref)
                continue
            conversation = {
                "messages": messages,
                "source": source_ref,
                "duplicate_sources": [],
                "sha256": digest,
            }
            by_hash[digest] = conversation
            conversations.append(conversation)

    if not conversations:
        raise ValueError("no valid conversations were found")
    return conversations, skipped


def _split_conversations(
    conversations: list[dict[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation and test fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")
    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    count = len(indices)
    test_count = int(count * test_fraction)
    validation_count = int(count * validation_fraction)
    if test_fraction > 0 and count >= 3:
        test_count = max(1, test_count)
    if validation_fraction > 0 and count - test_count >= 2:
        validation_count = max(1, validation_count)
    while test_count + validation_count >= count:
        if validation_count:
            validation_count -= 1
        elif test_count:
            test_count -= 1
        else:
            break
    test_indices = indices[:test_count]
    validation_indices = indices[test_count : test_count + validation_count]
    train_indices = indices[test_count + validation_count :]
    return {
        "train": [conversations[index] for index in train_indices],
        "validation": [conversations[index] for index in validation_indices],
        "test": [conversations[index] for index in test_indices],
    }


def _format_conversation(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|{message['role']}|>\n{message['content']}\n" for message in messages
    )


def _tokenizer_metadata(tokenizer: ByteTokenizer | BPETokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, BPETokenizer):
        return {
            "type": "byte-bpe",
            "vocab_size": tokenizer.vocab_size,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    return {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size}


def _encode_conversation_with_turns(
    messages: list[dict[str, str]], tokenizer: ByteTokenizer | BPETokenizer
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Encode a conversation and retain token spans for each assistant turn."""
    input_ids = [tokenizer.bos_id]
    labels = [IGNORE_INDEX]
    assistant_turns: list[dict[str, int]] = []
    for message in messages:
        header_ids = tokenizer.encode(f"<|{message['role']}|>\n")
        content_ids = tokenizer.encode(message["content"] + "\n")
        input_ids.extend(header_ids)
        labels.extend([IGNORE_INDEX] * len(header_ids))
        input_ids.extend(content_ids)
        if message["role"] == "assistant":
            target_start = len(labels)
            labels.extend(content_ids)
            input_ids.append(tokenizer.eos_id)
            labels.append(tokenizer.eos_id)
            assistant_turns.append(
                {
                    "turn_index": len(assistant_turns),
                    "target_start": target_start,
                    "target_end": len(labels),
                }
            )
        else:
            labels.extend([IGNORE_INDEX] * len(content_ids))
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("conversation has no assistant targets")
    return input_ids, labels, assistant_turns


def _encode_conversation(
    messages: list[dict[str, str]], tokenizer: ByteTokenizer | BPETokenizer
) -> tuple[list[int], list[int]]:
    """Backward-compatible conversation encoder without turn metadata."""
    input_ids, labels, _ = _encode_conversation_with_turns(messages, tokenizer)
    return input_ids, labels


def prepare_chat_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    tokenizer_type: str = "bpe",
    vocab_size: int = 1000,
) -> dict[str, Any]:
    """Prepare leakage-aware conversation splits and assistant-only labels."""
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"tokenizer_type must be one of {TOKENIZER_TYPES}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    conversations, skipped = _read_conversations(source, output)
    splits = _split_conversations(
        conversations,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not splits["train"]:
        raise ValueError("conversation split produced no training conversations")

    if tokenizer_type == "bpe":
        tokenizer: ByteTokenizer | BPETokenizer = BPETokenizer.train(
            (_format_conversation(item["messages"]) for item in splits["train"]),
            vocab_size=vocab_size,
        )
    else:
        tokenizer = ByteTokenizer()

    metadata = _tokenizer_metadata(tokenizer)
    fingerprint = _sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(tokenizer, BPETokenizer):
        tokenizer.save(output / "tokenizer.json")
    else:
        (output / "tokenizer.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    split_counts: dict[str, int] = {}
    input_token_counts: dict[str, int] = {}
    assistant_target_counts: dict[str, int] = {}
    for split_name, items in splits.items():
        split_counts[split_name] = len(items)
        token_path = output / f"{split_name}.jsonl"
        lines: list[str] = []
        input_count = 0
        target_count = 0
        for item in items:
            input_ids, labels, assistant_turns = _encode_conversation_with_turns(
                item["messages"], tokenizer
            )
            input_count += len(input_ids)
            target_count += sum(label != IGNORE_INDEX for label in labels)
            lines.append(
                json.dumps(
                    {
                        "source": item["source"],
                        "sha256": item["sha256"],
                        "tokenizer_fingerprint": fingerprint,
                        "message_count": len(item["messages"]),
                        "assistant_turns": assistant_turns,
                        "input_ids": input_ids,
                        "labels": labels,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        token_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        input_token_counts[split_name] = input_count
        assistant_target_counts[split_name] = target_count

    manifest = {
        "format_version": 1,
        "input": str(source),
        "seed": seed,
        "split_fractions": {
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "tokenizer": metadata,
        "tokenizer_fingerprint": fingerprint,
        "conversation_count": len(conversations),
        "duplicate_conversation_count": sum(
            len(item["duplicate_sources"]) for item in conversations
        ),
        "skipped_lines": skipped,
        "splits": split_counts,
        "input_token_counts": input_token_counts,
        "assistant_target_counts": assistant_target_counts,
        "documents": [
            {
                "source": item["source"],
                "duplicate_sources": item["duplicate_sources"],
                "sha256": item["sha256"],
                "split": split_name,
            }
            for split_name, items in splits.items()
            for item in items
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare conversation data for Entropy supervised fine-tuning."
    )
    parser.add_argument(
        "input", type=Path, help="JSONL file or directory of JSONL files"
    )
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=TOKENIZER_TYPES, default="bpe")
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()
    manifest = prepare_chat_dataset(
        args.input,
        args.output,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
    )
    print(
        json.dumps(
            {
                "conversations": manifest["conversation_count"],
                "splits": manifest["splits"],
                "assistant_targets": manifest["assistant_target_counts"],
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
