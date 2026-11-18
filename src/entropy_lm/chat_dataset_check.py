"""Integrity and split-leakage checks for prepared Entropy chat datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from entropy_lm.chat_data import IGNORE_INDEX


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object")
    return value


def check_prepared_chat_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate prepared split files, manifest counts, token labels, and leakage.

    This check is read-only. It reports aggregate counts and never emits
    conversation text or token sequences.
    """
    directory = Path(dataset_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    tokenizer_path = directory / "tokenizer.json"
    if not manifest_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            "prepared dataset requires manifest.json and tokenizer.json"
        )

    manifest = _read_json(manifest_path, "manifest.json")
    tokenizer = _read_json(tokenizer_path, "tokenizer.json")
    fingerprint = hashlib.sha256(
        json.dumps(tokenizer, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("tokenizer fingerprint does not match manifest")
    if manifest.get("tokenizer") != tokenizer:
        raise ValueError("tokenizer metadata does not match manifest")
    vocab_size = tokenizer.get("vocab_size")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 3
    ):
        raise ValueError("tokenizer vocabulary size is invalid")

    documents = manifest.get("documents")
    declared_splits = manifest.get("splits")
    if not isinstance(documents, list) or not isinstance(declared_splits, dict):
        raise ValueError("manifest documents or splits field is malformed")

    document_map: dict[str, tuple[str, str]] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise TypeError(f"manifest document {index} must be an object")
        digest = document.get("sha256")
        source = document.get("source")
        split = document.get("split")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(source, str)
            or not source
            or split not in ("train", "validation", "test")
        ):
            raise ValueError(f"manifest document {index} has invalid metadata")
        if digest in document_map:
            raise ValueError("manifest contains duplicate conversation hashes")
        document_map[digest] = (source, split)

    observed: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"prepared dataset is missing {path.name}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"{path.name} is not valid UTF-8") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"{path.name}:{line_number} must be an object")
            if record.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError(
                    f"{path.name}:{line_number} tokenizer fingerprint mismatch"
                )
            digest = record.get("sha256")
            source = record.get("source")
            if digest not in document_map:
                raise ValueError(f"{path.name}:{line_number} is absent from manifest")
            if document_map[digest] != (source, split):
                raise ValueError(f"{path.name}:{line_number} split/source mismatch")
            if digest in observed:
                raise ValueError(
                    "conversation hash occurs more than once across splits"
                )

            input_ids = record.get("input_ids")
            labels = record.get("labels")
            if (
                not isinstance(input_ids, list)
                or not isinstance(labels, list)
                or len(input_ids) < 2
                or len(input_ids) != len(labels)
            ):
                raise ValueError(
                    f"{path.name}:{line_number} token arrays are malformed"
                )
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 0 <= token < vocab_size
                for token in input_ids
            ):
                raise ValueError(
                    f"{path.name}:{line_number} input token is out of range"
                )
            if any(
                isinstance(label, bool)
                or not isinstance(label, int)
                or (label != IGNORE_INDEX and not 0 <= label < vocab_size)
                for label in labels
            ):
                raise ValueError(f"{path.name}:{line_number} label is out of range")
            if any(
                label != IGNORE_INDEX and label != token
                for token, label in zip(input_ids, labels)
            ):
                raise ValueError(f"{path.name}:{line_number} labels are misaligned")
            supervised = sum(label != IGNORE_INDEX for label in labels)
            if supervised == 0:
                raise ValueError(f"{path.name}:{line_number} has no assistant targets")

            observed[digest] = {"source": source, "split": split}
            counts[split] += 1
            token_counts[split] += len(input_ids)
            target_counts[split] += supervised

    if set(observed) != set(document_map):
        missing = len(set(document_map) - set(observed))
        raise ValueError(
            f"{missing} manifest conversations are missing from split files"
        )
    for split in ("train", "validation", "test"):
        declared = declared_splits.get(split)
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise ValueError(f"manifest split count for {split} is invalid")
        if counts[split] != declared:
            raise ValueError(f"{split} count does not match manifest")

    if manifest.get("conversation_count") != len(document_map):
        raise ValueError("manifest conversation_count does not match documents")

    return {
        "valid": True,
        "conversation_count": len(observed),
        "splits": {split: counts[split] for split in ("train", "validation", "test")},
        "input_tokens": {
            split: token_counts[split] for split in ("train", "validation", "test")
        },
        "assistant_targets": {
            split: target_counts[split] for split in ("train", "validation", "test")
        },
        "tokenizer_fingerprint": fingerprint,
        "split_hashes_disjoint": True,
        "content_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check prepared Entropy chat dataset integrity and split leakage."
    )
    parser.add_argument("dataset", type=Path, help="prepared chat dataset directory")
    args = parser.parse_args()
    print(json.dumps(check_prepared_chat_dataset(args.dataset), indent=2))


if __name__ == "__main__":
    main()
