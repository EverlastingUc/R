"""Deterministic, leakage-aware text corpus preparation for Entropy LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.tokenizer import ByteTokenizer

TOKENIZER_TYPES = ("byte", "bpe")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _discover_files(source: Path, output: Path) -> list[Path]:
    source = source.resolve()
    output = output.resolve()
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError("input file must have a .txt extension")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"corpus input does not exist: {source}")
    files = []
    for path in sorted(source.rglob("*.txt"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"symbolic links are not accepted: {path}")
        resolved = path.resolve()
        if output == resolved or output in resolved.parents:
            continue
        files.append(path)
    return files


def _read_documents(
    source: Path, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    documents: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    for path in _discover_files(source, output):
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"corpus file is not valid UTF-8: {path}") from error
        relative = (\n            path.name if source.is_file() else path.relative_to(source).as_posix()\n        )
        digest = _sha256(raw)
        if not text.strip():
            skipped.append({"path": relative, "reason": "empty_or_whitespace"})
            continue
        if digest in by_hash:
            by_hash[digest]["duplicate_sources"].append(relative)
            continue
        record = {
            "source": relative,
            "duplicate_sources": [],
            "sha256": digest,
            "text": text,
            "bytes": len(raw),
            "characters": len(text),
        }
        by_hash[digest] = record
        documents.append(record)
    if not documents:
        raise ValueError("no non-empty UTF-8 .txt documents were found")
    return documents, skipped


def _split_documents(
    documents: list[dict[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation and test fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")
    indices = list(range(len(documents)))
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
        "train": [documents[index] for index in train_indices],
        "validation": [documents[index] for index in validation_indices],
        "test": [documents[index] for index in test_indices],
    }


def _tokenizer_metadata(tokenizer: ByteTokenizer | BPETokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, BPETokenizer):
        return {
            "type": "byte-bpe",
            "vocab_size": tokenizer.vocab_size,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    return {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size}


def prepare_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    tokenizer_type: str = "bpe",
    vocab_size: int = 1000,
) -> dict[str, Any]:
    """Prepare deterministic document splits, tokenizer and token-ID caches.

    BPE is fitted exclusively on the training documents. Exact duplicate
    documents are collapsed before splitting to prevent cross-split leakage.
    Source text is preserved byte-for-byte after strict UTF-8 decoding.
    """
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"tokenizer_type must be one of {TOKENIZER_TYPES}")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    documents, skipped = _read_documents(source, output)
    splits = _split_documents(
        documents,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not splits["train"]:
        raise ValueError("corpus split produced no training documents")
    if tokenizer_type == "bpe":
        tokenizer: ByteTokenizer | BPETokenizer = BPETokenizer.train(
            (item["text"] for item in splits["train"]),
            vocab_size=vocab_size,
        )
    else:
        tokenizer = ByteTokenizer()
    metadata = _tokenizer_metadata(tokenizer)
    tokenizer_fingerprint = _sha256(
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
    token_counts: dict[str, int] = {}
    for split_name, records in splits.items():
        split_counts[split_name] = len(records)
        token_path = output / f"{split_name}.jsonl"
        token_lines = []
        total_tokens = 0
        for record in records:
            token_ids = tokenizer.encode(record["text"])
            total_tokens += len(token_ids)
            token_lines.append(
                json.dumps(
                    {
                        "source": record["source"],
                        "sha256": record["sha256"],
                        "tokenizer_fingerprint": tokenizer_fingerprint,
                        "token_ids": token_ids,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        token_path.write_text(
            "\n".join(token_lines) + ("\n" if token_lines else ""),
            encoding="utf-8",
        )
        token_counts[split_name] = total_tokens

    manifest = {
        "format_version": 1,
        "input": str(source),
        "seed": seed,
        "split_fractions": {
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "tokenizer": metadata,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "document_count": len(documents),
        "duplicate_document_count": sum(
            len(item["duplicate_sources"]) for item in documents
        ),
        "skipped_files": skipped,
        "splits": split_counts,
        "token_counts": token_counts,
        "documents": [
            {
                "source": item["source"],
                "duplicate_sources": item["duplicate_sources"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "characters": item["characters"],
                "split": split_name,
            }
            for split_name, records in splits.items()
            for item in records
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible Entropy language-model corpus."
    )
    parser.add_argument("input", type=Path, help="UTF-8 .txt file or directory")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=TOKENIZER_TYPES, default="bpe")
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()
    manifest = prepare_corpus(
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
                "documents": manifest["document_count"],
                "splits": manifest["splits"],
                "token_counts": manifest["token_counts"],
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
