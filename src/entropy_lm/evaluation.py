"""Held-out test-set evaluation for Entropy language-model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.tokenizer import ByteTokenizer


def _tokenizer_metadata(model: torch.nn.Module) -> dict[str, Any]:
    tokenizer = getattr(model, "tokenizer", None)
    if isinstance(tokenizer, BPETokenizer):
        return {
            "type": "byte-bpe",
            "vocab_size": tokenizer.vocab_size,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    if isinstance(tokenizer, ByteTokenizer):
        return {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size}
    raise TypeError("checkpoint does not contain a supported tokenizer")


def _load_test_documents(
    corpus_directory: Path, expected_tokenizer: dict[str, Any]
) -> tuple[list[list[int]], str, str]:
    manifest_path = corpus_directory / "manifest.json"
    tokenizer_path = corpus_directory / "tokenizer.json"
    test_path = corpus_directory / "test.jsonl"
    for path in (manifest_path, tokenizer_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"prepared corpus is missing {path.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        raise TypeError("prepared corpus metadata must contain JSON objects")
    if (
        metadata.get("type") == "byte-bpe"
        and metadata.get("format_version") == 1
        and "vocab_size" not in metadata
    ):
        merges = metadata.get("merges")
        if not isinstance(merges, list):
            raise ValueError("prepared BPE merge table is malformed")
        legacy_tokenizer = BPETokenizer(tuple(tuple(pair) for pair in merges))
        metadata = _tokenizer_metadata(legacy_tokenizer)
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("prepared corpus tokenizer fingerprint mismatch")
    if metadata != expected_tokenizer:
        raise ValueError("test corpus tokenizer does not match the model checkpoint")

    documents: list[list[int]] = []
    for line_number, line in enumerate(
        test_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid test JSONL record at line {line_number}"
            ) from error
        ids = record.get("token_ids") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("tokenizer_fingerprint") != fingerprint
        ):
            raise ValueError("test record tokenizer fingerprint mismatch")
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                or token_id >= metadata["vocab_size"]
                for token_id in ids
            )
        ):
            raise ValueError("test corpus contains invalid token IDs")
        documents.append(ids + [expected_tokenizer["vocab_size"] - 1])

    if not documents:
        raise ValueError("prepared corpus test split contains no documents")
    return (
        documents,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        fingerprint,
    )


@torch.no_grad()
def _score_document(
    model: torch.nn.Module, token_ids: list[int], *, device: torch.device
) -> tuple[float, int]:
    """Score each next-token target once, retaining a rolling context window."""
    context_length = model.config.context_length
    stride = max(1, context_length // 2)
    context_before = max(1, context_length - stride + 1)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    total_loss = 0.0
    scored_tokens = 0

    for target_start in range(1, len(tokens), stride):
        target_end = min(target_start + stride, len(tokens))
        input_start = max(0, target_start - context_before)
        input_end = target_end - 1
        inputs = tokens[input_start:input_end].unsqueeze(0).to(device)
        targets = tokens[target_start:target_end].to(device)
        logits, _ = model(inputs)
        score_start = target_start - input_start - 1
        score_end = target_end - input_start - 1
        selected_logits = logits[0, score_start:score_end, :]
        loss = F.cross_entropy(selected_logits, targets, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("test-set loss became non-finite")
        total_loss += float(loss.item())
        scored_tokens += int(targets.numel())

    return total_loss, scored_tokens


def evaluate_prepared_test_set(
    checkpoint_path: str | Path,
    corpus_directory: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one checkpoint against the untouched LM06 test split."""
    checkpoint_path = Path(checkpoint_path)
    corpus_directory = Path(corpus_directory)
    model = load_model_checkpoint(checkpoint_path, device=device)
    expected_tokenizer = _tokenizer_metadata(model)
    documents, manifest_sha256, tokenizer_fingerprint = _load_test_documents(
        corpus_directory, expected_tokenizer
    )

    target_device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    scored_documents = 0
    for document in documents:
        if len(document) < 2:
            continue
        document_loss, document_tokens = _score_document(
            model, document, device=target_device
        )
        total_loss += document_loss
        total_tokens += document_tokens
        scored_documents += 1
    if total_tokens == 0:
        raise ValueError("test split contains no next-token targets")

    mean_loss = total_loss / total_tokens
    return {
        "documents_scored": scored_documents,
        "tokens_scored": total_tokens,
        "cross_entropy_nats_per_token": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "context_length": model.config.context_length,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": manifest_sha256,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "split": "test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an Entropy checkpoint on a prepared corpus test split."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path, help="LM06 prepared-corpus directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_prepared_test_set(
        args.checkpoint, args.corpus, device=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
