"""Assistant-only held-out evaluation for Entropy supervised checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from entropy_lm.chat_training import (
    _evaluate,
    _load_chat_tokenizer,
    _load_chat_windows,
)
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.training import _tokenizer_metadata


def evaluate_chat_test_set(
    checkpoint_path: str | Path,
    dataset_directory: str | Path,
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score assistant response tokens on the untouched conversation test split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint_path)
    dataset_directory = Path(dataset_directory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("training"), dict
    ):
        raise ValueError("checkpoint does not contain SFT training metadata")
    training = checkpoint["training"]
    if (
        training.get("task") != "assistant-response-supervised-finetuning"
        or training.get("assistant_only_loss") is not True
    ):
        raise ValueError("checkpoint is not an assistant-only SFT model")

    tokenizer, fingerprint, corpus_sha256 = _load_chat_tokenizer(dataset_directory)
    model = load_model_checkpoint(checkpoint_path, device=device)
    if _tokenizer_metadata(model) != _tokenizer_metadata(tokenizer):
        raise ValueError("test dataset tokenizer does not match the SFT checkpoint")
    inputs, targets = _load_chat_windows(
        dataset_directory,
        "test",
        tokenizer,
        fingerprint,
        context_length=model.config.context_length,
    )
    assistant_targets = int((targets != -100).sum().item())
    if assistant_targets == 0:
        raise ValueError("test split contains no assistant response targets")

    loss = _evaluate(
        model,
        inputs,
        targets,
        batch_size=batch_size,
        device=torch.device(device),
    )
    return {
        "split": "test",
        "documents_or_conversations": inputs.shape[0],
        "assistant_targets_scored": assistant_targets,
        "assistant_cross_entropy_nats_per_token": loss,
        "assistant_perplexity": math.exp(min(loss, 80.0)),
        "context_length": model.config.context_length,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": corpus_sha256,
        "tokenizer_fingerprint": fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an Entropy SFT checkpoint on the held-out chat test split."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path, help="LM11 prepared chat dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_chat_test_set(
        args.checkpoint,
        args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
