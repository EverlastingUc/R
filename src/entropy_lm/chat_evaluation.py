"""Assistant-only held-out evaluation for Entropy supervised checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from entropy_lm.chat_data import IGNORE_INDEX
from entropy_lm.chat_training import (
    _evaluate,
    _load_chat_tokenizer,
    _load_chat_windows,
)
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.training import _tokenizer_metadata



@torch.no_grad()
def _evaluate_turn_positions(
    model: torch.nn.Module,
    dataset_directory: Path,
    *,
    tokenizer: Any,
    fingerprint: str,
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure assistant-token loss by first, interior, and final turn position."""
    totals = {
        name: {"loss_sum": 0.0, "tokens": 0, "turns": 0}
        for name in ("first", "middle", "final")
    }
    records = []
    path = dataset_directory / "test.jsonl"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid test JSONL record at line {line_number}") from error
        if not isinstance(record, dict) or record.get("tokenizer_fingerprint") != fingerprint:
            raise ValueError("test record metadata is invalid")
        token_ids, labels, turns = (
            record.get("input_ids"), record.get("labels"), record.get("assistant_turns")
        )
        if (
            not isinstance(token_ids, list)
            or not isinstance(labels, list)
            or len(token_ids) != len(labels)
            or not isinstance(turns, list)
            or not turns
        ):
            raise ValueError("test record is missing assistant-turn metadata")
        turn_map = [-1] * len(labels)
        previous_end = 0
        for index, turn in enumerate(turns):
            if (
                not isinstance(turn, dict)
                or turn.get("turn_index") != index
                or not isinstance(turn.get("target_start"), int)
                or not isinstance(turn.get("target_end"), int)
            ):
                raise ValueError("assistant-turn metadata is malformed")
            start, end = turn["target_start"], turn["target_end"]
            if not previous_end <= start < end <= len(labels):
                raise ValueError("assistant-turn spans overlap or exceed the sequence")
            if any(labels[position] == IGNORE_INDEX for position in range(start, end)):
                raise ValueError("assistant-turn span contains masked labels")
            for position in range(start, end):
                turn_map[position] = index
            previous_end = end
        if any(
            label != IGNORE_INDEX and turn_map[position] < 0
            for position, label in enumerate(labels)
        ):
            raise ValueError("assistant labels are not covered by turn metadata")
        records.append((token_ids, labels, turn_map, len(turns)))

    model.eval()
    for token_ids, labels, turn_map, turn_count in records:
        categories = {}
        for turn_index in range(turn_count):
            category = (
                "first" if turn_count == 1 or turn_index == 0
                else "final" if turn_index == turn_count - 1
                else "middle"
            )
            categories[turn_index] = category
            totals[category]["turns"] += 1

        windows = []
        stride = max(1, context_length // 2)
        score_start = 1
        while score_start < len(token_ids):
            score_end = min(len(token_ids), score_start + stride)
            context_start = max(0, score_start - (context_length - stride) - 1)
            window_inputs = token_ids[context_start : score_end - 1]
            window_targets = labels[context_start + 1 : score_end]
            window_turns = [
                turn_map[position] if position >= score_start else -1
                for position in range(context_start + 1, score_end)
            ]
            if any(
                label != IGNORE_INDEX and turn >= 0
                for label, turn in zip(window_targets, window_turns)
            ):
                padding = context_length - len(window_inputs)
                windows.append((
                    window_inputs + [tokenizer.eos_id] * padding,
                    window_targets + [IGNORE_INDEX] * padding,
                    window_turns + [-1] * padding,
                ))
            score_start = score_end

        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            inputs = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            turn_ids = torch.tensor([item[2] for item in batch], dtype=torch.long, device=device)
            logits, _ = model(inputs)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).reshape_as(targets)
            for row in range(len(batch)):
                for position in range(context_length):
                    turn_index = int(turn_ids[row, position].item())
                    if turn_index < 0 or int(targets[row, position].item()) == IGNORE_INDEX:
                        continue
                    bucket = totals[categories[turn_index]]
                    bucket["loss_sum"] += float(losses[row, position].item())
                    bucket["tokens"] += 1

    result = {"conversation_count": len(records)}
    for name, values in totals.items():
        count = values["tokens"]
        loss = values["loss_sum"] / count if count else None
        result[name] = {
            "assistant_turns": values["turns"],
            "assistant_tokens": count,
            "cross_entropy_nats_per_token": loss,
            "perplexity": math.exp(min(loss, 80.0)) if loss is not None else None,
        }
    return result


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
        raise TypeError("checkpoint does not contain SFT training metadata")
    training = checkpoint["training"]
    if (
        training.get("task") != "assistant-response-supervised-finetuning"
        or training.get("assistant_only_loss") is not True
    ):
        raise ValueError("checkpoint is not an assistant-only SFT model")

    tokenizer, fingerprint, corpus_sha256 = _load_chat_tokenizer(dataset_directory)
    model = load_model_checkpoint(checkpoint_path, device=device)
    if checkpoint.get("tokenizer") != _tokenizer_metadata(tokenizer):
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

    turn_position_metrics = _evaluate_turn_positions(
        model,
        dataset_directory,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        context_length=model.config.context_length,
        batch_size=batch_size,
        device=torch.device(device),
    )
    loss = _evaluate(
        model,
        inputs,
        targets,
        batch_size=batch_size,
        device=torch.device(device),
    )
    return {
        "split": "test",
        "windows_scored": inputs.shape[0],
        "assistant_targets_scored": assistant_targets,
        "assistant_cross_entropy_nats_per_token": loss,
        "assistant_perplexity": math.exp(min(loss, 80.0)),
        "context_length": model.config.context_length,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": corpus_sha256,
        "tokenizer_fingerprint": fingerprint,
        "turn_position_metrics": turn_position_metrics,
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
