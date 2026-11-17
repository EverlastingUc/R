"""Supervised fine-tuning on prepared conversations with assistant-only loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.chat_data import IGNORE_INDEX
from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.tokenizer import ByteTokenizer
from entropy_lm.training import _learning_rate_scale, _tokenizer_metadata


def _load_chat_dataset(
    directory: Path, *, context_length: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, ByteTokenizer | BPETokenizer, str]:
    """Validate prepared chat data and materialize supervised causal windows."""
    manifest_path = directory / "manifest.json"
    tokenizer_path = directory / "tokenizer.json"
    if not manifest_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            "prepared chat data requires manifest.json and tokenizer.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        raise TypeError("prepared chat metadata must contain JSON objects")

    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("prepared chat tokenizer fingerprint mismatch")
    if manifest.get("tokenizer") != metadata:
        raise ValueError("prepared chat tokenizer metadata does not match manifest")

    if metadata.get("type") == "utf8-byte":
        tokenizer: ByteTokenizer | BPETokenizer = ByteTokenizer()
    elif metadata.get("type") == "byte-bpe":
        merges = metadata.get("merges")
        if not isinstance(merges, list):
            raise ValueError("prepared chat BPE merge table is malformed")
        tokenizer = BPETokenizer(tuple(tuple(pair) for pair in merges))
        if metadata.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("prepared chat BPE vocabulary size mismatch")
    else:
        raise ValueError("prepared chat tokenizer type is unsupported")
    if _tokenizer_metadata(tokenizer) != metadata:
        raise ValueError("prepared chat tokenizer metadata is incompatible")

    def load_windows(split: str) -> tuple[Tensor, Tensor]:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"prepared chat data is missing {path.name}")
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid {split} JSONL record at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"{split} record at line {line_number} must be an object"
                )
            if record.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError(f"{split} record tokenizer fingerprint mismatch")
            token_ids = record.get("input_ids")
            labels = record.get("labels")
            if (
                not isinstance(token_ids, list)
                or not isinstance(labels, list)
                or len(token_ids) != len(labels)
                or len(token_ids) < 2
                or any(
                    not isinstance(token_id, int)
                    or isinstance(token_id, bool)
                    or token_id < 0
                    or token_id >= tokenizer.vocab_size
                    for token_id in token_ids
                )
                or any(
                    not isinstance(label, int)
                    or isinstance(label, bool)
                    or (label != IGNORE_INDEX and not 0 <= label < tokenizer.vocab_size)
                    for label in labels
                )
            ):
                raise ValueError(f"{split} record contains invalid token IDs or labels")
            if any(
                label != IGNORE_INDEX and label != token_id
                for token_id, label in zip(token_ids, labels)
            ):
                raise ValueError(f"{split} labels do not align with input token IDs")

            for start in range(0, len(token_ids) - 1, context_length):
                end = min(start + context_length, len(token_ids) - 1)
                window_inputs = token_ids[start:end]
                window_targets = labels[start + 1 : end + 1]
                if not any(label != IGNORE_INDEX for label in window_targets):
                    continue
                padding = context_length - len(window_inputs)
                inputs.append(window_inputs + [tokenizer.eos_id] * padding)
                targets.append(window_targets + [IGNORE_INDEX] * padding)

        if not inputs:
            raise ValueError(f"{split} split contains no supervised next-token windows")
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    train_inputs, train_targets = load_windows("train")
    validation_inputs, validation_targets = load_windows("validation")
    corpus_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        tokenizer,
        corpus_hash,
    )


@torch.no_grad()
def _evaluate(
    model: CausalLanguageModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        target_count = int((batch_targets != IGNORE_INDEX).sum().item())
        if target_count == 0:
            continue
        _, loss = model(batch_inputs, batch_targets, ignore_index=IGNORE_INDEX)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("assistant-only validation loss is non-finite")
        total_loss += float(loss.item()) * target_count
        total_targets += target_count
    if total_targets == 0:
        raise ValueError("validation split contains no assistant targets")
    return total_loss / total_targets


def train_on_conversations(
    dataset_directory: str | Path,
    *,
    config: TransformerConfig | None = None,
    steps: int = 1000,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    eval_interval: int = 100,
    warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.1,
    early_stopping_patience: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_path: str | Path = "entropy-chat-model.pt",
) -> list[dict[str, float | int]]:
    """Fine-tune a decoder-only Transformer using assistant-response targets."""
    if steps < 1 or batch_size < 1 or eval_interval < 1:
        raise ValueError("steps, batch_size, and eval_interval must be positive")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0 <= warmup_steps < steps:
        raise ValueError("warmup_steps must be non-negative and less than steps")
    if (
        not math.isfinite(minimum_learning_rate_ratio)
        or not 0.0 <= minimum_learning_rate_ratio <= 1.0
    ):
        raise ValueError("minimum_learning_rate_ratio must be finite and in [0, 1]")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when supplied")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    dataset = _load_chat_dataset(
        Path(dataset_directory),
        context_length=(config or TransformerConfig()).context_length,
    )
    (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        tokenizer,
        corpus_sha256,
    ) = dataset
    architecture = config or TransformerConfig(vocab_size=tokenizer.vocab_size)
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match the prepared chat tokenizer")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = CausalLanguageModel(architecture).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_scale(
            step,
            total_steps=steps,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_learning_rate_ratio,
        ),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    evaluations_without_improvement = 0
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            indices = torch.randint(
                len(train_inputs), (batch_size,), generator=generator
            )
            batch_inputs = train_inputs[indices].to(target_device)
            batch_targets = train_targets[indices].to(target_device)
            _, loss = model(batch_inputs, batch_targets, ignore_index=IGNORE_INDEX)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("assistant-only training loss is non-finite")
            accumulated_loss += float(loss.item())
            (loss / gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == steps:
            validation_loss = _evaluate(
                model,
                validation_inputs,
                validation_targets,
                batch_size=batch_size,
                device=target_device,
            )
            history.append(
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 80.0)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "mean_microbatch_loss": accumulated_loss
                    / gradient_accumulation_steps,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                evaluations_without_improvement = 0
                payload: dict[str, Any] = {
                    "format_version": 2,
                    "architecture": asdict(architecture),
                    "model_state": model.state_dict(),
                    "tokenizer": _tokenizer_metadata(tokenizer),
                    "training": {
                        "task": "assistant-response-supervised-finetuning",
                        "completed_steps": step,
                        "total_steps": steps,
                        "best_validation_loss": best_validation_loss,
                        "corpus_sha256": corpus_sha256,
                        "assistant_only_loss": True,
                        "history": history,
                        "seed": seed,
                    },
                }
                temporary = checkpoint.with_name(checkpoint.name + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
            else:
                evaluations_without_improvement += 1
            if (
                early_stopping_patience is not None
                and evaluations_without_improvement >= early_stopping_patience
            ):
                break

    if not checkpoint.is_file():
        raise RuntimeError("training did not produce a finite validation checkpoint")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune Entropy on prepared conversation data."
    )
    parser.add_argument("dataset", type=Path, help="LM11 prepared chat dataset")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("entropy-chat-model.pt")
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata = json.loads((args.dataset / "tokenizer.json").read_text(encoding="utf-8"))
    history = train_on_conversations(
        args.dataset,
        config=TransformerConfig(
            vocab_size=metadata["vocab_size"],
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            dropout=args.dropout,
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=(
            min(50, args.steps - 1) if args.warmup_steps is None else args.warmup_steps
        ),
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
