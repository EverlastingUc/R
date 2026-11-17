"""Next-token training for Entropy's optional PyTorch language model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.tokenizer import ByteTokenizer


def _sample_batch(
    tokens: Tensor,
    *,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    max_start = len(tokens) - context_length - 1
    starts = torch.randint(
        max_start + 1, (batch_size,), generator=generator, device="cpu"
    )
    offsets = torch.arange(context_length)
    positions = starts[:, None] + offsets[None, :]
    return tokens[positions].to(device), tokens[positions + 1].to(device)


@torch.no_grad()
def _estimate_loss(
    model: CausalLanguageModel,
    tokens: Tensor,
    *,
    batch_size: int,
    context_length: int,
    batches: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = _sample_batch(
            tokens,
            batch_size=batch_size,
            context_length=context_length,
            generator=generator,
            device=device,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return a loss for supplied targets")
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _tokenizer_metadata(tokenizer: ByteTokenizer | BPETokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, BPETokenizer):
        return {
            "type": "byte-bpe",
            "vocab_size": tokenizer.vocab_size,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
    return {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size}


def _learning_rate_scale(
    step: int, *, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    """Linear warmup followed by cosine decay to a configurable LR floor."""
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _save_checkpoint(
    path: Path,
    *,
    architecture: TransformerConfig,
    model: CausalLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer: ByteTokenizer | BPETokenizer,
    history: list[dict[str, float | int]],
    training: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "architecture": asdict(architecture),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training": training | {"history": history},
        "tokenizer": _tokenizer_metadata(tokenizer),
    }
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)



def _load_prepared_dataset(directory: Path):
    """Load LM06 splits and verify tokenizer fingerprints."""
    manifest_path = directory / "manifest.json"
    metadata_path = directory / "tokenizer.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "prepared corpus requires manifest.json and tokenizer.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("prepared corpus tokenizer fingerprint mismatch")
    if metadata.get("type") == "utf8-byte":
        tokenizer = ByteTokenizer()
    elif metadata.get("type") == "byte-bpe":
        merges = metadata.get("merges")
        if not isinstance(merges, list):
            raise ValueError("prepared BPE merge table is malformed")
        tokenizer = BPETokenizer(tuple(tuple(pair) for pair in merges))
        if metadata.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("prepared BPE vocabulary size mismatch")
    else:
        raise ValueError("prepared tokenizer type is unsupported")

    def load_split(name: str) -> Tensor:
        split_path = directory / (name + ".jsonl")
        if not split_path.is_file():
            raise FileNotFoundError(
                "prepared corpus is missing " + split_path.name
            )
        values: list[int] = []
        for line_number, line in enumerate(
            split_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSONL record at line " + str(line_number)
                ) from error
            if (
                not isinstance(record, dict)
                or record.get("tokenizer_fingerprint") != fingerprint
            ):
                raise ValueError("prepared dataset record has invalid metadata")
            ids = record.get("token_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    not isinstance(x, int)
                    or isinstance(x, bool)
                    or x < 0
                    or x >= tokenizer.vocab_size
                    for x in ids
                )
            ):
                raise ValueError("prepared dataset contains invalid token IDs")
            values.extend(ids)
            values.append(tokenizer.eos_id)
        return torch.tensor(values, dtype=torch.long)

    return (
        load_split("train"),
        load_split("validation"),
        tokenizer,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def train_language_model(
    text: str | Path,
    *,
    config: TransformerConfig | None = None,
    steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    validation_fraction: float = 0.1,
    eval_interval: int = 100,
    eval_batches: int = 10,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    tokenizer_type: Literal["byte", "bpe"] = "byte",
    bpe_vocab_size: int = 1000,
    warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int | None = None,
) -> list[dict[str, float | int]]:
    """Train a causal LM with warmup, cosine decay and best-validation saving.

    BPE vocabulary is learned only from the chronological training partition.
    Steps counts optimizer updates; gradient accumulation increases effective
    batch size without requiring a larger device-memory batch.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("training text must be a non-empty string")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if steps < 1 or batch_size < 1 or eval_interval < 1 or eval_batches < 1:
        raise ValueError("steps, batch_size, and evaluation settings must be positive")
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
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if isinstance(text, Path) and text.is_dir():
        train_tokens, validation_tokens, tokenizer, corpus_sha256 = (
            _load_prepared_dataset(text)
        )
    else:
        if isinstance(text, Path):
            text = text.read_text(encoding="utf-8")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("training text must be a non-empty string")
        split_at = int(len(text) * (1.0 - validation_fraction))
        train_text = text[:split_at]
        validation_text = text[split_at:]
        if tokenizer_type == "byte":
            tokenizer: ByteTokenizer | BPETokenizer = ByteTokenizer()
        elif tokenizer_type == "bpe":
            tokenizer = BPETokenizer.train([train_text], vocab_size=bpe_vocab_size)
        else:
            raise ValueError("tokenizer_type must be 'byte' or 'bpe'")

        train_tokens = torch.tensor(
            tokenizer.encode(train_text, add_bos=True, add_eos=True), dtype=torch.long
        )
        validation_tokens = torch.tensor(
            tokenizer.encode(validation_text, add_bos=True, add_eos=True),
            dtype=torch.long,
        )
    architecture = config or TransformerConfig(vocab_size=tokenizer.vocab_size)
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match the selected tokenizer")
    if (
        len(train_tokens) <= architecture.context_length
        or len(validation_tokens) <= architecture.context_length
    ):
        raise ValueError("corpus splits must each exceed context_length tokens")

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
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    training_eval_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    validation_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    evaluations_without_improvement = 0
    completed_steps = 0
    if not (isinstance(text, Path) and text.is_dir()):
        corpus_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    checkpoint_training = {
        "steps": steps,
        "completed_steps": 0,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "best_validation_loss": None,
    }

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            inputs, targets = _sample_batch(
                train_tokens,
                batch_size=batch_size,
                context_length=architecture.context_length,
                generator=train_generator,
                device=target_device,
            )
            _, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("training loss became non-finite")
            accumulated_loss += float(loss.item())
            (loss / gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        completed_steps = step

        if step % eval_interval == 0 or step == steps:
            training_loss = _estimate_loss(
                model,
                train_tokens,
                batch_size=batch_size,
                context_length=architecture.context_length,
                batches=eval_batches,
                generator=training_eval_generator,
                device=target_device,
            )
            validation_loss = _estimate_loss(
                model,
                validation_tokens,
                batch_size=batch_size,
                context_length=architecture.context_length,
                batches=eval_batches,
                generator=validation_generator,
                device=target_device,
            )
            current_learning_rate = float(optimizer.param_groups[0]["lr"])
            history.append(
                {
                    "step": step,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 80.0)),
                    "learning_rate": current_learning_rate,
                    "mean_microbatch_loss": accumulated_loss
                    / gradient_accumulation_steps,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                evaluations_without_improvement = 0
                checkpoint_training["completed_steps"] = completed_steps
                checkpoint_training["best_validation_loss"] = best_validation_loss
                if checkpoint_path is not None:
                    _save_checkpoint(
                        Path(checkpoint_path),
                        architecture=architecture,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        tokenizer=tokenizer,
                        history=history,
                        training=checkpoint_training,
                    )
            else:
                evaluations_without_improvement += 1
                if (
                    early_stopping_patience is not None
                    and evaluations_without_improvement >= early_stopping_patience
                ):
                    break

    if checkpoint_path is not None and not Path(checkpoint_path).exists():
        checkpoint_training["completed_steps"] = completed_steps
        checkpoint_training["best_validation_loss"] = (
            None if math.isinf(best_validation_loss) else best_validation_loss
        )
        _save_checkpoint(
            Path(checkpoint_path),
            architecture=architecture,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            history=history,
            training=checkpoint_training,
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Entropy's decoder-only language model."
    )
    parser.add_argument(
        "corpus", type=Path, help="UTF-8 text file or LM06 prepared-corpus directory"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("entropy-model.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--bpe-vocab-size", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.corpus.is_dir():
        metadata = json.loads(
            (args.corpus / "tokenizer.json").read_text(encoding="utf-8")
        )
        vocab_size = metadata["vocab_size"]
    else:
        corpus_text = args.corpus.read_text(encoding="utf-8")
        if args.tokenizer == "bpe":
            split_at = int(len(corpus_text) * 0.9)
            tokenizer = BPETokenizer.train(
                [corpus_text[:split_at]], vocab_size=args.bpe_vocab_size
            )
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = ByteTokenizer.vocab_size
    history = train_language_model(
        args.corpus,
        config=TransformerConfig(
            vocab_size=vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_heads=args.heads,
            num_layers=args.layers,
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
        tokenizer_type=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=(
            min(50, args.steps - 1) if args.warmup_steps is None else args.warmup_steps
        ),
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
