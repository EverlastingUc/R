"""Next-token training for Entropy's optional PyTorch language model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.tokenizer import ByteTokenizer


def _encode_and_split(
    text: str,
    validation_fraction: float,
    context_length: int,
    *,
    tokenizer_type: Literal["byte", "bpe"] = "byte",
    bpe_vocab_size: int = 1000,
) -> tuple[Tensor, Tensor, ByteTokenizer | BPETokenizer]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("training text must be a non-empty string")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    split_at = int(len(text) * (1.0 - validation_fraction))
    train_text = text[:split_at]
    validation_text = text[split_at:]
    if tokenizer_type == "byte":
        tokenizer: ByteTokenizer | BPETokenizer = ByteTokenizer()
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.train([train_text], vocab_size=bpe_vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'byte' or 'bpe'")

    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    validation_ids = tokenizer.encode(validation_text, add_bos=True, add_eos=True)
    train_tokens = torch.tensor(train_ids, dtype=torch.long)
    validation_tokens = torch.tensor(validation_ids, dtype=torch.long)
    if len(train_tokens) <= context_length or len(validation_tokens) <= context_length:
        raise ValueError("corpus splits must each exceed context_length tokens")
    return train_tokens, validation_tokens, tokenizer


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


def train_language_model(
    text: str,
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
) -> list[dict[str, float | int]]:
    """Train a compact causal LM; BPE vocabulary is learned on training text only."""
    if steps < 1 or batch_size < 1 or eval_interval < 1 or eval_batches < 1:
        raise ValueError("steps, batch_size, and evaluation settings must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    torch.manual_seed(seed)
    # Tokenizer learning uses only the chronological training partition.
    split_at = int(len(text) * (1.0 - validation_fraction))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("training text must be a non-empty string")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    train_text = text[:split_at]
    tokenizer: ByteTokenizer | BPETokenizer
    if tokenizer_type == "byte":
        tokenizer = ByteTokenizer()
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.train([train_text], vocab_size=bpe_vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'byte' or 'bpe'")

    train_tokens = torch.tensor(
        tokenizer.encode(train_text, add_bos=True, add_eos=True)
    )
    validation_text = text[split_at:]
    validation_tokens = torch.tensor(
        tokenizer.encode(validation_text, add_bos=True, add_eos=True)
    )
    architecture = config or TransformerConfig(vocab_size=tokenizer.vocab_size)
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match the selected tokenizer")
    if (
        len(train_tokens) <= architecture.context_length
        or len(validation_tokens) <= architecture.context_length
    ):
        raise ValueError("corpus splits must each exceed context_length tokens")

    target_device = torch.device(device)
    model = CausalLanguageModel(architecture).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    validation_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    history: list[dict[str, float | int]] = []

    for step in range(1, steps + 1):
        model.train()
        inputs, targets = _sample_batch(
            train_tokens,
            batch_size=batch_size,
            context_length=architecture.context_length,
            generator=train_generator,
            device=target_device,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == steps:
            training_loss = _estimate_loss(
                model,
                train_tokens,
                batch_size=batch_size,
                context_length=architecture.context_length,
                batches=eval_batches,
                generator=train_generator,
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
            history.append(
                {
                    "step": step,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 80.0)),
                }
            )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_metadata: dict[str, Any] = {
            "type": "utf8-byte", "vocab_size": tokenizer.vocab_size,
        }
        if isinstance(tokenizer, BPETokenizer):
            tokenizer_metadata = {
                "type": "byte-bpe",
                "vocab_size": tokenizer.vocab_size,
                "merges": [list(pair) for pair in tokenizer.merges],
            }
        checkpoint: dict[str, Any] = {
            "format_version": 1,
            "architecture": asdict(architecture),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training": {
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "seed": seed,
                "history": history,
            },
            "tokenizer": tokenizer_metadata,
        }
        with path.open("xb") as output_file:
            torch.save(checkpoint, output_file)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Entropy's decoder-only language model."
    )
    parser.add_argument("corpus", type=Path, help="UTF-8 plain-text training corpus")
    parser.add_argument("--checkpoint", type=Path, default=Path("entropy-model.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--bpe-vocab-size", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Resolve tokenizer vocabulary before constructing the architecture.
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
        corpus_text,
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
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
