"""Direct Preference Optimization for Entropy assistant checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from entropy_lm.chat_inference import build_chat_input_ids, format_chat_prompt
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.model import CausalLanguageModel
from entropy_lm.training import _tokenizer_metadata


def _read_preference_pairs(
    path: str | Path, model: CausalLanguageModel
) -> tuple[list[tuple[list[int], int, list[int], int]], str]:
    """Read prompt/chosen/rejected JSONL and tokenize within model context."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("preference dataset must be valid UTF-8") from error

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("preference training requires a checkpoint tokenizer")
    pairs: list[tuple[list[int], int, list[int], int]] = []
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid preference JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise TypeError(f"preference record at line {line_number} must be an object")
        prompt = record.get("prompt")
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(prompt, list):
            raise TypeError(f"preference prompt at line {line_number} must be messages")
        format_chat_prompt(prompt)
        if (
            not isinstance(chosen, str)
            or not chosen.strip()
            or not isinstance(rejected, str)
            or not rejected.strip()
        ):
            raise ValueError(f"chosen and rejected must be non-empty strings at line {line_number}")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"chosen and rejected must differ at line {line_number}")

        prompt_ids = build_chat_input_ids(
            tokenizer, prompt, context_length=model.config.context_length
        )
        available = model.config.context_length - len(prompt_ids)
        if available < 1:
            raise ValueError(f"prompt leaves no response capacity at line {line_number}")
        chosen_ids = tokenizer.encode(chosen) + [tokenizer.eos_id]
        rejected_ids = tokenizer.encode(rejected) + [tokenizer.eos_id]
        if len(chosen_ids) > available:
            chosen_ids = chosen_ids[: available - 1] + [tokenizer.eos_id]
        if len(rejected_ids) > available:
            rejected_ids = rejected_ids[: available - 1] + [tokenizer.eos_id]
        if not chosen_ids or not rejected_ids:
            raise ValueError(f"response tokenization is empty at line {line_number}")
        pairs.append(
            (
                prompt_ids + chosen_ids,
                len(prompt_ids) - 1,
                prompt_ids + rejected_ids,
                len(prompt_ids) - 1,
            )
        )

    if not pairs:
        raise ValueError("preference dataset contains no valid pairs")
    return pairs, hashlib.sha256(raw).hexdigest()


def _sequence_log_probabilities(
    model: CausalLanguageModel,
    sequences: list[tuple[list[int], int]],
    *,
    device: torch.device,
) -> Tensor:
    """Return summed response-token log probabilities for each sequence."""
    if not sequences:
        raise ValueError("at least one sequence is required")
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("model must have a tokenizer")
    max_input_length = max(len(tokens) - 1 for tokens, _ in sequences)
    if max_input_length < 1:
        raise ValueError("sequences must contain at least two tokens")

    inputs = torch.full(
        (len(sequences), max_input_length),
        tokenizer.eos_id,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full_like(inputs, tokenizer.eos_id)
    target_mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, (tokens, response_start) in enumerate(sequences):
        if len(tokens) < 2 or not 0 <= response_start < len(tokens) - 1:
            raise ValueError("sequence response boundary is invalid")
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long, device=device)
        inputs[row, : len(input_tokens)] = input_tokens
        targets[row, : len(target_tokens)] = target_tokens
        target_mask[row, response_start : len(target_tokens)] = True

    logits, _ = model(inputs)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * target_mask).sum(dim=-1)


def _dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    *,
    beta: float,
) -> Tensor:
    """Compute the standard sigmoid DPO objective for one preference batch."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()


@torch.no_grad()
def _evaluate_preferences(
    policy: CausalLanguageModel,
    reference: CausalLanguageModel,
    pairs: list[tuple[list[int], int, list[int], int]],
    *,
    batch_size: int,
    beta: float,
    device: torch.device,
) -> float:
    policy.eval()
    reference.eval()
    losses: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        chosen = [(item[0], item[1]) for item in batch]
        rejected = [(item[2], item[3]) for item in batch]
        policy_logps = _sequence_log_probabilities(
            policy, chosen + rejected, device=device
        )
        reference_logps = _sequence_log_probabilities(
            reference, chosen + rejected, device=device
        )
        count = len(batch)
        loss = _dpo_loss(
            policy_logps[:count],
            policy_logps[count:],
            reference_logps[:count],
            reference_logps[count:],
            beta=beta,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("preference validation loss is non-finite")
        losses.append(float(loss.item()) * count)
    return sum(losses) / len(pairs)


def train_with_preferences(
    base_checkpoint: str | Path,
    train_dataset: str | Path,
    validation_dataset: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    output_checkpoint: str | Path = "entropy-preference-model.pt",
) -> dict[str, Any]:
    """Align an SFT model to chosen/rejected responses using DPO."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    target_device = torch.device(device)
    torch.manual_seed(seed)
    policy = load_model_checkpoint(base_checkpoint, device=device)
    reference = copy.deepcopy(policy).to(target_device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    train_pairs, train_hash = _read_preference_pairs(train_dataset, policy)
    validation_pairs, validation_hash = _read_preference_pairs(
        validation_dataset, policy
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, epochs + 1):
        # Disable dropout for a stable policy/reference likelihood comparison.
        policy.eval()
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[index] for index in order[start : start + batch_size]]
            chosen = [(item[0], item[1]) for item in batch]
            rejected = [(item[2], item[3]) for item in batch]
            policy_logps = _sequence_log_probabilities(
                policy, chosen + rejected, device=target_device
            )
            with torch.no_grad():
                reference_logps = _sequence_log_probabilities(
                    reference, chosen + rejected, device=target_device
                )
            count = len(batch)
            loss = _dpo_loss(
                policy_logps[:count],
                policy_logps[count:],
                reference_logps[:count],
                reference_logps[count:],
                beta=beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("preference training loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * count
            seen += count

        validation_loss = _evaluate_preferences(
            policy,
            reference,
            validation_pairs,
            batch_size=batch_size,
            beta=beta,
            device=target_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("preference training did not produce a checkpoint")
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "architecture": asdict(policy.config),
        "model_state": best_state,
        "tokenizer": _tokenizer_metadata(policy.tokenizer),
        "training": {
            "task": "direct-preference-optimization",
            "base_checkpoint_sha256": hashlib.sha256(
                Path(base_checkpoint).read_bytes()
            ).hexdigest(),
            "train_dataset_sha256": train_hash,
            "validation_dataset_sha256": validation_hash,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "beta": beta,
            "seed": seed,
            "history": history,
            "best_validation_loss": best_validation_loss,
            "reference_policy": "frozen-copy-of-base-checkpoint",
        },
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {
        "checkpoint": str(destination),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align an Entropy SFT checkpoint with DPO preference pairs."
    )
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("train_dataset", type=Path, help="JSONL chosen/rejected pairs")
    parser.add_argument("validation_dataset", type=Path, help="held-out JSONL pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("entropy-preference-model.pt"))
    args = parser.parse_args()
    result = train_with_preferences(
        args.base_checkpoint,
        args.train_dataset,
        args.validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
        output_checkpoint=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
