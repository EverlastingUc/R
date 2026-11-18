"""Supervised fine-tuning on prepared conversations with assistant-only loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.chat_data import IGNORE_INDEX
from entropy_lm.model import (
    CausalLanguageModel,
    TransformerConfig,
    architecture_matches_checkpoint,
)
from entropy_lm.tokenizer import ByteTokenizer
from entropy_lm.training import (
    _autocast_context,
    _create_grad_scaler,
    _learning_rate_scale,
    _tokenizer_metadata,
    _validate_training_precision,
)


def _load_chat_tokenizer(
    directory: Path,
) -> tuple[ByteTokenizer | BPETokenizer, str, str]:
    """Validate prepared chat metadata and reconstruct its tokenizer."""
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

    return (
        tokenizer,
        fingerprint,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def _load_chat_windows(
    directory: Path,
    split: str,
    tokenizer: ByteTokenizer | BPETokenizer,
    fingerprint: str,
    *,
    context_length: int,
) -> tuple[Tensor, Tensor]:
    """Read and validate one split, then create masked next-token windows."""
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
            raise TypeError(f"{split} record at line {line_number} must be an object")
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

        # Score each next-token target exactly once, while carrying the
        # preceding context into overlapping windows. Only the newest stride
        # of target positions is supervised in each window.
        stride = max(1, context_length // 2)
        score_start = 1
        while score_start < len(token_ids):
            score_end = min(len(token_ids), score_start + stride)
            context_start = max(0, score_start - (context_length - stride) - 1)
            window_inputs = token_ids[context_start : score_end - 1]
            window_targets = labels[context_start + 1 : score_end]
            scored_targets = [
                label if context_start + offset + 1 >= score_start else IGNORE_INDEX
                for offset, label in enumerate(window_targets)
            ]
            if any(label != IGNORE_INDEX for label in scored_targets):
                padding = context_length - len(window_inputs)
                inputs.append(window_inputs + [tokenizer.eos_id] * padding)
                targets.append(scored_targets + [IGNORE_INDEX] * padding)
            score_start = score_end

    if not inputs:
        raise ValueError(f"{split} split contains no supervised next-token windows")
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def _load_chat_dataset(
    directory: Path, *, context_length: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, ByteTokenizer | BPETokenizer, str]:
    """Validate prepared chat data and materialize train/validation windows."""
    tokenizer, fingerprint, corpus_hash = _load_chat_tokenizer(directory)
    train_inputs, train_targets = _load_chat_windows(
        directory, "train", tokenizer, fingerprint, context_length=context_length
    )
    validation_inputs, validation_targets = _load_chat_windows(
        directory,
        "validation",
        tokenizer,
        fingerprint,
        context_length=context_length,
    )
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
    precision: str = "fp32",
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
        with _autocast_context(device, precision):
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
    resume_from: str | Path | None = None,
    max_steps_per_run: int | None = None,
    precision: Literal["fp32", "bf16", "fp16"] = "fp32",
    gradient_checkpointing: bool = False,
) -> list[dict[str, float | int]]:
    """Fine-tune on assistant targets, with exact CPU pause/resume support."""
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
    if max_steps_per_run is not None and max_steps_per_run < 1:
        raise ValueError("max_steps_per_run must be positive when supplied")
    if not isinstance(gradient_checkpointing, bool):
        raise TypeError("gradient_checkpointing must be a boolean")
    if resume_from is not None and checkpoint_path == "entropy-chat-model.pt":
        checkpoint_path = resume_from

    architecture = config or TransformerConfig()
    dataset = _load_chat_dataset(
        Path(dataset_directory), context_length=architecture.context_length
    )
    (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        tokenizer,
        corpus_sha256,
    ) = dataset
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match the prepared chat tokenizer")

    target_device = torch.device(device)
    _validate_training_precision(target_device, precision)
    run_config = {
        "steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "eval_interval": eval_interval,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "device": device,
        "precision": precision,
        "gradient_checkpointing": gradient_checkpointing,
        "corpus_sha256": corpus_sha256,
    }

    torch.manual_seed(seed)
    model = CausalLanguageModel(architecture).to(target_device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = _create_grad_scaler(precision)
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
    best_model_state: dict[str, Tensor] | None = None
    evaluations_without_improvement = 0
    completed_steps = 0
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError("SFT resume checkpoint does not exist")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("format_version") != 2:
            raise ValueError("SFT resume requires a version 2 checkpoint")
        if not architecture_matches_checkpoint(saved.get("architecture"), architecture):
            raise ValueError("SFT resume checkpoint architecture does not match")
        if saved.get("tokenizer") != _tokenizer_metadata(tokenizer):
            raise ValueError("SFT resume checkpoint tokenizer does not match")
        saved_training = saved.get("training")
        if not isinstance(saved_training, dict):
            raise ValueError("SFT resume checkpoint training metadata is malformed")
        if saved_training.get("corpus_sha256") != corpus_sha256:
            raise ValueError("SFT resume checkpoint corpus does not match")
        saved_run_config = saved_training.get("run_config")
        legacy_run_config = dict(run_config)
        legacy_run_config.pop("precision")
        legacy_run_config.pop("gradient_checkpointing")
        compatible_legacy = (
            precision == "fp32"
            and not gradient_checkpointing
            and saved_run_config == legacy_run_config
        )
        if saved_run_config != run_config and not compatible_legacy:
            raise ValueError(
                "SFT resume checkpoint training configuration does not match"
            )
        resume_state = saved.get("resume_state")
        if not isinstance(resume_state, dict):
            raise ValueError("SFT resume checkpoint is missing latest training state")
        required_states = ("model_state", "optimizer_state", "scheduler_state")
        if any(not isinstance(resume_state.get(key), dict) for key in required_states):
            raise ValueError("SFT resume checkpoint state is malformed")
        rng_state = resume_state.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("SFT resume checkpoint RNG state is missing")
        try:
            model.load_state_dict(resume_state["model_state"], strict=True)
            optimizer.load_state_dict(resume_state["optimizer_state"])
            scheduler.load_state_dict(resume_state["scheduler_state"])
            saved_scaler_state = resume_state.get("scaler_state")
            if saved_scaler_state is not None:
                scaler.load_state_dict(saved_scaler_state)
            elif scaler.is_enabled():
                raise ValueError(
                    "SFT resume checkpoint is missing gradient scaler state"
                )
            restored_history = saved_training.get("history")
            if not isinstance(restored_history, list):
                raise TypeError("SFT resume checkpoint history must be a list")
            history = restored_history
            completed_steps = int(saved_training["completed_steps"])
            if not 0 <= completed_steps <= steps:
                raise ValueError("SFT resume checkpoint progress is invalid")
            best_value = saved_training.get("best_validation_loss")
            best_validation_loss = math.inf if best_value is None else float(best_value)
            evaluations_without_improvement = int(
                saved_training.get("evaluations_without_improvement", 0)
            )
            if saved_training.get("has_best_checkpoint") is True:
                saved_best = saved.get("model_state")
                if not isinstance(saved_best, dict):
                    raise ValueError("SFT resume checkpoint best weights are missing")
                best_model_state = saved_best
            generator.set_state(rng_state["sampling_generator"])
            torch.set_rng_state(rng_state["torch_cpu"])
            cuda_states = rng_state.get("torch_cuda", [])
            if cuda_states and target_device.type == "cuda":
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError("SFT resume CUDA device count does not match")
                torch.cuda.set_rng_state_all(cuda_states)
        except (KeyError, TypeError, RuntimeError, OverflowError) as error:
            raise ValueError("SFT resume checkpoint state is invalid") from error

    def clone_model_state() -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def capture_rng_state() -> dict[str, Any]:
        return {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "sampling_generator": generator.get_state(),
        }

    def save_progress() -> None:
        training_metadata: dict[str, Any] = {
            "task": "assistant-response-supervised-finetuning",
            "completed_steps": completed_steps,
            "total_steps": steps,
            "best_validation_loss": (
                None if math.isinf(best_validation_loss) else best_validation_loss
            ),
            "evaluations_without_improvement": evaluations_without_improvement,
            "has_best_checkpoint": best_model_state is not None,
            "corpus_sha256": corpus_sha256,
            "assistant_only_loss": True,
            "history": history,
            "seed": seed,
            "run_config": run_config,
        }
        payload: dict[str, Any] = {
            "format_version": 2,
            "architecture": asdict(architecture),
            "model_state": (
                best_model_state
                if best_model_state is not None
                else clone_model_state()
            ),
            "tokenizer": _tokenizer_metadata(tokenizer),
            "training": training_metadata,
            "resume_state": {
                "model_state": clone_model_state(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            },
        }
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)

    start_step = completed_steps
    end_step = (
        steps
        if max_steps_per_run is None
        else min(steps, completed_steps + max_steps_per_run)
    )
    if (
        early_stopping_patience is not None
        and evaluations_without_improvement >= early_stopping_patience
    ):
        end_step = start_step

    for step in range(start_step + 1, end_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample_indices = torch.randint(
            len(train_inputs),
            (batch_size * gradient_accumulation_steps,),
            generator=generator,
        )
        sampled_targets = train_targets[sample_indices]
        supervised_target_count = int((sampled_targets != IGNORE_INDEX).sum().item())
        if supervised_target_count == 0:
            raise RuntimeError("sampled training batch contains no assistant targets")
        accumulated_loss = 0.0
        for microbatch_index in range(gradient_accumulation_steps):
            start = microbatch_index * batch_size
            end = start + batch_size
            indices = sample_indices[start:end]
            batch_inputs = train_inputs[indices].to(target_device)
            batch_targets = train_targets[indices].to(target_device)
            with _autocast_context(target_device, precision):
                logits, _ = model(batch_inputs)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, architecture.vocab_size),
                    batch_targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("assistant-only training loss is non-finite")
            accumulated_loss += float(loss_sum.item())
            scaler.scale(loss_sum / supervised_target_count).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
            scheduler.step()
        completed_steps = step

        if step % eval_interval == 0 or step == steps:
            validation_loss = _evaluate(
                model,
                validation_inputs,
                validation_targets,
                batch_size=batch_size,
                device=target_device,
                precision=precision,
            )
            history.append(
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 80.0)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "assistant_training_loss": accumulated_loss
                    / supervised_target_count,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = clone_model_state()
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            save_progress()
            if (
                early_stopping_patience is not None
                and evaluations_without_improvement >= early_stopping_patience
            ):
                break

    save_progress()
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
    parser.add_argument(
        "--position-encoding", choices=("learned", "rope"), default="learned"
    )
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--attention-window", type=int)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute Transformer activations to reduce training memory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the checkpoint path",
    )
    parser.add_argument(
        "--max-steps-this-run",
        type=int,
        help="pause after this many additional steps and save progress",
    )
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
            position_encoding=args.position_encoding,
            rope_base=args.rope_base,
            attention_window=args.attention_window,
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
        resume_from=args.checkpoint if args.resume else None,
        max_steps_per_run=args.max_steps_this_run,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
