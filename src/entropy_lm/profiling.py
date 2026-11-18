"""Checkpoint and inference profiling for Entropy language models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.inference import load_model_checkpoint


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_checkpoint(
    checkpoint_path: str | Path,
    *,
    prompt: str = "Entropy profiling prompt.",
    max_new_tokens: int = 32,
    batch_size: int = 1,
    warmup_steps: int = 2,
    seed: int = 0,
    device: str = "cpu",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure checkpoint size, prefill, decode latency, throughput and memory."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA profiling requires CUDA support")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError("model checkpoint does not exist")

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(seed)
        model = load_model_checkpoint(checkpoint, device=str(target_device))
        model.eval()
        tokenizer = model.tokenizer
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_id]
        prompt_ids = prompt_ids[-model.config.context_length :]
        input_ids = torch.tensor(
            [prompt_ids] * batch_size,
            dtype=torch.long,
            device=target_device,
        )

        for _ in range(warmup_steps):
            for _token in model.generate_stream(
                input_ids,
                max_new_tokens=min(2, max_new_tokens),
                temperature=1.0,
                top_k=1,
            ):
                pass
        _synchronize(target_device)

        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)

        prefill_started = time.perf_counter()
        prefill_logits, prefill_cache = model.forward_with_cache(input_ids)
        _synchronize(target_device)
        prefill_latency = time.perf_counter() - prefill_started
        del prefill_logits, prefill_cache

        stream = iter(
            model.generate_stream(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,
            )
        )
        generation_started = time.perf_counter()
        first_token_started = generation_started
        try:
            next(stream)
        except StopIteration as error:
            raise RuntimeError("model generation returned no tokens") from error
        _synchronize(target_device)
        first_token_latency = time.perf_counter() - first_token_started
        generated_steps = 1
        decode_step_latencies: list[float] = []

        while generated_steps < max_new_tokens:
            step_started = time.perf_counter()
            try:
                next(stream)
            except StopIteration:
                break
            _synchronize(target_device)
            decode_step_latencies.append(time.perf_counter() - step_started)
            generated_steps += 1
        generation_elapsed = time.perf_counter() - generation_started
        close = getattr(stream, "close", None)
        if callable(close):
            close()

        generated_tokens = generated_steps * batch_size
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        peak_cuda_memory = (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda"
            else None
        )
        if isinstance(tokenizer, BPETokenizer):
            tokenizer_type = "byte-bpe"
        else:
            tokenizer_type = "utf8-byte"

        report: dict[str, Any] = {
            "benchmark": "entropy-inference-profile",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "architecture": {
                "vocab_size": model.config.vocab_size,
                "context_length": model.config.context_length,
                "embedding_dim": model.config.embedding_dim,
                "num_heads": model.config.num_heads,
                "num_layers": model.config.num_layers,
                "dropout": model.config.dropout,
            },
            "tokenizer": tokenizer_type,
            "parameters": {
                "count": parameter_count,
                "storage_bytes": parameter_bytes,
                "dtype": str(next(model.parameters()).dtype),
            },
            "runtime": {
                "torch_version": torch.__version__,
                "device": str(target_device),
                "device_name": (
                    torch.cuda.get_device_name(target_device)
                    if target_device.type == "cuda"
                    else "CPU"
                ),
                "batch_size": batch_size,
                "prompt_tokens_per_sequence": len(prompt_ids),
                "warmup_steps": warmup_steps,
                "max_new_tokens_per_sequence": max_new_tokens,
                "seed": seed,
            },
            "latency": {
                "prefill_seconds": prefill_latency,
                "first_token_seconds": first_token_latency,
                "mean_decode_step_seconds": (
                    statistics.mean(decode_step_latencies)
                    if decode_step_latencies
                    else None
                ),
                "median_decode_step_seconds": (
                    statistics.median(decode_step_latencies)
                    if decode_step_latencies
                    else None
                ),
                "generation_seconds": generation_elapsed,
            },
            "throughput": {
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / generation_elapsed,
                "decode_steps_per_second": generated_steps / generation_elapsed,
            },
            "memory": {
                "peak_cuda_allocated_bytes": peak_cuda_memory,
            },
        }
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile an Entropy language-model checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Entropy profiling prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = profile_checkpoint(
        args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
