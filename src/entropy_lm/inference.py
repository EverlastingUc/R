"""Checkpoint loading and text generation for Entropy language models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.tokenizer import BOS_ID, ByteTokenizer


def load_model_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> CausalLanguageModel:
    """Load and validate a byte-token or BPE-token language-model checkpoint."""
    checkpoint: Any = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported or malformed language-model checkpoint")
    architecture = checkpoint.get("architecture")
    state = checkpoint.get("model_state")
    if not isinstance(architecture, dict) or not isinstance(state, dict):
        raise TypeError("checkpoint architecture and model weights must be mappings")
    metadata = checkpoint.get("tokenizer")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata is missing")
    if metadata.get("type") == "utf8-byte":
        tokenizer: ByteTokenizer | BPETokenizer = ByteTokenizer()
        expected = {"type": "utf8-byte", "vocab_size": tokenizer.vocab_size}
        if metadata != expected:
            raise ValueError("checkpoint byte-tokenizer metadata is incompatible")
    elif metadata.get("type") == "byte-bpe":
        merges = metadata.get("merges")
        if not isinstance(merges, list):
            raise ValueError("checkpoint BPE merge table is malformed")
        try:
            tokenizer = BPETokenizer(tuple(tuple(pair) for pair in merges))
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint BPE merge table is invalid") from error
        if metadata.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("checkpoint BPE vocabulary size is inconsistent")
    else:
        raise ValueError("checkpoint tokenizer type is unsupported")

    try:
        config = TransformerConfig(**architecture)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint architecture is invalid") from error
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match its tokenizer")
    model = CausalLanguageModel(config).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("checkpoint model weights do not match architecture") from error
    model.tokenizer = tokenizer
    model.eval()
    return model


def generate_text(
    model: CausalLanguageModel,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Generate text using the tokenizer stored with the loaded model."""
    tokenizer = getattr(model, "tokenizer", ByteTokenizer())
    prompt_ids = tokenizer.encode(prompt) or [tokenizer.bos_id if isinstance(tokenizer, BPETokenizer) else BOS_ID]
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist(), errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with an Entropy LM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model_checkpoint(args.checkpoint, device=args.device)
    print(generate_text(
        model, args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
