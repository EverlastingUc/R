"""Small decoder-only Transformer language model (optional PyTorch extra)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture settings for a causal decoder-only language model."""

    vocab_size: int = 258
    context_length: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_heads,
            self.num_layers,
        ) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalLanguageModel(nn.Module):
    """Predict the next token using masked self-attention.

    Inputs have shape (batch, sequence). The returned logits have shape
    (batch, sequence, vocabulary). Targets, when supplied, have the same
    shape as inputs and are scored with next-token cross entropy.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embedding_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> (
        tuple[Tensor, Tensor | None]
    ):
        """Return vocabulary logits and optional next-token cross-entropy."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("input_ids must be non-empty")
        if sequence_length > self.config.context_length:
            raise ValueError("sequence exceeds configured context_length")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        if torch.any(input_ids < 0) or torch.any(
            input_ids >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain an out-of-vocabulary token")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        # True entries are masked: a position cannot attend to future tokens.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        logits = self.output(self.final_norm(hidden))

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype != torch.long:
                raise TypeError("targets must use torch.long token IDs")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressively sample tokens, cropping to the context window."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary size")
        was_training = self.training
        self.eval()
        result = input_ids
        try:
            for _ in range(max_new_tokens):
                context = result[:, -self.config.context_length :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold.unsqueeze(-1), float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                result = torch.cat((result, next_token), dim=1)
        finally:
            self.train(was_training)
        return result
