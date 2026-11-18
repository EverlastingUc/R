"""Small decoder-only Transformer language model (optional PyTorch extra)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

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
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.embedding_dim,
                self.num_heads,
                self.num_layers,
            )
            < 1
        ):
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
        self._initialize_parameters()
        self.output.weight = self.token_embedding.weight

    def _initialize_parameters(self) -> None:
        """Initialize cloned Transformer blocks independently."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for block in self.blocks.layers:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        ignore_index: int = -100,
    ) -> tuple[Tensor, Tensor | None]:
        """Return vocabulary logits and optional next-token cross-entropy.

        Target positions equal to ignore_index are excluded from the loss. This
        supports assistant-only supervised fine-tuning while retaining the
        existing full-sequence language-model objective by default.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("input_ids must be non-empty")
        if sequence_length > self.config.context_length:
            raise ValueError("sequence exceeds configured context_length")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
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
            valid_targets = targets != ignore_index
            if not torch.any(valid_targets):
                raise ValueError("targets contain no supervised tokens")
            if torch.any(
                (targets[valid_targets] < 0)
                | (targets[valid_targets] >= self.config.vocab_size)
            ):
                raise ValueError("targets contain an out-of-vocabulary token")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: Tensor,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """Run an inference chunk and return reusable per-layer key/value states.

        Cached keys and values have shape (batch, heads, sequence, head_dim).
        This method is intended for evaluation/generation and does not compute
        a training loss or retain autograd graphs.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a rank-2 torch.long tensor")
        batch_size, sequence_length = input_ids.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("input_ids must be non-empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain an out-of-vocabulary token")

        layers = self.blocks.layers
        if past_key_values is not None and len(past_key_values) != len(layers):
            raise ValueError(
                "past_key_values must contain one entry per Transformer layer"
            )

        past_length = 0
        if past_key_values is not None:
            lengths: set[int] = set()
            expected_shape = (batch_size, self.config.num_heads)
            for key, value in past_key_values:
                if key.ndim != 4 or value.shape != key.shape:
                    raise ValueError("cached key/value tensors have invalid shapes")
                if key.shape[:2] != expected_shape:
                    raise ValueError(
                        "cached key/value batch or head dimensions mismatch"
                    )
                if key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("cached key/value tensors must match input device")
                if key.dtype != self.token_embedding.weight.dtype:
                    raise ValueError("cached key/value tensor dtype mismatch")
                lengths.add(key.shape[-2])
            if len(lengths) != 1:
                raise ValueError("all cached layers must have the same sequence length")
            past_length = lengths.pop()

        if past_length + sequence_length > self.config.context_length:
            raise ValueError("cached sequence exceeds configured context_length")

        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=input_ids.device,
        )
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        updated_cache: list[tuple[Tensor, Tensor]] = []

        for layer_index, layer in enumerate(layers):
            normalized = layer.norm1(hidden) if layer.norm_first else hidden
            attention = layer.self_attn
            if attention.in_proj_weight is None:
                raise RuntimeError("cached attention requires a packed QKV projection")
            projected = F.linear(
                normalized, attention.in_proj_weight, attention.in_proj_bias
            )
            query, key, value = projected.chunk(3, dim=-1)
            head_dim = self.config.embedding_dim // self.config.num_heads
            query = query.view(
                batch_size, sequence_length, self.config.num_heads, head_dim
            ).transpose(1, 2)
            key = key.view(
                batch_size, sequence_length, self.config.num_heads, head_dim
            ).transpose(1, 2)
            value = value.view(
                batch_size, sequence_length, self.config.num_heads, head_dim
            ).transpose(1, 2)

            if past_key_values is not None:
                previous_key, previous_value = past_key_values[layer_index]
                key = torch.cat((previous_key, key), dim=-2)
                value = torch.cat((previous_value, value), dim=-2)
            updated_cache.append((key, value))

            dropout_probability = attention.dropout if self.training else 0.0
            if past_length == 0:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_probability,
                    is_causal=True,
                )
            else:
                query_positions = torch.arange(
                    past_length,
                    past_length + sequence_length,
                    device=input_ids.device,
                )
                key_positions = torch.arange(
                    past_length + sequence_length, device=input_ids.device
                )
                causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=causal_mask,
                    dropout_p=dropout_probability,
                    is_causal=False,
                )

            attended = (
                attended.transpose(1, 2)
                .contiguous()
                .view(batch_size, sequence_length, self.config.embedding_dim)
            )
            attended = attention.out_proj(attended)
            if layer.norm_first:
                hidden = hidden + layer.dropout1(attended)
                feedforward_input = layer.norm2(hidden)
                feedforward = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(feedforward_input)))
                )
                hidden = hidden + layer.dropout2(feedforward)
            else:
                hidden = layer.norm1(hidden + layer.dropout1(attended))
                feedforward = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(hidden)))
                )
                hidden = layer.norm2(hidden + layer.dropout2(feedforward))

        logits = self.output(self.final_norm(hidden))
        return logits, tuple(updated_cache)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate a complete sequence, preserving the existing batch API."""
        result = input_ids
        for next_token in self.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        ):
            result = torch.cat((result, next_token), dim=1)
        return result

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Iterator[Tensor]:
        """Yield one newly sampled token tensor (batch, 1) at a time."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary size")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be within the vocabulary size")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be non-empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain an out-of-vocabulary token")

        was_training = self.training
        self.eval()
        result = input_ids
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        try:
            context = result[:, -self.config.context_length :]
            logits, past_key_values = self.forward_with_cache(context)
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold.unsqueeze(-1), float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                result = torch.cat((result, next_token), dim=1)
                yield next_token
                if eos_token_id is not None and bool(torch.all(finished)):
                    break

                cached_length = past_key_values[0][0].shape[-2]
                if cached_length < self.config.context_length:
                    logits, past_key_values = self.forward_with_cache(
                        next_token, past_key_values
                    )
                else:
                    context = result[:, -self.config.context_length :]
                    logits, past_key_values = self.forward_with_cache(context)
        finally:
            self.train(was_training)
