"""Small decoder-only Transformer language model (optional PyTorch extra)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _filter_generation_logits(
    logits: Tensor,
    token_history: Tensor,
    *,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float,
) -> Tensor:
    """Apply repetition, top-k, and nucleus filtering before token sampling."""
    if repetition_penalty != 1.0:
        repeated = torch.zeros_like(logits, dtype=torch.bool)
        repeated.scatter_(1, token_history, True)
        penalized = torch.where(
            logits < 0,
            logits * repetition_penalty,
            logits / repetition_penalty,
        )
        logits = torch.where(repeated, penalized, logits)

    if top_k is not None:
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1]
        logits = logits.masked_fill(logits < threshold.unsqueeze(-1), float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        remove_sorted = cumulative_probabilities > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_tokens = torch.zeros_like(remove_sorted).scatter(
            1, sorted_indices, remove_sorted
        )
        logits = logits.masked_fill(remove_tokens, float("-inf"))
    return logits


def _apply_rotary_position_embedding(
    query: Tensor,
    key: Tensor,
    positions: Tensor,
    *,
    base: float,
) -> tuple[Tensor, Tensor]:
    """Apply interleaved rotary position encoding to query and key heads."""
    head_dim = query.shape[-1]
    inverse_frequency = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=query.device)
            / head_dim
        )
    )
    angles = positions.to(device=query.device, dtype=torch.float32).unsqueeze(
        -1
    ) * inverse_frequency.unsqueeze(0)
    cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
    sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]

    query_even, query_odd = query[..., 0::2], query[..., 1::2]
    key_even, key_odd = key[..., 0::2], key[..., 1::2]
    rotated_query = torch.stack(
        (
            query_even * cosine - query_odd * sine,
            query_even * sine + query_odd * cosine,
        ),
        dim=-1,
    ).flatten(-2)
    rotated_key = torch.stack(
        (
            key_even * cosine - key_odd * sine,
            key_even * sine + key_odd * cosine,
        ),
        dim=-1,
    ).flatten(-2)
    return rotated_query, rotated_key


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture settings for a causal decoder-only language model."""

    vocab_size: int = 258
    context_length: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    position_encoding: Literal["learned", "rope"] = "learned"
    rope_base: float = 10000.0
    attention_window: int | None = None

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
        if self.position_encoding not in ("learned", "rope"):
            raise ValueError("position_encoding must be 'learned' or 'rope'")
        if (
            isinstance(self.rope_base, bool)
            or not isinstance(self.rope_base, (int, float))
            or not math.isfinite(self.rope_base)
            or self.rope_base <= 1.0
        ):
            raise ValueError("rope_base must be positive, finite, and greater than one")
        head_dim = self.embedding_dim // self.num_heads
        if self.position_encoding == "rope" and head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.attention_window is not None:
            if (
                isinstance(self.attention_window, bool)
                or not isinstance(self.attention_window, int)
                or not 1 <= self.attention_window <= self.context_length
            ):
                raise ValueError(
                    "attention_window must be between one and context_length"
                )
            if self.position_encoding != "rope":
                raise ValueError("sliding attention windows currently require RoPE")


def architecture_matches_checkpoint(
    saved_architecture: object, config: TransformerConfig
) -> bool:
    """Accept current metadata and legacy learned-position architecture metadata."""
    if not isinstance(saved_architecture, dict):
        return False
    expected = asdict(config)
    if saved_architecture == expected:
        return True
    legacy = dict(expected)
    legacy.pop("position_encoding")
    legacy.pop("rope_base")
    legacy.pop("attention_window")
    return (
        config.position_encoding == "learned"
        and config.rope_base == 10000.0
        and saved_architecture == legacy
    )


class CausalLanguageModel(nn.Module):
    """Predict the next token using masked self-attention.

    Inputs have shape (batch, sequence). The returned logits have shape
    (batch, sequence, vocabulary). Targets, when supplied, have the same
    shape as inputs and are scored with next-token cross entropy.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.embedding_dim)
            if config.position_encoding == "learned"
            else None
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
        if self.position_embedding is not None:
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

    def gradient_checkpointing_enable(self) -> None:
        """Recompute Transformer layer activations during training to save memory."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing and retain normal forward activations."""
        self.gradient_checkpointing = False

    def _forward_rope_layer(
        self,
        layer: nn.TransformerEncoderLayer,
        hidden: Tensor,
        positions: Tensor,
        attention_window: int | None = None,
    ) -> Tensor:
        """Run one pre-norm Transformer layer with rotary attention positions."""
        batch_size, sequence_length, _ = hidden.shape
        attention = layer.self_attn
        normalized = layer.norm1(hidden) if layer.norm_first else hidden
        if attention.in_proj_weight is None:
            raise RuntimeError("RoPE attention requires a packed QKV projection")
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
        query, key = _apply_rotary_position_embedding(
            query, key, positions, base=self.config.rope_base
        )
        if attention_window is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=attention.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            query_positions = torch.arange(sequence_length, device=hidden.device)
            key_positions = torch.arange(sequence_length, device=hidden.device)
            local_causal_mask = (
                key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            ) & (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1) - attention_window
            )
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=local_causal_mask,
                dropout_p=attention.dropout if self.training else 0.0,
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
            return hidden + layer.dropout2(feedforward)

        hidden = layer.norm1(hidden + layer.dropout1(attended))
        feedforward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(hidden)))
        )
        return layer.norm2(hidden + layer.dropout2(feedforward))

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
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)
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
        if self.config.position_encoding == "rope":
            for layer in self.blocks.layers:
                if self.training and self.gradient_checkpointing:
                    hidden = checkpoint(
                        lambda states, block=layer: self._forward_rope_layer(
                            block, states, positions, self.config.attention_window
                        ),
                        hidden,
                        use_reentrant=False,
                    )
                else:
                    hidden = self._forward_rope_layer(
                        layer, hidden, positions, self.config.attention_window
                    )
        elif self.training and self.gradient_checkpointing:
            for layer in self.blocks.layers:
                hidden = checkpoint(layer, hidden, causal_mask, use_reentrant=False)
        else:
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
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)
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
            if self.config.position_encoding == "rope":
                query, key = _apply_rotary_position_embedding(
                    query, key, positions, base=self.config.rope_base
                )

            if past_key_values is not None:
                previous_key, previous_value = past_key_values[layer_index]
                key = torch.cat((previous_key, key), dim=-2)
                value = torch.cat((previous_value, value), dim=-2)
            updated_cache.append((key, value))

            dropout_probability = attention.dropout if self.training else 0.0
            if past_length == 0 and self.config.attention_window is None:
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
                if self.config.attention_window is not None:
                    causal_mask = causal_mask & (
                        key_positions.unsqueeze(0)
                        > query_positions.unsqueeze(1) - self.config.attention_window
                    )
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
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate a complete sequence, preserving the existing batch API."""
        result = input_ids
        for next_token in self.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
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
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Iterator[Tensor]:
        """Yield one newly sampled token tensor (batch, 1) at a time."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be within the vocabulary size")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0.0 < top_p <= 1.0
        ):
            raise ValueError("top_p must be finite and in (0, 1]")
        if (
            isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0.0
        ):
            raise ValueError("repetition_penalty must be positive and finite")
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
                recent_history = result[:, -self.config.context_length :]
                next_logits = _filter_generation_logits(
                    next_logits,
                    recent_history,
                    top_k=top_k,
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
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
