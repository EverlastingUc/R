"""Trainable byte-pair encoding tokenizer for Entropy language-model experiments.

The tokenizer starts from UTF-8 bytes, learns deterministic adjacent-pair merges,
and reserves BOS/EOS IDs after the learned vocabulary. It has no third-party
runtime dependencies and stores its merge table as portable JSON.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

BOS_ID = 256
EOS_ID = 257
BASE_VOCAB_SIZE = 258


class BPETokenizer:
    """A deterministic byte-level BPE tokenizer.

    IDs 0..255 are raw bytes, learned merge IDs start at 258, and BOS/EOS
    occupy the final two IDs. The merge order is the tokenization priority.
    """

    def __init__(self, merges: Iterable[tuple[int, int]] = ()) -> None:
        self.merges = [(int(left), int(right)) for left, right in merges]
        self._validate_merges()
        self._rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.bos_id = BASE_VOCAB_SIZE + len(self.merges)
        self.eos_id = self.bos_id + 1
        self.vocab_size = self.eos_id + 1

    def _validate_merges(self) -> None:
        available = BASE_VOCAB_SIZE
        seen: set[tuple[int, int]] = set()
        for pair in self.merges:
            left, right = pair
            if left < 0 or right < 0 or left >= available or right >= available:
                raise ValueError("merge references a token that has not been defined")
            if pair in seen:
                raise ValueError("merge pairs must be unique")
            seen.add(pair)
            available += 1

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Learn up to vocab_size tokens from UTF-8 text using pair frequency."""
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        sequences: list[list[int]] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("all training documents must be strings")
            if text:
                sequences.append(list(text.encode("utf-8")))
        if not sequences:
            raise ValueError("at least one non-empty training document is required")

        merges: list[tuple[int, int]] = []
        next_id = BASE_VOCAB_SIZE
        while next_id < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(itertools.pairwise(sequence))
            if not counts:
                break
            best_pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_frequency:
                break
            left, right = best_pair
            merges.append(best_pair)
            for index, sequence in enumerate(sequences):
                merged: list[int] = []
                cursor = 0
                while cursor < len(sequence):
                    if (
                        cursor + 1 < len(sequence)
                        and sequence[cursor] == left
                        and sequence[cursor + 1] == right
                    ):
                        merged.append(next_id)
                        cursor += 2
                    else:
                        merged.append(sequence[cursor])
                        cursor += 1
                sequences[index] = merged
            next_id += 1
        return cls(merges)

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            candidates = [
                (self._rank[(tokens[i], tokens[i + 1])], i)
                for i in range(len(tokens) - 1)
                if (tokens[i], tokens[i + 1]) in self._rank
            ]
            if not candidates:
                break
            _, index = min(candidates)
            merge_id = BASE_VOCAB_SIZE + self._rank[(tokens[index], tokens[index + 1])]
            tokens[index : index + 2] = [merge_id]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        if errors not in ("strict", "replace", "ignore"):
            raise ValueError("errors must be strict, replace, or ignore")
        pieces: list[bytes] = []
        for raw_id in token_ids:
            if not isinstance(raw_id, int):
                raise TypeError("token IDs must be integers")
            token_id = raw_id
            if token_id in (self.bos_id, self.eos_id) and skip_special_tokens:
                continue
            if 0 <= token_id <= 255:
                pieces.append(bytes([token_id]))
            elif BASE_VOCAB_SIZE <= token_id < self.bos_id:
                stack = [token_id]
                expanded: list[int] = []
                while stack:
                    current = stack.pop()
                    if current < BASE_VOCAB_SIZE:
                        expanded.append(current)
                    else:
                        left, right = self.merges[current - BASE_VOCAB_SIZE]
                        stack.extend((right, left))
                pieces.append(bytes(expanded))
            else:
                raise ValueError(f"unknown or unexpected token ID: {token_id}")
        return b"".join(pieces).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 1,
            "type": "byte-bpe",
            "merges": [list(pair) for pair in self.merges],
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("type") != "byte-bpe":
            raise ValueError("unsupported BPE tokenizer file")
        merges = payload.get("merges")
        if not isinstance(merges, list):
            raise TypeError("tokenizer merges must be a list")
        return cls(tuple(pair) for pair in merges)
