"""UTF-8 byte tokenizer with explicit beginning/end-of-sequence tokens."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

BOS_ID = 256
EOS_ID = 257
VOCAB_SIZE = 258


class ByteTokenizer:
    """A deterministic byte-level tokenizer that needs no training corpus.

    Every UTF-8 byte maps to IDs 0..255. BOS and EOS occupy IDs 256 and 257.
    This is intentionally a baseline tokenizer, not a compression-oriented BPE.
    """

    vocab_size = VOCAB_SIZE
    bos_id = BOS_ID
    eos_id = EOS_ID

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode Unicode text as UTF-8 bytes, optionally adding boundary IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(text.encode("utf-8"))
        if add_bos:
            token_ids.insert(0, BOS_ID)
        if add_eos:
            token_ids.append(EOS_ID)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        """Decode byte IDs; reject invalid IDs and malformed UTF-8 sequences."""
        if errors not in ("strict", "replace", "ignore"):
            raise ValueError("errors must be strict, replace, or ignore")
        byte_values = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, (int, np.integer)):
                raise TypeError("token IDs must be integers")
            value = int(token_id)
            if 0 <= value <= 255:
                byte_values.append(value)
            elif value in (BOS_ID, EOS_ID) and skip_special_tokens:
                continue
            else:
                raise ValueError(f"unknown or unexpected special token ID: {value}")
        return bytes(byte_values).decode("utf-8", errors=errors)

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a collection of strings without padding or truncation."""
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
