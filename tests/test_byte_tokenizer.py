"""Tests for the deterministic byte tokenizer."""

import pytest

from entropy_lm.tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE, ByteTokenizer


def test_byte_tokenizer_round_trips_unicode_and_special_tokens() -> None:
    tokenizer = ByteTokenizer()
    text = "Entropy — नमस्ते 🌌"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == BOS_ID
    assert encoded[-1] == EOS_ID
    assert tokenizer.decode(encoded) == text
    assert tokenizer.vocab_size == VOCAB_SIZE == 258


def test_byte_tokenizer_encodes_each_utf8_byte() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("A") == [65]
    assert tokenizer.decode([65]) == "A"


def test_byte_tokenizer_rejects_invalid_ids_and_utf8() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError):
        tokenizer.decode([258])
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([255])


def test_byte_tokenizer_requires_strings_and_integer_ids() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(TypeError):
        tokenizer.encode(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenizer.decode([1.5])  # type: ignore[list-item]


def test_byte_tokenizer_batch_encoding() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.encode_batch(["a", "b"], add_eos=True) == [
        [97, EOS_ID],
        [98, EOS_ID],
    ]
