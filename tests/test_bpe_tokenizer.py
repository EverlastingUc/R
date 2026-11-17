from entropy_lm.bpe_tokenizer import BASE_VOCAB_SIZE, BPETokenizer


def test_bpe_round_trip_unicode_and_special_tokens():
    tokenizer = BPETokenizer.train(
        ["hello hello", "hello world", "नमस्ते नमस्ते"],
        vocab_size=BASE_VOCAB_SIZE + 40,
        min_frequency=2,
    )
    text = "hello 🌍 — नमस्ते"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_bpe_compresses_repeated_training_patterns():
    corpus = ["abababababab"] * 5
    tokenizer = BPETokenizer.train(
        corpus, vocab_size=BASE_VOCAB_SIZE + 8, min_frequency=2
    )
    assert len(tokenizer.encode(corpus[0])) < len(corpus[0].encode("utf-8"))


def test_bpe_is_deterministic():
    corpus = ["alpha beta alpha", "beta gamma"]
    first = BPETokenizer.train(corpus, vocab_size=270)
    second = BPETokenizer.train(corpus, vocab_size=270)
    assert first.merges == second.merges
    assert first.encode("alpha beta") == second.encode("alpha beta")


def test_bpe_save_load(tmp_path):
    tokenizer = BPETokenizer.train(["one two one two"], vocab_size=270)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.merges == tokenizer.merges
    assert restored.decode(restored.encode("one two")) == "one two"


def test_bpe_rejects_invalid_merge_and_input():
    import pytest

    with pytest.raises(ValueError):
        BPETokenizer([(300, 1)])
    with pytest.raises(ValueError):
        BPETokenizer.train(["abc"], vocab_size=257)
    with pytest.raises(ValueError):
        BPETokenizer.train(["", ""], vocab_size=270)
    with pytest.raises(TypeError):
        BPETokenizer.train([1], vocab_size=270)
