"""BPE training, checkpoint persistence, and generation integration tests."""

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.inference import generate_text, load_model_checkpoint
from entropy_lm.model import TransformerConfig
from entropy_lm.training import train_language_model


def test_bpe_training_saves_and_reloads_matching_tokenizer(tmp_path):
    corpus = "The quick brown fox learns language. " * 100
    checkpoint = tmp_path / "bpe-model.pt"
    split_at = int(len(corpus) * 0.9)
    tokenizer = BPETokenizer.train([corpus[:split_at]], vocab_size=280)
    history = train_language_model(
        corpus,
        config=TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_length=12,
            embedding_dim=16, num_heads=2, num_layers=1, dropout=0.0,
        ),
        steps=1, batch_size=2, eval_interval=1, eval_batches=1,
        seed=11, checkpoint_path=checkpoint,
        tokenizer_type="bpe", bpe_vocab_size=280,
    )
    assert len(history) == 1
    model = load_model_checkpoint(checkpoint)
    assert model.tokenizer.vocab_size == model.config.vocab_size
    generated = generate_text(model, "The fox", max_new_tokens=2, top_k=4)
    assert isinstance(generated, str)


def test_byte_checkpoint_remains_backward_compatible(tmp_path):
    checkpoint = tmp_path / "byte-model.pt"
    train_language_model(
        "Byte tokenizer compatibility test. " * 100,
        config=TransformerConfig(
            context_length=12, embedding_dim=16,
            num_heads=2, num_layers=1, dropout=0.0,
        ),
        steps=1, batch_size=2, eval_interval=1, eval_batches=1,
        checkpoint_path=checkpoint,
    )
    model = load_model_checkpoint(checkpoint)
    assert model.tokenizer.vocab_size == 258
