"""Optional PyTorch tests for next-token training."""

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.model import TransformerConfig
from entropy_lm.corpus import prepare_corpus
from entropy_lm.training import train_language_model


def test_training_loop_learns_and_saves_checkpoint(tmp_path) -> None:
    corpus = "the quick brown fox jumps over the lazy dog. " * 80
    checkpoint = tmp_path / "tiny-model.pt"
    history = train_language_model(
        corpus,
        config=TransformerConfig(
            context_length=16,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=2,
        batch_size=2,
        eval_interval=1,
        eval_batches=1,
        seed=5,
        checkpoint_path=checkpoint,
        warmup_steps=1,
        minimum_learning_rate_ratio=0.2,
        gradient_accumulation_steps=2,
    )
    assert len(history) == 2
    assert all(
        torch.isfinite(torch.tensor(item["validation_loss"])) for item in history
    )
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training"]["effective_batch_size"] == 4
    assert payload["training"]["gradient_accumulation_steps"] == 2
    assert payload["training"]["best_validation_loss"] is not None
    assert "scheduler_state" in payload


def test_training_rejects_too_short_corpus() -> None:
    with pytest.raises(ValueError, match="exceed context_length"):
        train_language_model(
            "short",
            config=TransformerConfig(
                context_length=8,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
            ),
            steps=1,
        )


def test_training_rejects_invalid_accumulation_and_warmup() -> None:
    corpus = "training and validation text. " * 20
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_language_model(corpus, gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        train_language_model(corpus, steps=2, warmup_steps=2)



def test_training_consumes_prepared_document_splits(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        (source / f"doc-{index}.txt").write_text(
            f"document {index} contains enough repeated training text. " * 8,
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    prepare_corpus(
        source,
        prepared,
        validation_fraction=0.25,
        test_fraction=0.125,
        tokenizer_type="byte",
        seed=11,
    )
    checkpoint = tmp_path / "prepared-model.pt"
    history = train_language_model(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        eval_batches=1,
        checkpoint_path=checkpoint,
    )
    assert len(history) == 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training"]["corpus_sha256"]


def test_prepared_dataset_rejects_stale_tokenizer_fingerprint(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        (source / f"doc-{index}.txt").write_text(
            "a sufficiently long document for this small test. " * 8,
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    prepare_corpus(source, prepared, tokenizer_type="byte")
    train_file = prepared / "train.jsonl"
    train_file.write_text(
        train_file.read_text(encoding="utf-8").replace(
            '"tokenizer_fingerprint":', '"tokenizer_fingerprint": "stale", "ignored":'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid metadata"):
        train_language_model(prepared, steps=1)
