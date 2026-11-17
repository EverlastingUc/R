"""Optional PyTorch tests for next-token training."""

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.model import TransformerConfig  # noqa: E402
from entropy_lm.training import train_language_model  # noqa: E402


def test_training_loop_learns_and_saves_checkpoint(tmp_path) -> None:
    corpus = ("the quick brown fox jumps over the lazy dog. " * 80)
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
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(item["validation_loss"])) for item in history)
    assert checkpoint.is_file()


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
