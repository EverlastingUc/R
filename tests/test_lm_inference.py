"""Optional PyTorch tests for safe checkpoint loading and generation."""

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.inference import generate_text, load_model_checkpoint
from entropy_lm.model import TransformerConfig
from entropy_lm.training import train_language_model


def test_checkpoint_loads_and_generates_text(tmp_path) -> None:
    checkpoint = tmp_path / "tiny-model.pt"
    train_language_model(
        "A small language model learns from text. " * 80,
        config=TransformerConfig(
            context_length=16,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        eval_batches=1,
        seed=3,
        checkpoint_path=checkpoint,
    )
    model = load_model_checkpoint(checkpoint)
    result = generate_text(model, "A", max_new_tokens=3, top_k=4)
    assert isinstance(result, str)
    assert len(result) >= 1


def test_checkpoint_loader_rejects_invalid_file(tmp_path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"format_version": 99}, checkpoint)
    with pytest.raises(ValueError, match="unsupported"):
        load_model_checkpoint(checkpoint)
