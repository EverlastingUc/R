"""Tests for held-out assistant-only SFT evaluation."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_evaluation import evaluate_chat_test_set
from entropy_lm.chat_training import train_on_conversations
from entropy_lm.model import TransformerConfig


def test_chat_evaluator_scores_only_assistant_targets_on_test_split(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(15):
        messages = [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": f"What is item {index}?"},
            {"role": "assistant", "content": f"Item {index} has value {index + 1}."},
        ]
        (source / f"{index:02d}.jsonl").write_text(
            json.dumps({"messages": messages}) + "\n",
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    manifest = prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=31,
    )
    checkpoint = tmp_path / "sft.pt"
    train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        checkpoint_path=checkpoint,
    )

    result = evaluate_chat_test_set(checkpoint, prepared, batch_size=2)
    assert result["split"] == "test"
    assert (
        result["assistant_targets_scored"]
        == manifest["assistant_target_counts"]["test"]
    )
    assert result["assistant_targets_scored"] > 0
    assert torch.isfinite(
        torch.tensor(result["assistant_cross_entropy_nats_per_token"])
    )
    assert result["assistant_perplexity"] > 1.0
    assert len(result["checkpoint_sha256"]) == 64


def test_chat_evaluator_rejects_non_sft_checkpoint(tmp_path):
    checkpoint = tmp_path / "base.pt"
    torch.save({"format_version": 2, "training": {"task": "pretraining"}}, checkpoint)
    with pytest.raises(ValueError, match="not an assistant-only SFT"):
        evaluate_chat_test_set(checkpoint, tmp_path / "missing")
