"""Optional PyTorch tests for assistant-response supervised fine-tuning."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_training import _load_chat_dataset, train_on_conversations
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.model import TransformerConfig


def _write_chat_corpus(path):
    path.mkdir(parents=True, exist_ok=True)
    for index in range(15):
        record = {
            "messages": [
                {"role": "system", "content": "Answer clearly."},
                {"role": "user", "content": f"What is the result for item {index}?"},
                {
                    "role": "assistant",
                    "content": f"The result for item {index} is {index * 2}.",
                },
            ]
        }
        (path / f"{index:02d}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )


def test_sft_loader_builds_supervised_windows_only(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=17,
    )
    train_inputs, train_targets, val_inputs, val_targets, tokenizer, corpus_hash = (
        _load_chat_dataset(prepared, context_length=16)
    )
    assert train_inputs.shape == train_targets.shape
    assert val_inputs.shape == val_targets.shape
    assert train_inputs.shape[1] == 16
    assert torch.any(train_targets != -100)
    assert torch.all(
        (train_targets == -100)
        | ((train_targets >= 0) & (train_targets < tokenizer.vocab_size))
    )
    assert len(corpus_hash) == 64


def test_sft_training_saves_inference_compatible_checkpoint(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=19,
    )
    checkpoint = tmp_path / "assistant-model.pt"
    history = train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=2,
        batch_size=2,
        gradient_accumulation_steps=2,
        eval_interval=1,
        warmup_steps=1,
        seed=23,
        checkpoint_path=checkpoint,
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(row["validation_loss"])) for row in history)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training"]["assistant_only_loss"] is True
    assert payload["training"]["task"] == "assistant-response-supervised-finetuning"
    assert payload["training"]["corpus_sha256"]
    model = load_model_checkpoint(checkpoint)
    assert model.config.vocab_size == 258


def test_sft_loader_rejects_label_id_mismatch(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=29,
    )
    train_path = prepared / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    supervised = next(
        label_index
        for label_index, value in enumerate(records[0]["labels"])
        if value != -100
    )
    records[0]["labels"][supervised] = 1
    train_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="do not align"):
        _load_chat_dataset(prepared, context_length=16)
