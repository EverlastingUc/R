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


def test_sft_training_resumes_exactly_after_a_non_evaluation_step(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=37,
    )
    config = TransformerConfig(
        vocab_size=258,
        context_length=16,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.15,
    )
    options = {
        "config": config,
        "steps": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "eval_interval": 2,
        "warmup_steps": 1,
        "seed": 41,
        "gradient_checkpointing": True,
    }
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    partial_path = tmp_path / "partial.pt"
    uninterrupted_history = train_on_conversations(
        prepared, checkpoint_path=uninterrupted_path, **options
    )
    partial_history = train_on_conversations(
        prepared,
        checkpoint_path=partial_path,
        max_steps_per_run=1,
        **options,
    )
    assert partial_history == []
    partial = torch.load(partial_path, map_location="cpu", weights_only=True)
    assert partial["training"]["completed_steps"] == 1
    assert partial["training"]["has_best_checkpoint"] is False
    assert set(partial["resume_state"]) == {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
    }
    assert set(partial["resume_state"]["rng_state"]) == {
        "torch_cpu",
        "torch_cuda",
        "sampling_generator",
    }

    resumed_history = train_on_conversations(
        prepared,
        checkpoint_path=partial_path,
        resume_from=partial_path,
        **options,
    )
    uninterrupted = torch.load(
        uninterrupted_path, map_location="cpu", weights_only=True
    )
    resumed = torch.load(partial_path, map_location="cpu", weights_only=True)
    assert resumed_history == uninterrupted_history
    assert resumed["training"]["completed_steps"] == 4
    for name, expected in uninterrupted["model_state"].items():
        assert torch.equal(resumed["model_state"][name], expected)
    for name, expected in uninterrupted["resume_state"]["model_state"].items():
        assert torch.equal(resumed["resume_state"]["model_state"][name], expected)


def test_sft_resume_rejects_mismatched_dataset_and_training_config(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=43,
    )
    checkpoint = tmp_path / "resume.pt"
    options = {
        "config": TransformerConfig(
            vocab_size=258,
            context_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        "steps": 3,
        "batch_size": 2,
        "eval_interval": 1,
        "checkpoint_path": checkpoint,
    }
    train_on_conversations(prepared, max_steps_per_run=1, **options)
    with pytest.raises(ValueError, match="training configuration"):
        train_on_conversations(
            prepared,
            resume_from=checkpoint,
            **(options | {"learning_rate": 1e-4}),
        )

    manifest = prepared / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus does not match"):
        train_on_conversations(prepared, resume_from=checkpoint, **options)


def test_sft_supports_bf16_and_activation_checkpointing(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=53,
    )
    checkpoint = tmp_path / "bf16-assistant.pt"
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
        steps=1,
        batch_size=2,
        eval_interval=1,
        precision="bf16",
        gradient_checkpointing=True,
        checkpoint_path=checkpoint,
    )
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["validation_loss"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training"]["run_config"]["precision"] == "bf16"
    assert payload["training"]["run_config"]["gradient_checkpointing"] is True
    assert payload["resume_state"]["scaler_state"] == {}


def test_sft_fp16_training_rejects_cpu_device(tmp_path):
    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=59,
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        train_on_conversations(prepared, precision="fp16")


def test_sft_supports_rope_position_encoding_and_checkpoint_loading(tmp_path):
    from entropy_lm.inference import load_model_checkpoint

    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=67,
    )
    checkpoint = tmp_path / "rope-assistant.pt"
    history = train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            position_encoding="rope",
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        checkpoint_path=checkpoint,
    )
    assert len(history) == 1
    model = load_model_checkpoint(checkpoint)
    assert model.config.position_encoding == "rope"
    assert model.position_embedding is None


def test_sft_supports_rope_sliding_attention(tmp_path):
    from entropy_lm.inference import load_model_checkpoint

    source = tmp_path / "source"
    _write_chat_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=71,
    )
    checkpoint = tmp_path / "assistant-window.pt"
    train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            position_encoding="rope",
            attention_window=4,
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        checkpoint_path=checkpoint,
    )
    loaded = load_model_checkpoint(checkpoint)
    assert loaded.config.attention_window == 4
