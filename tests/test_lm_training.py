"""Optional PyTorch tests for next-token training."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.corpus import prepare_corpus
from entropy_lm.model import TransformerConfig
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
    record = json.loads(train_file.read_text(encoding="utf-8").splitlines()[0])
    record["tokenizer_fingerprint"] = "stale"
    train_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid metadata"):
        train_language_model(prepared, steps=1)

def test_training_can_resume_deterministically(tmp_path) -> None:
    corpus = "A tiny corpus for deterministic checkpoint continuation. " * 100
    config = TransformerConfig(
        context_length=12,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.15,
    )
    common = {
        "config": config,
        "steps": 4,
        "batch_size": 2,
        "eval_interval": 2,
        "eval_batches": 2,
        "seed": 19,
        "learning_rate": 1e-3,
        "warmup_steps": 1,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
    }
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    partial_path = tmp_path / "partial.pt"

    uninterrupted_history = train_language_model(
        corpus, checkpoint_path=uninterrupted_path, **common
    )
    first_history = train_language_model(
        corpus,
        checkpoint_path=partial_path,
        max_steps_per_run=2,
        **common,
    )
    assert len(first_history) == 1
    partial_payload = torch.load(partial_path, map_location="cpu", weights_only=True)
    assert partial_payload["training"]["completed_steps"] == 2
    assert set(partial_payload["rng_state"]) >= {
        "torch_cpu",
        "torch_cuda",
        "train_generator",
        "training_eval_generator",
        "validation_generator",
    }

    resumed_history = train_language_model(
        corpus,
        checkpoint_path=partial_path,
        resume_from=partial_path,
        **common,
    )
    uninterrupted = torch.load(
        uninterrupted_path, map_location="cpu", weights_only=True
    )
    resumed = torch.load(partial_path, map_location="cpu", weights_only=True)
    assert resumed_history == uninterrupted_history
    assert resumed["training"]["completed_steps"] == 4
    for name, expected in uninterrupted["model_state"].items():
        assert torch.equal(resumed["model_state"][name], expected)

def test_resume_rejects_a_different_corpus_or_training_configuration(tmp_path) -> None:
    checkpoint = tmp_path / "resume.pt"
    config = TransformerConfig(
        context_length=8,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    options = {
        "config": config,
        "steps": 3,
        "batch_size": 2,
        "eval_interval": 1,
        "eval_batches": 1,
        "checkpoint_path": checkpoint,
    }
    corpus = "the same training source repeated many times. " * 40
    train_language_model(corpus, max_steps_per_run=1, **options)

    with pytest.raises(ValueError, match="corpus does not match"):
        train_language_model(
            corpus + "changed",
            resume_from=checkpoint,
            **(options | {"checkpoint_path": checkpoint}),
        )

    with pytest.raises(ValueError, match="configuration does not match"):
        train_language_model(
            corpus,
            resume_from=checkpoint,
            **(options | {"learning_rate": 1e-4}),
        )

def test_prepared_bpe_corpus_trains_and_evaluates_on_held_out_test(tmp_path) -> None:
    from entropy_lm.evaluation import evaluate_prepared_test_set

    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        (source / f"doc-{index}.txt").write_text(
            (
                f"document {index} repeats a small vocabulary for tokenizer training. "
                * 16
            ),
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    prepare_corpus(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="bpe",
        vocab_size=280,
        seed=23,
    )
    metadata = json.loads((prepared / "tokenizer.json").read_text(encoding="utf-8"))
    checkpoint = tmp_path / "bpe-model.pt"
    train_language_model(
        prepared,
        config=TransformerConfig(
            vocab_size=metadata["vocab_size"],
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

    result = evaluate_prepared_test_set(checkpoint, prepared)
    assert result["split"] == "test"
    assert result["documents_scored"] == 2
    assert result["tokens_scored"] > 0
    assert torch.isfinite(torch.tensor(result["cross_entropy_nats_per_token"]))
    assert result["perplexity"] > 1.0
    assert len(result["checkpoint_sha256"]) == 64

def test_training_supports_bf16_and_activation_checkpointing(tmp_path):
    corpus = "A small mixed precision language modeling corpus. " * 80
    checkpoint = tmp_path / "bf16-model.pt"
    history = train_language_model(
        corpus,
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
        tokenizer_type="byte",
        precision="bf16",
        gradient_checkpointing=True,
        checkpoint_path=checkpoint,
    )
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["validation_loss"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training"]["run_config"]["precision"] == "bf16"
    assert payload["training"]["run_config"]["gradient_checkpointing"] is True
    assert payload["scaler_state"] == {}

def test_fp16_training_rejects_cpu_device():
    corpus = "A corpus that is long enough to pass preprocessing. " * 80
    with pytest.raises(ValueError, match="requires a CUDA device"):
        train_language_model(corpus, precision="fp16")

def test_training_supports_rope_and_saves_architecture_metadata(tmp_path):
    from entropy_lm.inference import load_model_checkpoint

    corpus = "Rotary position embeddings are used by this tiny corpus. " * 80
    checkpoint = tmp_path / "rope-model.pt"
    history = train_language_model(
        corpus,
        config=TransformerConfig(
            vocab_size=258,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            position_encoding="rope",
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        eval_batches=1,
        tokenizer_type="byte",
        checkpoint_path=checkpoint,
    )
    assert len(history) == 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["architecture"]["position_encoding"] == "rope"
    assert "position_embedding.weight" not in payload["model_state"]
    model = load_model_checkpoint(checkpoint)
    assert model.config.position_encoding == "rope"

def test_resume_accepts_legacy_learned_position_architecture_metadata(tmp_path):
    corpus = "A compact legacy checkpoint training corpus. " * 80
    checkpoint = tmp_path / "legacy-learned.pt"
    options = {
        "config": TransformerConfig(
            vocab_size=258,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        "steps": 2,
        "batch_size": 2,
        "eval_interval": 1,
        "eval_batches": 1,
        "checkpoint_path": checkpoint,
    }
    train_language_model(corpus, max_steps_per_run=1, **options)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["architecture"].pop("position_encoding")
    payload["architecture"].pop("rope_base")
    torch.save(payload, checkpoint)

    history = train_language_model(
        corpus,
        resume_from=checkpoint,
        **options,
    )
    assert len(history) == 2
    assert history[-1]["step"] == 2

def test_training_supports_rope_sliding_attention_checkpoint(tmp_path):
    from entropy_lm.inference import load_model_checkpoint

    checkpoint = tmp_path / "rope-window.pt"
    train_language_model(
        "Sliding attention training uses a deterministic corpus. " * 80,
        config=TransformerConfig(
            vocab_size=258,
            context_length=8,
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
        eval_batches=1,
        tokenizer_type="byte",
        checkpoint_path=checkpoint,
    )
    loaded = load_model_checkpoint(checkpoint)
    assert loaded.config.attention_window == 4
