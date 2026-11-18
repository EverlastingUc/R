"""Optional PyTorch tests for the reference-based chat benchmark."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.chat_benchmark import (
    _contains_phrase,
    _read_benchmark_dataset,
    _score_case,
    _token_f1,
    evaluate_chat_benchmark,
)
from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_training import train_on_conversations
from entropy_lm.model import TransformerConfig


def _write_training_corpus(path):
    path.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        messages = [
            {"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": f"What is item {index}?"},
            {"role": "assistant", "content": f"Item {index} has value {index + 1}."},
        ]
        (path / f"{index:02d}.jsonl").write_text(
            json.dumps({"messages": messages}) + "\n", encoding="utf-8"
        )


def test_reference_scorer_reports_exact_match_and_phrase_checks():
    scores = _score_case(
        "The answer is FOUR.",
        "The answer is four.",
        ["four", "answer"],
        ["five"],
    )
    assert scores["normalized_exact_match"] is True
    assert scores["token_f1"] == 1.0
    assert scores["required_terms"] == {"four": True, "answer": True}
    assert scores["forbidden_terms_found"] == []


def test_reference_scorer_penalizes_missing_and_forbidden_terms():
    scores = _score_case(
        "The answer is five.", "The answer is four.", ["four"], ["five"]
    )
    assert scores["normalized_exact_match"] is False
    assert 0.0 < scores["token_f1"] < 1.0
    assert scores["required_terms"] == {"four": False}
    assert scores["forbidden_terms_found"] == ["five"]


def test_benchmark_dataset_rejects_duplicate_ids_and_invalid_roles(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    valid = {
        "id": "same",
        "messages": [{"role": "user", "content": "What is two plus two?"}],
        "reference": "4",
    }
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate benchmark case id"):
        _read_benchmark_dataset(path)

    invalid = valid | {
        "id": "bad-role",
        "messages": [{"role": "system", "content": "Only a system prompt."}],
    }
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="user message"):
        _read_benchmark_dataset(path)


def test_end_to_end_chat_benchmark_is_reproducible_and_preserves_rng(tmp_path):
    source = tmp_path / "training-source"
    _write_training_corpus(source)
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=17,
    )
    checkpoint = tmp_path / "assistant.pt"
    train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=64,
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

    benchmark = tmp_path / "benchmark.jsonl"
    cases = [
        {
            "id": "arithmetic-addition",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "reference": "4",
            "must_include": ["4"],
            "must_not_include": ["5"],
        },
        {
            "id": "capital",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "reference": "Paris",
        },
    ]
    benchmark.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )

    torch.manual_seed(987)
    rng_before = torch.get_rng_state().clone()
    output_path = tmp_path / "reports" / "benchmark.json"
    first = evaluate_chat_benchmark(
        checkpoint,
        benchmark,
        output_path=output_path,
        max_new_tokens=8,
        top_k=1,
        seed=123,
    )
    rng_after = torch.get_rng_state()
    second = evaluate_chat_benchmark(
        checkpoint,
        benchmark,
        max_new_tokens=8,
        top_k=1,
        seed=123,
    )

    assert torch.equal(rng_before, rng_after)
    assert first["cases"] == second["cases"]
    assert first["metrics"] == second["metrics"]
    assert first["samples"] == 2
    assert first["generation"]["top_p"] == 1.0
    assert first["generation"]["repetition_penalty"] == 1.0
    assert first["benchmark_dataset_sha256"]
    assert len(first["checkpoint_sha256"]) == 64
    assert 0.0 <= first["metrics"]["normalized_exact_match_rate"] <= 1.0
    assert 0.0 <= first["metrics"]["mean_token_f1"] <= 1.0
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == first


def test_phrase_checks_match_word_sequences_not_substrings():
    assert _contains_phrase("The assistant is helpful.", "assistant is")
    assert _contains_phrase("Paris, France", "paris france")
    assert _contains_phrase("Paris, France", "Paris")
    assert _contains_phrase("the answer", "he") is False
    assert _contains_phrase("a café is here", "café")
    assert _contains_phrase("the answer", "the answer")
    assert _contains_phrase("the answer", "answer the") is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_new_tokens": True},
        {"max_new_tokens": 0},
        {"max_new_tokens": 2.5},
        {"temperature": True},
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"top_k": True},
        {"top_k": 0},
        {"top_k": 1.5},
    ],
)
def test_chat_benchmark_rejects_invalid_generation_parameters(tmp_path, overrides):
    from entropy_lm.chat_benchmark import evaluate_chat_benchmark

    with pytest.raises(ValueError):
        evaluate_chat_benchmark(
            tmp_path / "missing.pt",
            tmp_path / "missing.jsonl",
            **overrides,
        )


def test_token_f1_returns_zero_when_nonempty_answers_do_not_overlap():
    assert _token_f1("cat", "dog") == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_p": 0.0},
        {"top_p": 1.01},
        {"top_p": float("nan")},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("inf")},
    ],
)
def test_chat_benchmark_rejects_invalid_sampling_controls(tmp_path, overrides):
    with pytest.raises(ValueError):
        evaluate_chat_benchmark(
            tmp_path / "missing.pt",
            tmp_path / "missing.jsonl",
            **overrides,
        )
