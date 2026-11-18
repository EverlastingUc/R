"""Tests for DPO preference alignment primitives."""

import torch

from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.preference_training import (
    _dpo_loss,
    _read_preference_pairs,
    _sequence_log_probabilities,
)
from entropy_lm.tokenizer import ByteTokenizer


def _tiny_model():
    tokenizer = ByteTokenizer()
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=64,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    model.tokenizer = tokenizer
    return model


def test_dpo_loss_is_finite_and_differentiable():
    chosen = torch.tensor([-2.0, -1.0], requires_grad=True)
    rejected = torch.tensor([-3.0, -2.0], requires_grad=True)
    reference_chosen = torch.tensor([-2.0, -1.0])
    reference_rejected = torch.tensor([-2.5, -1.5])
    loss = _dpo_loss(
        chosen,
        rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None


def test_sequence_log_probabilities_scores_only_response_tokens():
    model = _tiny_model()
    tokenizer = model.tokenizer
    prompt = [tokenizer.bos_id, *tokenizer.encode("prompt")]
    response = tokenizer.encode("answer") + [tokenizer.eos_id]
    sequence = prompt + response
    scores = _sequence_log_probabilities(
        model,
        [(sequence, len(prompt) - 1)],
        device=torch.device("cpu"),
    )
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()


def test_preference_reader_validates_and_tokenizes_jsonl(tmp_path):
    model = _tiny_model()
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"prompt":[{"role":"user","content":"Question?"}],'
        '"chosen":"Good answer.","rejected":"Bad answer."}\n',
        encoding="utf-8",
    )
    pairs, digest = _read_preference_pairs(path, model)
    assert len(pairs) == 1
    assert len(digest) == 64
    assert pairs[0][0] != pairs[0][2]


def test_preference_reader_rejects_identical_responses(tmp_path):
    model = _tiny_model()
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"prompt":[{"role":"user","content":"Question?"}],'
        '"chosen":"Same.","rejected":"Same."}\n',
        encoding="utf-8",
    )
    try:
        _read_preference_pairs(path, model)
    except ValueError as error:
        assert "must differ" in str(error)
    else:
        raise AssertionError("identical preference responses must be rejected")
