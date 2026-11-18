"""Optional PyTorch tests for the causal language model."""

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F

from entropy_lm.model import (
    CausalLanguageModel,
    TransformerConfig,
    _filter_generation_logits,
)


def test_transformer_forward_loss_and_generation() -> None:
    torch.manual_seed(7)
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = CausalLanguageModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 5, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    generated = model.generate(tokens[:, :2], max_new_tokens=3, top_k=5)
    assert generated.shape == (2, 5)


def test_transformer_cannot_see_future_tokens() -> None:
    torch.manual_seed(11)
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = CausalLanguageModel(config).eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 9, 8]], dtype=torch.long)
    first_logits, _ = model(first)
    changed_logits, _ = model(changed_future)
    assert torch.allclose(first_logits[:, :2], changed_logits[:, :2], atol=1e-6)


def test_transformer_validates_configuration_and_input() -> None:
    with pytest.raises(ValueError):
        TransformerConfig(embedding_dim=15, num_heads=4)
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
    )
    with pytest.raises(ValueError):
        model(torch.ones((1, 5), dtype=torch.long))


def test_transformer_ignores_masked_targets_in_loss() -> None:
    torch.manual_seed(17)
    config = TransformerConfig(
        vocab_size=20,
        context_length=6,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = CausalLanguageModel(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)
    labels = tokens.clone()
    labels[:, :3] = -100
    logits, loss = model(tokens, labels)
    expected = torch.nn.functional.cross_entropy(
        logits[:, 3:, :].reshape(-1, config.vocab_size),
        labels[:, 3:].reshape(-1),
    )
    assert loss is not None
    assert torch.allclose(loss, expected)


def test_transformer_rejects_all_masked_targets() -> None:
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
    )
    tokens = torch.ones((1, 4), dtype=torch.long)
    targets = torch.full_like(tokens, -100)
    with pytest.raises(ValueError, match="no supervised tokens"):
        model(tokens, targets)


def test_generation_stops_after_eos_token(monkeypatch) -> None:
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    tokens = torch.tensor([[1, 2]], dtype=torch.long)

    def sample_eos(probabilities, *, num_samples):
        return torch.full(
            (probabilities.shape[0], num_samples),
            3,
            dtype=torch.long,
            device=probabilities.device,
        )

    monkeypatch.setattr(torch, "multinomial", sample_eos)
    generated = model.generate(tokens, max_new_tokens=8, top_k=1, eos_token_id=3)
    assert generated.shape == (1, 3)
    assert generated[0, -1].item() == 3


def test_generation_rejects_invalid_eos_token():
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
    )
    with pytest.raises(ValueError, match="eos_token_id"):
        model.generate(
            torch.tensor([[1, 2]], dtype=torch.long),
            max_new_tokens=1,
            eos_token_id=16,
        )


def test_forward_with_cache_matches_full_sequence_logits():
    torch.manual_seed(101)
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=32,
            context_length=12,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
    ).eval()
    tokens = torch.randint(0, 32, (2, 7), dtype=torch.long)

    full_logits, _ = model(tokens)
    cached_prefill_logits, cache = model.forward_with_cache(tokens[:, :-1])
    cached_last_logits, _ = model.forward_with_cache(tokens[:, -1:], cache)

    assert torch.allclose(cached_prefill_logits, full_logits[:, :-1], atol=1e-5)
    assert torch.allclose(cached_last_logits[:, -1], full_logits[:, -1], atol=1e-5)


def test_generation_with_cache_matches_recomputed_sliding_window():
    torch.manual_seed(103)
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=24,
            context_length=6,
            embedding_dim=12,
            num_heads=3,
            num_layers=2,
            dropout=0.0,
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    torch.manual_seed(107)
    cached = model.generate(prompt, max_new_tokens=5, top_k=1)

    torch.manual_seed(107)
    expected = prompt.clone()
    for _ in range(5):
        context = expected[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        threshold = torch.topk(next_logits, 1, dim=-1).values[:, -1]
        next_logits = next_logits.masked_fill(
            next_logits < threshold.unsqueeze(-1), float("-inf")
        )
        probabilities = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        expected = torch.cat((expected, next_token), dim=1)

    assert torch.equal(cached, expected)


def test_forward_with_cache_rejects_incompatible_cache():
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    ).eval()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    _, cache = model.forward_with_cache(tokens)
    with pytest.raises(ValueError, match="one entry per Transformer layer"):
        model.forward_with_cache(tokens, cache * 2)
    with pytest.raises(ValueError, match="exceeds configured context_length"):
        model.forward_with_cache(torch.ones((1, 7), dtype=torch.long), cache)


def test_gradient_checkpointing_matches_normal_parameter_gradients():
    torch.manual_seed(211)
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    baseline = CausalLanguageModel(config).train()
    checkpointed = CausalLanguageModel(config).train()
    checkpointed.load_state_dict(baseline.state_dict())
    checkpointed.gradient_checkpointing_enable()

    tokens = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    _, baseline_loss = baseline(tokens, tokens)
    _, checkpointed_loss = checkpointed(tokens, tokens)
    assert baseline_loss is not None and checkpointed_loss is not None
    baseline_loss.backward()
    checkpointed_loss.backward()
    assert torch.allclose(baseline_loss, checkpointed_loss, atol=1e-6)
    for baseline_parameter, checkpointed_parameter in zip(
        baseline.parameters(), checkpointed.parameters()
    ):
        assert baseline_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.allclose(
            baseline_parameter.grad, checkpointed_parameter.grad, atol=1e-6
        )


def test_generation_filters_apply_top_p_and_repetition_penalty():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, -2.0]])
    history = torch.tensor([[1, 4]], dtype=torch.long)
    penalized = _filter_generation_logits(
        logits.clone(),
        history,
        top_k=None,
        top_p=1.0,
        repetition_penalty=2.0,
    )
    assert penalized[0, 1].item() == pytest.approx(1.5)
    assert penalized[0, 4].item() == pytest.approx(-4.0)

    nucleus = _filter_generation_logits(
        logits.clone(),
        history[:, :0],
        top_k=None,
        top_p=0.8,
        repetition_penalty=1.0,
    )
    assert torch.isfinite(nucleus[0, 0])
    assert torch.isfinite(nucleus[0, 1])
    assert torch.isneginf(nucleus[0, 2:]).all()


def test_generation_filters_keep_defaults_and_validate_sampling_controls():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    history = torch.tensor([[0, 1]], dtype=torch.long)
    unchanged = _filter_generation_logits(
        logits.clone(),
        history,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    assert torch.equal(unchanged, logits)

    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
    )
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    for kwargs in (
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"top_p": True},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("inf")},
        {"repetition_penalty": True},
    ):
        with pytest.raises(ValueError):
            model.generate(prompt, max_new_tokens=1, **kwargs)


def test_generation_repetition_penalty_uses_recent_context_window():
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=16,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=2,
        top_k=1,
        repetition_penalty=1.2,
    )
    assert generated.shape == (1, 6)



@pytest.mark.parametrize("position_encoding", ["learned", "rope"])
def test_cached_attention_matches_full_forward_for_position_modes(position_encoding):
    torch.manual_seed(307)
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=32,
            context_length=12,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            position_encoding=position_encoding,
        )
    ).eval()
    tokens = torch.randint(0, 32, (2, 7), dtype=torch.long)
    full_logits, _ = model(tokens)
    prefix_logits, cache = model.forward_with_cache(tokens[:, :-1])
    final_logits, _ = model.forward_with_cache(tokens[:, -1:], cache)
    assert torch.allclose(prefix_logits, full_logits[:, :-1], atol=1e-5)
    assert torch.allclose(final_logits[:, -1], full_logits[:, -1], atol=1e-5)


def test_rope_forward_is_causal_and_supports_cached_sliding_generation():
    torch.manual_seed(311)
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=24,
            context_length=6,
            embedding_dim=12,
            num_heads=3,
            num_layers=2,
            dropout=0.0,
            position_encoding="rope",
        )
    ).eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    future_changed = torch.tensor([[1, 2, 9, 8]], dtype=torch.long)
    first_logits, _ = model(first)
    changed_logits, _ = model(future_changed)
    assert torch.allclose(first_logits[:, :2], changed_logits[:, :2], atol=1e-5)

    torch.manual_seed(313)
    generated = model.generate(first, max_new_tokens=8, top_k=1)
    assert generated.shape == (1, 12)


def test_rope_validation_and_legacy_architecture_metadata():
    from dataclasses import asdict

    from entropy_lm.model import architecture_matches_checkpoint

    with pytest.raises(ValueError, match="even attention head dimension"):
        TransformerConfig(
            embedding_dim=9,
            num_heads=3,
            position_encoding="rope",
        )
    with pytest.raises(ValueError, match="rope_base"):
        TransformerConfig(rope_base=1.0)

    learned = TransformerConfig()
    legacy = asdict(learned)
    legacy.pop("position_encoding")
    legacy.pop("rope_base")
    assert architecture_matches_checkpoint(legacy, learned)
    rope = TransformerConfig(position_encoding="rope")
    assert not architecture_matches_checkpoint(legacy, rope)


def test_rope_gradient_checkpointing_matches_regular_gradients():
    torch.manual_seed(317)
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        position_encoding="rope",
    )
    baseline = CausalLanguageModel(config).train()
    checkpointed = CausalLanguageModel(config).train()
    checkpointed.load_state_dict(baseline.state_dict())
    checkpointed.gradient_checkpointing_enable()
    tokens = torch.randint(0, config.vocab_size, (2, 6), dtype=torch.long)
    _, baseline_loss = baseline(tokens, tokens)
    _, checkpointed_loss = checkpointed(tokens, tokens)
    assert baseline_loss is not None and checkpointed_loss is not None
    baseline_loss.backward()
    checkpointed_loss.backward()
    assert torch.allclose(baseline_loss, checkpointed_loss, atol=1e-6)
    for expected, actual in zip(baseline.parameters(), checkpointed.parameters()):
        if expected.grad is None or actual.grad is None:
            assert expected.grad is None and actual.grad is None
        else:
            assert torch.allclose(expected.grad, actual.grad, atol=1e-6)
