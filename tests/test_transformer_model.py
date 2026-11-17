"""Optional PyTorch tests for the causal language model."""

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.model import CausalLanguageModel, TransformerConfig


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
    generated = model.generate(
        tokens, max_new_tokens=8, top_k=1, eos_token_id=3
    )
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
