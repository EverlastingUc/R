"""Optional PyTorch tests for model profiling."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.model import TransformerConfig
from entropy_lm.profiling import profile_checkpoint
from entropy_lm.training import train_language_model


def test_profile_checkpoint_reports_latency_throughput_and_provenance(tmp_path):
    checkpoint = tmp_path / "profile-model.pt"
    train_language_model(
        "Profiling uses a small repeatable training corpus. " * 80,
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
        eval_batches=1,
        seed=31,
        checkpoint_path=checkpoint,
    )

    torch.manual_seed(901)
    rng_before = torch.get_rng_state().clone()
    output_path = tmp_path / "reports" / "profile.json"
    report = profile_checkpoint(
        checkpoint,
        prompt="Measure this model.",
        max_new_tokens=3,
        batch_size=2,
        warmup_steps=0,
        seed=73,
        output_path=output_path,
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    assert report["benchmark"] == "entropy-inference-profile"
    assert report["parameters"]["count"] > 0
    assert report["parameters"]["storage_bytes"] > 0
    assert report["checkpoint_bytes"] == checkpoint.stat().st_size
    assert len(report["checkpoint_sha256"]) == 64
    assert report["latency"]["prefill_seconds"] > 0.0
    assert report["latency"]["first_token_seconds"] > 0.0
    assert report["latency"]["generation_seconds"] > 0.0
    assert report["throughput"]["generated_tokens"] == 6
    assert report["throughput"]["tokens_per_second"] > 0.0
    assert report["memory"]["peak_cuda_allocated_bytes"] is None
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"warmup_steps": -1},
        {"warmup_steps": True},
    ],
)
def test_profile_rejects_invalid_measurement_settings(tmp_path, overrides):
    with pytest.raises(ValueError):
        profile_checkpoint(tmp_path / "missing.pt", **overrides)


def test_profile_rejects_cuda_when_unavailable(tmp_path, monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this test environment")
    with pytest.raises(ValueError, match="CUDA profiling requires"):
        profile_checkpoint(tmp_path / "missing.pt", device="cuda")
