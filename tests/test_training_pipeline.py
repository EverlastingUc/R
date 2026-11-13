"""Tests for batching, validation, deterministic resume, and model checkpoints."""

import numpy as np
import pytest

from nn.losses import mean_squared_error
from nn.network import FeedForwardNetwork
from nn.optimizers import SGD, Adam
from nn.trainer import Trainer


def make_network(seed: int = 4) -> FeedForwardNetwork:
    return FeedForwardNetwork([1, 3, 2, 1], ["tanh", "tanh", "identity"], seed)


def test_fit_uses_epochs_batches_and_validation() -> None:
    network = make_network()
    optimizer = SGD(learning_rate=0.03)
    trainer = Trainer(network, optimizer, batch_size=4, seed=21, shuffle=True)
    train_inputs = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    train_targets = 2.0 * train_inputs
    validation_inputs = np.array([[-0.75], [0.25], [0.8]], dtype=np.float64)
    validation_targets = 2.0 * validation_inputs
    initial_validation_loss = mean_squared_error(
        network.forward(validation_inputs), validation_targets
    )

    history = trainer.fit(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        epochs=100,
    )

    assert len(history) == 100
    assert [metrics.epoch for metrics in history] == list(range(1, 101))
    assert history[-1].training_loss < history[0].training_loss
    assert history[-1].validation_loss < initial_validation_loss
    assert history[-1].training_loss == pytest.approx(
        mean_squared_error(network.forward(train_inputs), train_targets)
    )
    assert all(
        np.isfinite([metrics.training_loss, metrics.validation_loss]).all()
        for metrics in history
    )


def test_checkpoint_restore_reproduces_the_next_epoch(tmp_path) -> None:
    train_inputs = np.linspace(-1.0, 1.0, 7).reshape(-1, 1)
    train_targets = 0.5 * train_inputs
    validation_inputs = np.array([[-0.6], [0.7]], dtype=np.float64)
    validation_targets = 0.5 * validation_inputs

    original = Trainer(make_network(9), Adam(learning_rate=0.01), 3, seed=37)
    original.fit(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        epochs=2,
    )
    checkpoint_path = tmp_path / "session.npz"
    original.save_checkpoint(checkpoint_path)
    expected_metrics = original.fit(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        epochs=1,
    )[0]
    expected_output = original.network.forward(validation_inputs)

    restored = Trainer(make_network(99), Adam(learning_rate=0.01), 3, seed=1)
    restored.restore_checkpoint(checkpoint_path)
    actual_metrics = restored.fit(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        epochs=1,
    )[0]
    actual_output = restored.network.forward(validation_inputs)

    assert actual_metrics.epoch == expected_metrics.epoch == 3
    assert actual_metrics.training_loss == pytest.approx(
        expected_metrics.training_loss, abs=0.0
    )
    assert actual_metrics.validation_loss == pytest.approx(
        expected_metrics.validation_loss, abs=0.0
    )
    np.testing.assert_array_equal(actual_output, expected_output)


def test_checkpoint_save_does_not_overwrite_an_existing_file(tmp_path) -> None:
    trainer = Trainer(make_network(), SGD(learning_rate=0.01), 2, seed=3)
    checkpoint_path = tmp_path / "versioned.npz"
    trainer.save_checkpoint(checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()

    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(checkpoint_path)

    assert checkpoint_path.read_bytes() == original_bytes
