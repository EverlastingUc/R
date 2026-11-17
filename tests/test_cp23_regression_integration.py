"""End-to-end tests for the CP23 nonlinear regression benchmark."""

import numpy as np

from experiments.cp23_regression_baseline import make_dataset, mean_squared_error
from nn.network import FeedForwardNetwork
from nn.optimizers import Adam
from nn.trainer import Trainer


def test_regression_network_learns_nonlinear_function() -> None:
    """Training must reduce loss and generalize to the untouched test split."""
    train_x, train_y, validation_x, validation_y, test_x, test_y, _ = make_dataset(
        seed=2026, samples=1200
    )
    network = FeedForwardNetwork(
        [4, 32, 16, 1],
        ["relu", "tanh", "identity"],
        seed=2026,
    )
    trainer = Trainer(
        network, Adam(learning_rate=0.003), batch_size=32, seed=2026
    )
    initial_validation_mse = mean_squared_error(network.forward(validation_x), validation_y)
    history = trainer.fit(
        train_x, train_y, validation_x, validation_y, epochs=220
    )
    test_mse = mean_squared_error(network.forward(test_x), test_y)

    assert history[-1].training_loss < history[0].training_loss
    assert history[-1].validation_loss < initial_validation_mse * 0.35
    assert test_mse < 0.35
    assert np.isfinite(test_mse)


def test_regression_dataset_is_deterministic_and_disjoint() -> None:
    """A fixed seed must reproduce the same normalized partitions."""
    first = make_dataset(seed=77, samples=300)
    second = make_dataset(seed=77, samples=300)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first[0].shape == (210, 4)
    assert first[2].shape == (45, 4)
    assert first[4].shape == (45, 4)
