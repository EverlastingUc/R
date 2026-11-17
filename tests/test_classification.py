"""Tests for Entropy's softmax classification path."""

import numpy as np
import pytest

from nn.classification import (
    ClassificationTrainer,
    backprop_categorical_cross_entropy,
)
from nn.network import FeedForwardNetwork
from nn.optimizers import Adam


def make_three_class_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(41)
    centers = np.array([[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]])
    inputs = np.vstack(
        [center + random.normal(0.0, 0.35, size=(30, 2)) for center in centers]
    )
    labels = np.repeat(np.arange(3, dtype=np.int64), 30)
    train_indices = np.concatenate(
        [np.arange(index * 30, index * 30 + 24) for index in range(3)]
    )
    validation_indices = np.concatenate(
        [np.arange(index * 30 + 24, index * 30 + 30) for index in range(3)]
    )
    return (
        inputs[train_indices],
        labels[train_indices],
        inputs[validation_indices],
        labels[validation_indices],
    )


def make_classifier(seed: int = 7) -> FeedForwardNetwork:
    return FeedForwardNetwork(
        [2, 12, 8, 3],
        ["relu", "tanh", "softmax"],
        seed=seed,
    )


def test_cross_entropy_backprop_matches_finite_difference() -> None:
    network = make_classifier()
    inputs = np.array([[-1.0, 0.2], [0.4, 1.1], [1.2, -0.7]])
    targets = np.eye(3, dtype=np.float64)[[0, 2, 1]]
    result = backprop_categorical_cross_entropy(network, inputs, targets)

    original = network.layers[0].weights[0, 0]
    epsilon = 1e-6
    network.layers[0].weights[0, 0] = original + epsilon
    plus = backprop_categorical_cross_entropy(network, inputs, targets).loss
    network.layers[0].weights[0, 0] = original - epsilon
    minus = backprop_categorical_cross_entropy(network, inputs, targets).loss
    network.layers[0].weights[0, 0] = original

    numerical = (plus - minus) / (2.0 * epsilon)
    analytical = result.weight_gradients[0][0, 0]
    assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-7)


def test_classifier_learns_separable_three_class_data() -> None:
    train_x, train_y, validation_x, validation_y = make_three_class_data()
    trainer = ClassificationTrainer(
        make_classifier(),
        Adam(learning_rate=0.02),
        batch_size=12,
        seed=19,
    )

    history = trainer.fit(train_x, train_y, validation_x, validation_y, epochs=100)
    probabilities = trainer.predict_proba(validation_x)
    predictions = trainer.predict(validation_x)

    assert len(history) == 100
    assert history[-1].training_loss < history[0].training_loss
    assert history[-1].validation_accuracy >= 0.95
    assert np.mean(predictions == validation_y) >= 0.95
    assert probabilities.shape == (len(validation_y), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_classifier_rejects_out_of_range_class_ids() -> None:
    trainer = ClassificationTrainer(make_classifier(), Adam())
    with pytest.raises(ValueError, match="class IDs"):
        trainer.fit(
            np.ones((3, 2)),
            np.array([0, 1, 3]),
            np.ones((3, 2)),
            np.array([0, 1, 2]),
            epochs=1,
        )


def test_classifier_requires_softmax_output() -> None:
    network = FeedForwardNetwork(
        [2, 5, 4, 3],
        ["relu", "tanh", "identity"],
    )
    with pytest.raises(ValueError, match="softmax output"):
        ClassificationTrainer(network, Adam())
