"""Tests for gradient descent on known quadratic functions."""

import numpy as np
import pytest

from nn.optimization import gradient_descent


def squared_distance(parameters: np.ndarray) -> float:
    return float(np.sum((parameters - 3.0) ** 2))


def squared_distance_gradient(parameters: np.ndarray) -> np.ndarray:
    return 2.0 * (parameters - 3.0)


def test_gradient_descent_converges_to_known_minimum() -> None:
    result = gradient_descent(
        objective=squared_distance,
        gradient=squared_distance_gradient,
        initial_parameters=np.array([-5.0]),
        learning_rate=0.1,
        steps=100,
    )
    assert result.parameters[0] == pytest.approx(3.0, abs=1e-8)
    assert result.loss_history[-1] < result.loss_history[0]


def test_loss_history_contains_initial_and_each_updated_loss() -> None:
    result = gradient_descent(
        squared_distance,
        squared_distance_gradient,
        np.array([0.0]),
        learning_rate=0.1,
        steps=4,
    )
    assert result.loss_history.shape == (5,)


def test_larger_stable_learning_rate_converges_faster_on_quadratic() -> None:
    initial = np.array([10.0])
    slow = gradient_descent(
        lambda values: float(0.5 * np.sum(values**2)),
        lambda values: values,
        initial,
        learning_rate=0.1,
        steps=20,
    )
    faster = gradient_descent(
        lambda values: float(0.5 * np.sum(values**2)),
        lambda values: values,
        initial,
        learning_rate=0.5,
        steps=20,
    )
    assert faster.loss_history[-1] < slow.loss_history[-1]


def test_oversized_learning_rate_increases_quadratic_loss() -> None:
    result = gradient_descent(
        lambda values: float(0.5 * np.sum(values**2)),
        lambda values: values,
        np.array([1.0]),
        learning_rate=2.1,
        steps=1,
    )
    assert result.loss_history[-1] > result.loss_history[0]


def test_gradient_descent_does_not_mutate_initial_parameters() -> None:
    initial = np.array([-5.0])
    before = initial.copy()
    gradient_descent(
        squared_distance,
        squared_distance_gradient,
        initial,
        learning_rate=0.1,
        steps=1,
    )
    np.testing.assert_array_equal(initial, before)


def test_rejects_gradient_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="same shape"):
        gradient_descent(
            lambda values: float(np.sum(values**2)),
            lambda values: np.array([1.0]),
            np.array([1.0, 2.0]),
            learning_rate=0.1,
            steps=1,
        )


def test_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        gradient_descent(
            squared_distance,
            squared_distance_gradient,
            np.array([0.0]),
            learning_rate=0.0,
            steps=1,
        )
