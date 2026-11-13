"""Hand-calculated checks for the artificial neuron."""

import math

import numpy as np
import pytest

from nn.neuron import artificial_neuron


@pytest.mark.parametrize(
    ("inputs", "weights", "bias", "expected"),
    [
        ([1.0], [2.0], 0.0, 2.0),
        ([1.0], [2.0], 1.0, 3.0),
        ([-1.0], [2.0], 1.0, -1.0),
        ([2.0, 3.0], [4.0, 5.0], 0.0, 23.0),
        ([2.0, 3.0], [4.0, 5.0], 1.0, 24.0),
        ([-2.0, 3.0], [4.0, -5.0], 0.0, -23.0),
        ([0.0, 0.0], [4.0, 5.0], 2.0, 2.0),
        ([1.0, 0.0, 0.0], [4.0, 5.0, 6.0], 2.0, 6.0),
        ([0.0, 1.0, 0.0], [4.0, 5.0, 6.0], 2.0, 7.0),
        ([0.0, 0.0, 1.0], [4.0, 5.0, 6.0], 2.0, 8.0),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], -6.0, 0.0),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.0, 6.0),
        ([0.5, 1.5], [2.0, 4.0], -1.0, 6.0),
        ([0.5, -1.0], [2.0, 3.0], 0.5, -1.5),
        ([2.0, -1.0, 0.5], [-3.0, 4.0, 2.0], 1.0, -8.0),
    ],
)
def test_identity_neuron_matches_hand_calculation(
    inputs: list[float], weights: list[float], bias: float, expected: float
) -> None:
    result = artificial_neuron(inputs, weights, bias, activation="identity")
    assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("preactivation", "expected"),
    [
        (0.0, 0.5),
        (math.log(3.0), 0.75),
        (-math.log(3.0), 0.25),
        (math.log(9.0), 0.9),
        (-math.log(9.0), 0.1),
        (math.log(99.0), 0.99),
        (-math.log(99.0), 0.01),
    ],
)
def test_sigmoid_neuron_matches_hand_calculation(
    preactivation: float, expected: float
) -> None:
    result = artificial_neuron([preactivation], [1.0], activation="sigmoid")
    assert result == pytest.approx(expected, abs=1e-12)


def test_sigmoid_is_the_default_activation() -> None:
    assert artificial_neuron([0.0], [1.0]) == pytest.approx(0.5)


def test_neuron_returns_a_python_float() -> None:
    result = artificial_neuron([1.0, 2.0], [3.0, 4.0], activation="identity")
    assert isinstance(result, float)


def test_rejects_different_input_and_weight_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        artificial_neuron([1.0, 2.0], [1.0])


def test_rejects_empty_vectors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        artificial_neuron([], [])


def test_rejects_non_vector_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        artificial_neuron([[1.0, 2.0]], [[3.0, 4.0]])


@pytest.mark.parametrize(
    ("inputs", "weights", "bias"),
    [
        ([float("nan")], [1.0], 0.0),
        ([1.0], [float("inf")], 0.0),
        ([1.0], [1.0], float("-inf")),
    ],
)
def test_rejects_non_finite_values(
    inputs: list[float], weights: list[float], bias: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        artificial_neuron(inputs, weights, bias)


def test_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        artificial_neuron([1.0], [1.0], activation="relu")


def test_large_finite_preactivations_have_finite_sigmoid_outputs() -> None:
    outputs = [
        artificial_neuron([value], [1.0], activation="sigmoid")
        for value in (-1000.0, 1000.0)
    ]
    assert np.isfinite(outputs).all()
    assert outputs[0] == 0.0
    assert outputs[1] == 1.0
