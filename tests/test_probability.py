"""Tests for probability calculations and deterministic sampling."""

import numpy as np
import pytest

from nn.probability import (
    conditional_probability,
    empirical_distribution,
    entropy_bits,
    expected_value,
    normalize_probabilities,
    sample_categorical,
    variance,
)


def test_probability_normalization_matches_hand_calculation() -> None:
    np.testing.assert_allclose(
        normalize_probabilities([2.0, 3.0, 5.0]), [0.2, 0.3, 0.5]
    )


def test_expected_value_matches_hand_calculation() -> None:
    assert expected_value([0.0, 1.0], [0.25, 0.75]) == pytest.approx(0.75)


def test_variance_matches_hand_calculation() -> None:
    assert variance([0.0, 2.0], [0.5, 0.5]) == pytest.approx(1.0)


def test_entropy_of_fair_binary_distribution_is_one_bit() -> None:
    assert entropy_bits([0.5, 0.5]) == pytest.approx(1.0)


def test_entropy_handles_zero_probability() -> None:
    assert entropy_bits([1.0, 0.0]) == pytest.approx(0.0)


def test_conditional_probability_matches_joint_table() -> None:
    joint = [[0.1, 0.2], [0.3, 0.4]]
    assert conditional_probability(
        joint, outcome_index=1, condition_index=1
    ) == pytest.approx(2.0 / 3.0)


def test_sampling_is_reproducible_with_a_fixed_seed() -> None:
    first = sample_categorical([0.25, 0.75], sample_count=100, seed=808)
    second = sample_categorical([0.25, 0.75], sample_count=100, seed=808)
    np.testing.assert_array_equal(first, second)


def test_sampling_frequencies_approximate_requested_probabilities() -> None:
    samples = sample_categorical([0.25, 0.75], sample_count=20_000, seed=808)
    frequencies = empirical_distribution(samples, category_count=2)
    np.testing.assert_allclose(frequencies, [0.25, 0.75], atol=0.01)


def test_probability_normalization_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="positive total"):
        normalize_probabilities([0.0, 0.0])


def test_conditional_probability_rejects_zero_probability_condition() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        conditional_probability([[0.5, 0.0], [0.5, 0.0]], 0, 1)
