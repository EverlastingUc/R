"""Tests for the small linear-algebra routines used in Entropy."""

import numpy as np
import pytest

from nn.linear_algebra import (
    dot_product,
    matrix_multiply,
    matrix_vector_multiply,
    transpose,
)


def test_dot_product_matches_hand_calculation() -> None:
    assert dot_product([1.0, 2.0, 3.0], [4.0, -1.0, 2.0]) == pytest.approx(8.0)


def test_dot_product_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        dot_product([1.0, 2.0], [1.0])


def test_transpose_matches_hand_calculation() -> None:
    assert transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == [
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ]


def test_transpose_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        transpose([[1.0, 2.0], [3.0]])


def test_matrix_multiply_matches_hand_calculation() -> None:
    result = matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
    assert result == [[19.0, 22.0], [43.0, 50.0]]


def test_matrix_multiply_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions do not align"):
        matrix_multiply([[1.0, 2.0]], [[1.0, 2.0]])


def test_matrix_vector_multiply_matches_hand_calculation() -> None:
    result = matrix_vector_multiply([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    assert result == [17.0, 39.0]


def test_matrix_operations_match_numpy_reference() -> None:
    random = np.random.default_rng(606)
    left = random.normal(size=(3, 4)).astype(np.float64)
    right = random.normal(size=(4, 2)).astype(np.float64)
    vector = random.normal(size=4).astype(np.float64)

    np.testing.assert_allclose(
        matrix_multiply(left.tolist(), right.tolist()), left @ right
    )
    np.testing.assert_allclose(
        matrix_vector_multiply(left.tolist(), vector.tolist()), left @ vector
    )
    assert dot_product(vector.tolist(), vector.tolist()) == pytest.approx(
        vector @ vector
    )
    np.testing.assert_array_equal(transpose(left.tolist()), left.T)
