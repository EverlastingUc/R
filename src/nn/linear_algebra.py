"""Small readable vector and matrix operations for the math checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _validated_matrix(values: Matrix, name: str) -> list[list[float]]:
    """Return a rectangular, non-empty finite matrix as Python floats."""
    matrix = [[float(value) for value in row] for row in values]
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be non-empty")
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def dot_product(left: Vector, right: Vector) -> float:
    """Return the dot product of two vectors with shape ``(n,)``."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values:
        raise ValueError("vectors must be non-empty")
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same length")
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("vectors must contain only finite values")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    )


def transpose(values: Matrix) -> list[list[float]]:
    """Return the transpose of a matrix with shape ``(rows, columns)``."""
    matrix = _validated_matrix(values, "matrix")
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(left: Matrix, right: Matrix) -> list[list[float]]:
    """Multiply ``(m, n)`` and ``(n, p)`` matrices to produce ``(m, p)``."""
    left_values = _validated_matrix(left, "left matrix")
    right_values = _validated_matrix(right, "right matrix")
    if len(left_values[0]) != len(right_values):
        raise ValueError("matrix dimensions do not align for multiplication")

    right_columns = transpose(right_values)
    return [
        [dot_product(row, column) for column in right_columns] for row in left_values
    ]


def matrix_vector_multiply(matrix: Matrix, vector: Vector) -> list[float]:
    """Multiply an ``(m, n)`` matrix by a vector of shape ``(n,)``."""
    matrix_values = _validated_matrix(matrix, "matrix")
    vector_values = [float(value) for value in vector]
    if len(matrix_values[0]) != len(vector_values):
        raise ValueError("matrix and vector dimensions do not align")
    return [dot_product(row, vector_values) for row in matrix_values]
