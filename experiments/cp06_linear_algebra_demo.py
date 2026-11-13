"""Run a small matrix multiplication example for Checkpoint 06."""

from nn.linear_algebra import matrix_multiply, transpose


def main() -> None:
    """Print a matrix, its transpose, and a hand-checkable product."""
    left = [[1.0, 2.0], [3.0, 4.0]]
    right = [[5.0, 6.0], [7.0, 8.0]]
    print("Left:", left)
    print("Left transposed:", transpose(left))
    print("Left × right:", matrix_multiply(left, right))


if __name__ == "__main__":
    main()
