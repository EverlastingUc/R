"""Demonstrate overflow in naive softmax and stable alternatives."""

import numpy as np

from nn.numerics import float_precision_example, logsumexp, stable_softmax


def main() -> None:
    """Print stable results beside the failures of naive arithmetic."""
    logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        exponentials = np.exp(logits)
        naive_probabilities = exponentials / exponentials.sum()
    print("Naive softmax stayed finite:", bool(np.isfinite(naive_probabilities).all()))
    print("Stable softmax:", stable_softmax(logits))
    print("Stable log-sum-exp:", logsumexp(logits))
    print("Precision example:", float_precision_example())


if __name__ == "__main__":
    main()
