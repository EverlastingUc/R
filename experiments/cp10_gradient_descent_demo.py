"""Compare learning rates while minimizing a known quadratic function."""

import numpy as np

from nn.optimization import gradient_descent


def main() -> None:
    """Print convergence results for three learning rates."""
    target = np.array([3.0], dtype=np.float64)
    initial = np.array([-5.0], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        return float(np.sum((parameters - target) ** 2))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return 2.0 * (parameters - target)

    for learning_rate in (0.1, 0.5, 1.1):
        result = gradient_descent(
            objective,
            gradient,
            initial,
            learning_rate=learning_rate,
            steps=20,
        )
        print(
            f"rate={learning_rate:.1f}: parameter={result.parameters[0]:.6f}, "
            f"initial loss={result.loss_history[0]:.3f}, "
            f"final loss={result.loss_history[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
