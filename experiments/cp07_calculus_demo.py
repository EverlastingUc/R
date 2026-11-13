"""Print derivatives for a small sigmoid neuron."""

import numpy as np

from nn.calculus import sigmoid, sigmoid_derivative, sigmoid_neuron_mse_gradients


def main() -> None:
    """Show a sigmoid value and chain-rule gradients for one example."""
    inputs = np.array([1.0, 2.0], dtype=np.float64)
    weights = np.array([0.2, -0.3], dtype=np.float64)
    weight_gradient, bias_gradient = sigmoid_neuron_mse_gradients(
        inputs, weights, bias=0.1, target=1.0
    )
    print(f"sigmoid(0) = {sigmoid(0.0):.3f}")
    print(f"sigmoid derivative at 0 = {sigmoid_derivative(0.0):.3f}")
    print("weight gradients:", weight_gradient)
    print(f"bias gradient: {bias_gradient:.6f}")


if __name__ == "__main__":
    main()
