"""Independent reference checks for multilayer forward propagation."""

import numpy as np

from nn.network import FeedForwardNetwork


def make_hand_calculated_network() -> FeedForwardNetwork:
    network = FeedForwardNetwork(
        layer_sizes=[2, 2, 2, 1],
        activations=["relu", "relu", "identity"],
        seed=1,
    )
    network.layers[0].weights = np.array([[1.0, -1.0], [0.5, 2.0]])
    network.layers[0].bias = np.array([[0.0, 0.5]])
    network.layers[1].weights = np.array([[1.0, 0.0], [-1.0, 2.0]])
    network.layers[1].bias = np.array([[1.0, -1.0]])
    network.layers[2].weights = np.array([[2.0], [-0.5]])
    network.layers[2].bias = np.array([[1.0]])
    return network


def test_multilayer_forward_result_matches_independent_calculation() -> None:
    network = make_hand_calculated_network()
    output = network.forward(np.array([[1.0, 2.0]], dtype=np.float64))
    np.testing.assert_allclose(output, [[-2.0]], atol=1e-12)


def test_cached_values_match_independent_layer_calculations() -> None:
    network = make_hand_calculated_network()
    output, caches = network.forward_with_cache([[1.0, 2.0]])

    np.testing.assert_allclose(caches[0].preactivation, [[2.0, 3.5]])
    np.testing.assert_allclose(caches[0].output, [[2.0, 3.5]])
    np.testing.assert_allclose(caches[1].preactivation, [[-0.5, 6.0]])
    np.testing.assert_allclose(caches[1].output, [[0.0, 6.0]])
    np.testing.assert_allclose(caches[2].preactivation, [[-2.0]])
    np.testing.assert_allclose(output, [[-2.0]])


def test_forward_propagation_handles_a_batch_of_examples() -> None:
    network = make_hand_calculated_network()
    inputs = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    expected = np.array([[-2.0], [2.0]], dtype=np.float64)
    np.testing.assert_allclose(network.forward(inputs), expected, atol=1e-12)
