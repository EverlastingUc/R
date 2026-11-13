"""Controlled convergence checks for the Phase 04 optimizers."""

import numpy as np
import pytest

from nn.optimizers import SGD, Adam, Momentum, RMSProp


@pytest.mark.parametrize(
    "optimizer",
    [
        SGD(learning_rate=0.05),
        Momentum(learning_rate=0.05, momentum=0.9),
        RMSProp(learning_rate=0.05, decay=0.9),
        Adam(learning_rate=0.05),
    ],
    ids=["sgd", "momentum", "rmsprop", "adam"],
)
def test_optimizer_reduces_same_quadratic_objective(optimizer: object) -> None:
    parameters = np.array([4.0, -3.0], dtype=np.float64)
    target = np.array([1.0, 2.0], dtype=np.float64)
    initial_loss = 0.5 * float(np.sum((parameters - target) ** 2))

    for _ in range(200):
        optimizer.step([parameters], [parameters - target])

    final_loss = 0.5 * float(np.sum((parameters - target) ** 2))
    assert final_loss < initial_loss
    assert final_loss < 0.01
    assert np.isfinite(parameters).all()


def test_optimizer_rejects_mismatched_parameter_and_gradient_shapes() -> None:
    optimizer = SGD(learning_rate=0.1)
    parameters = np.array([1.0, 2.0], dtype=np.float64)
    original = parameters.copy()

    with pytest.raises(ValueError, match="same shape"):
        optimizer.step([parameters], [np.array([1.0])])

    np.testing.assert_array_equal(parameters, original)


def test_sgd_and_momentum_first_steps_match_their_update_rules() -> None:
    sgd_parameter = np.array([1.0], dtype=np.float64)
    momentum_parameter = sgd_parameter.copy()
    gradient = np.array([2.0], dtype=np.float64)

    SGD(learning_rate=0.1).step([sgd_parameter], [gradient])
    Momentum(learning_rate=0.1, momentum=0.9).step([momentum_parameter], [gradient])

    np.testing.assert_allclose(sgd_parameter, [0.8])
    np.testing.assert_allclose(momentum_parameter, [0.8])


def test_rmsprop_first_step_uses_squared_gradient_average() -> None:
    parameter = np.array([1.0], dtype=np.float64)
    optimizer = RMSProp(learning_rate=0.1, decay=0.9, epsilon=1e-8)
    gradient = np.array([2.0], dtype=np.float64)

    optimizer.step([parameter], [gradient])

    expected = 1.0 - 0.1 * 2.0 / (np.sqrt(0.1 * 2.0**2) + 1e-8)
    np.testing.assert_allclose(parameter, [expected])


def test_adam_first_step_uses_bias_corrected_moments() -> None:
    parameter = np.array([1.0], dtype=np.float64)
    optimizer = Adam(learning_rate=0.1, epsilon=1e-8)
    gradient = np.array([2.0], dtype=np.float64)

    optimizer.step([parameter], [gradient])

    expected = 1.0 - 0.1 * 2.0 / (abs(2.0) + 1e-8)
    np.testing.assert_allclose(parameter, [expected])


def test_rmsprop_rejects_nonfinite_state_without_mutating_parameters() -> None:
    parameter = np.array([1.0], dtype=np.float64)
    optimizer = RMSProp(learning_rate=0.1)
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="state became non-finite"),
    ):
        optimizer.step([parameter], [np.array([1e200])])

    np.testing.assert_array_equal(parameter, [1.0])
    assert optimizer.mean_square == []


@pytest.mark.parametrize(
    ("name", "settings"),
    [
        ("sgd", {"learning_rate": 0.1}),
        ("momentum", {"learning_rate": 0.1, "momentum": 0.8}),
        ("rmsprop", {"learning_rate": 0.01, "decay": 0.9}),
        ("adam", {"learning_rate": 0.01, "beta1": 0.8}),
    ],
)
def test_optimizer_state_can_be_restored(name: str, settings: dict) -> None:
    constructors = {
        "sgd": SGD,
        "momentum": Momentum,
        "rmsprop": RMSProp,
        "adam": Adam,
    }
    optimizer = constructors[name](**settings)
    parameters = np.array([2.0, -1.0], dtype=np.float64)
    optimizer.step([parameters], [np.array([0.5, -0.25])])
    state = optimizer.state_dict()

    restored_optimizer = constructors[name](**settings)
    restored_optimizer.load_state_dict(state, [parameters.shape])
    restored_parameters = parameters.copy()
    next_gradient = parameters - np.array([0.25, 0.5])
    optimizer.step([parameters], [next_gradient])
    restored_optimizer.step([restored_parameters], [next_gradient])

    np.testing.assert_array_equal(restored_parameters, parameters)
