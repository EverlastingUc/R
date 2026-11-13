"""NumPy implementations of common parameter-update algorithms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class _Optimizer:
    """Shared validation and atomic parameter-update helpers."""

    state_names: tuple[str, ...] = ()
    stores_step_count = False

    def __init__(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)

    def configuration(self) -> dict[str, str | float]:
        """Return the algorithm name and hyperparameters needed to recreate it."""
        values: dict[str, str | float] = {
            "name": type(self).__name__,
            "learning_rate": self.learning_rate,
        }
        for name in ("momentum", "decay", "epsilon", "beta1", "beta2"):
            if hasattr(self, name):
                values[name] = float(getattr(self, name))
        return values

    def state_dict(self) -> dict[str, object]:
        """Copy optimizer configuration and moving arrays for checkpointing."""
        state: dict[str, object] = {
            "configuration": self.configuration(),
            "arrays": {
                name: [array.copy() for array in getattr(self, name)]
                for name in self.state_names
            },
        }
        if self.stores_step_count:
            state["step_count"] = int(self.step_count)
        return state

    def load_state_dict(
        self,
        state: Mapping[str, object],
        parameter_shapes: Sequence[tuple[int, ...]],
    ) -> None:
        """Restore optimizer statistics after checking config and array shapes."""
        if state.get("configuration") != self.configuration():
            raise ValueError("checkpoint optimizer configuration does not match")
        raw_arrays = state.get("arrays")
        if not isinstance(raw_arrays, Mapping) or set(raw_arrays) != set(
            self.state_names
        ):
            raise ValueError("checkpoint optimizer state fields do not match")

        restored: dict[str, list[FloatArray]] = {}
        for name in self.state_names:
            raw_values = raw_arrays[name]
            if not isinstance(raw_values, list):
                raise TypeError(f"checkpoint {name} state must be a list")
            if raw_values and len(raw_values) != len(parameter_shapes):
                raise ValueError(f"checkpoint {name} state count does not match")
            values: list[FloatArray] = []
            for value, expected_shape in zip(
                raw_values, parameter_shapes, strict=bool(raw_values)
            ):
                array = np.asarray(value, dtype=np.float64)
                if array.shape != expected_shape or not np.isfinite(array).all():
                    raise ValueError(f"checkpoint {name} state is invalid")
                values.append(array.copy())
            restored[name] = values

        next_step_count = 0
        if self.stores_step_count:
            raw_step_count = state.get("step_count")
            if not isinstance(raw_step_count, int) or raw_step_count < 0:
                raise ValueError("checkpoint optimizer step_count is invalid")
            next_step_count = raw_step_count
            arrays_initialized = bool(next(iter(restored.values()), []))
            if (next_step_count > 0) != arrays_initialized:
                raise ValueError("checkpoint Adam step count and state disagree")

        for name, values in restored.items():
            setattr(self, name, values)
        if self.stores_step_count:
            self.step_count = next_step_count

    @staticmethod
    def _validate(
        parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> tuple[list[FloatArray], list[FloatArray]]:
        if not parameters or len(parameters) != len(gradients):
            raise ValueError("provide matching non-empty parameter and gradient lists")
        parameter_values: list[FloatArray] = []
        gradient_values: list[FloatArray] = []
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter_array = np.asarray(parameter)
            gradient_array = np.asarray(gradient, dtype=np.float64)
            if not np.issubdtype(parameter_array.dtype, np.floating):
                raise TypeError("parameters must use a floating-point dtype")
            if not parameter_array.flags.writeable:
                raise ValueError("parameters must be writable")
            if parameter_array.shape != gradient_array.shape:
                raise ValueError(
                    "each gradient must have the same shape as its parameter"
                )
            if not np.isfinite(parameter_array).all():
                raise ValueError("parameters must be finite")
            if not np.isfinite(gradient_array).all():
                raise ValueError("gradients must be finite")
            parameter_values.append(parameter_array)
            gradient_values.append(gradient_array)
        return parameter_values, gradient_values

    @staticmethod
    def _state_arrays(
        state: list[FloatArray], parameters: Sequence[FloatArray], name: str
    ) -> list[FloatArray]:
        if not state:
            return [
                np.zeros_like(parameter, dtype=np.float64) for parameter in parameters
            ]
        if len(state) != len(parameters) or any(
            current.shape != parameter.shape
            for current, parameter in zip(state, parameters, strict=True)
        ):
            raise ValueError(f"{name} state does not match the parameter shapes")
        return state

    @staticmethod
    def _commit(
        parameters: Sequence[FloatArray], candidates: Sequence[FloatArray]
    ) -> None:
        if any(not np.isfinite(candidate).all() for candidate in candidates):
            raise ValueError("optimizer update produced non-finite parameters")
        for parameter, candidate in zip(parameters, candidates, strict=True):
            parameter[...] = candidate

    @staticmethod
    def _ensure_finite_state(state: Sequence[FloatArray], name: str) -> None:
        if any(not np.isfinite(value).all() for value in state):
            raise ValueError(f"optimizer {name} state became non-finite")


class SGD(_Optimizer):
    """Stochastic gradient descent over a supplied gradient batch.

    ``parameters`` and ``gradients`` are matching lists of arrays. In this
    project gradients are computed for the full batch by backpropagation.
    """

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Apply ``parameter -= learning_rate * gradient`` in place."""
        parameter_values, gradient_values = self._validate(parameters, gradients)
        candidates = [
            parameter - self.learning_rate * gradient
            for parameter, gradient in zip(
                parameter_values, gradient_values, strict=True
            )
        ]
        self._commit(parameter_values, candidates)


class Momentum(_Optimizer):
    """Gradient descent with an exponentially smoothed velocity."""

    state_names = ("velocity",)

    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be finite and in [0, 1)")
        self.momentum = float(momentum)
        self.velocity: list[FloatArray] = []

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Update velocity, then add it to each parameter in place."""
        parameter_values, gradient_values = self._validate(parameters, gradients)
        velocity = self._state_arrays(self.velocity, parameter_values, "velocity")
        next_velocity = [
            self.momentum * current - self.learning_rate * gradient
            for current, gradient in zip(velocity, gradient_values, strict=True)
        ]
        self._ensure_finite_state(next_velocity, "velocity")
        candidates = [
            parameter + update
            for parameter, update in zip(parameter_values, next_velocity, strict=True)
        ]
        self._commit(parameter_values, candidates)
        self.velocity = next_velocity


class RMSProp(_Optimizer):
    """Adaptive gradient descent using a moving average of squared gradients."""

    state_names = ("mean_square",)

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite and in [0, 1)")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.mean_square: list[FloatArray] = []

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Scale each update by its moving root-mean-square gradient."""
        parameter_values, gradient_values = self._validate(parameters, gradients)
        mean_square = self._state_arrays(
            self.mean_square, parameter_values, "mean-square"
        )
        next_mean_square = [
            self.decay * current + (1.0 - self.decay) * gradient**2
            for current, gradient in zip(mean_square, gradient_values, strict=True)
        ]
        self._ensure_finite_state(next_mean_square, "mean-square")
        candidates = [
            parameter
            - self.learning_rate * gradient / (np.sqrt(average) + self.epsilon)
            for parameter, gradient, average in zip(
                parameter_values, gradient_values, next_mean_square, strict=True
            )
        ]
        self._commit(parameter_values, candidates)
        self.mean_square = next_mean_square


class Adam(_Optimizer):
    """Adaptive moment estimation with bias-corrected first and second moments."""

    state_names = ("first_moment", "second_moment")
    stores_step_count = True

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not math.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be finite and in [0, 1)")
        if not math.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be finite and in [0, 1)")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moment: list[FloatArray] = []
        self.second_moment: list[FloatArray] = []
        self.step_count = 0

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Apply Adam's bias-corrected moment update to each parameter."""
        parameter_values, gradient_values = self._validate(parameters, gradients)
        first_moment = self._state_arrays(
            self.first_moment, parameter_values, "first-moment"
        )
        second_moment = self._state_arrays(
            self.second_moment, parameter_values, "second-moment"
        )
        next_step = self.step_count + 1
        next_first = [
            self.beta1 * current + (1.0 - self.beta1) * gradient
            for current, gradient in zip(first_moment, gradient_values, strict=True)
        ]
        next_second = [
            self.beta2 * current + (1.0 - self.beta2) * gradient**2
            for current, gradient in zip(second_moment, gradient_values, strict=True)
        ]
        self._ensure_finite_state(next_first, "first-moment")
        self._ensure_finite_state(next_second, "second-moment")
        first_correction = 1.0 - self.beta1**next_step
        second_correction = 1.0 - self.beta2**next_step
        candidates = [
            parameter
            - self.learning_rate
            * (first / first_correction)
            / (np.sqrt(second / second_correction) + self.epsilon)
            for parameter, first, second in zip(
                parameter_values, next_first, next_second, strict=True
            )
        ]
        self._commit(parameter_values, candidates)
        self.first_moment = next_first
        self.second_moment = next_second
        self.step_count = next_step
