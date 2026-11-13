"""Mini-batch training, validation, and safe resumable checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.backprop import backprop_mse
from nn.losses import mean_squared_error
from nn.network import FeedForwardNetwork

FloatArray = NDArray[np.float64]
CHECKPOINT_FORMAT_VERSION = 1


class StatefulOptimizer(Protocol):
    """The optimizer operations required by ``Trainer``."""

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Update parameters in place."""

    def state_dict(self) -> dict[str, object]:
        """Return configuration and moving state arrays."""

    def load_state_dict(
        self,
        state: Mapping[str, object],
        parameter_shapes: Sequence[tuple[int, ...]],
    ) -> None:
        """Restore saved state after validating shapes and configuration."""


@dataclass(frozen=True)
class EpochMetrics:
    """Mean training and validation loss measured for a completed epoch."""

    epoch: int
    training_loss: float
    validation_loss: float


class Trainer:
    """Train a feedforward network with seeded shuffling and mini-batches."""

    def __init__(
        self,
        network: FeedForwardNetwork,
        optimizer: StatefulOptimizer,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        """Create a trainer; ``batch_size`` is the maximum rows per update."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.network = network
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.completed_epochs = 0
        self._random = np.random.default_rng(seed)

    @staticmethod
    def _dataset(
        inputs: ArrayLike, targets: ArrayLike, name: str
    ) -> tuple[FloatArray, FloatArray]:
        input_values = np.asarray(inputs, dtype=np.float64)
        target_values = np.asarray(targets, dtype=np.float64)
        if input_values.ndim != 2 or target_values.ndim != 2:
            raise ValueError(f"{name} inputs and targets must be two-dimensional")
        if (
            input_values.shape[0] == 0
            or input_values.shape[0] != target_values.shape[0]
        ):
            raise ValueError(f"{name} inputs and targets must have matching rows")
        if not np.isfinite(input_values).all() or not np.isfinite(target_values).all():
            raise ValueError(f"{name} inputs and targets must be finite")
        return input_values, target_values

    def fit(
        self,
        train_inputs: ArrayLike,
        train_targets: ArrayLike,
        validation_inputs: ArrayLike,
        validation_targets: ArrayLike,
        epochs: int,
    ) -> list[EpochMetrics]:
        """Train for ``epochs`` and evaluate validation loss after each epoch.

        Training arrays have shapes ``(samples, features)`` and
        ``(samples, outputs)``. The final short batch is included. Training
        and validation losses are measured over their full splits after each
        epoch has completed.
        """
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        train_x, train_y = self._dataset(train_inputs, train_targets, "training")
        validation_x, validation_y = self._dataset(
            validation_inputs, validation_targets, "validation"
        )
        if train_x.shape[1] != self.network.layer_sizes[0]:
            raise ValueError("training input feature count does not match the network")
        if validation_x.shape[1] != self.network.layer_sizes[0]:
            raise ValueError(
                "validation input feature count does not match the network"
            )
        if self.network.forward(train_x).shape != train_y.shape:
            raise ValueError("training targets must match network output shape")
        if self.network.forward(validation_x).shape != validation_y.shape:
            raise ValueError("validation targets must match network output shape")

        history: list[EpochMetrics] = []
        for _ in range(epochs):
            order = np.arange(len(train_x))
            if self.shuffle:
                self._random.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                gradients = backprop_mse(self.network, batch_x, batch_y)
                parameters: list[FloatArray] = []
                gradient_values: list[FloatArray] = []
                for layer, weight_gradient, bias_gradient in zip(
                    self.network.layers,
                    gradients.weight_gradients,
                    gradients.bias_gradients,
                    strict=True,
                ):
                    parameters.extend((layer.weights, layer.bias))
                    gradient_values.extend((weight_gradient, bias_gradient))
                self.optimizer.step(parameters, gradient_values)

            self.completed_epochs += 1
            training_loss = mean_squared_error(self.network.forward(train_x), train_y)
            validation_loss = mean_squared_error(
                self.network.forward(validation_x), validation_y
            )
            if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
                raise ValueError("training produced a non-finite loss")
            history.append(
                EpochMetrics(
                    epoch=self.completed_epochs,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )
            )
        return history

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model, optimizer, epoch, and shuffle state without overwriting.

        The `.npz` archive uses numeric arrays plus JSON metadata and loads with
        pickle disabled. `path` must not already exist.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model_arrays: dict[str, FloatArray] = {}
        for index, layer in enumerate(self.network.layers):
            model_arrays[f"weight_{index}"] = layer.weights.copy()
            model_arrays[f"bias_{index}"] = layer.bias.copy()

        optimizer_state = self.optimizer.state_dict()
        optimizer_arrays = optimizer_state.get("arrays")
        if not isinstance(optimizer_arrays, Mapping):
            raise TypeError("optimizer state_dict must contain named arrays")
        optimizer_metadata = {
            key: value for key, value in optimizer_state.items() if key != "arrays"
        }
        array_references: dict[str, list[str]] = {}
        for field, values in optimizer_arrays.items():
            if not isinstance(values, list):
                raise TypeError("optimizer state arrays must be lists")
            names = []
            for index, value in enumerate(values):
                name = f"optimizer_{field}_{index}"
                model_arrays[name] = np.asarray(value, dtype=np.float64).copy()
                names.append(name)
            array_references[str(field)] = names
        optimizer_metadata["arrays"] = array_references

        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "network": {
                "layer_sizes": list(self.network.layer_sizes),
                "activations": list(self.network.activations),
            },
            "trainer": {
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "completed_epochs": self.completed_epochs,
                "random_state": self._random.bit_generator.state,
            },
            "optimizer": optimizer_metadata,
        }
        model_arrays["metadata"] = np.asarray(json.dumps(metadata))

        # Exclusive creation prevents a rerun from silently replacing evidence.
        with checkpoint_path.open("xb") as output_file:
            np.savez_compressed(output_file, **model_arrays)
        return checkpoint_path

    def restore_checkpoint(self, path: str | Path) -> None:
        """Restore a compatible saved session and its exact shuffle sequence."""
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            if "metadata" not in archive:
                raise ValueError("checkpoint is missing metadata")
            try:
                metadata = json.loads(str(archive["metadata"].item()))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("checkpoint metadata is invalid") from error

            if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("unsupported checkpoint format version")
            network_metadata = metadata.get("network", {})
            if network_metadata != {
                "layer_sizes": list(self.network.layer_sizes),
                "activations": list(self.network.activations),
            }:
                raise ValueError("checkpoint network architecture does not match")

            trainer_metadata = metadata.get("trainer", {})
            if (
                trainer_metadata.get("batch_size") != self.batch_size
                or trainer_metadata.get("shuffle") != self.shuffle
            ):
                raise ValueError("checkpoint training settings do not match")
            try:
                completed_epochs = trainer_metadata["completed_epochs"]
                restored_seed = trainer_metadata["seed"]
                if (
                    not isinstance(completed_epochs, int)
                    or completed_epochs < 0
                    or not isinstance(restored_seed, int)
                ):
                    raise ValueError("checkpoint epoch or seed is invalid")
                restored_random = np.random.default_rng(restored_seed)
                restored_random.bit_generator.state = trainer_metadata["random_state"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("checkpoint shuffle state is invalid") from error

            model_parameters: list[tuple[FloatArray, FloatArray]] = []
            for index, layer in enumerate(self.network.layers):
                weight_name = f"weight_{index}"
                bias_name = f"bias_{index}"
                if weight_name not in archive or bias_name not in archive:
                    raise ValueError("checkpoint is missing model parameters")
                weights = np.asarray(archive[weight_name], dtype=np.float64)
                bias = np.asarray(archive[bias_name], dtype=np.float64)
                if (
                    weights.shape != layer.weights.shape
                    or bias.shape != layer.bias.shape
                    or not np.isfinite(weights).all()
                    or not np.isfinite(bias).all()
                ):
                    raise ValueError("checkpoint model parameters are invalid")
                model_parameters.append((weights.copy(), bias.copy()))

            optimizer_metadata = metadata.get("optimizer", {})
            references = optimizer_metadata.get("arrays")
            if not isinstance(references, Mapping):
                raise TypeError("checkpoint optimizer array references are invalid")
            restored_optimizer_state = {
                key: value
                for key, value in optimizer_metadata.items()
                if key != "arrays"
            }
            restored_arrays: dict[str, list[FloatArray]] = {}
            for field, names in references.items():
                if not isinstance(names, list):
                    raise TypeError("checkpoint optimizer array names are invalid")
                values = []
                for name in names:
                    if not isinstance(name, str) or name not in archive:
                        raise ValueError("checkpoint optimizer array is missing")
                    values.append(np.asarray(archive[name], dtype=np.float64).copy())
                restored_arrays[str(field)] = values
            restored_optimizer_state["arrays"] = restored_arrays

        parameter_shapes = [
            array.shape
            for layer in self.network.layers
            for array in (layer.weights, layer.bias)
        ]
        self.optimizer.load_state_dict(restored_optimizer_state, parameter_shapes)
        for layer, (weights, bias) in zip(
            self.network.layers, model_parameters, strict=True
        ):
            layer.weights[...] = weights
            layer.bias[...] = bias
        self.completed_epochs = completed_epochs
        self.seed = restored_seed
        self._random = restored_random
