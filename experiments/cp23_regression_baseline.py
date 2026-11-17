"""Train Entropy on a deterministic nonlinear regression benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from nn.network import FeedForwardNetwork
from nn.optimizers import Adam
from nn.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(seed: int = 2026, samples: int = 1200) -> tuple[np.ndarray, ...]:
    """Create a noisy nonlinear regression dataset with a fixed held-out test set."""
    if samples < 100:
        raise ValueError("samples must be at least 100")
    random = np.random.default_rng(seed)
    inputs = random.uniform(-2.0, 2.0, size=(samples, 4))
    clean_targets = (
        np.sin(inputs[:, 0])
        + 0.35 * inputs[:, 1] ** 2
        - 0.6 * inputs[:, 2] * inputs[:, 3]
        + 0.2 * np.cos(inputs[:, 1] * inputs[:, 2])
    )
    targets = clean_targets + random.normal(0.0, 0.04, size=samples)
    order = random.permutation(samples)
    train_end = int(samples * 0.70)
    validation_end = int(samples * 0.85)
    train_indices = order[:train_end]
    validation_indices = order[train_end:validation_end]
    test_indices = order[validation_end:]

    mean = inputs[train_indices].mean(axis=0)
    scale = inputs[train_indices].std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    normalized_inputs = (inputs - mean) / scale
    target_mean = targets[train_indices].mean()
    target_scale = targets[train_indices].std()
    normalized_targets = (targets - target_mean) / target_scale

    return (
        normalized_inputs[train_indices],
        normalized_targets[train_indices, None],
        normalized_inputs[validation_indices],
        normalized_targets[validation_indices, None],
        normalized_inputs[test_indices],
        normalized_targets[test_indices, None],
        np.asarray([target_mean, target_scale], dtype=np.float64),
    )


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate mean squared error in normalized target units."""
    return float(np.mean(np.square(predictions - targets)))


def main() -> None:
    """Run training and report untouched held-out test performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive")

    started = time.perf_counter()
    train_x, train_y, validation_x, validation_y, test_x, test_y, target_stats = (
        make_dataset(arguments.seed)
    )
    network = FeedForwardNetwork(
        [4, 32, 16, 1],
        ["relu", "tanh", "identity"],
        seed=arguments.seed,
    )
    trainer = Trainer(
        network,
        Adam(learning_rate=arguments.learning_rate),
        batch_size=32,
        seed=arguments.seed,
    )
    history = trainer.fit(
        train_x, train_y, validation_x, validation_y, epochs=arguments.epochs
    )

    # Test data is not used for gradient updates or model selection.
    test_predictions = network.forward(test_x)
    test_mse = mean_squared_error(test_predictions, test_y)
    target_scale = float(target_stats[1])
    report = {
        "checkpoint": "CP23",
        "experiment": "synthetic_nonlinear_regression",
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "optimizer": {
            "name": "Adam",
            "learning_rate": arguments.learning_rate,
            "batch_size": 32,
        },
        "architecture": {
            "layer_sizes": [4, 32, 16, 1],
            "activations": ["relu", "tanh", "identity"],
        },
        "metrics": {
            "training_mse_normalized": history[-1].training_loss,
            "validation_mse_normalized": history[-1].validation_loss,
            "test_mse_normalized": test_mse,
            "test_rmse_original_units": float(np.sqrt(test_mse) * target_scale),
        },
        "split_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "runtime": {
            "platform": platform.platform(),
            "seconds": time.perf_counter() - started,
        },
        "limitations": [
            "Synthetic benchmark; not a real-world forecasting result.",
            "One fixed architecture and hyperparameter setting; no tuning claim.",
            "The model is a feedforward regressor, not a language model.",
        ],
    }
    report_directory = ROOT / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / "cp23_regression_baseline.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evidence report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
