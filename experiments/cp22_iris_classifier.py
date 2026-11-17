"""Train and evaluate Entropy's first UCI Iris classifier."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from nn.classification import ClassificationTrainer
from nn.network import FeedForwardNetwork
from nn.optimizers import Adam
from nn.uci_iris import load_uci_iris

ROOT = Path(__file__).resolve().parents[1]


def next_report_path(directory: Path) -> Path:
    """Choose a fresh report filename without replacing earlier evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    for run_number in range(1, 10000):
        candidate = directory / f"cp22_iris_run_{run_number:03d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a new CP22 report filename")


def main() -> None:
    """Train on the training split, monitor validation, and score test once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive")

    started = time.perf_counter()
    dataset = load_uci_iris(ROOT / "data" / "raw", seed=arguments.seed)
    network = FeedForwardNetwork(
        [4, 24, 16, 3],
        ["relu", "tanh", "softmax"],
        seed=arguments.seed,
    )
    trainer = ClassificationTrainer(
        network,
        Adam(learning_rate=arguments.learning_rate),
        batch_size=16,
        seed=arguments.seed,
    )
    history = trainer.fit(
        dataset.train_inputs,
        dataset.train_labels,
        dataset.validation_inputs,
        dataset.validation_labels,
        epochs=arguments.epochs,
    )

    # The test partition is not used for training or model selection.
    test_predictions = trainer.predict(dataset.test_inputs)
    test_accuracy = float(np.mean(test_predictions == dataset.test_labels))
    final = history[-1]
    report = {
        "checkpoint": "CP22",
        "experiment": "uci_iris_multiclass_classification",
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "optimizer": {
            "name": "Adam",
            "learning_rate": arguments.learning_rate,
            "batch_size": 16,
        },
        "architecture": {
            "layer_sizes": [4, 24, 16, 3],
            "activations": ["relu", "tanh", "softmax"],
        },
        "metrics": {
            "training_loss": final.training_loss,
            "training_accuracy": final.training_accuracy,
            "validation_loss": final.validation_loss,
            "validation_accuracy": final.validation_accuracy,
            "test_accuracy": test_accuracy,
        },
        "dataset": dataset.metadata(),
        "runtime": {
            "platform": platform.platform(),
            "seconds": time.perf_counter() - started,
        },
        "limitations": [
            "One fixed architecture and hyperparameter setting; no tuning claim.",
            "Test metrics are reported once after training and validation.",
            "Iris is a small benchmark dataset, not evidence of language ability.",
        ],
    }
    report_path = next_report_path(ROOT / "reports")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evidence report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
