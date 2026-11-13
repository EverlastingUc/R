"""Compare learning an XOR pattern with memorizing random labels."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_network import EntropyNetwork

FloatArray = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cp04_memorization_generalization.json"


@dataclass(frozen=True)
class Scores:
    """Training and held-out measurements for one experiment condition."""

    train_accuracy: float
    test_accuracy: float
    initial_train_loss: float
    final_train_loss: float


@dataclass(frozen=True)
class ExperimentResult:
    """Scores and evidence files produced by one fixed-seed experiment."""

    generalization: Scores
    memorization: Scores
    plot_path: Path
    metrics_path: Path


def load_config(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], bytes]:
    """Load the JSON config and return its parsed values and original bytes."""
    config_bytes = config_path.read_bytes()
    return json.loads(config_bytes), config_bytes


def xor_targets(inputs: FloatArray) -> FloatArray:
    """Return XOR labels for inputs shaped ``(batch_size, 2)``."""
    first_is_high = inputs[:, 0] >= 0.5
    second_is_high = inputs[:, 1] >= 0.5
    labels = np.logical_xor(first_is_high, second_is_high)
    return labels.astype(np.float64).reshape(-1, 1)


def make_structured_data(
    seed: int = 2026,
    train_examples_per_corner: int = 32,
    test_examples_per_corner: int = 64,
    jitter: float = 0.1,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Create distinct train/test clouds around the four XOR input corners."""
    if train_examples_per_corner < 1 or test_examples_per_corner < 1:
        raise ValueError("each split must contain at least one example per corner")
    if not 0.0 < jitter < 0.35:
        raise ValueError("jitter must be greater than 0 and less than 0.35")

    random = np.random.default_rng(seed)
    train_parts: list[FloatArray] = []
    test_parts: list[FloatArray] = []
    for first_bit in (0, 1):
        for second_bit in (0, 1):
            center = np.array(
                [0.15 if first_bit == 0 else 0.85, 0.15 if second_bit == 0 else 0.85],
                dtype=np.float64,
            )
            train_parts.append(
                center
                + random.uniform(-jitter, jitter, size=(train_examples_per_corner, 2))
            )
            test_parts.append(
                center
                + random.uniform(-jitter, jitter, size=(test_examples_per_corner, 2))
            )

    train_inputs = np.concatenate(train_parts, axis=0)
    test_inputs = np.concatenate(test_parts, axis=0)
    train_inputs = train_inputs[random.permutation(train_inputs.shape[0])]
    test_inputs = test_inputs[random.permutation(test_inputs.shape[0])]
    return (
        train_inputs,
        xor_targets(train_inputs),
        test_inputs,
        xor_targets(test_inputs),
    )


def make_random_label_data(
    seed: int,
    train_count: int,
    test_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Create independent random inputs and labels for a memorization check."""
    if train_count < 1 or test_count < 1:
        raise ValueError("train_count and test_count must be positive")

    random = np.random.default_rng(seed)
    train_inputs = random.uniform(0.0, 1.0, size=(train_count, 2))
    train_labels = random.integers(0, 2, size=(train_count, 1)).astype(np.float64)
    test_inputs = random.uniform(0.0, 1.0, size=(test_count, 2))
    test_labels = random.integers(0, 2, size=(test_count, 1)).astype(np.float64)
    return train_inputs, train_labels, test_inputs, test_labels


def _measure(
    network: EntropyNetwork,
    train_inputs: FloatArray,
    train_labels: FloatArray,
    test_inputs: FloatArray,
    test_labels: FloatArray,
    epochs: int,
    learning_rate: float,
) -> Scores:
    """Fit one model and compute accuracy on train and held-out examples."""
    losses = network.fit(
        train_inputs,
        train_labels,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    train_accuracy = float(np.mean(network.predict(train_inputs) == train_labels))
    test_accuracy = float(np.mean(network.predict(test_inputs) == test_labels))
    return Scores(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        initial_train_loss=float(losses[0]),
        final_train_loss=float(losses[-1]),
    )


def _save_plot(generalization: Scores, memorization: Scores, plot_path: Path) -> None:
    """Save a compact plot comparing training and held-out accuracy."""
    labels = ["Structured XOR", "Random labels"]
    train_scores = [generalization.train_accuracy, memorization.train_accuracy]
    test_scores = [generalization.test_accuracy, memorization.test_accuracy]
    positions = np.arange(len(labels))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(positions - bar_width / 2, train_scores, bar_width, label="Train")
    axis.bar(positions + bar_width / 2, test_scores, bar_width, label="Held-out")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Learning a pattern vs memorizing random answers")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def run_experiment(
    output_dir: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> ExperimentResult:
    """Train both models, save a plot and a JSON record, and return scores."""
    config, config_bytes = load_config(config_path)
    seed = int(config["seed"])
    learning_rate = float(config["learning_rate"])
    output_directory = output_dir or PROJECT_ROOT / "reports"
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = output_directory / "memorization_vs_generalization.png"
    metrics_path = output_directory / "memorization_vs_generalization.json"

    start_time = time.perf_counter()
    structured_config = config["structured"]
    structured_data = make_structured_data(
        seed=seed,
        train_examples_per_corner=int(structured_config["train_examples_per_corner"]),
        test_examples_per_corner=int(structured_config["test_examples_per_corner"]),
        jitter=float(structured_config["jitter"]),
    )
    structured_network = EntropyNetwork(
        input_size=2,
        hidden_size=int(structured_config["hidden_size"]),
        seed=seed + 1,
    )
    generalization_scores = _measure(
        structured_network,
        *structured_data,
        epochs=int(structured_config["epochs"]),
        learning_rate=learning_rate,
    )

    memorization_config = config["memorization"]
    random_label_data = make_random_label_data(
        seed=seed + 2,
        train_count=int(memorization_config["train_examples"]),
        test_count=int(memorization_config["test_examples"]),
    )
    memorization_network = EntropyNetwork(
        input_size=2,
        hidden_size=int(memorization_config["hidden_size"]),
        seed=seed + 3,
    )
    memorization_scores = _measure(
        memorization_network,
        *random_label_data,
        epochs=int(memorization_config["epochs"]),
        learning_rate=learning_rate,
    )
    _save_plot(generalization_scores, memorization_scores, plot_path)
    duration_seconds = time.perf_counter() - start_time

    metrics = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seed": seed,
        "dataset": "synthetic XOR clusters and independently random labels",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "duration_seconds": duration_seconds,
        "generalization": asdict(generalization_scores),
        "memorization": asdict(memorization_scores),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return ExperimentResult(
        generalization=generalization_scores,
        memorization=memorization_scores,
        plot_path=plot_path,
        metrics_path=metrics_path,
    )


def main() -> None:
    """Run the configured experiment and print its measured scores."""
    result = run_experiment()
    print("Structured XOR: learn a repeatable pattern")
    print(
        f"  train accuracy: {result.generalization.train_accuracy:.3f}; "
        f"held-out accuracy: {result.generalization.test_accuracy:.3f}"
    )
    print("Random labels: memorize examples with no learnable pattern")
    print(
        f"  train accuracy: {result.memorization.train_accuracy:.3f}; "
        f"held-out accuracy: {result.memorization.test_accuracy:.3f}"
    )
    print(f"Plot: {result.plot_path}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
