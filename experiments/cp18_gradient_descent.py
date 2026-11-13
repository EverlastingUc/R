"""Train a tiny NumPy network on a reproducible linear regression task."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from nn.losses import mean_squared_error
from nn.network import FeedForwardNetwork
from nn.training import gradient_descent_step

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cp18_gradient_descent.json"
REPORT_PATH = ROOT / "reports" / "cp18_training_metrics.json"


def main() -> None:
    """Run the configured full-batch experiment and save measured metrics."""
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    dataset = config["dataset"]
    inputs = np.linspace(
        dataset["input_min"],
        dataset["input_max"],
        dataset["sample_count"],
        dtype=np.float64,
    ).reshape(-1, 1)
    targets = dataset["target_slope"] * inputs + dataset["target_intercept"]
    network = FeedForwardNetwork(
        config["layer_sizes"], config["activations"], seed=config["seed"]
    )

    start = time.perf_counter()
    initial_loss = mean_squared_error(network.forward(inputs), targets)
    checkpoints = {"0": initial_loss}
    for epoch in range(1, config["epochs"] + 1):
        gradient_descent_step(
            network,
            inputs,
            targets,
            learning_rate=config["learning_rate"],
        )
        if epoch in {1, 10, 100, config["epochs"]}:
            checkpoints[str(epoch)] = mean_squared_error(
                network.forward(inputs), targets
            )
    duration_seconds = time.perf_counter() - start
    final_loss = mean_squared_error(network.forward(inputs), targets)

    report = {
        "experiment": config["name"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seed": config["seed"],
        "dataset_manifest": dataset["manifest"],
        "sample_count": len(inputs),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_seconds": duration_seconds,
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "initial_mse": initial_loss,
        "loss_checkpoints": checkpoints,
        "final_mse": final_loss,
        "loss_reduction": initial_loss - final_loss,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
