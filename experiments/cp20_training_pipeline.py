"""Run a reproducible mini-batch training and checkpoint-resume demonstration."""

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
from nn.optimizers import Adam
from nn.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cp20_training_pipeline.json"
REPORT_DIRECTORY = ROOT / "reports"
CHECKPOINT_DIRECTORY = ROOT / "reports" / "checkpoints" / "cp20"


def _next_run_paths(config_hash: str) -> tuple[Path, Path]:
    """Choose new numbered report and checkpoint paths without overwriting."""
    for run_number in range(1, 10000):
        checkpoint_path = CHECKPOINT_DIRECTORY / (
            f"cp20-{config_hash[:12]}-run-{run_number:03d}.npz"
        )
        report_path = (
            REPORT_DIRECTORY / f"cp20_training_pipeline_run_{run_number:03d}.json"
        )
        if not checkpoint_path.exists() and not report_path.exists():
            return checkpoint_path, report_path
    raise RuntimeError("could not find an unused Checkpoint 20 output name")


def main() -> None:
    """Train, save, restore, verify predictions, and write measured metadata."""
    config_bytes = CONFIG_PATH.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    dataset = config["dataset"]

    train_inputs = np.linspace(
        dataset["train_input_min"],
        dataset["train_input_max"],
        dataset["train_sample_count"],
        dtype=np.float64,
    ).reshape(-1, 1)
    validation_inputs = np.linspace(
        dataset["validation_input_min"],
        dataset["validation_input_max"],
        dataset["validation_sample_count"],
        dtype=np.float64,
    ).reshape(-1, 1)
    train_targets = dataset["target_slope"] * train_inputs + dataset["target_intercept"]
    validation_targets = (
        dataset["target_slope"] * validation_inputs + dataset["target_intercept"]
    )

    network = FeedForwardNetwork(
        config["layer_sizes"], config["activations"], seed=config["seed"]
    )
    optimizer_settings = {
        key: value for key, value in config["optimizer"].items() if key != "name"
    }
    optimizer = Adam(**optimizer_settings)
    trainer = Trainer(
        network,
        optimizer,
        batch_size=config["batch_size"],
        seed=config["shuffle_seed"],
        shuffle=True,
    )
    initial_train_loss = mean_squared_error(
        network.forward(train_inputs), train_targets
    )
    initial_validation_loss = mean_squared_error(
        network.forward(validation_inputs), validation_targets
    )

    start = time.perf_counter()
    history = trainer.fit(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        epochs=config["epochs"],
    )
    checkpoint_path, report_path = _next_run_paths(config_hash)
    trainer.save_checkpoint(checkpoint_path)
    expected_predictions = network.forward(validation_inputs)

    restored_network = FeedForwardNetwork(
        config["layer_sizes"], config["activations"], seed=0
    )
    restored_trainer = Trainer(
        restored_network,
        Adam(**optimizer_settings),
        batch_size=config["batch_size"],
        seed=0,
        shuffle=True,
    )
    restored_trainer.restore_checkpoint(checkpoint_path)
    restored_predictions = restored_network.forward(validation_inputs)
    predictions_match = bool(np.array_equal(expected_predictions, restored_predictions))
    if not predictions_match:
        raise RuntimeError("restored network predictions did not match")
    duration_seconds = time.perf_counter() - start

    report = {
        "experiment": config["name"],
        "config_sha256": config_hash,
        "dataset_manifest": dataset["manifest"],
        "seed": config["seed"],
        "shuffle_seed": config["shuffle_seed"],
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_seconds": duration_seconds,
        "optimizer": optimizer.configuration(),
        "training_loss_definition": "full training split evaluated after each epoch",
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "train_sample_count": len(train_inputs),
        "validation_sample_count": len(validation_inputs),
        "initial_train_mse": initial_train_loss,
        "initial_validation_mse": initial_validation_loss,
        "first_epoch": {
            "train_mse": history[0].training_loss,
            "validation_mse": history[0].validation_loss,
        },
        "final_epoch": {
            "train_mse": history[-1].training_loss,
            "validation_mse": history[-1].validation_loss,
        },
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "restored_epoch": restored_trainer.completed_epochs,
        "restored_predictions_match": predictions_match,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
