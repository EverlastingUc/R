"""Compare four NumPy optimizers on the same known quadratic objective."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from nn.optimizers import SGD, Adam, Momentum, RMSProp

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cp19_optimizer_comparison.json"
REPORT_PATH = ROOT / "reports" / "cp19_optimizer_comparison.json"


def make_optimizer(name: str, settings: dict[str, float]) -> Any:
    """Construct the named optimizer from its versioned configuration."""
    constructors = {
        "sgd": SGD,
        "momentum": Momentum,
        "rmsprop": RMSProp,
        "adam": Adam,
    }
    try:
        constructor = constructors[name]
    except KeyError as error:
        raise ValueError(f"unknown optimizer: {name}") from error
    return constructor(**settings)


def main() -> None:
    """Run each optimizer and save the complete comparison evidence."""
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    initial = np.asarray(config["initial_parameters"], dtype=np.float64)
    target = np.asarray(config["target_parameters"], dtype=np.float64)
    steps = int(config["steps"])
    report_optimizers: dict[str, dict[str, Any]] = {}

    start = time.perf_counter()
    for name, settings in config["optimizers"].items():
        parameters = initial.copy()
        optimizer = make_optimizer(name, settings)
        checkpoints = {"0": 0.5 * float(np.sum((parameters - target) ** 2))}
        for step in range(1, steps + 1):
            gradient = parameters - target
            optimizer.step([parameters], [gradient])
            if step in {1, 10, 50, 100, steps}:
                checkpoints[str(step)] = 0.5 * float(np.sum((parameters - target) ** 2))
        report_optimizers[name] = {
            "settings": settings,
            "final_parameters": parameters.tolist(),
            "loss_checkpoints": checkpoints,
            "final_loss": checkpoints[str(steps)],
        }
    duration_seconds = time.perf_counter() - start

    report = {
        "experiment": config["name"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "problem_manifest": config["problem_manifest"],
        "initial_parameters": initial.tolist(),
        "target_parameters": target.tolist(),
        "steps_per_optimizer": steps,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_seconds": duration_seconds,
        "optimizers": report_optimizers,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
