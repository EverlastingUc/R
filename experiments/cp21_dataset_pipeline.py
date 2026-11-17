"""Download, prepare and archive Entropy's approved CP21 UCI datasets."""

from __future__ import annotations

import json
from pathlib import Path

from nn.datasets import (
    BIKE_URL,
    IRIS_URL,
    download_to_cache,
    prepare_bike_hourly,
    prepare_iris,
    save_prepared_dataset,
    sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cp21_dataset_pipeline.json"


def main() -> None:
    """Run both datasets and print a compact split/provenance audit."""
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    config_hash = sha256_bytes(config_bytes)
    split = config["split"]
    specs = config["datasets"]
    sources = (
        (
            "iris",
            IRIS_URL,
            prepare_iris,
            {
                "uci_id": 53,
                "doi": "10.24432/C56C76",
                "citation": "Fisher, R. A. (1936). Iris. UCI Machine Learning Repository.",
                "attribution": "R. A. Fisher; distributed by the UCI Machine Learning Repository.",
            },
        ),
        (
            "bike_sharing_hourly",
            BIKE_URL,
            prepare_bike_hourly,
            {
                "uci_id": 275,
                "doi": "10.24432/C5W894",
                "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository.",
                "attribution": "Hadi Fanaee-T; distributed by the UCI Machine Learning Repository.",
            },
        ),
    )
    audit: list[dict[str, object]] = []
    for name, source_url, prepare, citation in sources:
        settings = specs[name]
        raw_path = download_to_cache(source_url, ROOT / settings["raw_path"])
        raw_bytes = raw_path.read_bytes()
        dataset = prepare(raw_bytes, **({"seed": config["seed"]} if name == "iris" else {}), **split)
        provenance = {
            **citation,
            "source_url": source_url,
            "source_sha256": sha256_bytes(raw_bytes),
            "config_sha256": config_hash,
            "license": "CC BY 4.0",
        }
        archive, metadata = save_prepared_dataset(
            dataset, ROOT / settings["processed_path"], provenance
        )
        audit.append({
            "name": dataset.name,
            "task": dataset.task,
            "train_rows": len(dataset.train_inputs),
            "validation_rows": len(dataset.validation_inputs),
            "test_rows": len(dataset.test_inputs),
            "features": len(dataset.feature_names),
            "archive": str(archive.relative_to(ROOT)),
            "metadata": str(metadata.relative_to(ROOT)),
        })
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
