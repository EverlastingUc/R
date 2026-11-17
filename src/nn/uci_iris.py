"""Small, deterministic loader for the official UCI Iris classification data."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

IRIS_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
FEATURE_NAMES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
MAX_ARCHIVE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class IrisSplits:
    """Train, validation, and test arrays with train-only normalization."""

    train_inputs: FloatArray
    train_labels: IntArray
    validation_inputs: FloatArray
    validation_labels: IntArray
    test_inputs: FloatArray
    test_labels: IntArray
    mean: FloatArray
    scale: FloatArray
    archive_sha256: str

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe source, split, and preprocessing metadata."""
        return {
            "dataset": "UCI Iris",
            "dataset_id": 53,
            "doi": "10.24432/C56C76",
            "license": "CC BY 4.0",
            "archive_sha256": self.archive_sha256,
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(CLASS_NAMES),
            "split_counts": {
                "train": len(self.train_labels),
                "validation": len(self.validation_labels),
                "test": len(self.test_labels),
            },
            "normalization": {
                "method": "standard score",
                "fit_partition": "train",
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
            },
        }


def _validate_archive(path: Path) -> None:
    """Reject invalid, oversized, or unsafe ZIP archives before parsing."""
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Iris archive exceeds the configured size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members or len(members) > 20:
                raise ValueError("Iris archive has an unexpected file count")
            for member in members:
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("Iris archive contains an unsafe path")
            archive.testzip()
    except zipfile.BadZipFile as error:
        raise ValueError("Iris archive is not a valid ZIP file") from error


def _get_archive(cache_directory: str | Path) -> Path:
    """Download the official UCI archive once, using an atomic cache write."""
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "uci-iris.zip"
    if destination.exists():
        _validate_archive(destination)
        return destination

    request = urllib.request.Request(
        IRIS_URL, headers={"User-Agent": "Entropy-Iris-loader/1.0"}
    )
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, suffix=".part", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                total = 0
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Iris download exceeds the size limit")
                    temporary.write(chunk)
        _validate_archive(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_uci_iris(
    cache_directory: str | Path = "data/raw", seed: int = 2026
) -> IrisSplits:
    """Load UCI Iris and create deterministic stratified 70/16/14 splits."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    archive_path = _get_archive(cache_directory)
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name in {"iris.data", "bezdekIris.data"}
        ]
        if len(candidates) != 1:
            raise ValueError("Iris archive must contain exactly one data file")
        with archive.open(candidates[0]) as source:
            content = source.read(MAX_ARCHIVE_BYTES + 1)
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Iris data file exceeds the size limit")

    features: list[list[float]] = []
    labels: list[int] = []
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    for row_number, row in enumerate(reader, start=1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 5:
            raise ValueError(f"Iris row {row_number} must contain five fields")
        values = [float(value) for value in row[:4]]
        label = row[4].strip()
        if label not in class_ids:
            raise ValueError(f"Iris row {row_number} has an unknown class")
        if not np.isfinite(values).all():
            raise ValueError(f"Iris row {row_number} contains non-finite features")
        features.append(values)
        labels.append(class_ids[label])

    inputs = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if inputs.shape != (150, 4) or not np.array_equal(
        np.bincount(targets, minlength=3), [50, 50, 50]
    ):
        raise ValueError("Iris source does not match the documented 150-row dataset")

    random = np.random.default_rng(seed)
    partitions: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for class_id in range(3):
        indices = random.permutation(np.flatnonzero(targets == class_id))
        partitions["train"].extend(indices[:35].tolist())
        partitions["validation"].extend(indices[35:43].tolist())
        partitions["test"].extend(indices[43:].tolist())

    split_indices = {
        name: random.permutation(np.asarray(indices, dtype=np.int64))
        for name, indices in partitions.items()
    }
    mean = inputs[split_indices["train"]].mean(axis=0)
    scale = inputs[split_indices["train"]].std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    normalized = (inputs - mean) / scale
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    return IrisSplits(
        train_inputs=normalized[split_indices["train"]].copy(),
        train_labels=targets[split_indices["train"]].copy(),
        validation_inputs=normalized[split_indices["validation"]].copy(),
        validation_labels=targets[split_indices["validation"]].copy(),
        test_inputs=normalized[split_indices["test"]].copy(),
        test_labels=targets[split_indices["test"]].copy(),
        mean=mean,
        scale=scale,
        archive_sha256=digest,
    )
