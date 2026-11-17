"""Leakage-safe, reproducible loaders for Entropy's first real datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IRIS_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
IRIS_CLASSES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
IRIS_FEATURES = ("sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm")
BIKE_FEATURES = (
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
)
BIKE_REQUIRED = ("instant", "dteday", *BIKE_FEATURES, "casual", "registered", "cnt")


@dataclass(frozen=True)
class DatasetSplits:
    """Prepared arrays; scaler statistics are fitted on training data only."""

    name: str
    task: str
    feature_names: tuple[str, ...]
    target_name: str
    train_inputs: FloatArray
    train_targets: NDArray[Any]
    validation_inputs: FloatArray
    validation_targets: NDArray[Any]
    test_inputs: FloatArray
    test_targets: NDArray[Any]
    scaler_mean: FloatArray
    scaler_scale: FloatArray
    classes: tuple[str, ...] = ()
    split_metadata: dict[str, Any] | None = None


def sha256_bytes(content: bytes) -> str:
    """Return a SHA-256 digest for dataset provenance."""
    return hashlib.sha256(content).hexdigest()


def _is_official_uci_https(url: str) -> bool:
    """Require the exact UCI hostname, HTTPS, and no embedded credentials."""
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname == "archive.ics.uci.edu"
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def download_to_cache(url: str, destination: str | Path) -> Path:
    """Download a bounded archive only from the official UCI HTTPS host."""
    if not _is_official_uci_https(url):
        raise ValueError("downloads must use the official UCI HTTPS host")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("cached archive exceeds the 10 MiB size limit")
        return path
    request = urllib.request.Request(
        url, headers={"User-Agent": "Entropy-Research/0.1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if not _is_official_uci_https(response.geturl()):
            raise ValueError("download redirected away from the official UCI host")
        content = response.read(10 * 1024 * 1024 + 1)
    if not content or len(content) > 10 * 1024 * 1024:
        raise ValueError("download is empty or exceeds the 10 MiB size limit")
    with path.open("xb") as output:
        output.write(content)
    return path


def _zip_member(archive_bytes: bytes, member_name: str) -> bytes:
    """Read exactly one named member without extracting ZIP paths."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [
                item for item in archive.infolist()
                if PurePosixPath(item.filename).name.lower() == member_name.lower()
            ]
            if len(matches) != 1:
                raise ValueError(f"archive must contain exactly one {member_name} member")
            if matches[0].file_size > 5 * 1024 * 1024:
                raise ValueError("archive member exceeds the 5 MiB size limit")
            return archive.read(matches[0])
    except zipfile.BadZipFile as error:
        raise ValueError("source is not a valid ZIP archive") from error


def _fractions(train: float, validation: float, test: float) -> None:
    values = (train, validation, test)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("split fractions must be finite and positive")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1.0")


def _scale(
    train: FloatArray, validation: FloatArray, test: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit a standard scaler on train and apply unchanged statistics elsewhere."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return (train - mean) / scale, (validation - mean) / scale, (test - mean) / scale, mean, scale


def _stratified_split(
    labels: NDArray[np.int64], seed: int, train: float, validation: float, test: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Make deterministic per-class splits while keeping every class represented."""
    _fractions(train, validation, test)
    rng = np.random.default_rng(seed)
    groups: list[tuple[NDArray[np.int64], int, int, float]] = []
    total_remaining = 0
    base_validation = 0
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        train_count = int(np.floor(len(indices) * train))
        remaining = len(indices) - train_count
        exact_validation = remaining * validation / (validation + test)
        validation_count = int(np.floor(exact_validation))
        groups.append((indices, train_count, validation_count, exact_validation - validation_count))
        total_remaining += remaining
        base_validation += validation_count
    target_validation = int(np.floor(total_remaining * validation / (validation + test) + 0.5))
    extra = target_validation - base_validation
    order = sorted(range(len(groups)), key=lambda i: (-groups[i][3], i))
    validation_counts = [group[2] for group in groups]
    for index in order[:extra]:
        validation_counts[index] += 1
    train_parts, validation_parts, test_parts = [], [], []
    for (indices, train_count, _, _), validation_count in zip(groups, validation_counts, strict=True):
        train_parts.append(indices[:train_count])
        validation_parts.append(indices[train_count:train_count + validation_count])
        test_parts.append(indices[train_count + validation_count:])
    train_idx = np.concatenate(train_parts)
    validation_idx = np.concatenate(validation_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(validation_idx)
    rng.shuffle(test_idx)
    return train_idx, validation_idx, test_idx


def prepare_iris(
    archive_bytes: bytes, *, seed: int = 42, train_fraction: float = 0.70,
    validation_fraction: float = 0.15, test_fraction: float = 0.15,
) -> DatasetSplits:
    """Load UCI Iris, stratify by class and standardise using training rows only."""
    content = _zip_member(archive_bytes, "iris.data").decode("utf-8-sig")
    rows = [row for row in csv.reader(io.StringIO(content)) if row and any(v.strip() for v in row)]
    if len(rows) != 150 or any(len(row) != 5 for row in rows):
        raise ValueError("official Iris data must contain 150 five-column rows")
    class_map = {name: index for index, name in enumerate(IRIS_CLASSES)}
    features = np.empty((150, 4), dtype=np.float64)
    labels = np.empty(150, dtype=np.int64)
    for index, row in enumerate(rows):
        try:
            features[index] = [float(value) for value in row[:4]]
            labels[index] = class_map[row[4].strip()]
        except (ValueError, KeyError) as error:
            raise ValueError(f"invalid Iris observation at row {index + 1}") from error
    if not np.isfinite(features).all() or not np.array_equal(np.bincount(labels), [50, 50, 50]):
        raise ValueError("Iris data must have finite values and 50 rows per class")
    train_idx, validation_idx, test_idx = _stratified_split(
        labels, seed, train_fraction, validation_fraction, test_fraction
    )
    train_x, validation_x, test_x, mean, scale = _scale(
        features[train_idx], features[validation_idx], features[test_idx]
    )
    split_targets = {
        "train": labels[train_idx], "validation": labels[validation_idx], "test": labels[test_idx]
    }
    counts = {
        split: {name: int(np.count_nonzero(values == class_id))
                for class_id, name in enumerate(IRIS_CLASSES)}
        for split, values in split_targets.items()
    }
    return DatasetSplits(
        "iris", "classification", IRIS_FEATURES, "class_id",
        train_x, labels[train_idx], validation_x, labels[validation_idx],
        test_x, labels[test_idx], mean, scale, IRIS_CLASSES,
        {"strategy": "stratified_random", "seed": seed, "class_counts": counts},
    )


def prepare_bike_hourly(
    archive_bytes: bytes, *, train_fraction: float = 0.70,
    validation_fraction: float = 0.15, test_fraction: float = 0.15,
) -> DatasetSplits:
    """Load hourly UCI Bike Sharing with chronological, whole-day splits."""
    content = _zip_member(archive_bytes, "hour.csv").decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None or not set(BIKE_REQUIRED).issubset(reader.fieldnames):
        raise ValueError("Bike Sharing CSV is missing required columns")
    feature_rows: list[list[float]] = []
    targets: list[float] = []
    dates: list[str] = []
    previous: date | None = None
    for row_number, row in enumerate(reader, start=2):
        try:
            current = date.fromisoformat(row["dteday"])
            values = [float(row[name]) for name in BIKE_FEATURES]
            casual, registered, target = int(row["casual"]), int(row["registered"]), int(row["cnt"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed Bike Sharing row {row_number}") from error
        if previous is not None and current < previous:
            raise ValueError("Bike Sharing rows must be chronological")
        if min(casual, registered, target) < 0 or target != casual + registered:
            raise ValueError(f"row {row_number} violates non-negative counts or cnt identity")
        if not np.isfinite(values).all():
            raise ValueError(f"row {row_number} contains non-finite features")
        feature_rows.append(values)
        targets.append(float(target))
        dates.append(current.isoformat())
        previous = current
    if not feature_rows:
        raise ValueError("Bike Sharing CSV contains no observations")
    features = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    date_values = np.asarray(dates, dtype="U10")
    unique_dates = np.unique(date_values)
    _fractions(train_fraction, validation_fraction, test_fraction)
    train_end = int(np.floor(len(unique_dates) * train_fraction))
    validation_end = train_end + int(np.floor(len(unique_dates) * validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(unique_dates):
        raise ValueError("split fractions produce an empty chronological split")
    date_groups = (unique_dates[:train_end], unique_dates[train_end:validation_end], unique_dates[validation_end:])
    indices = tuple(np.flatnonzero(np.isin(date_values, group)) for group in date_groups)
    train_idx, validation_idx, test_idx = indices
    train_x, validation_x, test_x, mean, scale = _scale(
        features[train_idx], features[validation_idx], features[test_idx]
    )
    return DatasetSplits(
        "bike_sharing_hourly", "regression", BIKE_FEATURES, "cnt",
        train_x, y[train_idx], validation_x, y[validation_idx], test_x, y[test_idx],
        mean, scale, (),
        {
            "strategy": "chronological_by_calendar_day",
            "date_ranges": {
                "train": [str(date_groups[0][0]), str(date_groups[0][-1])],
                "validation": [str(date_groups[1][0]), str(date_groups[1][-1])],
                "test": [str(date_groups[2][0]), str(date_groups[2][-1])],
            },
            "excluded_columns": ["instant", "dteday", "casual", "registered"],
        },
    )


def save_prepared_dataset(
    dataset: DatasetSplits, destination: str | Path, provenance: dict[str, Any]
) -> tuple[Path, Path]:
    """Write numeric arrays and provenance JSON without overwriting prior results."""
    path = Path(destination)
    metadata_path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing prepared dataset")
    metadata = {
        **provenance, "name": dataset.name, "task": dataset.task,
        "feature_names": list(dataset.feature_names), "target_name": dataset.target_name,
        "classes": list(dataset.classes), "split_metadata": dataset.split_metadata,
        "shapes": {
            name: list(getattr(dataset, name).shape)
            for name in (
                "train_inputs", "train_targets", "validation_inputs",
                "validation_targets", "test_inputs", "test_targets",
            )
        },
    }
    with path.open("xb") as output:
        np.savez_compressed(
            output, train_inputs=dataset.train_inputs, train_targets=dataset.train_targets,
            validation_inputs=dataset.validation_inputs, validation_targets=dataset.validation_targets,
            test_inputs=dataset.test_inputs, test_targets=dataset.test_targets,
            scaler_mean=dataset.scaler_mean, scaler_scale=dataset.scaler_scale,
        )
    try:
        with metadata_path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path, metadata_path
