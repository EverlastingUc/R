"""Tests for CP21 acquisition, preprocessing, split hygiene and persistence."""

import csv
import io
import json
import zipfile
from datetime import date, timedelta

import numpy as np
import pytest

from nn.datasets import (
    BIKE_FEATURES,
    IRIS_CLASSES,
    download_to_cache,
    prepare_bike_hourly,
    prepare_iris,
    save_prepared_dataset,
    sha256_bytes,
)


def make_zip(name: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def iris_archive() -> bytes:
    rows = []
    for class_id, label in enumerate(IRIS_CLASSES):
        for index in range(50):
            values = (
                4.0 + class_id + index * 0.01,
                2.0 + class_id * 0.2 + index * 0.005,
                1.0 + class_id * 1.5 + index * 0.02,
                0.1 + class_id * 0.5 + index * 0.01,
            )
            rows.append(",".join([*(f"{value:.3f}" for value in values), label]))
    return make_zip("iris.data", "\n".join(rows) + "\n")


def bike_archive(day_count: int = 20, rows_per_day: int = 2) -> bytes:
    columns = ["instant", "dteday", *BIKE_FEATURES, "casual", "registered", "cnt"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    instant = 1
    for day_index in range(day_count):
        day = (date(2011, 1, 1) + timedelta(days=day_index)).isoformat()
        for hour in range(rows_per_day):
            values = {name: 0 for name in BIKE_FEATURES}
            values.update({
                "season": 1, "yr": 0, "mnth": 1, "hr": hour,
                "weekday": day_index % 7, "temp": day_index / day_count,
                "atemp": 0.25, "hum": 0.5, "windspeed": 0.1,
            })
            writer.writerow({
                "instant": instant, "dteday": day, **values,
                "casual": 1, "registered": 2, "cnt": 3,
            })
            instant += 1
    return make_zip("hour.csv", buffer.getvalue())


def test_iris_stratification_reproducibility_and_train_only_scaling() -> None:
    archive = iris_archive()
    first = prepare_iris(archive, seed=19)
    second = prepare_iris(archive, seed=19)
    assert first.train_inputs.shape == (105, 4)
    assert first.validation_inputs.shape == (23, 4)
    assert first.test_inputs.shape == (22, 4)
    assert first.classes == IRIS_CLASSES
    assert all(
        count > 0
        for split in first.split_metadata["class_counts"].values()
        for count in split.values()
    )
    assert first.split_metadata["class_counts"]["train"] == {
        label: 35 for label in IRIS_CLASSES
    }
    np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
    np.testing.assert_array_equal(first.train_targets, second.train_targets)
    np.testing.assert_allclose(first.train_inputs.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(first.train_inputs.std(axis=0), 1.0, atol=1e-12)


def test_iris_rejects_unknown_labels_and_duplicate_zip_members() -> None:
    invalid = make_zip(
        "iris.data",
        "\n".join(
            [f"5.1,3.5,1.4,0.2,{label}" for label in IRIS_CLASSES for _ in range(50)]
        ).replace("Iris-setosa", "Iris-unknown"),
    )
    with pytest.raises(ValueError, match="invalid Iris observation"):
        prepare_iris(invalid)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("a/iris.data", "one")
        zipped.writestr("b/iris.data", "two")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_iris(archive.getvalue())


def test_bike_split_is_chronological_and_excludes_target_leakage() -> None:
    prepared = prepare_bike_hourly(bike_archive())
    assert prepared.train_inputs.shape == (28, len(BIKE_FEATURES))
    assert prepared.validation_inputs.shape == (6, len(BIKE_FEATURES))
    assert prepared.test_inputs.shape == (6, len(BIKE_FEATURES))
    assert prepared.train_targets.shape == (28, 1)
    assert prepared.feature_names == BIKE_FEATURES
    assert not {"casual", "registered", "instant", "dteday"}.intersection(
        prepared.feature_names
    )
    ranges = prepared.split_metadata["date_ranges"]
    assert ranges["train"][1] < ranges["validation"][0]
    assert ranges["validation"][1] < ranges["test"][0]
    np.testing.assert_allclose(prepared.train_inputs.mean(axis=0), 0.0, atol=1e-12)
    assert np.isfinite(prepared.test_inputs).all()


def test_bike_rejects_target_identity_violation_and_unsorted_dates() -> None:
    archive = bike_archive(day_count=10, rows_per_day=1)
    csv_text = zipfile.ZipFile(io.BytesIO(archive)).read("hour.csv").decode()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    rows[0]["cnt"] = "99"
    corrupted = io.StringIO(newline="")
    writer = csv.DictWriter(corrupted, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match="cnt identity"):
        prepare_bike_hourly(make_zip("hour.csv", corrupted.getvalue()))
    lines = csv_text.splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    with pytest.raises(ValueError, match="chronological"):
        prepare_bike_hourly(make_zip("hour.csv", "\n".join(lines) + "\n"))


def test_constant_training_features_remain_finite() -> None:
    prepared = prepare_bike_hourly(bike_archive())
    for values in (
        prepared.train_inputs, prepared.validation_inputs, prepared.test_inputs
    ):
        assert np.isfinite(values).all()
    assert np.all(prepared.scaler_scale > 0)


def test_persistence_records_provenance_and_refuses_overwrite(tmp_path) -> None:
    prepared = prepare_iris(iris_archive())
    raw = iris_archive()
    archive_path = tmp_path / "iris.npz"
    archive, metadata_path = save_prepared_dataset(
        prepared,
        archive_path,
        {"source_sha256": sha256_bytes(raw), "license": "CC BY 4.0"},
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == sha256_bytes(raw)
    assert metadata["license"] == "CC BY 4.0"
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["train_targets"], prepared.train_targets)
    with pytest.raises(FileExistsError):
        save_prepared_dataset(prepared, archive_path, metadata)


def test_download_cache_requires_exact_uci_host_and_preserves_existing(tmp_path) -> None:
    cached = tmp_path / "archive.zip"
    cached.write_bytes(b"original")
    assert download_to_cache(
        "https://archive.ics.uci.edu/static/example.zip", cached
    ) == cached
    assert cached.read_bytes() == b"original"
    for url in (
        "http://archive.ics.uci.edu/example.zip",
        "https://archive.ics.uci.edu.evil.example/data.zip",
        "https://user:pass@archive.ics.uci.edu/data.zip",
    ):
        with pytest.raises(ValueError, match="official UCI HTTPS"):
            download_to_cache(url, tmp_path / "rejected.zip")
