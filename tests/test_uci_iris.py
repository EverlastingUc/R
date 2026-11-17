"""Tests for deterministic UCI Iris loading and preprocessing."""

import csv
import io
import zipfile

import numpy as np
import pytest

from nn.uci_iris import CLASS_NAMES, load_uci_iris


def write_iris_archive(path) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for class_id, class_name in enumerate(CLASS_NAMES):
        for index in range(50):
            base = 1.0 + class_id * 2.0 + index * 0.01
            writer.writerow(
                [base, base + 0.2, base + 0.4, base + 0.6, class_name]
            )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("iris.data", buffer.getvalue())


def test_iris_loader_is_reproducible_and_stratified(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_iris_archive(raw / "uci-iris.zip")

    first = load_uci_iris(raw, seed=31)
    second = load_uci_iris(raw, seed=31)

    assert first.train_inputs.shape == (105, 4)
    assert first.validation_inputs.shape == (24, 4)
    assert first.test_inputs.shape == (21, 4)
    np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
    np.testing.assert_array_equal(first.train_labels, second.train_labels)
    np.testing.assert_array_equal(
        np.bincount(first.train_labels, minlength=3), [35, 35, 35]
    )
    np.testing.assert_array_equal(
        np.bincount(first.validation_labels, minlength=3), [8, 8, 8]
    )
    np.testing.assert_array_equal(
        np.bincount(first.test_labels, minlength=3), [7, 7, 7]
    )
    np.testing.assert_allclose(first.train_inputs.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(first.train_inputs.std(axis=0), 1.0, atol=1e-12)
    assert first.metadata()["normalization"]["fit_partition"] == "train"


def test_iris_loader_rejects_corrupt_cached_archive(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "uci-iris.zip").write_bytes(b"not a zip archive")

    with pytest.raises(ValueError, match="valid ZIP"):
        load_uci_iris(raw)


def test_iris_loader_rejects_unsafe_archive_paths(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "uci-iris.zip", "w") as archive:
        archive.writestr("../iris.data", "invalid")

    with pytest.raises(ValueError, match="unsafe path"):
        load_uci_iris(raw)


def test_iris_metadata_records_source_provenance(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_iris_archive(raw / "uci-iris.zip")

    metadata = load_uci_iris(raw).metadata()

    assert metadata["dataset_id"] == 53
    assert metadata["doi"] == "10.24432/C56C76"
    assert metadata["license"] == "CC BY 4.0"
    assert len(metadata["archive_sha256"]) == 64
