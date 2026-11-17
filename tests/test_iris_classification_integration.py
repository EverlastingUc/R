"""End-to-end integration of the UCI Iris loader and classifier."""

import csv
import io
import zipfile

import numpy as np

from nn.classification import ClassificationTrainer
from nn.network import FeedForwardNetwork
from nn.optimizers import Adam
from nn.uci_iris import CLASS_NAMES, load_uci_iris


def make_separable_iris_archive(path) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for class_id, class_name in enumerate(CLASS_NAMES):
        for sample_id in range(50):
            base = 1.0 + class_id * 2.0 + sample_id * 0.01
            writer.writerow([base, base + 0.2, base + 0.4, base + 0.6, class_name])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("iris.data", buffer.getvalue())


def test_iris_data_trains_a_multiclass_entropy_network(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    make_separable_iris_archive(raw_directory / "uci-iris.zip")
    dataset = load_uci_iris(raw_directory, seed=23)

    network = FeedForwardNetwork(
        [4, 12, 8, 3],
        ["relu", "tanh", "softmax"],
        seed=23,
    )
    trainer = ClassificationTrainer(
        network,
        Adam(learning_rate=0.02),
        batch_size=16,
        seed=23,
    )
    history = trainer.fit(
        dataset.train_inputs,
        dataset.train_labels,
        dataset.validation_inputs,
        dataset.validation_labels,
        epochs=100,
    )

    predictions = trainer.predict(dataset.test_inputs)
    test_accuracy = float(np.mean(predictions == dataset.test_labels))

    assert history[-1].training_loss < history[0].training_loss
    assert history[-1].validation_accuracy >= 0.95
    assert test_accuracy >= 0.95
