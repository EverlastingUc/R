"""Acceptance checks for the memorization/generalization experiment."""

import numpy as np
import pytest

from experiments.memorization_vs_generalization import (
    make_random_label_data,
    make_structured_data,
    run_experiment,
)


@pytest.fixture(scope="module")
def experiment_result(tmp_path_factory: pytest.TempPathFactory):
    output_directory = tmp_path_factory.mktemp("entropy-experiment")
    return run_experiment(output_dir=output_directory)


def test_structured_dataset_is_reproducible_and_splits_are_distinct() -> None:
    first = make_structured_data(seed=21)
    second = make_structured_data(seed=21)

    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)

    train_inputs, _, test_inputs, _ = first
    train_rows = {tuple(row) for row in train_inputs}
    test_rows = {tuple(row) for row in test_inputs}
    assert train_rows.isdisjoint(test_rows)


def test_random_label_data_is_reproducible() -> None:
    first = make_random_label_data(seed=11, train_count=12, test_count=32)
    second = make_random_label_data(seed=11, train_count=12, test_count=32)

    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)


def test_structured_xor_task_generalizes(experiment_result) -> None:
    scores = experiment_result.generalization
    assert scores.train_accuracy >= 0.95
    assert scores.test_accuracy >= 0.95
    assert scores.final_train_loss < scores.initial_train_loss


def test_random_labels_are_memorized_but_do_not_generalize(experiment_result) -> None:
    scores = experiment_result.memorization
    assert scores.train_accuracy >= 0.95
    assert 0.30 <= scores.test_accuracy <= 0.70
    assert scores.train_accuracy - scores.test_accuracy >= 0.25


def test_experiment_saves_plot_and_metrics(experiment_result) -> None:
    assert experiment_result.plot_path.is_file()
    assert experiment_result.plot_path.stat().st_size > 0
    assert experiment_result.metrics_path.is_file()
