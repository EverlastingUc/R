"""Tests for instruction JSONL normalization."""

import json

import pytest

from entropy_lm.instruction_data import (
    normalize_instruction_dataset,
    normalize_instruction_record,
)


def test_normalizes_prompt_completion_and_optional_system():
    messages = normalize_instruction_record(
        {"system": "Be precise.", "prompt": "2 + 2?", "completion": "4"},
        location="sample:1",
    )
    assert messages == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_normalizes_instruction_input_output():
    messages = normalize_instruction_record(
        {"instruction": "Translate.", "input": "Hello", "output": "Hola"},
        location="sample:1",
    )
    assert messages == [
        {"role": "user", "content": "Translate.\n\nHello"},
        {"role": "assistant", "content": "Hola"},
    ]


def test_accepts_existing_multiturn_messages():
    messages = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Second answer."},
    ]
    assert normalize_instruction_record(
        {"messages": messages}, location="sample:1"
    ) == messages


@pytest.mark.parametrize(
    "record",
    [
        {"prompt": "missing completion"},
        {"instruction": "missing output"},
        {"instruction": "Do it", "output": "Done", "input": 3},
        {"prompt": "", "completion": "answer"},
        {"messages": [{"role": "user", "content": "No assistant"}]},
    ],
)
def test_rejects_ambiguous_or_invalid_records(record):
    with pytest.raises((TypeError, ValueError)):
        normalize_instruction_record(record, location="sample:1")


def test_dataset_normalization_is_atomic_and_counts_blank_lines(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "nested" / "chat.jsonl"
    source.write_text(
        json.dumps({"instruction": "Explain", "input": "gravity", "output": "Attraction."})
        + "\n\n"
        + json.dumps({"prompt": "Hi", "completion": "Hello"})
        + "\n",
        encoding="utf-8",
    )

    report = normalize_instruction_dataset(source, output)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["records_written"] == 2
    assert report["blank_lines_skipped"] == 1
    assert records[0]["messages"][0]["content"] == "Explain\n\ngravity"
    assert records[1]["messages"][-1]["content"] == "Hello"
    assert report["content_included"] is False


def test_malformed_input_does_not_replace_existing_output(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text('{"prompt":"unfinished"}\n', encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must appear together"):
        normalize_instruction_dataset(source, output)

    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_rejects_same_input_and_output_path(tmp_path):
    source = tmp_path / "same.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different"):
        normalize_instruction_dataset(source, source)
