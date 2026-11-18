"""Tests for prepared chat dataset integrity and leakage checks."""

import json

import pytest

from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_dataset_check import check_prepared_chat_dataset


def _conversation(index: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"Question {index}?"},
        {"role": "assistant", "content": f"Answer {index}."},
    ]


def _prepared(tmp_path):
    source = tmp_path / "conversations.jsonl"
    source.write_text(
        "".join(
            json.dumps({"messages": _conversation(index)}) + "\n" for index in range(12)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        output,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=17,
    )
    return output


def test_check_prepared_chat_dataset_reports_integrity_without_content(tmp_path):
    output = _prepared(tmp_path)

    report = check_prepared_chat_dataset(output)

    assert report["valid"] is True
    assert report["conversation_count"] == 12
    assert report["split_hashes_disjoint"] is True
    assert sum(report["splits"].values()) == 12
    assert report["assistant_targets"]["train"] > 0
    assert report["content_included"] is False


def test_check_detects_split_record_tampering(tmp_path):
    output = _prepared(tmp_path)
    path = output / "train.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    record["tokenizer_fingerprint"] = "0" * 64
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        check_prepared_chat_dataset(output)


def test_check_detects_cross_split_conversation_duplication(tmp_path):
    output = _prepared(tmp_path)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    train_record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    validation_records = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
    ]
    validation_records[0]["sha256"] = train_record["sha256"]
    validation_records[0]["source"] = train_record["source"]
    validation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in validation_records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split/source mismatch|more than once"):
        check_prepared_chat_dataset(output)



def test_check_detects_corrupted_assistant_turn_spans(tmp_path):
    output = _prepared(tmp_path)
    path = output / "train.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["assistant_turns"][0]["target_end"] = len(records[0]["labels"]) + 1
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assistant-turn spans are invalid"):
        check_prepared_chat_dataset(output)
