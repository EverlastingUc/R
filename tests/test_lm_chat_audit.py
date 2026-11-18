"""Tests for the non-destructive conversation dataset auditor."""

import json

import pytest

from entropy_lm.chat_audit import audit_chat_dataset


def _record(user: str, assistant: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    )


def test_audit_counts_duplicates_short_responses_and_does_not_emit_content(tmp_path):
    source = tmp_path / "data.jsonl"
    first = _record("What is gravity?", "A force.")
    second = _record("Explain stars.", "Stars are luminous spheres of plasma.")
    source.write_text(first + "\n" + first + "\n" + second + "\n\n", encoding="utf-8")

    report = audit_chat_dataset(source, short_response_chars=10)

    assert report["valid_unique_conversations"] == 2
    assert report["exact_duplicate_conversations"] == 1
    assert report["blank_lines"] == 1
    assert report["assistant_response_chars"]["count"] == 2
    assert report["assistant_response_chars"]["below_short_threshold"] == 1
    assert report["content_included"] is False
    assert "A force." not in json.dumps(report)


def test_audit_reports_invalid_json_and_schema_records(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        "{invalid json}\n"
        + json.dumps({"messages": [{"role": "user", "content": "Missing reply"}]})
        + "\n",
        encoding="utf-8",
    )

    report = audit_chat_dataset(source)

    assert report["valid_unique_conversations"] == 0
    assert report["invalid_records_or_files"] == 2
    assert report["issues"]


def test_audit_reports_turn_structure_and_writes_atomic_json_report(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer one"},
                    {"role": "assistant", "content": "Answer one"},
                    {"role": "user", "content": "Follow up"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "reports" / "audit.json"

    report = audit_chat_dataset(source, report_path=destination)

    assert report["repeated_assistant_messages"] == 1
    assert report["conversations_without_final_assistant"] == 1
    assert report["consecutive_same_role_transitions"] == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == report


def test_audit_rejects_invalid_thresholds(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(_record("Question", "Answer") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="long_response_chars"):
        audit_chat_dataset(source, short_response_chars=20, long_response_chars=20)
