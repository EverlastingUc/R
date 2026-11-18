"""Tests for leakage-aware assistant conversation data preparation."""

import hashlib
import json

import pytest

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.chat_data import (
    IGNORE_INDEX,
    _encode_conversation,
    prepare_chat_dataset,
)
from entropy_lm.tokenizer import ByteTokenizer


def _messages(index: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer clearly and accurately."},
        {"role": "user", "content": f"Question number {index}?"},
        {"role": "assistant", "content": f"Answer number {index}."},
        {"role": "user", "content": f"Another question about {index}."},
        {"role": "assistant", "content": f"Another answer for {index}."},
    ]


def _write_conversations(path, conversations):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"messages": messages}) + "\n" for messages in conversations
        ),
        encoding="utf-8",
    )


def test_chat_labels_train_assistant_content_and_mask_prompt_tokens():
    tokenizer = ByteTokenizer()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
    input_ids, labels = _encode_conversation(messages, tokenizer)
    expected_targets = tokenizer.encode("4\n") + [tokenizer.eos_id]
    assert len(input_ids) == len(labels)
    assert labels[0] == IGNORE_INDEX
    assert [token for token in labels if token != IGNORE_INDEX] == expected_targets
    assert labels.count(IGNORE_INDEX) > len(expected_targets)


def test_chat_preparation_splits_deduplicates_and_masks_prompt_tokens(tmp_path):
    source = tmp_path / "conversations.jsonl"
    conversations = [_messages(index) for index in range(10)]
    _write_conversations(source, conversations + [conversations[0]])
    output = tmp_path / "prepared"

    manifest = prepare_chat_dataset(
        source,
        output,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
        seed=13,
    )

    assert manifest["conversation_count"] == 10
    assert manifest["duplicate_conversation_count"] == 1
    assert sum(manifest["splits"].values()) == 10
    assert manifest["assistant_target_counts"]["train"] > 0

    assignments = {item["sha256"]: item["split"] for item in manifest["documents"]}
    assert len(assignments) == 10
    assert len({item["sha256"] for item in manifest["documents"]}) == 10

    record = json.loads(
        (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["tokenizer_fingerprint"] == manifest["tokenizer_fingerprint"]
    assert len(record["input_ids"]) == len(record["labels"])
    assert IGNORE_INDEX in record["labels"]
    assert any(label != IGNORE_INDEX for label in record["labels"])
    assert len(record["assistant_turns"]) == 2
    assert [turn["turn_index"] for turn in record["assistant_turns"]] == [0, 1]
    for turn in record["assistant_turns"]:
        assert 0 <= turn["target_start"] < turn["target_end"] <= len(record["labels"])
        assert all(
            label != IGNORE_INDEX
            for label in record["labels"][turn["target_start"] : turn["target_end"]]
        )


def test_chat_bpe_metadata_matches_manifest_and_is_loadable(tmp_path):
    source = tmp_path / "conversations.jsonl"
    _write_conversations(source, [_messages(index) for index in range(10)])
    output = tmp_path / "prepared"
    manifest = prepare_chat_dataset(
        source,
        output,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="bpe",
        vocab_size=280,
        seed=5,
    )

    metadata = json.loads((output / "tokenizer.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert metadata == manifest["tokenizer"]
    assert fingerprint == manifest["tokenizer_fingerprint"]
    assert (
        BPETokenizer.load(output / "tokenizer.json").vocab_size
        == metadata["vocab_size"]
    )


def test_chat_preparation_rejects_conversation_without_assistant(tmp_path):
    source = tmp_path / "invalid.jsonl"
    _write_conversations(
        source,
        [
            [
                {"role": "system", "content": "Answer questions."},
                {"role": "user", "content": "No response is present."},
            ]
        ],
    )
    with pytest.raises(ValueError, match="needs user and assistant"):
        prepare_chat_dataset(source, tmp_path / "prepared", tokenizer_type="byte")
