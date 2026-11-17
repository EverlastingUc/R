"""Tests for conversation formatting and local chat generation."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.chat_inference import format_chat_prompt, generate_chat_response
from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_training import train_on_conversations
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.model import TransformerConfig


def test_format_chat_prompt_preserves_roles_and_appends_assistant_prefix():
    prompt = format_chat_prompt(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "How are you?"},
        ]
    )
    assert prompt.startswith("<|system|>\nBe concise.\n")
    assert "<|assistant|>\nHi.\n<|user|>\nHow are you?\n" in prompt
    assert prompt.endswith("<|assistant|>\n")


def test_format_chat_prompt_requires_user_at_end():
    with pytest.raises(ValueError, match="end with user"):
        format_chat_prompt(
            [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi."},
            ]
        )


def test_chat_response_uses_sft_checkpoint_and_eos_stopping(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        (source / f"{index}.jsonl").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "Answer clearly."},
                        {"role": "user", "content": f"Question {index}?"},
                        {"role": "assistant", "content": f"Answer {index}."},
                    ]
                }
            ) + "\n",
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source, prepared, validation_fraction=0.2, test_fraction=0.2, tokenizer_type="byte"
    )
    checkpoint = tmp_path / "assistant.pt"
    train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=24,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        steps=1,
        batch_size=2,
        eval_interval=1,
        checkpoint_path=checkpoint,
    )
    model = load_model_checkpoint(checkpoint)
    response = generate_chat_response(
        model,
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello."},
        ],
        max_new_tokens=2,
        top_k=1,
    )
    assert isinstance(response, str)
