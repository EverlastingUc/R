"""Tests for conversation formatting and local chat generation."""

import json

import pytest

torch = pytest.importorskip("torch")

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.chat_data import prepare_chat_dataset
from entropy_lm.chat_inference import (
    build_chat_input_ids,
    format_chat_prompt,
    generate_chat_response,
    stream_chat_response,
)
from entropy_lm.chat_training import train_on_conversations
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.model import CausalLanguageModel, TransformerConfig
from entropy_lm.tokenizer import ByteTokenizer


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
            )
            + "\n",
            encoding="utf-8",
        )
    prepared = tmp_path / "prepared"
    prepare_chat_dataset(
        source,
        prepared,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer_type="byte",
    )
    checkpoint = tmp_path / "assistant.pt"
    train_on_conversations(
        prepared,
        config=TransformerConfig(
            vocab_size=258,
            context_length=32,
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


def test_model_generate_stream_reconstructs_the_complete_sequence():
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=258,
            context_length=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    input_ids = torch.tensor([[256, 65, 66]], dtype=torch.long)
    torch.manual_seed(101)
    expected = model.generate(input_ids, max_new_tokens=5, top_k=1, eos_token_id=257)
    torch.manual_seed(101)
    generated = list(
        model.generate_stream(input_ids, max_new_tokens=5, top_k=1, eos_token_id=257)
    )
    actual = torch.cat([input_ids, *generated], dim=1)
    assert torch.equal(actual, expected)
    assert model.training is True

    interrupted = model.generate_stream(
        input_ids, max_new_tokens=5, top_k=1, eos_token_id=257
    )
    next(interrupted)
    assert model.training is False
    interrupted.close()
    assert model.training is True


@pytest.mark.parametrize(
    "tokenizer",
    [ByteTokenizer(), BPETokenizer([(195, 169)])],
)
def test_streaming_chat_decodes_utf8_safely(tokenizer, tmp_path):
    model = CausalLanguageModel(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=64,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    model.tokenizer = tokenizer
    expected_text = "café 🌍 — streaming"
    generated_ids = tokenizer.encode(expected_text) + [tokenizer.eos_id]

    def fixed_stream(input_ids, **kwargs):
        for token_id in generated_ids:
            yield torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)

    model.generate_stream = fixed_stream
    messages = [{"role": "user", "content": "Return the test phrase."}]
    chunks = list(
        stream_chat_response(
            model, messages, top_k=1, top_p=0.9, repetition_penalty=1.1
        )
    )
    assert "".join(chunks) == expected_text
    assert (
        generate_chat_response(
            model, messages, top_k=1, top_p=0.9, repetition_penalty=1.1
        )
        == expected_text
    )


def test_context_packing_preserves_system_and_latest_user_turn():
    tokenizer = ByteTokenizer()
    messages = [
        {"role": "system", "content": "Follow the important system rule."},
        {"role": "user", "content": "Old question that should be removed."},
        {"role": "assistant", "content": "Old answer that should be removed."},
        {
            "role": "user",
            "content": "A very long current question with a distinctive ending marker.",
        },
    ]
    token_ids = build_chat_input_ids(tokenizer, messages, context_length=96)
    decoded = tokenizer.decode(token_ids)
    assert len(token_ids) <= 96
    assert decoded.startswith("<|system|>\nFollow")
    assert "<|user|>\n" in decoded
    assert decoded.endswith("<|assistant|>\n")
    assert "ending marker." in decoded
    assert "Old question" not in decoded
    assert "Old answer" not in decoded


def test_context_packing_keeps_recent_complete_history_when_it_fits():
    tokenizer = ByteTokenizer()
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Current question?"},
    ]
    token_ids = build_chat_input_ids(tokenizer, messages, context_length=128)
    decoded = tokenizer.decode(token_ids)
    assert "Be precise." in decoded
    assert "Earlier question." in decoded
    assert "Earlier answer." in decoded
    assert decoded.endswith("<|user|>\nCurrent question?\n<|assistant|>\n")


def test_context_packing_rejects_context_too_small_for_chat_prefix():
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="too small"):
        build_chat_input_ids(
            tokenizer,
            [{"role": "user", "content": "Hello."}],
            context_length=4,
        )



def test_stream_until_stop_handles_markers_split_across_chunks_and_closes():
    from entropy_lm.chat_inference import _stream_until_stop

    closed = []

    def source():
        try:
            yield "The answer is EN"
            yield "D_MARKERhidden content"
        finally:
            closed.append(True)

    chunks = list(_stream_until_stop(source(), ["END_MARKER"]))
    assert "".join(chunks) == "The answer is "
    assert closed == [True]


def test_stream_until_stop_preserves_partial_prefix_when_no_stop_completes():
    from entropy_lm.chat_inference import _stream_until_stop

    chunks = list(_stream_until_stop(iter(["visible", " END", " partial"]), ["END_MARKER"]))
    assert "".join(chunks) == "visible END partial"


def test_stream_until_stop_chooses_earliest_marker_and_validates_values():
    from entropy_lm.chat_inference import _stream_until_stop

    chunks = list(
        _stream_until_stop(iter(["before STOP then END"]), ["END", "STOP"])
    )
    assert "".join(chunks) == "before "

    with pytest.raises(ValueError, match="non-empty strings"):
        list(_stream_until_stop(iter(["text"]), [""]))
    with pytest.raises(ValueError, match="non-empty strings"):
        list(_stream_until_stop(iter(["text"]), ["ok", 3]))
