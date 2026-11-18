"""Conversation-formatted streaming inference and a local terminal chat interface."""

from __future__ import annotations

import argparse
import codecs
import sys
from collections.abc import Iterator

import torch

from entropy_lm.bpe_tokenizer import BPETokenizer
from entropy_lm.inference import load_model_checkpoint
from entropy_lm.model import CausalLanguageModel
from entropy_lm.tokenizer import ByteTokenizer

_ALLOWED_ROLES = {"system", "user", "assistant"}


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Serialize conversation history and leave an assistant response prefix."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    formatted: list[str] = []
    has_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be non-empty")
        has_user |= role == "user"
        formatted.append(f"<|{role}|>\n{content}\n")
    if not has_user or messages[-1].get("role") != "user":
        raise ValueError("conversation must contain a user message and end with user")
    formatted.append("<|assistant|>\n")
    return "".join(formatted)


def _token_to_bytes(tokenizer: ByteTokenizer | BPETokenizer, token_id: int) -> bytes:
    """Expand one ordinary tokenizer ID to UTF-8 bytes for incremental decoding."""
    if token_id in (tokenizer.bos_id, tokenizer.eos_id):
        return b""
    if 0 <= token_id <= 255:
        return bytes((token_id,))
    if isinstance(tokenizer, BPETokenizer) and 258 <= token_id < tokenizer.bos_id:
        pending = [token_id]
        expanded = bytearray()
        while pending:
            current = pending.pop()
            if current < 256:
                expanded.append(current)
                continue
            if current < 258:
                raise ValueError(f"invalid reserved BPE token ID: {current}")
            merge_index = current - 258
            if merge_index >= len(tokenizer.merges):
                raise ValueError(f"invalid BPE token ID: {current}")
            left, right = tokenizer.merges[merge_index]
            pending.extend((right, left))
        return bytes(expanded)
    raise ValueError(f"unexpected special or out-of-range token ID: {token_id}")


def _decode_token_stream(
    tokenizer: ByteTokenizer | BPETokenizer, token_stream: Iterator[torch.Tensor]
) -> Iterator[str]:
    """Incrementally decode token bytes without splitting Unicode characters."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        for token in token_stream:
            if token.numel() != 1:
                raise ValueError("chat streaming expects one generated sequence")
            token_id = int(token.item())
            if token_id == tokenizer.eos_id:
                break
            chunk = decoder.decode(_token_to_bytes(tokenizer, token_id), final=False)
            if chunk:
                yield chunk
        remainder = decoder.decode(b"", final=True)
        if remainder:
            yield remainder
    finally:
        close = getattr(token_stream, "close", None)
        if callable(close):
            close()


def build_chat_input_ids(
    tokenizer: ByteTokenizer | BPETokenizer,
    messages: list[dict[str, str]],
    *,
    context_length: int,
) -> list[int]:
    """Pack chat context while prioritizing the system and latest user message.

    The first system message is retained with priority. The latest user message
    is always retained; if it is too long, its header and trailing content are
    kept. Older messages are included only when each complete message fits.
    """
    format_chat_prompt(messages)
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("context_length must be an integer")
    if context_length < 1:
        raise ValueError("context_length must be positive")

    assistant_prefix = tokenizer.encode("<|assistant|>\n")
    available = context_length - 1 - len(assistant_prefix)
    latest_user = messages[-1]
    user_header = tokenizer.encode("<|user|>\n")
    user_body = tokenizer.encode(latest_user["content"] + "\n")
    minimum_user_length = len(user_header) + min(1, len(user_body))
    if available < minimum_user_length:
        raise ValueError(
            "context_length is too small for a user message and assistant prefix"
        )

    has_system = messages[0]["role"] == "system"
    system_header: list[int] = []
    system_body: list[int] = []
    if has_system:
        system_header = tokenizer.encode("<|system|>\n")
        system_body = tokenizer.encode(messages[0]["content"] + "\n")
    full_system = system_header + system_body
    full_user = user_header + user_body

    if len(full_system) + len(full_user) <= available:
        system_tokens = full_system
        user_tokens = full_user
    else:
        system_capacity = min(
            len(full_system),
            max(len(system_header), available // 4),
            available - minimum_user_length,
        )
        if system_capacity >= len(system_header):
            system_tokens = (
                system_header + system_body[: system_capacity - len(system_header)]
            )
        else:
            system_tokens = []

        user_capacity = available - len(system_tokens)
        body_capacity = user_capacity - len(user_header)
        if body_capacity < 1:
            raise ValueError("context_length leaves no room for latest user content")
        user_tokens = user_header + user_body[-body_capacity:]

    remaining = available - len(system_tokens) - len(user_tokens)
    history_blocks: list[list[int]] = []
    history_start = 1 if has_system else 0
    for message in reversed(messages[history_start:-1]):
        block = tokenizer.encode(f"<|{message['role']}|>\n")
        block.extend(tokenizer.encode(message["content"] + "\n"))
        if len(block) > remaining:
            break
        history_blocks.append(block)
        remaining -= len(block)

    history_tokens = [
        token_id for block in reversed(history_blocks) for token_id in block
    ]
    return (
        [tokenizer.bos_id]
        + system_tokens
        + history_tokens
        + user_tokens
        + assistant_prefix
    )


def stream_chat_response(
    model: CausalLanguageModel,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> Iterator[str]:
    """Yield decoded assistant text as the model generates each token."""
    tokenizer = getattr(model, "tokenizer", None)
    if not isinstance(tokenizer, (ByteTokenizer, BPETokenizer)):
        raise TypeError("model must be loaded with a supported tokenizer")
    prompt_ids = build_chat_input_ids(
        tokenizer, messages, context_length=model.config.context_length
    )
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
    )
    generated_tokens = model.generate_stream(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
    )
    yield from _decode_token_stream(tokenizer, generated_tokens)


def generate_chat_response(
    model: CausalLanguageModel,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Generate a complete response using the same UTF-8-safe streaming path."""
    return "".join(
        stream_chat_response(
            model,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a locally trained Entropy assistant checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--system-prompt",
        default="You are Entropy, a helpful and accurate AI assistant.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="wait for the complete response before printing it",
    )
    args = parser.parse_args()
    model = load_model_checkpoint(args.checkpoint, device=args.device)
    conversation = [{"role": "system", "content": args.system_prompt}]
    print("Entropy chat. Use /reset to clear history or /exit to quit.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            conversation = [{"role": "system", "content": args.system_prompt}]
            print("Conversation reset.")
            continue
        conversation.append({"role": "user", "content": user_text})
        if args.no_stream:
            response = generate_chat_response(
                model,
                conversation,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"Entropy: {response}")
        else:
            print("Entropy: ", end="", flush=True)
            response_chunks: list[str] = []
            for chunk in stream_chat_response(
                model,
                conversation,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                response_chunks.append(chunk)
            print()
            response = "".join(response_chunks).strip()
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
