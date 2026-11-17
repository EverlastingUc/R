"""Conversation-formatted inference and a local terminal chat interface."""

from __future__ import annotations

import argparse

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


@torch.inference_mode()
def generate_chat_response(
    model: CausalLanguageModel,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Generate one assistant turn from system/user/assistant conversation history."""
    tokenizer = getattr(model, "tokenizer", None)
    if not isinstance(tokenizer, (ByteTokenizer, BPETokenizer)):
        raise TypeError("model must be loaded with a supported tokenizer")
    prompt = format_chat_prompt(messages)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    prompt_ids = prompt_ids[-model.config.context_length :]
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
    )
    new_tokens = generated[0, len(prompt_ids) :].tolist()
    return tokenizer.decode(new_tokens, errors="replace").strip()


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
        response = generate_chat_response(
            model,
            conversation,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        conversation.append({"role": "assistant", "content": response})
        print(f"Entropy: {response}")


if __name__ == "__main__":
    main()
