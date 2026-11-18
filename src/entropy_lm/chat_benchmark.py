"""Reference-based behavioral benchmark for Entropy chat checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from entropy_lm.chat_inference import format_chat_prompt, generate_chat_response
from entropy_lm.inference import load_model_checkpoint


def _normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without removing meaningful punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a phrase as consecutive Unicode word tokens, not a substring."""
    text_tokens = re.findall(r"\w+", _normalize_text(text))
    phrase_tokens = re.findall(r"\w+", _normalize_text(phrase))
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _token_f1(prediction: str, reference: str) -> float:
    """Compute whitespace/punctuation-insensitive multiset token F1."""
    prediction_tokens = Counter(re.findall(r"\w+", _normalize_text(prediction)))
    reference_tokens = Counter(re.findall(r"\w+", _normalize_text(reference)))
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((prediction_tokens & reference_tokens).values())
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2.0 * precision * recall / (precision + recall)


def _read_benchmark_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load validated JSONL benchmark cases and return the source-file hash."""
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("benchmark dataset must be valid UTF-8") from error

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid benchmark JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise TypeError(f"benchmark case at line {line_number} must be an object")

        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"benchmark case at line {line_number} needs a non-empty id"
            )
        if case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)

        messages = record.get("messages")
        if not isinstance(messages, list):
            raise TypeError(f"benchmark case {case_id} messages must be a list")
        format_chat_prompt(messages)

        reference = record.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"benchmark case {case_id} needs a non-empty reference")

        normalized_terms: dict[str, list[str]] = {}
        for field in ("must_include", "must_not_include"):
            terms = record.get(field, [])
            if not isinstance(terms, list) or any(
                not isinstance(term, str) or not term.strip() for term in terms
            ):
                raise ValueError(f"benchmark case {case_id} {field} must be strings")
            normalized_terms[field] = terms

        cases.append(
            {
                "id": case_id,
                "messages": messages,
                "reference": reference,
                "must_include": normalized_terms["must_include"],
                "must_not_include": normalized_terms["must_not_include"],
            }
        )

    if not cases:
        raise ValueError("benchmark dataset contains no cases")
    return cases, hashlib.sha256(raw).hexdigest()


def _score_case(
    prediction: str,
    reference: str,
    must_include: list[str],
    must_not_include: list[str],
) -> dict[str, Any]:
    normalized_prediction = _normalize_text(prediction)
    normalized_reference = _normalize_text(reference)
    required = {term: _contains_phrase(prediction, term) for term in must_include}
    forbidden = [
        term for term in must_not_include if _contains_phrase(prediction, term)
    ]
    return {
        "normalized_exact_match": normalized_prediction == normalized_reference,
        "token_f1": _token_f1(prediction, reference),
        "required_terms": required,
        "forbidden_terms_found": forbidden,
    }


def evaluate_chat_benchmark(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = 1,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate and score responses on an independent prompt/reference JSONL set."""
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError("temperature must be positive and finite")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer when supplied")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    checkpoint_path = Path(checkpoint_path)
    dataset_path = Path(dataset_path)
    cases, dataset_sha256 = _read_benchmark_dataset(dataset_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(seed)
        model = load_model_checkpoint(checkpoint_path, device=device)
        if top_k is not None and top_k > model.config.vocab_size:
            raise ValueError("top_k cannot exceed the model vocabulary size")

        scored_cases: list[dict[str, Any]] = []
        exact_matches = 0
        token_f1_total = 0.0
        required_case_count = 0
        required_case_passes = 0
        forbidden_case_count = 0
        forbidden_case_violations = 0

        for case in cases:
            prediction = generate_chat_response(
                model,
                case["messages"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            scores = _score_case(
                prediction,
                case["reference"],
                case["must_include"],
                case["must_not_include"],
            )
            exact_matches += int(scores["normalized_exact_match"])
            token_f1_total += scores["token_f1"]
            if scores["required_terms"]:
                required_case_count += 1
                required_case_passes += int(all(scores["required_terms"].values()))
            if case["must_not_include"]:
                forbidden_case_count += 1
                forbidden_case_violations += int(bool(scores["forbidden_terms_found"]))
            scored_cases.append(
                {
                    "id": case["id"],
                    "reference": case["reference"],
                    "prediction": prediction,
                    **scores,
                }
            )

        count = len(scored_cases)
        report: dict[str, Any] = {
            "benchmark": "reference-based-chat",
            "split": "external-benchmark",
            "samples": count,
            "metrics": {
                "normalized_exact_match_rate": exact_matches / count,
                "mean_token_f1": token_f1_total / count,
                "required_terms_case_pass_rate": (
                    required_case_passes / required_case_count
                    if required_case_count
                    else None
                ),
                "forbidden_terms_case_violation_rate": (
                    forbidden_case_violations / forbidden_case_count
                    if forbidden_case_count
                    else None
                ),
            },
            "generation": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "seed": seed,
                "device": device,
            },
            "checkpoint_sha256": checkpoint_sha256,
            "benchmark_dataset_sha256": dataset_sha256,
            "cases": scored_cases,
        }
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an Entropy chat checkpoint on reference-based prompts."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path, help="JSONL benchmark cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_chat_benchmark(
        args.checkpoint,
        args.dataset,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
