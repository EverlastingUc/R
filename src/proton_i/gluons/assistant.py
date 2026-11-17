"""Plain-text answer orchestration for Proton-I's Gluons stage."""

from __future__ import annotations

import re
from collections.abc import Sequence

from proton_i.gluons.search import DuckDuckGoSearch, SearchError, SearchResult

_UNSUPPORTED_TERMS = re.compile(
    r"\b(javascript|node\.?js|typescript|react|vue|angular)\b", re.IGNORECASE
)
_MARKDOWN_MARKERS = re.compile(r"[`*_#>{}\[\]]")
_MAX_ANSWER_CHARS = 520


class GluonsAssistant:
    """A short-answer web assistant with an intentionally narrow scope.

    Gluons does not contain a trained language model in this first baseline.
    It retrieves a web result, uses the result snippet as a concise answer,
    and includes a plain-text source URL. A later Proton-I stage can replace
    this baseline with a trained model while preserving the public interface.
    """

    def __init__(self, search: DuckDuckGoSearch | None = None) -> None:
        self.search = search or DuckDuckGoSearch()

    def answer(self, prompt: str) -> str:
        """Return one short plain-text paragraph for a user prompt."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        query = " ".join(prompt.split())
        if not query:
            raise ValueError("prompt must not be empty")
        if len(query) > 500:
            raise ValueError("prompt must be 500 characters or fewer")

        if _UNSUPPORTED_TERMS.search(query):
            return (
                "Gluons currently supports basic questions and web search, but "
                "JavaScript and other programming-language requests are outside "
                "this stage."
            )

        try:
            results = self.search.search(query, max_results=5)
        except SearchError:
            return "Website search is unavailable right now. Please try again later."

        if not results:
            return (
                "I could not find a useful result. You can try this search: "
                f"{self.search.search_url(query)}"
            )

        summary = _short_summary(query, results)
        source = next((item.url for item in results if item.url), "")
        if not summary:
            summary = (
                "I found a relevant website, but it did not provide a readable summary."
            )
        if source:
            summary = f"{summary} Source: {source}"
        return summary


def _short_summary(query: str, results: Sequence[SearchResult]) -> str:
    """Choose a concise relevant sentence from returned search snippets."""
    query_terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9+#.-]+", query)
        if len(term) > 1
    }
    candidates: list[tuple[int, int, str]] = []
    order = 0

    for result in results:
        text = _plain_text(result.snippet)
        if not text:
            text = _plain_text(result.title)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = {
                term.lower()
                for term in re.findall(r"[a-zA-Z0-9+#.-]+", sentence)
                if len(term) > 1
            }
            score = len(query_terms & words)
            candidates.append((score, -order, sentence))
            order += 1

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    selected = candidates[0][2]
    if len(selected) > _MAX_ANSWER_CHARS:
        selected = selected[: _MAX_ANSWER_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return selected


def _plain_text(value: str) -> str:
    """Remove markup-like artifacts and normalize whitespace."""
    text = re.sub(r"<[^>]*>", " ", value)
    text = _MARKDOWN_MARKERS.sub("", text)
    return " ".join(text.split()).strip()
