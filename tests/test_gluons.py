"""Tests for the first Gluons assistant baseline."""

from __future__ import annotations

from io import BytesIO
from urllib.parse import parse_qs, urlsplit

import pytest

from proton_i.gluons.assistant import GluonsAssistant
from proton_i.gluons.search import (
    DuckDuckGoSearch,
    SearchError,
    SearchResult,
    _DuckDuckGoHTMLParser,
)

SAMPLE_HTML = """
<div class="result results_links">
  <div class="result__title">
    <a class="result__a"
       href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdeveloper.mozilla.org%2Fen-US%2Fdocs%2FWeb%2FCSS%2Fflex">
      CSS Flexible Box Layout
    </a>
  </div>
  <a class="result__snippet">
    Flexbox is a one-dimensional layout method for arranging items in rows or columns.
  </a>
</div>
"""


def test_parser_extracts_title_snippet_and_redirected_url() -> None:
    parser = _DuckDuckGoHTMLParser()
    parser.feed(SAMPLE_HTML)
    parser.close()

    assert parser.results == [
        SearchResult(
            title="CSS Flexible Box Layout",
            url="https://developer.mozilla.org/en-US/docs/Web/CSS/flex",
            snippet=(
                "Flexbox is a one-dimensional layout method for arranging items "
                "in rows or columns."
            ),
        )
    ]


def test_search_encodes_query_and_returns_parsed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse(BytesIO):
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(SAMPLE_HTML.encode("utf-8"))

    monkeypatch.setattr("proton_i.gluons.search.urlopen", fake_urlopen)
    results = DuckDuckGoSearch(timeout=3.5).search("css flexbox", max_results=1)

    assert len(results) == 1
    assert captured["timeout"] == 3.5
    assert "q=css+flexbox" in captured["request"].full_url  # type: ignore[attr-defined]


def test_search_url_encodes_special_characters() -> None:
    query = "HTML forms & labels"
    parsed = urlsplit(DuckDuckGoSearch.search_url(query))
    assert parse_qs(parsed.query)["q"] == [query]


def test_search_validates_query_and_result_count() -> None:
    search = DuckDuckGoSearch()
    with pytest.raises(ValueError):
        search.search("   ")
    with pytest.raises(ValueError):
        search.search("HTML", max_results=11)


def test_assistant_returns_one_plain_text_paragraph_with_source() -> None:
    class FakeSearch:
        @staticmethod
        def search(_query: str, max_results: int = 5) -> list[SearchResult]:
            assert max_results == 5
            return [
                SearchResult(
                    title="CSS Flexbox",
                    url=(
                        "https://developer.mozilla.org/en-US/docs/Web/CSS/"
                        "CSS_flexible_box_layout"
                    ),
                    snippet=(
                        "Flexbox is a one-dimensional CSS layout method. "
                        "It can arrange items in rows or columns."
                    ),
                )
            ]

        @staticmethod
        def search_url(query: str) -> str:
            return DuckDuckGoSearch.search_url(query)

    assistant = GluonsAssistant(search=FakeSearch())  # type: ignore[arg-type]
    answer = assistant.answer("What is CSS Flexbox?")

    assert answer.startswith("Flexbox is a one-dimensional CSS layout method.")
    assert "Source: https://developer.mozilla.org/" in answer
    assert "\n" not in answer
    assert "**" not in answer and "```" not in answer


def test_assistant_rejects_javascript_without_search() -> None:
    class SearchMustNotRun:
        def search(self, *_args: object, **_kwargs: object) -> list[SearchResult]:
            raise AssertionError(
                "search must not run for unsupported JavaScript requests"
            )

    assistant = GluonsAssistant(search=SearchMustNotRun())  # type: ignore[arg-type]
    answer = assistant.answer("Write JavaScript for a button")
    assert "outside this stage" in answer


def test_assistant_handles_empty_search_results() -> None:
    class EmptySearch:
        @staticmethod
        def search(_query: str, max_results: int = 5) -> list[SearchResult]:
            return []

        @staticmethod
        def search_url(query: str) -> str:
            return DuckDuckGoSearch.search_url(query)

    assistant = GluonsAssistant(search=EmptySearch())  # type: ignore[arg-type]
    answer = assistant.answer("CSS grid")
    assert answer.startswith("I could not find a useful result.")
    assert DuckDuckGoSearch.search_url("CSS grid") in answer
    assert "\n" not in answer


def test_assistant_handles_search_failure() -> None:
    class BrokenSearch:
        def search(self, *_args: object, **_kwargs: object) -> list[SearchResult]:
            raise SearchError("offline")

    assistant = GluonsAssistant(search=BrokenSearch())  # type: ignore[arg-type]
    answer = assistant.answer("What is HTML?")
    assert answer == "Website search is unavailable right now. Please try again later."


def test_assistant_validates_prompt() -> None:
    assistant = GluonsAssistant()
    with pytest.raises(ValueError):
        assistant.answer("  ")
    with pytest.raises(ValueError):
        assistant.answer("x" * 501)
