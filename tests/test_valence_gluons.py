"""Tests for the Gluons retrieval integration in Valence."""

from __future__ import annotations

from collections.abc import Sequence

from proton_i.gluons.search import SearchError, SearchResult
from proton_i.valence.integrations.gluons import (
    create_gluons_tool,
    register_gluons_tool,
)
from proton_i.valence.tools import (
    CapabilityRegistry,
    ExecutionPolicy,
    ToolDefinition,
    ToolExecutor,
    ToolHandlerError,
)


class FakeSearchProvider:
    """Deterministic provider for testing the Gluons tool without the internet."""

    def __init__(self, results: Sequence[SearchResult] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.queries.append((query, max_results))
        return self.results[:max_results]


def sample_results() -> list[SearchResult]:
    """Return two controlled source records for the adapter tests."""
    return [
        SearchResult(
            title="MDN: CSS Flexbox",
            url="https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_flexible_box_layout",
            snippet="Flexbox is a one-dimensional CSS layout model.",
        ),
        SearchResult(
            title="CSS Tricks: A Complete Guide",
            url="https://css-tricks.com/snippets/css/a-guide-to-flexbox/",
            snippet="The main axis and cross axis define item placement.",
        ),
    ]


def test_gluons_tool_returns_bounded_source_attributed_results() -> None:
    provider = FakeSearchProvider(sample_results())
    tool = create_gluons_tool(provider)
    registry = CapabilityRegistry()
    registry.register(tool)
    policy = ExecutionPolicy(frozenset({"search_web"}))

    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "  CSS   Flexbox  "},
        policy,
    )

    assert result.status == "success"
    assert result.data["query"] == "CSS Flexbox"
    assert result.data["result_count"] == 2
    assert result.data["content_trust"] == "untrusted_web_content"
    assert len(result.data["results"]) == 2
    assert result.data["results"][0]["source_url"].startswith("https://")
    assert provider.queries == [("CSS Flexbox", 5)]


def test_gluons_tool_is_not_exposed_without_policy_permission() -> None:
    registry = CapabilityRegistry()
    register_gluons_tool(registry, FakeSearchProvider(sample_results()))

    assert registry.catalog(ExecutionPolicy()) == ()

    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "CSS Flexbox"},
        ExecutionPolicy(),
    )
    assert result.status == "denied"


def test_gluons_tool_preserves_empty_search_result_state() -> None:
    registry = CapabilityRegistry()
    register_gluons_tool(registry, FakeSearchProvider())
    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "no matching pages"},
        ExecutionPolicy(frozenset({"search_web"})),
    )

    assert result.status == "success"
    assert result.data["result_count"] == 0
    assert result.data["results"] == []


def test_gluons_tool_returns_sanitized_provider_failure() -> None:
    class BrokenProvider:
        def search(self, _query: str, max_results: int = 5) -> list[SearchResult]:
            raise SearchError("provider internal details")

    registry = CapabilityRegistry()
    register_gluons_tool(registry, BrokenProvider())
    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "HTML"},
        ExecutionPolicy(frozenset({"search_web"})),
    )

    assert result.status == "failed"
    assert result.error == "Website search is unavailable right now."
    assert "provider internal details" not in result.error


def test_gluons_tool_rejects_blank_query_without_searching() -> None:
    provider = FakeSearchProvider(sample_results())
    registry = CapabilityRegistry()
    register_gluons_tool(registry, provider)

    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "   "},
        ExecutionPolicy(frozenset({"search_web"})),
    )

    assert result.status == "failed"
    assert result.error == "Search query must not be empty."
    assert provider.queries == []


def test_gluons_tool_rejects_oversized_query_before_provider_call() -> None:
    provider = FakeSearchProvider(sample_results())
    registry = CapabilityRegistry()
    register_gluons_tool(registry, provider)

    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "x" * 501},
        ExecutionPolicy(frozenset({"search_web"})),
    )

    assert result.status == "invalid"
    assert provider.queries == []


def test_gluons_tool_filters_non_http_and_oversized_source_urls() -> None:
    provider = FakeSearchProvider(
        [
            SearchResult("Local file", "file:///etc/passwd", "Not a web source."),
            SearchResult(
                "Credentials",
                "https://user:pass@example.com/article",
                "A source with embedded credentials.",
            ),
            SearchResult(
                "Invalid port",
                "https://example.com:99999/article",
                "A source with an invalid port.",
            ),
            SearchResult("Long URL", "https://example.com/" + "x" * 2100, "Too long."),
            SearchResult(
                "Valid",
                "https://example.com/article",
                "A valid source snippet.",
            ),
        ]
    )
    registry = CapabilityRegistry()
    register_gluons_tool(registry, provider)

    result = ToolExecutor(registry).execute(
        "search_web",
        {"query": "example"},
        ExecutionPolicy(frozenset({"search_web"})),
    )

    assert result.status == "success"
    assert result.data["result_count"] == 1
    assert result.data["results"][0]["source_url"] == "https://example.com/article"


def test_tool_handler_error_reaches_the_orchestrator_as_a_safe_message() -> None:
    def fail_safely(_arguments: dict[str, object]) -> object:
        raise ToolHandlerError("A safe message.")

    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="safe_error",
            description="Returns a safe tool failure.",
            parameters=(),
            handler=fail_safely,
        )
    )
    result = ToolExecutor(registry).execute(
        "safe_error",
        {},
        ExecutionPolicy(frozenset({"safe_error"})),
    )

    assert result.status == "failed"
    assert result.error == "A safe message."
