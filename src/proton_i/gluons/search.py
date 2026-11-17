"""Small dependency-free web search adapter used by Gluons.

The adapter reads DuckDuckGo's HTML search page. This is a lightweight
best-effort integration rather than a guaranteed stable search API; callers
must handle timeouts, HTTP failures, and changes to the result-page markup.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen


_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
_USER_AGENT = "Proton-I-Gluons/0.1 (+https://github.com/EverlastingUc/R)"
_MAX_RESPONSE_BYTES = 2_000_000
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class SearchError(RuntimeError):
    """Raised when the remote search page cannot be fetched or parsed."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One web result returned by the search provider."""

    title: str
    url: str
    snippet: str


class _DuckDuckGoHTMLParser(HTMLParser):
    """Extract result titles, destination links, and snippets from DDG HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._current: dict[str, object] | None = None
        self._element_stack: list[str] = []
        self._capture_title = False
        self._capture_snippet = False
        self._snippet_end_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if self._current is None:
            if tag == "div" and "result" in classes:
                self._current = {"title": [], "snippet": [], "href": ""}
                self._element_stack = [tag]
            return

        if tag not in _VOID_TAGS:
            self._element_stack.append(tag)
        if tag == "a" and "result__a" in classes:
            self._current["href"] = attributes.get("href") or ""
            self._capture_title = True
        if "result__snippet" in classes:
            self._capture_snippet = True
            self._snippet_end_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return

        if tag == "a" and self._capture_title:
            self._capture_title = False
        if self._capture_snippet and tag == self._snippet_end_tag:
            self._capture_snippet = False
            self._snippet_end_tag = None

        # Remove the matching element and any malformed descendants after it.
        if tag in self._element_stack:
            reverse_index = self._element_stack[::-1].index(tag)
            index = len(self._element_stack) - reverse_index - 1
            del self._element_stack[index:]

        if not self._element_stack:
            self._finish_result()

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._capture_title:
            title = self._current["title"]
            assert isinstance(title, list)
            title.append(data)
        elif self._capture_snippet:
            snippet = self._current["snippet"]
            assert isinstance(snippet, list)
            snippet.append(data)

    def _finish_result(self) -> None:
        if self._current is None:
            return
        title_parts = self._current["title"]
        snippet_parts = self._current["snippet"]
        href = self._current["href"]
        assert isinstance(title_parts, list)
        assert isinstance(snippet_parts, list)
        assert isinstance(href, str)

        title = " ".join(" ".join(title_parts).split())
        snippet = " ".join(" ".join(snippet_parts).split())
        url = _destination_url(href)
        if title and url:
            self.results.append(SearchResult(title=title, url=url, snippet=snippet))
        self._current = None
        self._element_stack = []
        self._capture_title = False
        self._capture_snippet = False
        self._snippet_end_tag = None


def _destination_url(href: str) -> str:
    """Resolve DDG redirect links and reject non-web destinations."""
    if not href:
        return ""
    absolute = urljoin("https://duckduckgo.com/", href)
    parsed = urlsplit(absolute)
    if parsed.scheme not in {"http", "https"}:
        return ""

    if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"}:
        redirected = parse_qs(parsed.query).get("uddg", [""])[0]
        if redirected:
            target = urlsplit(redirected)
            if target.scheme in {"http", "https"} and target.netloc:
                return redirected
            return ""
    return absolute


class DuckDuckGoSearch:
    """Search the public web through DuckDuckGo's HTML results page."""

    def __init__(self, timeout: float = 8.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)

    @staticmethod
    def search_url(query: str) -> str:
        """Return a browser-search URL for the supplied query."""
        return _SEARCH_ENDPOINT + "?" + urlencode({"q": query})

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Fetch and parse up to ``max_results`` search results."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if len(cleaned_query) > 500:
            raise ValueError("query must be 500 characters or fewer")
        if not isinstance(max_results, int) or not 1 <= max_results <= 10:
            raise ValueError("max_results must be an integer between 1 and 10")

        request = Request(
            self.search_url(cleaned_query),
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError, URLError, ValueError) as error:
            raise SearchError("web search request failed") from error

        if len(body) > _MAX_RESPONSE_BYTES:
            raise SearchError("search response exceeded the 2 MB safety limit")

        parser = _DuckDuckGoHTMLParser()
        try:
            parser.feed(body.decode("utf-8", errors="replace"))
            parser.close()
        except (AssertionError, ValueError) as error:
            raise SearchError("could not parse the search response") from error

        return parser.results[:max_results]
