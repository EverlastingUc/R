"""Gluons: Proton-I's first, deliberately small web-assisted assistant."""

from proton_i.gluons.assistant import GluonsAssistant
from proton_i.gluons.search import DuckDuckGoSearch, SearchError, SearchResult

__all__ = ["DuckDuckGoSearch", "GluonsAssistant", "SearchError", "SearchResult"]
