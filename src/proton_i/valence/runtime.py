"""Application-owned default capabilities for Valence."""

from __future__ import annotations

from proton_i.valence.integrations.gluons import WebSearchProvider, register_gluons_tool
from proton_i.valence.integrations.mathematics import register_mathematics_tool
from proton_i.valence.tools import CapabilityRegistry, ExecutionPolicy


def build_tool_environment(
    enable_web_search: bool = False,
    searcher: WebSearchProvider | None = None,
) -> tuple[CapabilityRegistry, ExecutionPolicy]:
    """Build the default calculator environment and optional Gluons search.

    Deterministic basic arithmetic is enabled by default. Website retrieval is
    external I/O and is only registered/allowed when explicitly requested.
    """
    if not isinstance(enable_web_search, bool):
        raise TypeError("enable_web_search must be a boolean")

    registry = CapabilityRegistry()
    register_mathematics_tool(registry)
    allowed_tools = {"calculate_expression"}

    if enable_web_search:
        register_gluons_tool(registry, searcher=searcher)
        allowed_tools.add("search_web")

    return registry, ExecutionPolicy(frozenset(allowed_tools))
