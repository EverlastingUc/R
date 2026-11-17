"""Tests for Valence's default mathematics and optional search policy."""

from __future__ import annotations

import pytest

from proton_i.valence.runtime import build_tool_environment


def test_basic_calculator_is_enabled_by_default() -> None:
    registry, policy = build_tool_environment()

    assert [tool["name"] for tool in registry.catalog(policy)] == [
        "calculate_expression"
    ]
    assert policy.allowed_tools == frozenset({"calculate_expression"})


def test_web_search_adds_gluons_without_removing_calculator() -> None:
    registry, policy = build_tool_environment(enable_web_search=True)

    assert [tool["name"] for tool in registry.catalog(policy)] == [
        "calculate_expression",
        "search_web",
    ]
    assert policy.allowed_tools == frozenset({"calculate_expression", "search_web"})


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_runtime_rejects_non_boolean_web_search_flag(value: object) -> None:
    with pytest.raises(TypeError, match="boolean"):
        build_tool_environment(value)  # type: ignore[arg-type]
