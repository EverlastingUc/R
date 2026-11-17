"""Tests for Valence's typed capability registry and guarded tool executor."""

from __future__ import annotations

import pytest

from proton_i.valence.tools import (
    CapabilityRegistry,
    ExecutionPolicy,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
)


def add_numbers(arguments: dict[str, object]) -> object:
    """Small deterministic handler used only by the registry tests."""
    return {"sum": arguments["left"] + arguments["right"]}  # type: ignore[operator]


def build_add_tool() -> ToolDefinition:
    """Return an application-owned numeric addition tool."""
    return ToolDefinition(
        name="add_numbers",
        description="Add two finite numbers and return their sum.",
        parameters=(
            ToolParameter(
                "left",
                "number",
                "First addend.",
                minimum=-1_000_000,
                maximum=1_000_000,
            ),
            ToolParameter(
                "right",
                "number",
                "Second addend.",
                minimum=-1_000_000,
                maximum=1_000_000,
            ),
        ),
        handler=add_numbers,
    )


def test_registry_starts_empty_and_registers_tool() -> None:
    registry = CapabilityRegistry()
    assert len(registry) == 0

    registry.register(build_add_tool())

    assert len(registry) == 1
    assert registry.get("add_numbers") is not None


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = CapabilityRegistry()
    registry.register(build_add_tool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(build_add_tool())


def test_registry_rejects_non_tool_definitions() -> None:
    registry = CapabilityRegistry()
    with pytest.raises(TypeError, match="ToolDefinition"):
        registry.register("add_numbers")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["AddNumbers", "tool name", "1x", "../shell"])
def test_tool_definition_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="lowercase identifiers"):
        ToolDefinition(
            name=name,
            description="A test tool.",
            parameters=(),
            handler=lambda _arguments: None,
        )


def test_tool_definition_rejects_duplicate_parameter_names() -> None:
    parameter = ToolParameter("value", "string", "A value.")
    with pytest.raises(ValueError, match="unique"):
        ToolDefinition(
            name="duplicate_args",
            description="Invalid duplicate parameters.",
            parameters=(parameter, parameter),
            handler=lambda _arguments: None,
        )


def test_tool_parameter_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="minimum cannot exceed"):
        ToolParameter(
            "count",
            "integer",
            "A count.",
            minimum=10,
            maximum=1,
        )
    with pytest.raises(ValueError, match="numeric parameter"):
        ToolParameter(
            "name",
            "string",
            "A name.",
            minimum=1,
        )


def test_default_policy_denies_all_tools() -> None:
    registry = CapabilityRegistry()
    registry.register(build_add_tool())
    executor = ToolExecutor(registry)

    result = executor.execute("add_numbers", {"left": 2, "right": 3}, ExecutionPolicy())

    assert result.status == "denied"
    assert "not allowed" in result.error
    assert result.data is None


def test_catalog_only_discloses_tools_allowed_by_policy() -> None:
    registry = CapabilityRegistry()
    registry.register(build_add_tool())
    registry.register(
        ToolDefinition(
            name="disabled_tool",
            description="This tool is currently disabled.",
            parameters=(),
            handler=lambda _arguments: "should not run",
            enabled=False,
        )
    )

    catalog = registry.catalog(
        ExecutionPolicy(frozenset({"add_numbers", "disabled_tool"}))
    )

    assert len(catalog) == 1
    assert catalog[0]["name"] == "add_numbers"
    schema = catalog[0]["parameters"]
    assert schema["additionalProperties"] is False
    assert "handler" not in catalog[0]


def test_allowed_tool_executes_with_validated_arguments() -> None:
    registry = CapabilityRegistry()
    registry.register(build_add_tool())
    executor = ToolExecutor(registry)
    policy = ExecutionPolicy(frozenset({"add_numbers"}))

    result = executor.execute("add_numbers", {"left": 2, "right": 3}, policy)

    assert result.status == "success"
    assert result.data == {"sum": 5}
    assert result.as_payload() == {
        "tool_name": "add_numbers",
        "status": "success",
        "data": {"sum": 5},
    }


def test_executor_rejects_missing_and_unexpected_arguments() -> None:
    registry = CapabilityRegistry()
    registry.register(build_add_tool())
    executor = ToolExecutor(registry)
    policy = ExecutionPolicy(frozenset({"add_numbers"}))

    missing = executor.execute("add_numbers", {"left": 2}, policy)
    extra = executor.execute(
        "add_numbers", {"left": 2, "right": 3, "command": "rm -rf"}, policy
    )

    assert missing.status == "invalid"
    assert "missing" in missing.error
    assert extra.status == "invalid"
    assert "Unexpected" in extra.error


def test_executor_rejects_boolean_where_integer_is_expected() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="repeat_count",
            description="Return a count.",
            parameters=(ToolParameter("count", "integer", "A whole number."),),
            handler=lambda arguments: {"count": arguments["count"]},
        )
    )
    result = ToolExecutor(registry).execute(
        "repeat_count",
        {"count": True},
        ExecutionPolicy(frozenset({"repeat_count"})),
    )

    assert result.status == "invalid"
    assert "wrong type" in result.error


def test_executor_enforces_numeric_ranges_enums_and_string_lengths() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="set_profile",
            description="Set one profile field.",
            parameters=(
                ToolParameter(
                    "display_name",
                    "string",
                    "Short display name.",
                    max_length=8,
                ),
                ToolParameter(
                    "theme",
                    "string",
                    "Allowed theme.",
                    enum=("light", "dark"),
                ),
                ToolParameter(
                    "age",
                    "integer",
                    "Allowed age.",
                    minimum=1,
                    maximum=120,
                ),
            ),
            handler=lambda arguments: arguments,
        )
    )
    executor = ToolExecutor(registry)
    policy = ExecutionPolicy(frozenset({"set_profile"}))

    too_long = executor.execute(
        "set_profile",
        {"display_name": "name-too-long", "theme": "light", "age": 20},
        policy,
    )
    invalid_enum = executor.execute(
        "set_profile",
        {"display_name": "Ever", "theme": "neon", "age": 20},
        policy,
    )
    invalid_range = executor.execute(
        "set_profile",
        {"display_name": "Ever", "theme": "dark", "age": 121},
        policy,
    )

    assert too_long.status == "invalid"
    assert invalid_enum.status == "invalid"
    assert invalid_range.status == "invalid"


def test_sensitive_tool_requires_explicit_approval() -> None:
    calls: list[bool] = []

    def sensitive_handler(_arguments: dict[str, object]) -> object:
        calls.append(True)
        return {"changed": True}

    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="change_setting",
            description="Change a local setting.",
            parameters=(),
            handler=sensitive_handler,
            requires_approval=True,
        )
    )
    executor = ToolExecutor(registry)
    policy = ExecutionPolicy(frozenset({"change_setting"}))

    blocked = executor.execute("change_setting", {}, policy)
    allowed = executor.execute("change_setting", {}, policy, approved=True)

    assert blocked.status == "denied"
    assert calls == [True]
    assert allowed.status == "success"


def test_disabled_tool_never_runs_even_when_policy_allows_it() -> None:
    calls: list[bool] = []
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="disabled_tool",
            description="Unavailable.",
            parameters=(),
            handler=lambda _arguments: calls.append(True),
            enabled=False,
        )
    )

    result = ToolExecutor(registry).execute(
        "disabled_tool",
        {},
        ExecutionPolicy(frozenset({"disabled_tool"})),
    )

    assert result.status == "denied"
    assert calls == []


def test_unknown_tool_and_invalid_name_return_structured_results() -> None:
    executor = ToolExecutor(CapabilityRegistry())
    policy = ExecutionPolicy(frozenset())

    unknown = executor.execute("missing_tool", {}, policy)
    invalid = executor.execute("NOT VALID", {}, policy)

    assert unknown.status == "failed"
    assert invalid.status == "invalid"
    assert "invalid" in invalid.error


def test_handler_exceptions_are_redacted_from_tool_result() -> None:
    def broken_handler(_arguments: dict[str, object]) -> object:
        raise RuntimeError("private file path or secret must not be returned")

    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="broken_tool",
            description="A tool that fails.",
            parameters=(),
            handler=broken_handler,
        )
    )

    result = ToolExecutor(registry).execute(
        "broken_tool", {}, ExecutionPolicy(frozenset({"broken_tool"}))
    )

    assert result.status == "failed"
    assert result.error == "Tool execution failed."
    assert "private file path" not in result.error


def test_executor_rejects_non_json_and_oversized_tool_outputs() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="bad_output",
            description="Returns an unsupported Python object.",
            parameters=(),
            handler=lambda _arguments: object(),
        )
    )
    registry.register(
        ToolDefinition(
            name="large_output",
            description="Returns a long string.",
            parameters=(),
            handler=lambda _arguments: "x" * 200,
        )
    )
    executor = ToolExecutor(registry)

    non_json = executor.execute(
        "bad_output",
        {},
        ExecutionPolicy(frozenset({"bad_output"})),
    )
    oversized = executor.execute(
        "large_output",
        {},
        ExecutionPolicy(frozenset({"large_output"}), max_output_bytes=32),
    )

    assert non_json.status == "failed"
    assert "cannot be encoded" in non_json.error
    assert oversized.status == "failed"
    assert "size limit" in oversized.error


def test_executor_enforces_argument_payload_size() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="accept_text",
            description="Accept a text input.",
            parameters=(ToolParameter("text", "string", "Input text."),),
            handler=lambda arguments: {"length": len(arguments["text"])},
        )
    )
    result = ToolExecutor(registry).execute(
        "accept_text",
        {"text": "x" * 200},
        ExecutionPolicy(frozenset({"accept_text"}), max_argument_bytes=64),
    )

    assert result.status == "invalid"
    assert "size limit" in result.error


def test_tool_parameter_schema_contains_constraints() -> None:
    parameter = ToolParameter(
        "query",
        "string",
        "A short search query.",
        required=True,
        enum=("html", "css"),
        max_length=20,
    )

    schema = parameter.as_schema()

    assert schema == {
        "type": "string",
        "description": "A short search query.",
        "enum": ["html", "css"],
        "maxLength": 20,
    }


def test_policy_rejects_invalid_limits_and_tool_names() -> None:
    with pytest.raises(ValueError, match="allowed_tools"):
        ExecutionPolicy(frozenset({"Bad Name"}))
    with pytest.raises(ValueError, match="max_output_bytes"):
        ExecutionPolicy(frozenset(), max_output_bytes=0)


def test_policy_rejects_a_string_instead_of_a_tool_name_collection() -> None:
    with pytest.raises(TypeError, match="iterable of tool names"):
        ExecutionPolicy("add_numbers")  # type: ignore[arg-type]


def test_executor_rejects_non_string_argument_keys() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="accept_object",
            description="Accept an object with named fields.",
            parameters=(ToolParameter("value", "object", "An object."),),
            handler=lambda arguments: arguments,
        )
    )
    result = ToolExecutor(registry).execute(
        "accept_object",
        {1: "not a JSON property name"},  # type: ignore[dict-item]
        ExecutionPolicy(frozenset({"accept_object"})),
    )

    assert result.status == "invalid"
    assert "names must be strings" in result.error


def test_executor_rejects_extremely_large_integer_as_number() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="accept_number",
            description="Accept a finite number.",
            parameters=(ToolParameter("value", "number", "A finite number."),),
            handler=lambda arguments: arguments,
        )
    )
    result = ToolExecutor(registry).execute(
        "accept_number",
        {"value": 10**1000},
        ExecutionPolicy(frozenset({"accept_number"})),
    )

    assert result.status == "invalid"
    assert "wrong type" in result.error


def test_executor_rejects_circular_argument_objects() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ToolDefinition(
            name="accept_text",
            description="Accept a text input.",
            parameters=(ToolParameter("text", "string", "Input text."),),
            handler=lambda arguments: arguments,
        )
    )
    circular: dict[str, object] = {}
    circular["self"] = circular

    result = ToolExecutor(registry).execute(
        "accept_text",
        circular,
        ExecutionPolicy(frozenset({"accept_text"})),
    )

    assert result.status == "invalid"
    assert "valid JSON" in result.error


def test_tool_parameter_rejects_extremely_large_numeric_bounds() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        ToolParameter(
            "value",
            "number",
            "A bounded value.",
            minimum=10**1000,
        )
