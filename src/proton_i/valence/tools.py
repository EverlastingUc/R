"""Typed, allowlisted tool contracts for Valence.

Only application-registered Python handlers can run. Model output is treated
as untrusted input: it never becomes Python code, shell commands, or imports.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

ParameterType = Literal["string", "integer", "number", "boolean", "object", "array"]
ToolStatus = Literal["success", "denied", "invalid", "failed"]
ToolHandler = Callable[[dict[str, object]], object]

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_POLICY_BYTES = 1_048_576


def _is_tool_name(value: object) -> bool:
    """Return whether a tool identifier follows the stable lowercase format."""
    return isinstance(value, str) and _TOOL_NAME_PATTERN.fullmatch(value) is not None


def _json_bytes(value: object) -> bytes:
    """Serialize a value as strict UTF-8 JSON, rejecting NaN and custom objects."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """One top-level JSON argument accepted by a registered tool."""

    name: str
    value_type: ParameterType
    description: str
    required: bool = True
    enum: tuple[object, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid or contradictory parameter definitions."""
        if not _is_tool_name(self.name):
            raise ValueError("parameter names must be lowercase identifiers")
        if self.value_type not in {
            "string", "integer", "number", "boolean", "object", "array"
        }:
            raise ValueError("parameter value_type is not supported")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("parameter description must not be empty")
        if len(self.description) > 500:
            raise ValueError("parameter description must be 500 characters or fewer")
        if not isinstance(self.required, bool):
            raise TypeError("parameter required must be a boolean")

        if self.enum is not None:
            if not isinstance(self.enum, tuple) or not self.enum:
                raise ValueError("enum must be a non-empty tuple")
            try:
                for item in self.enum:
                    _json_bytes(item)
            except (TypeError, ValueError) as error:
                raise ValueError("enum values must be JSON-compatible") from error

        if self.minimum is not None or self.maximum is not None:
            if self.value_type not in {"integer", "number"}:
                raise ValueError("minimum and maximum require a numeric parameter")
            for bound in (self.minimum, self.maximum):
                if bound is not None and (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(bound)
                ):
                    raise ValueError("numeric bounds must be finite numbers")
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ValueError("minimum cannot exceed maximum")

        if self.max_length is not None and (
            not isinstance(self.max_length, int)
            or isinstance(self.max_length, bool)
            or self.max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")
        if self.max_length is not None and self.value_type not in {
            "string", "array", "object"
        }:
            raise ValueError("max_length requires a string, array, or object parameter")

    def as_schema(self) -> dict[str, object]:
        """Return the JSON-Schema-like shape exposed to a model or UI."""
        schema: dict[str, object] = {
            "type": self.value_type,
            "description": self.description.strip(),
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.max_length is not None:
            size_key = {
                "string": "maxLength",
                "array": "maxItems",
                "object": "maxProperties",
            }[self.value_type]
            schema[size_key] = self.max_length
        return schema


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A trusted application-owned handler and its validated argument contract."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    handler: ToolHandler
    requires_approval: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the definition before it becomes visible in a registry."""
        if not _is_tool_name(self.name):
            raise ValueError("tool names must be lowercase identifiers")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must not be empty")
        if len(self.description) > 1_000:
            raise ValueError("tool description must be 1,000 characters or fewer")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple of ToolParameter objects")
        if any(not isinstance(item, ToolParameter) for item in self.parameters):
            raise TypeError("parameters must contain ToolParameter objects")
        parameter_names = [item.name for item in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("tool parameter names must be unique")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        if not isinstance(self.requires_approval, bool):
            raise TypeError("requires_approval must be a boolean")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")

    def as_schema(self) -> dict[str, object]:
        """Return a provider-neutral function schema without exposing its handler."""
        properties = {item.name: item.as_schema() for item in self.parameters}
        required = [item.name for item in self.parameters if item.required]
        return {
            "name": self.name,
            "description": self.description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "requires_approval": self.requires_approval,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Per-request allowlist and resource limits; default policy denies all tools."""

    allowed_tools: frozenset[str] = frozenset()
    max_argument_bytes: int = 16_384
    max_output_bytes: int = 32_768

    def __post_init__(self) -> None:
        """Validate the allowlist and keep payload limits within a hard ceiling."""
        try:
            names = frozenset(self.allowed_tools)
        except TypeError as error:
            raise TypeError("allowed_tools must be an iterable of tool names") from error
        if any(not _is_tool_name(name) for name in names):
            raise ValueError("allowed_tools contains an invalid tool name")
        object.__setattr__(self, "allowed_tools", names)

        for name, value in (
            ("max_argument_bytes", self.max_argument_bytes),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= MAX_POLICY_BYTES
            ):
                raise ValueError(f"{name} must be between 1 byte and 1 MB")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured outcome returned to the conversation orchestrator."""

    tool_name: str
    status: ToolStatus
    data: object | None = None
    error: str | None = None

    def as_payload(self) -> dict[str, object]:
        """Return the JSON response shape without stack traces or handler details."""
        payload: dict[str, object] = {
            "tool_name": self.tool_name,
            "status": self.status,
        }
        if self.status == "success":
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error
        return payload


class CapabilityRegistry:
    """Thread-safe registry of application-owned tools and their input schemas."""

    def __init__(self) -> None:
        """Create an empty registry; no tools are enabled by default."""
        from threading import RLock

        self._lock = RLock()
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """Register a trusted tool definition without replacing existing names."""
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        with self._lock:
            if definition.name in self._tools:
                raise ValueError(f"tool is already registered: {definition.name}")
            self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        """Return a registered definition, or None when the name is unknown."""
        with self._lock:
            return self._tools.get(name)

    def catalog(self, policy: ExecutionPolicy) -> tuple[dict[str, object], ...]:
        """Expose only enabled schemas that the current policy explicitly allows."""
        if not isinstance(policy, ExecutionPolicy):
            raise TypeError("policy must be an ExecutionPolicy")
        with self._lock:
            visible = [
                tool.as_schema()
                for name, tool in sorted(self._tools.items())
                if tool.enabled and name in policy.allowed_tools
            ]
        return tuple(visible)

    def __len__(self) -> int:
        """Return the number of registered tool definitions."""
        with self._lock:
            return len(self._tools)


class ToolExecutor:
    """Validate and invoke explicitly allowed application-owned tool handlers.

    This executor does not evaluate model-generated code. Handlers are trusted
    functions supplied by the application, and each handler must impose its
    own native timeout on any network or other blocking operation it performs.
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        """Create an executor bound to one capability registry."""
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be a CapabilityRegistry")
        self.registry = registry

    def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
        policy: ExecutionPolicy,
        approved: bool = False,
    ) -> ToolResult:
        """Execute one tool only after allowlist, approval, schema, and size checks."""
        if not isinstance(policy, ExecutionPolicy):
            raise TypeError("policy must be an ExecutionPolicy")
        if not isinstance(approved, bool):
            raise TypeError("approved must be a boolean")
        if not isinstance(name, str) or not _is_tool_name(name):
            return ToolResult(str(name)[:64], "invalid", error="Tool name is invalid.")

        definition = self.registry.get(name)
        if definition is None:
            return ToolResult(name, "failed", error="Tool is not registered.")
        if not definition.enabled:
            return ToolResult(name, "denied", error="Tool is disabled.")
        if name not in policy.allowed_tools:
            return ToolResult(name, "denied", error="Tool is not allowed by this policy.")
        if definition.requires_approval and not approved:
            return ToolResult(
                name,
                "denied",
                error="This tool requires explicit application-side user approval.",
            )
        if not isinstance(arguments, Mapping):
            return ToolResult(name, "invalid", error="Tool arguments must be a JSON object.")

        try:
            encoded_arguments = _json_bytes(dict(arguments))
        except (TypeError, ValueError):
            return ToolResult(name, "invalid", error="Tool arguments must be valid JSON.")
        if len(encoded_arguments) > policy.max_argument_bytes:
            return ToolResult(name, "invalid", error="Tool arguments exceed the size limit.")

        normalized_arguments = json.loads(encoded_arguments.decode("utf-8"))
        validation_error = _validate_arguments(definition, normalized_arguments)
        if validation_error is not None:
            return ToolResult(name, "invalid", error=validation_error)

        try:
            output = definition.handler(normalized_arguments)
        except Exception:
            return ToolResult(name, "failed", error="Tool execution failed.")

        try:
            encoded_output = _json_bytes(output)
        except (TypeError, ValueError):
            return ToolResult(
                name,
                "failed",
                error="Tool returned data that cannot be encoded as JSON.",
            )
        if len(encoded_output) > policy.max_output_bytes:
            return ToolResult(name, "failed", error="Tool output exceeds the size limit.")

        return ToolResult(
            name,
            "success",
            data=json.loads(encoded_output.decode("utf-8")),
        )


def _validate_arguments(
    definition: ToolDefinition,
    arguments: dict[str, object],
) -> str | None:
    """Validate top-level names, required parameters, types, bounds, and enums."""
    schemas = {item.name: item for item in definition.parameters}
    unexpected = set(arguments) - set(schemas)
    if unexpected:
        return f"Unexpected tool argument: {sorted(unexpected)[0]}."

    for parameter in definition.parameters:
        if parameter.name not in arguments:
            if parameter.required:
                return f"Required tool argument is missing: {parameter.name}."
            continue

        value = arguments[parameter.name]
        kind = parameter.value_type
        valid_type = {
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
            ),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
        }[kind]
        if not valid_type(value):
            return f"Tool argument has the wrong type: {parameter.name}."

        if parameter.enum is not None and value not in parameter.enum:
            return f"Tool argument is not an allowed value: {parameter.name}."

        if parameter.minimum is not None and value < parameter.minimum:
            return f"Tool argument is below its minimum: {parameter.name}."
        if parameter.maximum is not None and value > parameter.maximum:
            return f"Tool argument exceeds its maximum: {parameter.name}."

        if parameter.max_length is not None and len(value) > parameter.max_length:
            return f"Tool argument exceeds its length limit: {parameter.name}."

    return None
