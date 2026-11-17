"""Deterministic arithmetic adapter exposed through Valence's tool registry."""

from __future__ import annotations

from proton_i.valence.mathematics import MathematicsEngine, MathematicsError
from proton_i.valence.tools import (
    CapabilityRegistry,
    ToolDefinition,
    ToolHandlerError,
    ToolParameter,
)


def create_mathematics_tool(
    mathematics: MathematicsEngine | None = None,
) -> ToolDefinition:
    """Create a calculator tool backed by Valence's safe exact arithmetic engine."""
    engine = mathematics or MathematicsEngine()
    if not isinstance(engine, MathematicsEngine):
        raise TypeError("mathematics must be a MathematicsEngine or None")

    def calculate_expression(arguments: dict[str, object]) -> object:
        """Evaluate one bounded expression and return exact and decimal forms."""
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise ToolHandlerError("Mathematical expression must be text.")
        try:
            result = engine.evaluate(expression)
        except MathematicsError as error:
            raise ToolHandlerError("Mathematical calculation failed.") from error
        return {
            "expression": result.expression,
            "exact_result": result.exact_result,
            "decimal_approximation": result.decimal_approximation,
            "answer": result.answer,
        }

    return ToolDefinition(
        name="calculate_expression",
        description=(
            "Evaluate a mathematical expression using exact rational values where "
            "possible. Supports +, -, *, /, %, bounded integer powers, unary signs, "
            "decimals, parentheses, and the abs, sqrt, gcd, lcm, ratio, and percent "
            "functions. Irrational square roots are explicitly approximate."
        ),
        parameters=(
            ToolParameter(
                name="expression",
                value_type="string",
                description="A basic arithmetic expression.",
                required=True,
                max_length=512,
            ),
        ),
        handler=calculate_expression,
        requires_approval=False,
        enabled=True,
    )


def register_mathematics_tool(
    registry: CapabilityRegistry,
    mathematics: MathematicsEngine | None = None,
) -> ToolDefinition:
    """Register exact arithmetic without granting execution-policy permission."""
    if not isinstance(registry, CapabilityRegistry):
        raise TypeError("registry must be a CapabilityRegistry")
    definition = create_mathematics_tool(mathematics)
    registry.register(definition)
    return definition
