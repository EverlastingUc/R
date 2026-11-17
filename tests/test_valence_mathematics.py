"""Tests for Valence's first deterministic mathematics milestone."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

from proton_i.valence.backend import ChatMessage
from proton_i.valence.engine import ValenceEngine, ValenceError
from proton_i.valence.integrations.mathematics import (
    create_mathematics_tool,
    register_mathematics_tool,
)
from proton_i.valence.expressions import BinaryNode, NumberNode, UnaryNode
from proton_i.valence.mathematics import MathematicsEngine, MathematicsError
from proton_i.valence.tools import CapabilityRegistry, ExecutionPolicy, ToolExecutor


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", Fraction(4)),
        ("10 - 3 * 2", Fraction(4)),
        ("(10 - 3) * 2", Fraction(14)),
        ("8 / 4 / 2", Fraction(1)),
        ("1/3 + 1/6", Fraction(1, 2)),
        ("0.1 + 0.2", Fraction(3, 10)),
        ("-2 * (3 + 4)", Fraction(-14)),
        ("--5", Fraction(5)),
        ("-(2 + 3) / 4", Fraction(-5, 4)),
        (".5 + .25", Fraction(3, 4)),
    ],
)
def test_exact_arithmetic_matches_hand_calculated_results(
    expression: str, expected: Fraction
) -> None:
    result = MathematicsEngine().evaluate(expression)
    assert result.value == expected
    assert result.exact_result == str(expected)


def test_decimal_literals_are_converted_to_exact_rationals() -> None:
    result = MathematicsEngine().evaluate("0.1 + 0.2")
    assert result.value == Fraction(3, 10)
    assert result.exact_result == "3/10"
    assert result.decimal_approximation == "0.3"


def test_answer_formatter_includes_exact_and_decimal_forms() -> None:
    result = MathematicsEngine().evaluate("1 / 3")
    assert result.answer == "1 / 3 = 1/3 (≈ 0.333333333333333)"


def test_integer_answer_does_not_add_redundant_decimal_approximation() -> None:
    result = MathematicsEngine().evaluate("2 + 2")
    assert result.answer == "2 + 2 = 4"


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "1 +",
        "(1 + 2",
        "1 + 2)",
        "2 ** 3",
        "1 // 2",
        "2(3 + 4)",
        "__import__('os').system('whoami')",
        "1 + unknown",
    ],
)
def test_parser_rejects_malformed_or_executable_input(expression: str) -> None:
    with pytest.raises(MathematicsError):
        MathematicsEngine().evaluate(expression)


def test_division_by_zero_is_a_clear_math_error() -> None:
    with pytest.raises(MathematicsError, match="Division by zero"):
        MathematicsEngine().evaluate("10 / (3 - 3)")


def test_parser_enforces_expression_length_limit() -> None:
    with pytest.raises(MathematicsError, match="512 characters"):
        MathematicsEngine().evaluate("1" * 513)


def test_parser_enforces_token_limit() -> None:
    expression = "+".join(["1"] * 65)
    with pytest.raises(MathematicsError, match="128 tokens"):
        MathematicsEngine().evaluate(expression)


def test_parser_enforces_parenthesis_depth_limit() -> None:
    expression = "(" * 33 + "1" + ")" * 33
    with pytest.raises(MathematicsError, match="nesting limit"):
        MathematicsEngine().evaluate(expression)


def test_parser_rejects_excessively_long_numeric_literal() -> None:
    with pytest.raises(MathematicsError, match="128 digits"):
        MathematicsEngine().evaluate("9" * 129)


def test_parser_rejects_results_that_exceed_integer_size_limit() -> None:
    expression = "*".join(["9" * 120] * 4)
    with pytest.raises(MathematicsError, match="result size limit"):
        MathematicsEngine().evaluate(expression)


def test_calculation_prompt_recognition_does_not_capture_general_questions() -> None:
    engine = MathematicsEngine()
    assert engine.solve_prompt("What is 2 + 2?").exact_result == "4"
    assert engine.solve_prompt("calculate (1/3) + (1/6)").exact_result == "1/2"
    assert engine.solve_prompt("2+2?").exact_result == "4"
    assert engine.solve_prompt("What is 2026?") is None
    assert engine.solve_prompt("42") is None
    assert engine.solve_prompt("Explain CSS Flexbox") is None
    assert engine.solve_prompt("What is a neural network?") is None
    assert engine.solve_prompt("What is the area of a circle?") is None


class RecordingBackend:
    """Backend that records calls so direct arithmetic can bypass the model."""

    def __init__(self, replies: Sequence[str] = ()) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("model backend should not run for direct arithmetic")
        return self.replies.pop(0)


def test_valence_chat_answers_direct_arithmetic_without_model_call() -> None:
    backend = RecordingBackend()
    assistant = ValenceEngine(backend)

    answer = assistant.ask("What is 2 + 2?")

    assert answer == "2 + 2 = 4"
    assert backend.calls == []
    assert [message.role for message in assistant.history] == ["user", "assistant"]


def test_valence_chat_uses_model_for_non_math_questions() -> None:
    backend = RecordingBackend(["Flexbox is a CSS layout model."])
    assistant = ValenceEngine(backend)

    answer = assistant.ask("Explain CSS Flexbox")

    assert answer == "Flexbox is a CSS layout model."
    assert len(backend.calls) == 1


def test_valence_chat_returns_math_errors_without_asking_model() -> None:
    backend = RecordingBackend()
    assistant = ValenceEngine(backend)

    with pytest.raises(ValenceError, match="Division by zero"):
        assistant.ask("10 / 0")

    assert backend.calls == []
    assert assistant.history == ()


def test_mathematics_tool_executes_only_when_policy_allows_it() -> None:
    registry = CapabilityRegistry()
    register_mathematics_tool(registry)
    executor = ToolExecutor(registry)

    denied = executor.execute(
        "calculate_expression", {"expression": "2 + 2"}, ExecutionPolicy()
    )
    allowed = executor.execute(
        "calculate_expression",
        {"expression": "1/3 + 1/6"},
        ExecutionPolicy(frozenset({"calculate_expression"})),
    )

    assert denied.status == "denied"
    assert allowed.status == "success"
    assert allowed.data["exact_result"] == "1/2"
    assert allowed.data["decimal_approximation"] == "0.5"


def test_mathematics_tool_returns_sanitized_errors() -> None:
    registry = CapabilityRegistry()
    registry.register(create_mathematics_tool())
    result = ToolExecutor(registry).execute(
        "calculate_expression",
        {"expression": "1 / 0"},
        ExecutionPolicy(frozenset({"calculate_expression"})),
    )

    assert result.status == "failed"
    assert result.error == "Mathematical calculation failed."


def test_parser_builds_an_immutable_expression_tree_with_precedence() -> None:
    tree = MathematicsEngine().parse("2 + 3 * 4")

    assert isinstance(tree, BinaryNode)
    assert tree.operator == "+"
    assert isinstance(tree.left, NumberNode)
    assert tree.left.value == Fraction(2)
    assert isinstance(tree.right, BinaryNode)
    assert tree.right.operator == "*"
    assert tree.right.left.value == Fraction(3)
    assert tree.right.right.value == Fraction(4)


def test_parser_preserves_unary_nodes_in_the_expression_tree() -> None:
    tree = MathematicsEngine().parse("-(2 + 3)")

    assert isinstance(tree, UnaryNode)
    assert tree.operator == "-"
    assert isinstance(tree.operand, BinaryNode)
    assert tree.operand.operator == "+"


def test_expression_tree_nodes_are_immutable() -> None:
    tree = MathematicsEngine().parse("2 + 3")
    assert isinstance(tree, BinaryNode)

    with pytest.raises((AttributeError, TypeError)):
        tree.operator = "-"  # type: ignore[misc]


def test_explicit_parse_and_evaluate_are_separate_operations() -> None:
    engine = MathematicsEngine()
    tree = engine.parse("(1/3 + 1/6) * 6")

    assert isinstance(tree, BinaryNode)
    assert engine._evaluate_node(tree) == Fraction(3)
    assert engine.evaluate("(1/3 + 1/6) * 6").value == Fraction(3)
