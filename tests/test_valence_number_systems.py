"""Tests for Valence VQ-02 number-system operations."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from proton_i.valence.expressions import BinaryNode, FunctionNode
from proton_i.valence.mathematics import MathematicsEngine, MathematicsError


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 ** 3", Fraction(8)),
        ("2**3**2", Fraction(512)),
        ("-2**2", Fraction(-4)),
        ("(-2)**2", Fraction(4)),
        ("2**-3", Fraction(1, 8)),
        ("(3/2)**2", Fraction(9, 4)),
        ("(-2)**3", Fraction(-8)),
        ("5 % 2", Fraction(1)),
        ("-5 % 3", Fraction(1)),
        ("abs(-7/3)", Fraction(7, 3)),
        ("gcd(54, 24)", Fraction(6)),
        ("lcm(12, 18)", Fraction(36)),
        ("sqrt(81/100)", Fraction(9, 10)),
        ("sqrt(0)", Fraction(0)),
        ("ratio(2, 3)", Fraction(2, 3)),
        ("percent(15)", Fraction(3, 20)),
        ("percent(15, 200)", Fraction(30)),
    ],
)
def test_number_operations_match_exact_results(
    expression: str, expected: Fraction
) -> None:
    result = MathematicsEngine().evaluate(expression)

    assert result.value == expected
    assert result.exact_result == str(expected)


def test_irrational_square_root_is_explicitly_approximate() -> None:
    result = MathematicsEngine().evaluate("sqrt(2)")

    assert isinstance(result.value, Decimal)
    assert result.exact_result is None
    assert result.decimal_approximation.startswith("1.41421356237309")
    assert "≈" in result.answer


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(4/9)", Fraction(2, 3)),
        ("sqrt(2) * sqrt(2)", pytest.approx(2.0)),
    ],
)
def test_square_root_composes_with_other_exact_or_approximate_values(
    expression: str, expected: object
) -> None:
    result = MathematicsEngine().evaluate(expression)
    if isinstance(expected, Fraction):
        assert result.value == expected
    else:
        assert float(result.decimal_approximation) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 1001",
        "2 ** -1001",
        "0 ** -1",
        "2 ** (1/2)",
        "2 ** 1.5",
        "2 ** 3.2",
        "5 % 0",
        "5.5 % 2",
        "sqrt(-1)",
        "sqrt(-4/9)",
        "gcd(2.5, 5)",
        "lcm(2, 3.5)",
        "ratio(1, 0)",
        "percent(1, 0)",
        "unknown(2)",
        "pow(2, 3)",
    ],
)
def test_number_operations_reject_unsupported_or_undefined_cases(
    expression: str,
) -> None:
    with pytest.raises(MathematicsError):
        MathematicsEngine().evaluate(expression)


def test_number_operations_enforce_result_size_after_power() -> None:
    with pytest.raises(MathematicsError, match="result size limit"):
        MathematicsEngine().evaluate("9 ** 1000")


def test_power_and_modulo_operators_are_present_in_the_expression_tree() -> None:
    tree = MathematicsEngine().parse("2 ** 3 % 5")

    assert isinstance(tree, BinaryNode)
    assert tree.operator == "%"
    assert isinstance(tree.left, BinaryNode)
    assert tree.left.operator == "**"


def test_whitelisted_functions_are_present_in_the_expression_tree() -> None:
    tree = MathematicsEngine().parse("gcd(12, 18)")

    assert isinstance(tree, FunctionNode)
    assert tree.name == "gcd"
    assert len(tree.arguments) == 2


@pytest.mark.parametrize(
    "prompt",
    [
        "What is sqrt(2)?",
        "calculate gcd(12, 18)",
        "compute 2**10",
        "evaluate percent(15, 200)",
    ],
)
def test_valence_recognizes_number_system_prompts(prompt: str) -> None:
    result = MathematicsEngine().solve_prompt(prompt)

    assert result is not None


def test_factual_number_question_is_not_stolen_by_number_solver() -> None:
    engine = MathematicsEngine()

    assert engine.solve_prompt("What is 2026?") is None
    assert engine.solve_prompt("What is the square root?") is None


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "sqrt.__call__(2)",
        "abs.__globals__",
        "exec(1)",
        "open('/etc/passwd')",
        "sqrt(2); 3",
    ],
)
def test_function_parser_never_accepts_python_execution_syntax(
    expression: str,
) -> None:
    with pytest.raises(MathematicsError):
        MathematicsEngine().evaluate(expression)
