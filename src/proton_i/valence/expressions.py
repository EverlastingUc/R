"""Immutable expression-tree nodes for Valence's mathematics language."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias


ArithmeticOperator: TypeAlias = Literal["+", "-", "*", "/"]
UnaryOperator: TypeAlias = Literal["+", "-"]


@dataclass(frozen=True, slots=True)
class NumberNode:
    """An exact rational literal in a parsed mathematics expression."""

    value: Fraction


@dataclass(frozen=True, slots=True)
class UnaryNode:
    """A unary sign applied to one child expression."""

    operator: UnaryOperator
    operand: ExpressionNode


@dataclass(frozen=True, slots=True)
class BinaryNode:
    """A binary arithmetic operation with left and right child expressions."""

    operator: ArithmeticOperator
    left: ExpressionNode
    right: ExpressionNode


ExpressionNode: TypeAlias = NumberNode | UnaryNode | BinaryNode
