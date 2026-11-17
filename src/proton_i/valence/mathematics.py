"""Safe deterministic mathematics with exact rational arithmetic.

The parser accepts a small mathematical grammar and creates immutable
expression nodes. Evaluation never interprets Python source, identifiers, or
arbitrary callable objects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

from proton_i.valence.expressions import (
    BinaryNode,
    ExpressionNode,
    FunctionNode,
    NumberNode,
    UnaryNode,
)

_MAX_EXPRESSION_CHARS = 512
_MAX_TOKENS = 128
_MAX_NESTING_DEPTH = 32
_MAX_NUMBER_DIGITS = 128
_MAX_RESULT_DIGITS = 384
_MAX_RESULT_BITS = (_MAX_RESULT_DIGITS * 3322 // 1000) + 1
_MAX_POWER_EXPONENT = 1000
_DECIMAL_PRECISION = 36
_ALLOWED_FUNCTIONS = {"abs", "gcd", "lcm", "sqrt", "ratio", "percent"}

_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>(?:\d+(?:\.\d+)?|\.\d+))|"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<symbol>\*\*|[()+\-*/%,]))"
)
_MATH_PROMPT_PREFIX = re.compile(
    r"^(?:what\s+is|calculate|compute|evaluate)\s+(.+)$",
    re.IGNORECASE,
)
_DIRECT_EXPRESSION = re.compile(r"^[0-9\s.+*/()%\-*]+$")
_FUNCTION_EXPRESSION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(")

NumericValue = Fraction | Decimal


class MathematicsError(ValueError):
    """Raised when a mathematical expression is invalid or unsupported."""


@dataclass(frozen=True, slots=True)
class _Token:
    """One token emitted by the validated mathematics lexer."""

    kind: Literal["number", "name", "symbol"]
    text: str


@dataclass(frozen=True, slots=True)
class MathematicsResult:
    """An exact rational or explicitly approximate decimal result."""

    expression: str
    value: NumericValue

    @property
    def exact_result(self) -> str | None:
        """Return a reduced exact fraction, or None for approximate values."""
        if isinstance(self.value, Decimal):
            return None
        return str(self.value)

    @property
    def decimal_approximation(self) -> str:
        """Return a 15-significant-digit decimal representation."""
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            decimal_value = _as_decimal(self.value)
            return format(decimal_value, ".15g")

    @property
    def answer(self) -> str:
        """Return exact form when possible and label numerical approximations."""
        if isinstance(self.value, Decimal):
            return f"{self.expression} = ≈ {self.decimal_approximation}"
        if self.value.denominator == 1:
            return f"{self.expression} = {self.exact_result}"
        return (
            f"{self.expression} = {self.exact_result} "
            f"(≈ {self.decimal_approximation})"
        )


class _MathematicsParser:
    """Recursive-descent parser for exact arithmetic and whitelisted functions."""

    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.position = 0
        self.nesting_depth = 0

    def parse(self) -> ExpressionNode:
        """Build an immutable syntax tree and reject leftover tokens."""
        if not self.tokens:
            raise MathematicsError("Expression must not be empty.")
        result = self._parse_expression()
        if self.position != len(self.tokens):
            token = self.tokens[self.position]
            raise MathematicsError(f"Unexpected token: {token.text}")
        return result

    def _peek(self) -> _Token | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def _consume(self) -> _Token:
        token = self._peek()
        if token is None:
            raise MathematicsError("Expression ended unexpectedly.")
        self.position += 1
        return token

    def _parse_expression(self) -> ExpressionNode:
        value = self._parse_term()
        while (token := self._peek()) is not None and token.text in {"+", "-"}:
            operator = self._consume().text
            value = BinaryNode(operator, value, self._parse_term())
        return value

    def _parse_term(self) -> ExpressionNode:
        value = self._parse_unary()
        while (token := self._peek()) is not None and token.text in {"*", "/", "%"}:
            operator = self._consume().text
            value = BinaryNode(operator, value, self._parse_unary())
        return value

    def _parse_unary(self) -> ExpressionNode:
        token = self._peek()
        if token is not None and token.text in {"+", "-"}:
            operator = self._consume().text
            return UnaryNode(operator, self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> ExpressionNode:
        value = self._parse_primary()
        token = self._peek()
        if token is not None and token.text == "**":
            self._consume()
            # Parse the right operand as unary -> power, making exponentiation
            # right-associative while still allowing a negative exponent.
            return BinaryNode("**", value, self._parse_unary())
        return value

    def _parse_primary(self) -> ExpressionNode:
        token = self._peek()
        if token is None:
            raise MathematicsError(
                "Expected a number, function, or parenthesized expression."
            )

        if token.kind == "number":
            self._consume()
            try:
                value = Fraction(token.text)
            except (ValueError, ZeroDivisionError) as error:
                raise MathematicsError("Invalid numeric literal.") from error
            _check_result_size(value)
            return NumberNode(value)

        if token.kind == "name":
            return self._parse_function()

        if token.text == "(":
            self._consume()
            self.nesting_depth += 1
            if self.nesting_depth > _MAX_NESTING_DEPTH:
                raise MathematicsError(
                    f"Parenthesis nesting limit is {_MAX_NESTING_DEPTH}."
                )
            value = self._parse_expression()
            closing = self._peek()
            if closing is None or closing.text != ")":
                raise MathematicsError("A closing parenthesis is missing.")
            self._consume()
            self.nesting_depth -= 1
            return value

        raise MathematicsError(
            f"Expected a number, function, or '(', received: {token.text}"
        )

    def _parse_function(self) -> FunctionNode:
        """Parse a whitelisted function call and its comma-separated arguments."""
        name = self._consume().text.lower()
        if name not in _ALLOWED_FUNCTIONS:
            raise MathematicsError(f"Unsupported mathematical function: {name}")
        opening = self._peek()
        if opening is None or opening.text != "(":
            raise MathematicsError(f"Function {name} must be followed by parentheses.")
        self._consume()
        self.nesting_depth += 1
        if self.nesting_depth > _MAX_NESTING_DEPTH:
            raise MathematicsError(f"Function nesting limit is {_MAX_NESTING_DEPTH}.")

        arguments: list[ExpressionNode] = []
        closing = self._peek()
        if closing is not None and closing.text == ")":
            self._consume()
            self.nesting_depth -= 1
            self._validate_function_arity(name, 0)
            return FunctionNode(name, ())

        while True:
            arguments.append(self._parse_expression())
            separator = self._peek()
            if separator is not None and separator.text == ",":
                self._consume()
                continue
            if separator is None or separator.text != ")":
                raise MathematicsError("A closing parenthesis is missing.")
            self._consume()
            self.nesting_depth -= 1
            break

        self._validate_function_arity(name, len(arguments))
        return FunctionNode(name, tuple(arguments))

    @staticmethod
    def _validate_function_arity(name: str, count: int) -> None:
        """Reject unsupported argument counts before evaluation."""
        allowed = {
            "abs": {1},
            "gcd": {2},
            "lcm": {2},
            "sqrt": {1},
            "ratio": {2},
            "percent": {1, 2},
        }
        if count not in allowed[name]:
            raise MathematicsError(
                f"Function {name} expects {sorted(allowed[name])} argument(s)."
            )


class MathematicsEngine:
    """Evaluate bounded arithmetic and number-system expressions."""

    def evaluate(self, expression: str) -> MathematicsResult:
        """Parse and evaluate an expression with exact rational values where possible."""
        tokens = self._tokenize_input(expression)
        syntax_tree = _MathematicsParser(tokens).parse()
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            value = self._evaluate_node(syntax_tree)
        return MathematicsResult(
            expression=self._format_expression(tokens),
            value=value,
        )

    def parse(self, expression: str) -> ExpressionNode:
        """Parse a bounded expression into immutable expression nodes."""
        return _MathematicsParser(self._tokenize_input(expression)).parse()

    def _tokenize_input(self, expression: str) -> list[_Token]:
        """Validate expression type and size before tokenization."""
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        cleaned = expression.strip()
        if not cleaned:
            raise MathematicsError("Expression must not be empty.")
        if len(cleaned) > _MAX_EXPRESSION_CHARS:
            raise MathematicsError(
                f"Expression must be {_MAX_EXPRESSION_CHARS} characters or fewer."
            )
        return self._tokenize(cleaned)

    def _evaluate_node(self, node: ExpressionNode) -> NumericValue:
        """Evaluate only immutable, explicitly supported expression node types."""
        if isinstance(node, NumberNode):
            value: NumericValue = node.value
        elif isinstance(node, UnaryNode):
            operand = self._evaluate_node(node.operand)
            value = operand if node.operator == "+" else -operand
        elif isinstance(node, BinaryNode):
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            value = self._evaluate_binary(node.operator, left, right)
        elif isinstance(node, FunctionNode):
            arguments = [self._evaluate_node(argument) for argument in node.arguments]
            value = self._evaluate_function(node.name, arguments)
        else:
            raise MathematicsError("Unsupported expression node.")

        _check_result_size(value)
        return value

    def _evaluate_binary(
        self,
        operator: str,
        left: NumericValue,
        right: NumericValue,
    ) -> NumericValue:
        """Evaluate one binary operator while retaining exact rational values."""
        if operator == "%":
            left_integer = _require_integer(left, "Modulo")
            right_integer = _require_integer(right, "Modulo")
            if right_integer == 0:
                raise MathematicsError("Modulo by zero is undefined.")
            return Fraction(left_integer % right_integer)

        if operator == "**":
            exponent = _require_integer(right, "Exponentiation")
            if abs(exponent) > _MAX_POWER_EXPONENT:
                raise MathematicsError(
                    f"Power exponent magnitude is limited to {_MAX_POWER_EXPONENT}."
                )
            if isinstance(left, Fraction):
                if left == 0 and exponent < 0:
                    raise MathematicsError("Zero cannot have a negative exponent.")
                return left**exponent
            if left == 0 and exponent < 0:
                raise MathematicsError("Zero cannot have a negative exponent.")
            return left**exponent

        if operator == "/" and right == 0:
            raise MathematicsError("Division by zero is undefined.")

        if isinstance(left, Decimal) or isinstance(right, Decimal):
            left_decimal = _as_decimal(left)
            right_decimal = _as_decimal(right)
            if operator == "+":
                return left_decimal + right_decimal
            if operator == "-":
                return left_decimal - right_decimal
            if operator == "*":
                return left_decimal * right_decimal
            return left_decimal / right_decimal

        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        return left / right

    def _evaluate_function(
        self,
        name: str,
        arguments: list[NumericValue],
    ) -> NumericValue:
        """Evaluate one of the explicitly supported deterministic math functions."""
        if name == "abs":
            return abs(arguments[0])

        if name in {"gcd", "lcm"}:
            left = _require_integer(arguments[0], name.upper())
            right = _require_integer(arguments[1], name.upper())
            result = math.gcd(left, right) if name == "gcd" else math.lcm(left, right)
            return Fraction(result)

        if name == "sqrt":
            value = arguments[0]
            if value < 0:
                raise MathematicsError(
                    "Square root of a negative real number is undefined."
                )
            if isinstance(value, Fraction):
                numerator_root = math.isqrt(value.numerator)
                denominator_root = math.isqrt(value.denominator)
                if (
                    numerator_root * numerator_root == value.numerator
                    and denominator_root * denominator_root == value.denominator
                ):
                    return Fraction(numerator_root, denominator_root)
            return _as_decimal(value).sqrt()

        if name == "ratio":
            numerator, denominator = arguments
            if denominator == 0:
                raise MathematicsError("A ratio denominator must not be zero.")
            if isinstance(numerator, Decimal) or isinstance(denominator, Decimal):
                return _as_decimal(numerator) / _as_decimal(denominator)
            return numerator / denominator

        if name == "percent":
            percent = arguments[0]
            if len(arguments) == 1:
                return percent / 100
            value = arguments[1]
            if isinstance(percent, Decimal) or isinstance(value, Decimal):
                return (_as_decimal(percent) / Decimal(100)) * _as_decimal(value)
            return (percent * value) / 100

        raise MathematicsError(f"Unsupported mathematical function: {name}")

    def solve_prompt(self, prompt: str) -> MathematicsResult | None:
        """Recognize supported math expressions without stealing factual questions."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        cleaned = prompt.strip()
        if not cleaned:
            return None

        match = _MATH_PROMPT_PREFIX.fullmatch(cleaned)
        candidate = match.group(1).strip() if match is not None else cleaned
        candidate = re.sub(r"[?!.]+\s*$", "", candidate).strip()
        if not any(character.isdigit() for character in candidate):
            return None

        starts_function_call = _FUNCTION_EXPRESSION.match(candidate) is not None
        has_operator = any(character in candidate for character in "+-*/%().")
        if not starts_function_call and not has_operator:
            return None
        if (
            any(character.isalpha() for character in candidate)
            and not starts_function_call
        ):
            return None
        if (
            match is None
            and not starts_function_call
            and _DIRECT_EXPRESSION.fullmatch(candidate) is None
        ):
            return None

        return self.evaluate(candidate)

    @staticmethod
    def _tokenize(expression: str) -> list[_Token]:
        """Convert input to a bounded stream of numeric, name, and symbol tokens."""
        tokens: list[_Token] = []
        position = 0
        while position < len(expression):
            match = _TOKEN_PATTERN.match(expression, position)
            if match is None:
                if expression[position:].isspace():
                    break
                raise MathematicsError(
                    f"Unsupported character at position {position + 1}."
                )

            number = match.group("number")
            name = match.group("name")
            symbol = match.group("symbol")
            if number is not None:
                digit_count = sum(character.isdigit() for character in number)
                if digit_count > _MAX_NUMBER_DIGITS:
                    raise MathematicsError(
                        f"Numeric literals are limited to {_MAX_NUMBER_DIGITS} digits."
                    )
                tokens.append(_Token("number", number))
            elif name is not None:
                tokens.append(_Token("name", name))
            elif symbol is not None:
                tokens.append(_Token("symbol", symbol))
            position = match.end()

            if len(tokens) > _MAX_TOKENS:
                raise MathematicsError(
                    f"Expressions are limited to {_MAX_TOKENS} tokens."
                )

        return tokens

    @staticmethod
    def _format_expression(tokens: list[_Token]) -> str:
        """Normalize token spacing without changing expression meaning."""
        parts: list[str] = []
        for token in tokens:
            value = token.text
            if value == "(":
                if (
                    parts
                    and parts[-1] not in {" ", "("}
                    and not parts[-1].isidentifier()
                ):
                    parts.append(" ")
                parts.append("(")
            elif value == ")":
                if parts and parts[-1] == " ":
                    parts.pop()
                parts.append(")")
            elif value == ",":
                if parts and parts[-1] == " ":
                    parts.pop()
                parts.extend([",", " "])
            else:
                if parts and parts[-1] not in {" ", "("}:
                    parts.append(" ")
                parts.append(value)
                if value in {"+", "-", "*", "/", "%", "**"}:
                    parts.append(" ")
        return "".join(parts).strip()


def _require_integer(value: NumericValue, operation: str) -> int:
    """Require an exact integer argument for operations defined on integers."""
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise MathematicsError(f"{operation} requires integer arguments.")
        return int(value)
    if value.denominator != 1:
        raise MathematicsError(f"{operation} requires integer arguments.")
    return value.numerator


def _as_decimal(value: NumericValue) -> Decimal:
    """Convert exact rational input to a bounded-precision Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(value.numerator) / Decimal(value.denominator)


def _check_result_size(value: NumericValue) -> None:
    """Bound exact integer and approximate decimal growth."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise MathematicsError("Mathematical result is not finite.")
        if (
            len(value.as_tuple().digits) > _MAX_RESULT_DIGITS
            or abs(value.adjusted()) > _MAX_RESULT_DIGITS
        ):
            raise MathematicsError(
                f"Mathematical result exceeds the {_MAX_RESULT_DIGITS}-digit result size limit."
            )
        return

    if (
        abs(value.numerator).bit_length() > _MAX_RESULT_BITS
        or value.denominator.bit_length() > _MAX_RESULT_BITS
    ):
        raise MathematicsError(
            f"Mathematical result exceeds the {_MAX_RESULT_DIGITS}-digit result size limit."
        )
    numerator_digits = len(str(abs(value.numerator)))
    denominator_digits = len(str(value.denominator))
    if max(numerator_digits, denominator_digits) > _MAX_RESULT_DIGITS:
        raise MathematicsError(
            f"Mathematical result exceeds the {_MAX_RESULT_DIGITS}-digit result size limit."
        )
