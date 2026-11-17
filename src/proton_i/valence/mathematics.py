"""Safe deterministic arithmetic with exact rational results.

User input is tokenized and parsed by a small allowlisted grammar. No Python
source is evaluated and no names, attributes, calls, or implicit multiplication
are accepted in this first milestone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

_MAX_EXPRESSION_CHARS = 512
_MAX_TOKENS = 128
_MAX_NESTING_DEPTH = 32
_MAX_NUMBER_DIGITS = 128
_MAX_RESULT_DIGITS = 2_048

_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>(?:\d+(?:\.\d+)?|\.\d+))|(?P<symbol>[()+\-*/]))"
)
_MATH_PROMPT_PREFIX = re.compile(
    r"^(?:what\s+is|calculate|compute|evaluate)\s+(.+)$",
    re.IGNORECASE,
)
_DIRECT_EXPRESSION = re.compile(r"^[0-9\s.+*/()\-]+$")


class MathematicsError(ValueError):
    """Raised when a mathematical expression is invalid or unsupported."""


@dataclass(frozen=True, slots=True)
class _Token:
    """One token emitted by the validated arithmetic lexer."""

    kind: Literal["number", "symbol"]
    text: str


@dataclass(frozen=True, slots=True)
class MathematicsResult:
    """Exact rational arithmetic result plus a readable decimal approximation."""

    expression: str
    value: Fraction

    @property
    def exact_result(self) -> str:
        """Return the reduced exact fraction or integer."""
        return str(self.value)

    @property
    def decimal_approximation(self) -> str:
        """Return a 15-significant-digit decimal approximation."""
        with localcontext() as context:
            context.prec = 18
            decimal_value = Decimal(self.value.numerator) / Decimal(
                self.value.denominator
            )
            return format(decimal_value, ".15g")

    @property
    def answer(self) -> str:
        """Return the exact answer and include a decimal form for non-integers."""
        if self.value.denominator == 1:
            return f"{self.expression} = {self.exact_result}"
        return (
            f"{self.expression} = {self.exact_result} "
            f"(≈ {self.decimal_approximation})"
        )


class _ArithmeticParser:
    """Recursive-descent parser for +, -, *, /, unary signs, and parentheses."""

    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.position = 0
        self.nesting_depth = 0

    def parse(self) -> Fraction:
        """Parse one complete expression, rejecting leftover tokens."""
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

    def _parse_expression(self) -> Fraction:
        value = self._parse_term()
        while (token := self._peek()) is not None and token.text in {"+", "-"}:
            operator = self._consume().text
            right = self._parse_term()
            value = value + right if operator == "+" else value - right
            self._check_result_size(value)
        return value

    def _parse_term(self) -> Fraction:
        value = self._parse_unary()
        while (token := self._peek()) is not None and token.text in {"*", "/"}:
            operator = self._consume().text
            right = self._parse_unary()
            if operator == "*":
                value *= right
            else:
                if right == 0:
                    raise MathematicsError("Division by zero is undefined.")
                value /= right
            self._check_result_size(value)
        return value

    def _parse_unary(self) -> Fraction:
        token = self._peek()
        if token is not None and token.text in {"+", "-"}:
            operator = self._consume().text
            value = self._parse_unary()
            return value if operator == "+" else -value
        return self._parse_primary()

    def _parse_primary(self) -> Fraction:
        token = self._peek()
        if token is None:
            raise MathematicsError("Expected a number or parenthesized expression.")

        if token.kind == "number":
            self._consume()
            try:
                value = Fraction(token.text)
            except (ValueError, ZeroDivisionError) as error:
                raise MathematicsError("Invalid numeric literal.") from error
            self._check_result_size(value)
            return value

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

        raise MathematicsError(f"Expected a number or '(', received: {token.text}")

    @staticmethod
    def _check_result_size(value: Fraction) -> None:
        """Bound integer growth before a user expression consumes excess memory."""
        numerator_digits = len(str(abs(value.numerator)))
        denominator_digits = len(str(value.denominator))
        if max(numerator_digits, denominator_digits) > _MAX_RESULT_DIGITS:
            raise MathematicsError(
                f"Mathematical result exceeds the {_MAX_RESULT_DIGITS}-digit result size limit."
            )


class MathematicsEngine:
    """Evaluate a deliberately small exact arithmetic language."""

    def evaluate(self, expression: str) -> MathematicsResult:
        """Parse and evaluate an expression, returning an exact Fraction."""
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        cleaned = expression.strip()
        if not cleaned:
            raise MathematicsError("Expression must not be empty.")
        if len(cleaned) > _MAX_EXPRESSION_CHARS:
            raise MathematicsError(
                f"Expression must be {_MAX_EXPRESSION_CHARS} characters or fewer."
            )

        tokens = self._tokenize(cleaned)
        parser = _ArithmeticParser(tokens)
        value = parser.parse()
        return MathematicsResult(
            expression=self._format_expression(tokens),
            value=value,
        )

    def solve_prompt(self, prompt: str) -> MathematicsResult | None:
        """Recognize direct arithmetic prompts without capturing ordinary questions."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        cleaned = prompt.strip()
        if not cleaned:
            return None

        match = _MATH_PROMPT_PREFIX.fullmatch(cleaned)
        candidate = match.group(1).strip() if match is not None else cleaned
        candidate = re.sub(r"[?!.]+\s*$", "", candidate).strip()

        # Do not divert natural-language questions to the calculator.
        if not any(character.isdigit() for character in candidate):
            return None
        if any(character.isalpha() for character in candidate):
            return None
        if match is None and _DIRECT_EXPRESSION.fullmatch(candidate) is None:
            return None
        return self.evaluate(candidate)

    @staticmethod
    def _tokenize(expression: str) -> list[_Token]:
        """Convert input to a bounded stream of numeric and operator tokens."""
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
            symbol = match.group("symbol")
            if number is not None:
                digit_count = sum(character.isdigit() for character in number)
                if digit_count > _MAX_NUMBER_DIGITS:
                    raise MathematicsError(
                        f"Numeric literals are limited to {_MAX_NUMBER_DIGITS} digits."
                    )
                tokens.append(_Token("number", number))
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
                if parts and parts[-1] not in {" ", "("}:
                    parts.append(" ")
                parts.append("(")
            elif value == ")":
                if parts and parts[-1] == " ":
                    parts.pop()
                parts.append(")")
            else:
                if parts and parts[-1] not in {" ", "("}:
                    parts.append(" ")
                parts.append(value)
                if value in {"+", "-", "*", "/"}:
                    parts.append(" ")
        return "".join(parts).strip()
