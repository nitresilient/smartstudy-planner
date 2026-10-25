
import ast
import operator


def calculator(expression: str) -> str:
    """Safely calculate a mathematical expression using basic arithmetic.

    Args:
        expression: A mathematical expression using numbers and
            the operators +, -, *, /, and **.

    Returns:
        The calculation result or an error message.
    """

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent is too large.")

            return operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


import re


def text_analyzer(text: str) -> str:
    """Analyze text and return its character, word, and sentence counts.

    Args:
        text: The text to analyze.

    Returns:
        Character count, word count, and sentence count.
    """
    characters = len(text)
    words = len(text.split())
    sentences = len(
        [part for part in re.split(r"[.!?]+", text) if part.strip()]
    )

    return (
        f"Characters: {characters}\n"
        f"Words: {words}\n"
        f"Sentences: {sentences}"
    )
