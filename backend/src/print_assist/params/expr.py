"""A safe evaluator for parameter formulas: `height * 0.8`, `base_d / (2 * cos(30))`, `max(wall, 2 * nozzle)`.

The formula is parsed with Python's `ast` and walked against an allow-list of node types; anything else (attribute
access, subscripts, lambdas, comprehensions, keyword arguments, calls to anything but the functions below) is refused.
Nothing is ever passed to `eval`. Angles are in degrees: `sin(30)` is 0.5 and `atan(1)` is 45.
"""

from __future__ import annotations

import ast
import keyword
import math
import operator
from collections.abc import Mapping

MAX_LEN = 500
MAX_DEPTH = 50
MAX_EXPONENT = 64


class ExprError(ValueError):
    pass


def _deg(f):
    return lambda x: f(math.radians(x))


def _adeg(f):
    return lambda x: math.degrees(f(x))


FUNCS = {
    "sin": _deg(math.sin), "cos": _deg(math.cos), "tan": _deg(math.tan),
    "asin": _adeg(math.asin), "acos": _adeg(math.acos), "atan": _adeg(math.atan),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sqrt": math.sqrt, "abs": abs, "min": min, "max": max, "round": round, "floor": math.floor, "ceil": math.ceil,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
}
CONSTANTS = {"pi": math.pi, "e": math.e, "mm": 1.0, "cm": 10.0, "inch": 25.4, "deg": 1.0}
RESERVED = set(FUNCS) | set(CONSTANTS) | {"true", "false", "True", "False"}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}


def valid_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) and name not in RESERVED and not name.startswith("__")


def parse(src: str) -> ast.Expression:
    if not isinstance(src, str) or not src.strip():
        raise ExprError("empty formula")
    if len(src) > MAX_LEN:
        raise ExprError(f"formula longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot read {src.strip()!r}: {exc.msg}") from None
    _check(tree.body, 0)
    return tree


def _check(node: ast.AST, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise ExprError("formula is nested too deeply")
    match node:
        case ast.Constant(value=v):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ExprError(f"only numbers are allowed, not {v!r}")
        case ast.Name(id=name):
            if name.startswith("__"):
                raise ExprError(f"name {name!r} is not allowed")
        case ast.BinOp(left=a, op=op, right=b):
            if type(op) not in _BIN:
                raise ExprError(f"operator {type(op).__name__} is not allowed")
            _check(a, depth + 1)
            _check(b, depth + 1)
        case ast.UnaryOp(op=op, operand=a):
            if type(op) not in _UNARY:
                raise ExprError(f"operator {type(op).__name__} is not allowed")
            _check(a, depth + 1)
        case ast.Compare(left=a, ops=ops, comparators=rest):
            if any(type(o) not in _CMP for o in ops):
                raise ExprError("only < <= > >= == != comparisons are allowed")
            for x in (a, *rest):
                _check(x, depth + 1)
        case ast.BoolOp(values=vals):
            for x in vals:
                _check(x, depth + 1)
        case ast.IfExp(test=t, body=b, orelse=o):
            for x in (t, b, o):
                _check(x, depth + 1)
        case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]):
            if fname not in FUNCS:
                raise ExprError(f"unknown function {fname!r}")
            for x in args:
                if isinstance(x, ast.Starred):
                    raise ExprError("*args is not allowed")
                _check(x, depth + 1)
        case _:
            raise ExprError(f"{type(node).__name__} is not allowed in a formula")


def names(src: str) -> set[str]:
    """The parameter names a formula reads (functions and constants left out)."""
    tree = parse(src)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTANTS:
            out.add(n.id)
    return out


def is_literal(src: str) -> bool:
    try:
        float(src)
        return True
    except (TypeError, ValueError):
        return False


def evaluate(src: str, env: Mapping[str, float] | None = None) -> float:
    """The formula's value, with `env` giving each parameter's value. Raises ExprError on any problem."""
    tree = parse(src)
    env = env or {}

    def ev(node):
        match node:
            case ast.Constant(value=v):
                return v
            case ast.Name(id=name):
                if name in env:
                    v = env[name]
                    if isinstance(v, str):
                        raise ExprError(f"{name} is text, not a number")
                    return v
                if name in CONSTANTS:
                    return CONSTANTS[name]
                raise ExprError(f"unknown name {name!r}")
            case ast.BinOp(left=a, op=op, right=b):
                x, y = ev(a), ev(b)
                if isinstance(op, ast.Pow) and (abs(y) > MAX_EXPONENT or abs(x) > 1e9):
                    raise ExprError("power is too large")
                try:
                    return _BIN[type(op)](x, y)
                except ZeroDivisionError:
                    raise ExprError("division by zero") from None
            case ast.UnaryOp(op=op, operand=a):
                return _UNARY[type(op)](ev(a))
            case ast.Compare(left=a, ops=ops, comparators=rest):
                left = ev(a)
                for o, c in zip(ops, rest):
                    right = ev(c)
                    if not _CMP[type(o)](left, right):
                        return False
                    left = right
                return True
            case ast.BoolOp(op=op, values=vals):
                if isinstance(op, ast.And):
                    return all(ev(x) for x in vals)
                return any(ev(x) for x in vals)
            case ast.IfExp(test=t, body=b, orelse=o):
                return ev(b) if ev(t) else ev(o)
            case ast.Call(func=ast.Name(id=fname), args=args):
                try:
                    return FUNCS[fname](*(ev(x) for x in args))
                except (ValueError, TypeError) as exc:
                    raise ExprError(f"{fname}(): {exc}") from None
        raise ExprError(f"{type(node).__name__} is not allowed in a formula")  # unreachable after _check

    try:
        value = ev(tree.body)
    except (OverflowError, RecursionError):
        raise ExprError("the result is too large") from None
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExprError("the result is not a finite number")
    return float(value)
