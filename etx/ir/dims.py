"""Symbolic dimensions.

A dimension is an int or a string expression over symbol names, e.g. "B*H" or
"cdiv(S, 128)".  Expressions are evaluated with a restricted AST walker: no
attribute access, no calls except the whitelisted helpers, integer arithmetic
only.  The same evaluator serves edge maps, where expressions may also index
runtime tensors ("indptr[i+1]").
"""
from __future__ import annotations

import ast
import math
from typing import Any, Mapping

Dim = int | str

_FUNCS = {
    "cdiv": lambda a, b: -(-a // b),
    "max": max,
    "min": min,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _binop(op: ast.operator, a: int, b: int) -> int:
    if isinstance(op, ast.Add):
        return a + b
    if isinstance(op, ast.Sub):
        return a - b
    if isinstance(op, ast.Mult):
        return a * b
    if isinstance(op, (ast.FloorDiv, ast.Div)):
        return a // b
    if isinstance(op, ast.Mod):
        return a % b
    if isinstance(op, ast.Pow):
        return a ** b
    raise ValueError(f"unsupported operator {type(op).__name__}")


def eval_expr(src: str, env: Mapping[str, int], runtime: Mapping[str, Any] | None = None) -> int:
    """Evaluate an integer expression. `env` binds symbols and loop letters;
    `runtime` binds runtime tensors (nested lists) for subscript expressions."""
    node = ast.parse(str(src).strip(), mode="eval")
    runtime = runtime or {}

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in env:
                return int(env[n.id])
            raise KeyError(f"unbound symbol {n.id!r} in {src!r}")
        if isinstance(n, ast.BinOp):
            return _binop(n.op, ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return int(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            name = n.value.id
            if name not in runtime:
                raise KeyError(f"runtime tensor {name!r} not provided for {src!r}")
            idx = n.slice
            indices = [ev(e) for e in idx.elts] if isinstance(idx, ast.Tuple) else [ev(idx)]
            val: Any = runtime[name]
            for i in indices:
                val = val[i]
            return int(val)
        raise ValueError(f"unsupported expression node {type(n).__name__} in {src!r}")

    return ev(node)


def eval_dim(d: Dim, bindings: Mapping[str, int]) -> int:
    if isinstance(d, bool):
        raise TypeError("bool is not a dimension")
    if isinstance(d, int):
        return d
    v = eval_expr(d, bindings)
    if v < 0:
        raise ValueError(f"dimension {d!r} evaluated to {v} < 0")
    return v


def free_symbols(d: Dim) -> set[str]:
    if isinstance(d, int):
        return set()
    out: set[str] = set()
    for n in ast.walk(ast.parse(str(d), mode="eval")):
        if isinstance(n, ast.Name) and n.id not in _FUNCS:
            out.add(n.id)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            out.discard(n.value.id)
    return out


def eval_shape(shape: tuple[Dim, ...], bindings: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(eval_dim(d, bindings) for d in shape)
