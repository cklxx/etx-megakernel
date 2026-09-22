"""Edge maps: task coordinate -> event coordinate(s).

Four forms (design §07 6.2):
  affine / einsum subset   "ij->i"    "h->bh" (letter only on the right = broadcast)
  index arithmetic         "ij->(i/2)j"   parenthesised integer expressions over lhs letters
  runtime range            "i->range(indptr[i], indptr[i+1])"
  runtime gather           "i->topk[i,:]"
plus the internal form "*" (all coordinates), used by the event-elimination pass
when a consumer inherits a producer's dependencies conservatively.

`targets()` enumerates the event coordinates for one task coordinate (used by
verification, placement and the simulator).  `to_c()` emits the equivalent C
loop for the generated kernel, calling a caller-supplied macro with the linear
event index.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dims import eval_expr

_RANGE_RE = re.compile(r"^range\((.+),(.+)\)$")
_GATHER_RE = re.compile(r"^(\w+)\[(.+),\s*:\s*\]$")


@dataclass(frozen=True)
class EdgeMap:
    text: str
    kind: str                                   # affine | range | gather | all
    lhs: tuple[str, ...] = ()
    items: tuple[tuple[str, str], ...] = ()     # affine: (idx|expr|bcast, payload)
    runtime: tuple[str, ...] = ()               # runtime tensors referenced
    range_lo: str = ""
    range_hi: str = ""
    gather_tensor: str = ""
    gather_row: str = ""

    # ------------------------------------------------------------------ parse
    @staticmethod
    def parse(text: str) -> "EdgeMap":
        t = text.replace(" ", "")
        if t == "*":
            return EdgeMap(text=text, kind="all")
        if "->" not in t:
            raise ValueError(f"edge map {text!r} must contain '->'")
        lhs_s, rhs = t.split("->", 1)
        lhs = tuple(lhs_s)
        if not all(c.isalpha() for c in lhs):
            raise ValueError(f"edge map lhs must be single letters: {text!r}")
        m = _RANGE_RE.match(rhs)
        if m:
            lo, hi = m.group(1), m.group(2)
            rt = tuple(sorted(set(re.findall(r"(\w+)\[", lo + " " + hi))))
            return EdgeMap(text=text, kind="range", lhs=lhs, runtime=rt, range_lo=lo, range_hi=hi)
        m = _GATHER_RE.match(rhs)
        if m:
            return EdgeMap(text=text, kind="gather", lhs=lhs, runtime=(m.group(1),),
                           gather_tensor=m.group(1), gather_row=m.group(2))
        if rhs == "()":
            return EdgeMap(text=text, kind="affine", lhs=lhs, items=())
        items: list[tuple[str, str]] = []
        i = 0
        while i < len(rhs):
            c = rhs[i]
            if c == "(":
                depth, j = 1, i + 1
                while j < len(rhs) and depth:
                    depth += rhs[j] == "("
                    depth -= rhs[j] == ")"
                    j += 1
                if depth:
                    raise ValueError(f"unbalanced parentheses in {text!r}")
                expr = rhs[i + 1:j - 1]
                # names in an expression are lhs letters or graph symbols (checked at instantiation)
                for name in re.findall(r"[A-Za-z_]\w*", expr):
                    if len(name) == 1 and name not in lhs and name.islower():
                        raise ValueError(f"expression letter {name!r} not in lhs of {text!r}")
                items.append(("expr", expr))
                i = j
            elif c.isalpha():
                items.append(("idx", c) if c in lhs else ("bcast", c))
                i += 1
            else:
                raise ValueError(f"unexpected {c!r} in edge map {text!r}")
        return EdgeMap(text=text, kind="affine", lhs=lhs, items=tuple(items))

    # -------------------------------------------------------------- semantics
    @property
    def is_runtime(self) -> bool:
        return self.kind in ("range", "gather")

    def event_rank(self) -> int | None:
        if self.kind == "affine":
            return len(self.items)
        if self.kind in ("range", "gather"):
            return 1
        return None

    def targets(self, coord: Sequence[int], event_shape: Sequence[int],
                runtime: Mapping[str, Any] | None = None,
                bindings: Mapping[str, int] | None = None) -> list[tuple[int, ...]]:
        env: dict[str, int] = dict(bindings or {})
        env.update({letter: int(v) for letter, v in zip(self.lhs, coord)})   # coordinates shadow symbols
        if self.kind == "all":
            return list(itertools.product(*[range(n) for n in event_shape]))
        if self.kind == "affine":
            if len(self.items) != len(event_shape):
                raise ValueError(f"{self.text!r} produces rank {len(self.items)} but event has rank {len(event_shape)}")
            axes: list[list[int]] = []
            for (kind, payload), n in zip(self.items, event_shape):
                if kind == "idx":
                    axes.append([env[payload]])
                elif kind == "expr":
                    axes.append([eval_expr(payload, env)])
                else:
                    axes.append(list(range(n)))
            return list(itertools.product(*axes))
        if self.kind == "range":
            lo = eval_expr(self.range_lo, env, runtime)
            hi = eval_expr(self.range_hi, env, runtime)
            return [(t,) for t in range(lo, hi)]
        if self.kind == "gather":
            if runtime is None or self.gather_tensor not in runtime:
                raise KeyError(f"runtime tensor {self.gather_tensor!r} required by {self.text!r}")
            row = runtime[self.gather_tensor][eval_expr(self.gather_row, env)]
            return [(int(v),) for v in row]
        raise AssertionError(self.kind)

    # ---------------------------------------------------------------- codegen
    def coord_exprs_c(self, coord_vars: Sequence[str], symbols: Mapping[str, str] | None = None) -> list[str]:
        """Per-dimension C expressions of the target coordinate (affine maps without broadcast)."""
        if self.kind != "affine" or any(k == "bcast" for k, _ in self.items):
            raise ValueError(f"{self.text!r}: only affine maps without broadcast give a single coordinate")
        env: dict[str, str] = dict(symbols or {})
        env.update({letter: v for letter, v in zip(self.lhs, coord_vars)})
        out = []
        for kind, payload in self.items:
            if kind == "idx":
                out.append(env[payload])
            else:
                e = payload
                for letter, v in env.items():
                    e = re.sub(rf"\b{letter}\b", f"({v})", e)
                out.append(e.replace("cdiv(", "etx_cdiv("))
        return out

    def to_c(self, coord_vars: Sequence[str], ev_shape_expr: str, callback: str,
             runtime_ptrs: Mapping[str, str] | None = None, gather_width: str = "",
             symbols: Mapping[str, str] | None = None) -> str:
        """Emit C that calls `callback(linear_index)` for every target.
        `ev_shape_expr` names an int32 array with the event shape; `symbols`
        maps graph symbols to C expressions (e.g. p.shape[2])."""
        env: dict[str, str] = dict(symbols or {})
        env.update({letter: v for letter, v in zip(self.lhs, coord_vars)})
        rp = runtime_ptrs or {}

        def lin(coords: list[str], rank: int) -> str:
            acc = coords[0]
            for d in range(1, rank):
                acc = f"(({acc}) * {ev_shape_expr}[{d}] + ({coords[d]}))"
            return acc

        if self.kind == "all":
            return (f"  for (int _t = 0; _t < etx_ev_numel({ev_shape_expr}); ++_t) {{ {callback}(_t); }}\n")
        if self.kind == "affine":
            if not self.items:
                return f"  {callback}(0);\n"
            lines, coords, indent = [], [], "  "
            for d, (kind, payload) in enumerate(self.items):
                if kind == "idx":
                    coords.append(env[payload])
                elif kind == "expr":
                    e = payload
                    for letter, v in env.items():
                        e = re.sub(rf"\b{letter}\b", f"({v})", e)
                    coords.append(e.replace("cdiv(", "etx_cdiv("))
                else:
                    var = f"_b{d}"
                    lines.append(f"{indent}for (int {var} = 0; {var} < {ev_shape_expr}[{d}]; ++{var}) {{")
                    indent += "  "
                    coords.append(var)
            lines.append(f"{indent}{callback}({lin(coords, len(self.items))});")
            for _ in range(len(indent) // 2 - 1):
                indent = indent[:-2]
                lines.append(f"{indent}}}")
            return "\n".join(lines) + "\n"
        if self.kind == "range":
            lo, hi = self.range_lo, self.range_hi
            for letter, v in env.items():
                lo = re.sub(rf"\b{letter}\b", f"({v})", lo)
                hi = re.sub(rf"\b{letter}\b", f"({v})", hi)
            for name, ptr in rp.items():
                lo = lo.replace(f"{name}[", f"{ptr}[")
                hi = hi.replace(f"{name}[", f"{ptr}[")
            return f"  for (int _t = {lo}; _t < {hi}; ++_t) {{ {callback}(_t); }}\n"
        if self.kind == "gather":
            row = self.gather_row
            for letter, v in env.items():
                row = re.sub(rf"\b{letter}\b", f"({v})", row)
            ptr = rp.get(self.gather_tensor, self.gather_tensor)
            return (f"  for (int _k = 0; _k < {gather_width}; ++_k) "
                    f"{{ {callback}({ptr}[({row}) * {gather_width} + _k]); }}\n")
        raise AssertionError(self.kind)
