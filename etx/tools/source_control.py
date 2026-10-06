"""Parse control paths for the fixed source protocols in shared_state.

This parser accepts a small C++ statement subset. Expressions stay as tokens.
Unknown control statements fail closed. This is not a C++ type or alias proof.
"""
from dataclasses import dataclass
import re


def tokens(source):
    source = re.sub(r"^\s*#pragma[^\n]*", "", source, flags=re.M)
    return re.findall(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|\d+|[^\s]''', source)


def compact(source):
    return "".join(tokens(source))


@dataclass
class Statement:
    kind: str
    text: str = ""
    children: tuple = ()
    lexical: tuple = ()


class Parser:
    def __init__(self, source):
        self.ts = tokens(source)
        self.at = 0

    def peek(self):
        return self.ts[self.at] if self.at < len(self.ts) else ""

    def pop(self):
        if not self.peek():
            raise ValueError("incomplete statement")
        token = self.peek()
        self.at += 1
        return token

    def balanced(self):
        pairs = {"(": ")", "[": "]", "{": "}"}
        opening = self.pop()
        closing = pairs[opening]
        result = [opening]
        while self.peek() != closing:
            if self.peek() in pairs:
                result.extend(self.balanced())
            else:
                token = self.pop()
                if token in pairs.values():
                    raise ValueError("unbalanced expression")
                result.append(token)
        result.append(self.pop())
        return result

    def sequence(self, end=""):
        result = []
        while self.peek() != end:
            result.append(self.statement())
        return tuple(result)

    def statement(self):
        head = self.peek()
        if head == "{":
            self.pop()
            result = Statement("block", children=self.sequence("}"))
            self.pop()
            return result
        if head in ("if", "for", "while"):
            self.pop()
            if self.peek() != "(":
                raise ValueError(f"unsupported {head} syntax")
            condition = "".join(self.balanced()[1:-1])
            children = [self.statement()]
            if head == "if" and self.peek() == "else":
                self.pop()
                children.append(self.statement())
            return Statement(head, condition, tuple(children))
        if head in ("switch", "case", "default", "do", "try", "catch", "else"):
            raise ValueError(f"unsupported control statement: {head}")
        result = []
        while self.peek() != ";":
            if self.peek() in ("(", "[", "{"):
                result.extend(self.balanced())
            else:
                token = self.pop()
                if token == "}":
                    raise ValueError("statement lacks semicolon")
                result.append(token)
        result.append(self.pop())
        # Labels can bypass the structured paths. Ternary expressions stay valid.
        if len(result) > 1 and result[1] == ":":
            raise ValueError("unsupported label")
        return Statement("leaf", "".join(result), lexical=tuple(result))


def leaves(statements, path=()):
    for stmt in statements:
        if stmt.kind == "leaf":
            yield path, stmt
        elif stmt.kind == "block":
            yield from leaves(stmt.children, path)
        else:
            yield from leaves(stmt.children[:1], path + ((stmt.kind, stmt.text),))
            if len(stmt.children) == 2:
                yield from leaves(stmt.children[1:], path + (("else", stmt.text),))


# These loop bounds and branch conditions are uniform in the supported callers.
# A new path needs a contract review, even if the new path is also uniform.
MODE = (("if", "d.mode==4"),)
ROWS = (("for", "intr=r0;r<r1;r+=64"),)
BARRIERS = {
    "dsl_stage": ("xs", [()]),
    "dsl_stage_norm_at": ("xs", [MODE, ()]),
    "dsl_topk_lds": ("ids/wts", [()]),
    "dsl_gemv_rows_nb": ("part", [(), (), ()]),
    "dsl_gemv_split": ("part", [(), (), ()]),
    "dsl_gemv_bigk": ("xs/part", [ROWS + (("for", "intk0=0;k0<K;k0+=4096"),), ROWS]),
    "etx_block_sum": ("red", [(), ()]),
    "etx_block_max": ("red", [(), ()]),
    "vl_argmax": ("sv/si", [(), (("for", "ints=blockDim.x/2;s>0;s>>=1"),)]),
}

THREAD_ZERO = (("if", "threadIdx.x==0"),)
NATIVE_STORES = {
    "vl_argmax": [
        (THREAD_ZERO, "VP.token_out[t] = si[0];"),
        (THREAD_ZERO + (("if", "VP.write_next"),), "VP.token_in[t + 1] = si[0];"),
        (THREAD_ZERO, "VP.token_time[2 * t + 1] = wall_clock64();"),
    ],
    "vl_prologue": [
        (THREAD_ZERO, "VP.token_time[2 * t] = wall_clock64();"),
        (THREAD_ZERO, "VP.positions[0] = VP.pos;"),
        (THREAD_ZERO, "VP.seq_lens[0] = VP.pos + 1;"),
        (THREAD_ZERO, "VP.slot_mapping[0] = VP.pos;"),
        ((("for", "inti=threadIdx.x;i<n8*VP.nrep;i+=blockDim.x"),), "dst[i] = src[i % n8];"),
    ],
}


def check_control(fs):
    """Require the reviewed barrier paths and native writer statements."""
    errors = []
    for name in sorted(BARRIERS.keys() | NATIVE_STORES.keys()):
        try:
            stmts = Parser(fs.get(name, "")).sequence()
            parsed = list(leaves(stmts))
            entries = [(path, stmt.text) for path, stmt in parsed]
        except ValueError as error:
            errors.append(f"{name}: control parse failed: {error}")
            continue
        # No path can leave a protocol early. The reduction's final return is safe.
        for index, (path, stmt) in enumerate(parsed):
            jumps = set(stmt.lexical) & {"return", "break", "continue", "goto", "throw"}
            if jumps and not (name in ("etx_block_sum", "etx_block_max")
                              and index == len(entries) - 1 and not path and stmt.text == "returnt;"):
                errors.append(f"{name}: unsupported control exit")
        if name in BARRIERS:
            site, expected = BARRIERS[name]
            actual = [path for path, text in entries if text == "__syncthreads();"]
            hidden = any("__syncthreads" in tokens(text) and text != "__syncthreads();"
                         for _, text in entries)
            # Keep the mean barrier before all work in the LayerNorm branch.
            mean_entries = [text for path, text in entries if path[:1] == MODE]
            late_mean = name == "dsl_stage_norm_at" and mean_entries[:1] != ["__syncthreads();"]
            if actual != expected or hidden or late_mean:
                errors.append(f"{name}:{site}: barrier control path differs from the required protocol")
        for path, store in NATIVE_STORES.get(name, []):
            if entries.count((path, compact(store))) != 1:
                errors.append(f"{name}: native output needs one writer on its required path: {store}")
    return errors
