"""Shared LLVM text parsing and kernel argument ABI records."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

ARENA = "etx_lds_arena"                    # the megakernel's shared LDS arena (etx/import.h)
SLOT_BARRIER = "etx_imp_slot_barrier"      # void (i32 slot, i32 nwaves), etx/import.h

_SIZES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "half": 2, "bfloat": 2, "float": 4, "double": 8}


class ImportError_(Exception):
    pass


@dataclass
class Param:
    ty: str          # IR type, e.g. "i32", "ptr addrspace(1)"
    attrs: str       # parameter attributes as written (kept on the body)
    name: str        # %0 ...
    size: int
    align: int
    offset: int = 0


@dataclass
class ImportInfo:
    export: str
    kernel: str
    block: tuple[int, int, int]
    threads: int
    slots: int
    params: list[Param] = field(default_factory=list)
    grid_offset: int = 0                   # byte offset of u32 grid[3] + u32 nblocks in the argument block
    arg_bytes: int = 0
    lds_bytes_per_slot: int = 0
    lds_bytes: int = 0                     # arena bytes this import needs (all slots)
    barriers: int = 0
    slot_threads: int = 0                  # threads per slot (the kernel's block rounded up to whole waves)
    slot_barrier: bool = False             # s_barrier replaced by the slot barrier

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["params"] = [p.__dict__ for p in self.params]
        return d


# ------------------------------------------------------------------------- module text helpers

def _split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def _matching_paren(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ImportError_("unbalanced parentheses in a function header")


def _functions(text: str) -> list[tuple[int, int, str]]:
    """(start, end, name) of every `define` block (the header line through the closing '}' line)."""
    out = []
    for m in re.finditer(r"^define [^\n]*?@(\"[^\"]+\"|[\w.$]+)\(", text, flags=re.M):
        end = text.find("\n}\n", m.start())
        if end < 0:
            end = len(text) - 2
        out.append((m.start(), end + 3, m.group(1).strip('"')))
    return out


def _param_type(p: str) -> tuple[str, str, str]:
    """'ptr addrspace(1) noalias noundef readonly %6' -> ('ptr addrspace(1)', 'noalias noundef readonly', '%6')."""
    p = p.strip()
    name = ""
    m = re.search(r"\s(%[\w.\"-]+)$", p)
    if m:
        name = m.group(1); p = p[: m.start()]
    m = re.match(r"(ptr(?: addrspace\(\d+\))?|<\d+ x [\w ()]+>|i\d+|half|bfloat|float|double)(?=\s|$)(.*)", p)
    if not m:
        raise ImportError_(f"unsupported kernel parameter type: {p!r}")
    return m.group(1), m.group(2).strip(), name


def _size_align(ty: str) -> tuple[int, int]:
    if ty.startswith("ptr"):
        return 8, 8
    m = re.match(r"<(\d+) x (\w+)>", ty)
    if m:
        n, e = int(m.group(1)), _SIZES[m.group(2)]
        sz = n * e
        al = 1 << (sz - 1).bit_length()
        return sz, min(al, 16)
    if ty in _SIZES:
        return _SIZES[ty], _SIZES[ty]
    raise ImportError_(f"no layout for parameter type {ty}")


# ------------------------------------------------------------------------- the transformation

_ID_CALL = re.compile(
    r"^(\s*)(%[\w.\"-]+) = (?:tail |musttail |notail )?call (i64|i32) "
    r"@(__ockl_get_(?:group_id|local_id|local_size|num_groups|global_id|global_size|global_offset)"
    r"|llvm\.amdgcn\.(?:workgroup|workitem)\.id\.[xyz])\((?:i32 (?:noundef )?(\d+))?\)[^\n]*$",
    flags=re.M)
_BARRIER = re.compile(r"^(\s*)(?:tail |notail )?call void @llvm\.amdgcn\.s\.barrier\(\)[^\n]*$", flags=re.M)
_REFUSE = re.compile(r"@llvm\.amdgcn\.(implicitarg\.ptr|dispatch\.ptr|kernarg\.segment\.ptr|queue\.ptr|"
                     r"lds\.kernel\.id|dispatch\.id|s\.barrier\.signal|cluster\.id)|@__ockl_get_(enqueued|work_dim)")


def _lds_globals(text: str) -> list[tuple[str, int, int, str]]:
    """(name, bytes, align, full line) of every static LDS variable."""
    types = {m.group(1): m.group(2) for m in re.finditer(r'^(%"[^"]+"|%[\w.$]+) = type (\{[^\n]*\}|<\{[^\n]*\}>)', text, flags=re.M)}
    out = []
    for m in re.finditer(r'^@("[^"]+"|[\w.$]+) = [^\n]*?addrspace\(3\) global (\[[^\n]*?\]|%"[^"]+"|[%\w.<> ]+?) (undef|poison|zeroinitializer)[^\n]*?(?:, align (\d+))?[^\n]*$',
                         text, flags=re.M):
        name, ty = m.group(1).strip('"'), m.group(2)
        out.append((name, _type_bytes(ty, types)[0], int(m.group(4) or 4), m.group(0)))
    if re.search(r"addrspace\(3\) global \[0 x", text):
        raise ImportError_("dynamic (extern) LDS is not supported yet")
    return out


def _type_bytes(ty: str, types: dict[str, str]) -> tuple[int, int]:
    """(size, align) of an IR type; named structs are looked up in the module's type table."""
    ty = ty.strip()
    m = re.match(r"\[(\d+) x (.+)\]$", ty)
    if m:
        sz, al = _type_bytes(m.group(2), types)
        return int(m.group(1)) * sz, al
    m = re.match(r"<(\d+) x (.+)>$", ty)
    if m:
        sz, al = _type_bytes(m.group(2), types)
        n = int(m.group(1)) * sz
        return n, min(1 << (n - 1).bit_length(), 16)
    if ty in _SIZES:
        return _SIZES[ty], _SIZES[ty]
    if ty == "ptr" or ty.startswith("ptr "):
        return 8, 8
    if ty in types:
        return _type_bytes(types[ty], types)
    m = re.match(r"(<)?\{(.*)\}(>)?$", ty)
    if m:
        packed = bool(m.group(1))
        off, al_max = 0, 1
        for f in _split_top(m.group(2)):
            sz, al = _type_bytes(f, types)
            if packed:
                al = 1
            off = (off + al - 1) // al * al + sz
            al_max = max(al_max, al)
        return (off + al_max - 1) // al_max * al_max, al_max
    raise ImportError_(f"LDS variable of an unsupported type: {ty}")
