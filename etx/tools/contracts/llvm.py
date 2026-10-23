"""Llvm: acquire contract."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from etx.tools.contracts.source_control import compact

from .schema import ATTN_EXPORT_PREFIX, CALLEES, CONSUMER, CONTRACT, CROSS, ETX_PARAMS, Function, GRID_OFFSETS, OPCODES, PRODUCER, Param, Refusal, WG_FENCE

def split_top(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_param(text: str):
    m = re.match(r"^(ptr addrspace\(\d+\)|ptr|i\d+|float|double|half|bfloat)\s+(.*?)(%[\w.]+)$", text)
    if not m:
        raise Refusal(f"unparsed parameter {text!r}")
    attrs = tuple(re.findall(r"\w+(?:\([^)]*\))?", m.group(2)))
    return m.group(1), attrs, m.group(3)[1:]


def parse_module(text: str) -> dict[str, Function]:
    funcs, cur = {}, None
    for line in text.splitlines():
        if line.startswith("define "):
            m = re.match(r"^define [^@]*@([\w.$]+)\((.*)\)[^()]*\{\s*$", line)
            try:
                params = [parse_param(p) for p in split_top(m.group(2))] if m else None
            except Refusal:
                params = None
            name = m.group(1) if m else re.search(r"@([\w.$]+)\(", line).group(1)
            cur = Function(name, line, params)   # params None: unparsed, an error only if the contract reads it
            funcs[cur.name] = cur
        elif line == "}":
            cur = None
        elif cur is not None:
            cur.lines.append(line)
    return funcs


def statements(fn: Function):
    """(line number in the body, instruction text without metadata and trailing comments)."""
    for n, raw in enumerate(fn.lines, 1):
        s = raw.split(";", 1)[0].strip() if not raw.lstrip().startswith(";") else ""
        s = re.sub(r",\s*![\w.]+ !\d+", "", s)
        if not s or re.match(r"^[\w.$]+:$", s):
            continue
        yield n, s, raw


def ll_type(p: Param) -> str:
    return "ptr addrspace(1)" if p.ty == "ptr" else p.ty


def check_wrapper(funcs, export, errors) -> list[str]:
    name = f"etx_imp_{export}"
    fn = funcs.get(name)
    if fn is None:
        errors.append(f"IR: export {name} is not defined")
        return []
    contract = CONTRACT[export]
    if fn.params is None or [(t, n) for t, _, n in fn.params] != [("ptr", "args"), ("i32", "blk"), ("i32", "tid"),
                                                                   ("i32", "slot")]:
        errors.append(f"IR {name}: wrapper signature {fn.params} is not (args, blk, tid, slot)")
    want = ["%args4 = addrspacecast ptr %args to ptr addrspace(4)"]
    for k, p in enumerate(contract):
        want += [f"%a{k}.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {p.offset}",
                 f"%a{k} = load {ll_type(p)}, ptr addrspace(4) %a{k}.p, align {8 if p.ty == 'ptr' else 4}"]
    for g, off in zip(("gx", "gy", "gz"), GRID_OFFSETS):
        want += [f"%{g}.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {off}",
                 f"%{g} = load i32, ptr addrspace(4) %{g}.p, align 4"]
    want += ["%bx = urem i32 %blk, %gx", "%b1 = udiv i32 %blk, %gx", "%by = urem i32 %b1, %gy",
             "%bz = udiv i32 %b1, %gy"]
    args = ", ".join([f"{ll_type(p)} %a{k}" for k, p in enumerate(contract)]
                     + [f"i32 %{v}" for v in ("bx", "by", "bz", "gx", "gy", "gz", "tid", "slot")])
    call = re.compile(rf"^call void @({re.escape(name)}\.body\.s\d+)\({re.escape(args)}\)$")
    other = re.compile(r"^(switch i32 %slot, label %s0 \[|i32 \d+, label %s\d+|\]|br label %done|ret void)$")
    got = [s for _, s, _ in statements(fn)]
    if got[:len(want)] != want:
        for k, (w, g) in enumerate(zip(want, got)):
            if w != g:
                errors.append(f"IR {name}: argument load {k}: expected {w!r}, found {g!r}")
                break
        else:
            errors.append(f"IR {name}: wrapper has {len(got)} statements, the argument block needs {len(want)}")
        return []
    bodies = []
    for s in got[len(want):]:
        m = call.match(s)
        if m:
            bodies.append(m.group(1))
        elif not other.match(s):
            errors.append(f"IR {name}: unknown wrapper statement {s!r}")
    defined = sorted(n for n in funcs if n.startswith(f"{name}.body."))
    if not bodies:
        errors.append(f"IR {name}: no call to a body with the contract's arguments")
    if sorted(set(bodies)) != defined:
        errors.append(f"IR {name}: bodies called {sorted(set(bodies))} differ from bodies defined {defined}")
    return [b for b in defined if b in bodies]


PTR1 = r"ptr addrspace\(1\)(?: (?:noundef|nonnull|noalias|readonly|writeonly|align \d+|dereferenceable\(\d+\)|captures\([^)]*\)))*"


def check_body(funcs, export, body, errors) -> dict:
    """Parameter attributes, opcode / callee allowlist, and the root parameter of every global access."""
    fn = funcs[body]
    contract = CONTRACT[export]
    where = f"IR {body}"
    np_ = len(contract)
    if fn.params is None:
        errors.append(f"{where}: unparsed definition {fn.header[:120]!r}")
        return {}
    if len(fn.params) != np_ + len(ETX_PARAMS) or [n for _, _, n in fn.params[np_:]] != list(ETX_PARAMS):
        errors.append(f"{where}: {len(fn.params)} parameters, expected {np_} contract + {len(ETX_PARAMS)} etx")
        return {}
    params = {}
    for k, (p, (ty, attrs, pname)) in enumerate(zip(contract, fn.params)):
        if ty != ll_type(p):
            errors.append(f"{where}: parameter {k} ({p.name}) has type {ty}, expected {ll_type(p)}")
            continue
        if p.ty != "ptr":
            continue
        params[pname] = k
        access = {"readonly": "read", "writeonly": "write", "readnone": "none"}
        got = [access[a] for a in attrs if a in access] or ["readwrite"]
        if got != [p.access]:
            errors.append(f"{where}: parameter {k} ({p.name}) is {got[0]}, the contract says {p.access}")
        if ("noalias" in attrs) != p.noalias:
            errors.append(f"{where}: parameter {k} ({p.name}) noalias is {'noalias' in attrs}, the contract says {p.noalias}")
    defs: dict[str, list[str]] = {}
    accesses = []   # (line, pointer, kind)
    for n, s, raw in statements(fn):
        at = f"{where} line {n}"
        m = re.match(r"^(?:%([\w.]+) = )?(?:tail |musttail |notail )?(\w+)", s)
        op = m.group(2)
        if op not in OPCODES:
            errors.append(f"{at}: opcode {op!r} is not in the contract's allowlist: {s[:100]}")
            continue
        if re.search(r"\bvolatile\b|\batomic\b|\basm\b", s):
            errors.append(f"{at}: volatile, atomic or inline asm access: {s[:100]}")
            continue
        if re.search(r"(?<!\w)ptr (?!addrspace)|addrspace\((?![13]\))\d+\)", s):
            errors.append(f"{at}: memory outside the global (1) and LDS (3) address spaces: {s[:100]}")
            continue
        if re.search(r"x ptr", s):
            errors.append(f"{at}: vector of pointers: {s[:100]}")
            continue
        uses = re.findall(PTR1 + r" (%[\w.]+)", s)
        res = m.group(1)
        if op == "fence":
            if not WG_FENCE.match(s):
                errors.append(f"{at}: fence outside the workgroup scope: {s}")
        elif op == "call":
            c = re.search(r"@([\w.$]+)\(", s)
            if not c or c.group(1) not in CALLEES:
                errors.append(f"{at}: call to {c.group(1) if c else 'an indirect target'!s} is not in the allowlist")
            elif c.group(1) == "llvm.memcpy.p3.p1.i64":
                if not re.search(r"\(ptr addrspace\(3\)", s) or len(uses) != 1:
                    errors.append(f"{at}: memcpy is not global -> LDS: {s[:100]}")
                else:
                    accesses.append((n, uses[0][1:], "memcpy"))
            elif uses:
                errors.append(f"{at}: a global pointer is passed to {c.group(1)}")
        elif op == "getelementptr":
            g = re.match(r"^%[\w.]+ = getelementptr (?:inbounds |nuw |nusw )*[^,]+, " + PTR1 + r" (%[\w.]+)", s)
            if g:
                defs[res] = [g.group(1)[1:]]
            elif uses:
                errors.append(f"{at}: unparsed global getelementptr: {s[:100]}")
        elif op == "phi" and s.startswith(f"%{res} = phi ptr addrspace(1) "):
            defs[res] = re.findall(r"\[ %([\w.]+), %[\w.]+ \]", s)
            if re.search(r"\[ (?!%)", s):
                errors.append(f"{at}: constant incoming global pointer: {s[:100]}")
        elif op == "select" and re.search(r", ptr addrspace\(1\) ", s):
            defs[res] = [u[1:] for u in uses]
        elif op == "load":
            ld = re.match(r"^%[\w.]+ = load ([^,]+), " + PTR1 + r" (%[\w.]+)", s)
            if ld and ld.group(1).startswith("ptr"):
                errors.append(f"{at}: a pointer is loaded from memory (unknown alias): {s[:100]}")
            elif ld:
                accesses.append((n, ld.group(2)[1:], "nontemporal" if "!nontemporal" in raw else "plain"))
            elif uses:
                errors.append(f"{at}: unparsed global load: {s[:100]}")
        elif op == "store":
            st = re.match(r"^store (.+), " + PTR1 + r" (%[\w.]+), align \d+$", s)
            if st and (st.group(1).startswith("ptr") or "!nontemporal" in raw):
                errors.append(f"{at}: pointer or non-temporal store: {s[:100]}")
            elif st:
                accesses.append((n, st.group(2)[1:], "store"))
            elif len(uses) > 0 or "addrspace(1)" in s:
                errors.append(f"{at}: unparsed global store: {s[:100]}")
        elif op == "icmp" and uses:
            if not re.match(r"^%[\w.]+ = icmp (eq|ne) ptr addrspace\(1\) %[\w.]+, null$", s):
                errors.append(f"{at}: global pointer compared with something other than null: {s[:100]}")
        elif uses or (res and re.match(rf"^%{re.escape(res)} = \w+ ptr addrspace\(1\)", s)):
            errors.append(f"{at}: global pointer through {op!r} (unknown origin or escape): {s[:100]}")

    memo: dict[str, frozenset] = {}

    def roots(v, seen=()):
        if v in params:
            return frozenset({params[v]})
        if v in memo:
            return memo[v]
        if v in seen:
            return frozenset()
        if v not in defs:
            return None
        out = set()
        for u in defs[v]:
            r = roots(u, seen + (v,))
            if r is None:
                return None
            out |= r
        memo[v] = frozenset(out)
        return memo[v]

    inventory = {k: {} for k, p in enumerate(contract) if p.ty == "ptr"}
    for n, ptr, kind in accesses:
        r = roots(ptr)
        if not r:
            errors.append(f"{where} line {n}: global {kind} through %{ptr} of unknown origin")
            continue
        if len(r) > 1:
            errors.append(f"{where} line {n}: global {kind} through %{ptr} may address parameters {sorted(r)}")
            continue
        k = next(iter(r))
        p = contract[k]
        need = "write" if kind == "store" else "read"
        if p.access != need:
            errors.append(f"{where} line {n}: {kind} through parameter {k} ({p.name}), which the contract marks {p.access}")
        inventory[k][kind] = inventory[k].get(kind, 0) + 1
    for k, kinds in inventory.items():
        p = contract[k]
        if set(kinds) != set(p.kinds):
            errors.append(f"{where}: parameter {k} ({p.name}) is accessed as {sorted(kinds) or 'nothing'}, "
                          f"the contract says {sorted(p.kinds)}")
    return inventory


def check_attn_out_readnone(funcs, errors) -> list[str]:
    """The attention export must not touch attn_out (spec argument 16): its body parameter is readnone."""
    out = []
    for name, fn in funcs.items():
        if not re.match(rf"etx_imp_{ATTN_EXPORT_PREFIX}\w*\.body\.s\d+$", name):
            continue
        if fn.params is None or len(fn.params) <= 16 or "readnone" not in fn.params[16][1]:
            errors.append(f"IR {name}: parameter 16 (bound to {CROSS}) is not readnone")
        out.append(name)
    return out


def load_ir(path: Path) -> str:
    if not path.is_file():
        raise Refusal(f"missing input: IR {path}")
    if path.suffix == ".ll":
        return path.read_text()
    from etx.importer.bundle import llvm_tool
    try:
        r = subprocess.run([llvm_tool("llvm-dis"), str(path), "-o", "-"], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise Refusal(f"missing input: llvm-dis to read {path} ({e}); pass a .ll file or set ETX_LLVM")
    if r.returncode:
        raise Refusal(f"llvm-dis {path}: {r.stderr.strip()[-400:]}")
    return r.stdout


def check_ir(text: str, errors) -> dict:
    funcs = parse_module(text)
    report = {}
    for export in (PRODUCER, CONSUMER):
        for body in check_wrapper(funcs, export, errors):
            report[body] = check_body(funcs, export, body, errors)
    report["attn_bodies"] = check_attn_out_readnone(funcs, errors)
    if not report["attn_bodies"]:
        errors.append(f"IR: no {ATTN_EXPORT_PREFIX}* body: the writer set of {CROSS} cannot be confirmed")
    return report
