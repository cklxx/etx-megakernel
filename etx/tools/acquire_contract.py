"""Memory contract check for one imported producer/consumer pair: attn_reduce -> wvsplitk_y2 (o_proj).

docs/core/CONTRACT-acquire-2026-10-05.md states the contract. This tool checks that a build still matches it:

  python -m etx.tools.acquire_contract --build <build dir> --cflags "<device -D flags of that build>"

The build directory holds imports.bc (or pass --ir <file.ll|file.bc>), megakernel_d0.hip, imp_tiles.hip,
etx_lowering.h and plan.json. The tool checks:

  1. IR: the wrapper argument offsets and types, every specialisation body's parameter attributes, an allowlist of
     opcodes and callees, and the root parameter of every global access. A global pointer must come from a
     parameter through getelementptr, phi or select. Anything else is an unknown alias pattern.
  2. Bindings: every operator argument that names attn_out in etx.importer.spec.
  3. Emitted source: per layer, the producer's arrive and the consumer's wait use the same DEVICE event, and the
     code order is wait -> ACQUIRE_DEVICE -> barrier -> body -> all-wave drain -> barrier -> thread 0 arrive.
  4. Plan: event scope, producer count and placement, consumer inputs.
  5. Lowering and runtime: the DEVICE release / acquire / poll / arrive macros and the arrive function text.
  6. Compile flags: only known -D macros. Any flag that removes a fence fails.

Fail-closed: an unknown opcode, callee, pointer origin, attribute, source line or flag is an error.
Exit 0: contract holds. Exit 1: violations, one per line. Exit 2: refused (an input is missing).
This is a static check of what was emitted. It does not prove the GPU memory model.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from etx.tools.source_control import compact

ROOT = Path(__file__).resolve().parents[2]


class Refusal(Exception):
    pass


@dataclass(frozen=True)
class Param:
    offset: int
    ty: str            # "ptr" (addrspace 1) or "i32"
    name: str
    binding: str       # the host expression (etx/importer/spec.py ARGUMENTS)
    access: str = ""   # ptr only: "read", "write" or "none"
    noalias: bool = False
    kinds: frozenset = frozenset()   # load / store kinds the body must show: plain, nontemporal, memcpy, store
    role: str = ""     # weight, activation, step-input, null


def P(offset, name, binding, access, noalias, kinds, role):
    return Param(offset, "ptr", name, binding, access, noalias, frozenset(kinds), role)


def I(offset, name, binding):
    return Param(offset, "i32", name, binding)


PRODUCER, CONSUMER = "attn_reduce", "wvsplitk_y2"
PRODUCER_TILE, CONSUMER_TILE = "etx_tile_attn_reduce", "etx_tile_wvsplitk_y2"
CROSS = "attn_out"

# Observed in the Qwen3-30B-A3B and Mixtral-8x7B imported builds (build/rocm22/*_xcd_qknr, mixtral-8x7b_imported).
CONTRACT = {
    PRODUCER: (
        P(0, "out", "attn_out", "write", True, {"store"}, "activation"),
        P(8, "exp_sums", "exp_sums", "read", True, {"plain"}, "activation"),
        P(16, "max_logits", "max_logits", "read", True, {"plain"}, "activation"),
        P(24, "tmp_out", "tmp_out", "read", True, {"plain"}, "activation"),
        P(32, "seq_lens", "seq_lens", "read", True, {"plain"}, "step-input"),
        P(40, "ptr5", "nullptr", "read", True, {"plain"}, "null"),
        I(48, "max_num_partitions", "parts"),
        P(56, "ptr7", "nullptr", "read", True, {"plain"}, "null"),
    ),
    CONSUMER: (
        I(0, "K", "x.K"), I(4, "Kbp", "x.K"), I(8, "Kap", "x.K"), I(12, "M", "xM"), I(16, "Bx", "b ? xM : 1"),
        I(20, "By", "1"),
        P(24, "B", "w", "read", False, {"nontemporal"}, "weight"),
        P(32, "A", "in", "read", True, {"memcpy"}, "activation"),
        P(40, "BIAS", "b", "read", True, {"plain"}, "null"),
        P(48, "C", "out", "write", False, {"store"}, "activation"),
        I(56, "WvPrGrp", "x.wv"), I(60, "CuCount", "x.gx * x.gy * x.gz"),
    ),
}
GRID_OFFSETS = (64, 68, 72)
ETX_PARAMS = ("etx.bx", "etx.by", "etx.bz", "etx.gx", "etx.gy", "etx.gz", "etx.tid", "etx.slot")
# Every reference to attn_out in the operator spec: (role, argument index, the export's parameter access).
CROSS_USERS = {("attn", 16): "none", ("reduce", 0): "write"}
ATTN_EXPORT_PREFIX = "attn_mfma"

OPCODES = {
    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr", "and", "or", "xor",
    "fadd", "fsub", "fmul", "fdiv", "fneg", "zext", "sext", "trunc", "fpext", "fptrunc", "sitofp", "uitofp",
    "fptosi", "fptoui", "bitcast", "icmp", "fcmp", "select", "phi", "br", "ret", "freeze", "getelementptr",
    "extractelement", "insertelement", "shufflevector", "load", "store", "call", "fence", "unreachable",
}
CALLEES = {
    "llvm.umin.i32", "llvm.umax.i32", "llvm.smin.i32", "llvm.smax.i32", "llvm.maxnum.f32", "llvm.minnum.f32",
    "llvm.exp.f32", "llvm.exp2.f32", "llvm.amdgcn.exp2.f32", "llvm.fabs.f32", "llvm.amdgcn.s.barrier",
    "llvm.amdgcn.wave.barrier", "llvm.amdgcn.sched.barrier", "llvm.amdgcn.update.dpp.i32",
    "llvm.amdgcn.mfma.f32.4x4x4bf16.1k", "llvm.amdgcn.mbcnt.lo", "llvm.amdgcn.mbcnt.hi",
    "llvm.amdgcn.ds.bpermute", "llvm.memcpy.p3.p1.i64", "llvm.lifetime.start.p5", "llvm.lifetime.end.p5",
    "llvm.assume", "etx_imp_slot_barrier",
}
WG_FENCE = re.compile(r'^fence syncscope\("workgroup"\) (acquire|release)$')

# -D macros a build of this pair may carry. Anything else fails: ETX_COHERENT_DEVICE_DATA, ETX_VL_UC and
# ETX_NO_ALLWAVE_DRAIN remove a fence this contract depends on; an unknown macro may do the same.
ALLOWED_DEFINES = {"ETX_DSL_NT", "ETX_DSL_VECNORM", "ETX_VL_CHECK", "ETX_DSL_PREP_EARLY", "ETX_DSL_UNIFORM",
                   "ETX_NO_PRE_WAIT", "ETX_NO_TOUCH", "ETX_BACKOFF_STMT"}

LOWERING = {
    "ETX_RELEASE_DOMAIN()": 'do { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); } while (0)',
    "ETX_RELEASE_DEVICE()": 'do { asm volatile("buffer_wbl2 sc1\\n\\ts_waitcnt vmcnt(0)" ::: "memory"); } while (0)',
    "ETX_ACQUIRE_DEVICE()": 'do { asm volatile("buffer_inv sc1" ::: "memory"); } while (0)',
    "ETX_POLL_DEVICE(ptr)": "(__hip_atomic_load((ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))",
    "ETX_ARRIVE_DEVICE(ptr)": "(__hip_atomic_fetch_sub((ptr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))",
}

ARRIVE_FLUSH = r"""
static __device__ __forceinline__ int32_t etx_arrive_flush_DEVICE_s(const etx_params& p, int32_t idx, uint32_t domain, int32_t share) {
  if (!p.ev_share || share <= 1) { ETX_RELEASE_DEVICE(); return ETX_ARRIVE_DEVICE(p.events + ETX_EVW(idx)) - 1; }
  ETX_RELEASE_DOMAIN();
  const int32_t old = atomicAdd(p.ev_sub + (size_t)domain * p.event_words + idx, 1);
  if (old + 1 != share) return 1;
  ETX_RELEASE_DEVICE();
  return __hip_atomic_fetch_sub(p.events + ETX_EVW(idx), share, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - share;
}
"""
WAIT_SCOPE = r"""
  static __device__ __forceinline__ void etx_wait_##S(etx_event* e, const int32_t* abort_flag) {
    uint32_t n = 0;
    while (ETX_POLL_##S(e) > 0) {
      ETX_BACKOFF();
      if ((++n & 255u) == 0 && ETX_POLL_DEVICE(abort_flag) != 0) return;
    }
  }
"""
PRE_WAIT = ('extern "C" __device__ __attribute__((always_inline)) void etx_pre_wvsplitk_y2(const etx_ctx* c) '
            '{ etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), 128); }')
IMP_TOUCH = r"""
static __device__ __forceinline__ void etx_imp_touch(const unsigned char* a, int bytes) {
  const __attribute__((address_space(4))) uint32_t* a4 = (const __attribute__((address_space(4))) uint32_t*)(uintptr_t)a;
  uint32_t acc = a4[(bytes - 4) / 4];
  for (int o = 0; o < bytes; o += 64) acc ^= a4[o / 4];
  asm volatile("" :: "s"(acc));
}
"""


# ---------------------------------------------------------------- IR

@dataclass
class Function:
    name: str
    header: str
    params: list            # (type, attrs tuple, name)
    lines: list = field(default_factory=list)


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


# ---------------------------------------------------------------- bindings

def check_bindings(errors, spec=None):
    """Every operator argument naming attn_out, and the host dispatch of o_proj, against CROSS_USERS."""
    if spec is None:
        from etx.importer import spec
    users = {}
    for role, args in spec.ARGUMENTS.items():
        for k, (_, expr) in enumerate(args):
            if re.search(rf"\b{CROSS}\b", expr):
                users[(role, k)] = expr
    for k, p in enumerate(CONTRACT[PRODUCER]):
        if spec.ARGUMENTS["reduce"][k][1] != p.binding:
            errors.append(f"bindings: reduce argument {k} is {spec.ARGUMENTS['reduce'][k][1]!r}, contract {p.binding!r}")
    for k, p in enumerate(CONTRACT[CONSUMER]):
        if spec.ARGUMENTS["gemv"][k][1] != p.binding:
            errors.append(f"bindings: gemv argument {k} is {spec.ARGUMENTS['gemv'][k][1]!r}, contract {p.binding!r}")
    if set(users) != set(CROSS_USERS):
        errors.append(f"bindings: arguments naming {CROSS} are {sorted(users)}, the contract lists {sorted(CROSS_USERS)}")
    gemvs = [ln.strip() for ln in spec.HOST_DISPATCH.splitlines() if re.search(rf"\b{CROSS}\b", ln)]
    if gemvs != ['else if (r == "o_proj") gemv(W(pl + "wo"), attn_out, o, nullptr);']:
        errors.append(f"bindings: host dispatch lines naming {CROSS}: {gemvs}")
    return sorted(users)


# ---------------------------------------------------------------- emitted source

P1 = "switch (t.type) {   // phase 1: immediates, waits, acquire"
P2 = "etx_call_body(sym, &ctx);   // phase 2: the only call site of each body"
P3 = "switch (t.type) {   // phase 3: releases, arrives, pushes"
BETWEEN = [r"default: break;", r"\}", r"__syncthreads\(\);", re.escape(P2), r"#ifndef ETX_NO_ALLWAVE_DRAIN",
           r"ETX_RELEASE_DOMAIN\(\);   // every wave drains its own stores before the barrier: .*", r"#endif",
           r"__syncthreads\(\);", r"if \(threadIdx\.x != 0\) return;", re.escape(P3)]
TRACE = re.compile(r"^if \(tr && threadIdx\.x == 0\) (tr\[1\] = \(uint64_t\)ETX_TIMER\(\);"
                   r"|\{ tr\[2\] = \(uint64_t\)ETX_TIMER\(\); tr\[3\] = worker; \})$")


def cases(lines: list[str]) -> dict[str, tuple[int, list[str]]]:
    out, cur = {}, None
    for s in lines:
        m = re.match(r"^case (\d+): \{ // (\S+)", s)
        if m:
            cur = m.group(2)
            if cur in out:
                raise Refusal(f"source: case {cur} appears twice in one phase")
            out[cur] = (int(m.group(1)), [s])
        elif s == "default: break;":
            break
        elif cur is not None:
            out[cur][1].append(s)
    return out


CONSUMER_BLOCK = [
    r"case (?P<id>\d+): \{ // (?P<name>o_proj_\d+) \[static\] grid=\((?P<grid>\d+),\)",
    r"ctx\.cst\[0\] = \d+; ctx\.cst\[1\] = \d+; ctx\.cst\[2\] = -?\d+; ctx\.cst\[3\] = \d+;",
    r"#ifndef ETX_NO_PRE_WAIT?",
    r"if \(threadIdx\.x < 64\) etx_pre_wvsplitk_y2\(&ctx\);   // before the wait: does not depend on the event?",
    r"#endif?",
    r"if \(threadIdx\.x == 0\) \{",
    r"// wait (?P<event>E_\w+) via '[^']*' \(DEVICE\)",
    r"#define ETX_TARGET\(idx\) etx_wait_DEVICE\(p\.events \+ ETX_EVW\((?P<word>\d+) \+ \(idx\)\), p\.ctrl_abort\)",
    r"ETX_TARGET\(0\);",
    r"#undef ETX_TARGET",
    r"ETX_ACQUIRE_DEVICE\(\);",
    r"\}",
    rf"sym = \d+;   // {CONSUMER_TILE}",
    r"break; \}",
]
PRODUCER_BLOCK = [
    r"case (?P<id>\d+): \{ // (?P<name>reduce_\d+)",
    r"// arrive (?P<event>E_\w+) via '[^']*' \(DEVICE\)",
    r"#define ETX_TARGET\(idx\) \(void\)etx_arrive_flush_DEVICE_s\(p, (?P<word>\d+) \+ \(idx\), domain, (?P<share>\d+)\)",
    r"ETX_TARGET\(0\);",
    r"#undef ETX_TARGET",
    r"break; \}",
]


def match_block(template, lines, where, errors):
    """Lines must match the template in order. A template entry ending in '?' is an optional line."""
    got, i = {}, 0
    for pat in template:
        optional = pat.endswith("?") and not pat.endswith("\\?")
        rx = re.compile("^" + (pat[:-1] if optional else pat) + "$")
        m = rx.match(lines[i]) if i < len(lines) else None
        if m:
            got.update(m.groupdict())
            i += 1
        elif not optional:
            errors.append(f"{where}: expected {pat!r}, found {lines[i] if i < len(lines) else 'end of block'!r}")
            return None
    if i != len(lines):
        errors.append(f"{where}: unexpected line {lines[i]!r}")
        return None
    return got


def check_source(src: str, plan: dict, errors) -> list[dict]:
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    for macro in ("ETX_COHERENT_DEVICE_DATA", "ETX_NO_ALLWAVE_DRAIN", "ETX_VL_UC"):
        if any(re.match(rf"#define {macro}\b", s) for s in lines):
            errors.append(f"source: #define {macro} removes a fence this contract needs")
    idx = {}
    for name, marker in (("phase 1", P1), ("phase 2", P2), ("phase 3", P3)):
        hits = [k for k, s in enumerate(lines) if s == marker]
        if len(hits) != 1:
            errors.append(f"source: {name} marker found {len(hits)} times, expected once")
            return []
        idx[name] = hits[0]
    if not idx["phase 1"] < idx["phase 2"] < idx["phase 3"]:
        errors.append("source: phase markers out of order")
        return []
    phase1 = cases(lines[idx["phase 1"] + 1:idx["phase 2"]])
    phase3 = cases(lines[idx["phase 3"] + 1:])
    end1 = max(k for k in range(idx["phase 1"], idx["phase 2"]) if lines[k] == "default: break;")
    between = [s for s in lines[end1:idx["phase 3"] + 1] if not TRACE.match(s)]
    if len(between) != len(BETWEEN) or not all(re.fullmatch(p, s) for p, s in zip(BETWEEN, between)):
        errors.append(f"source: the code between the waits and the arrives is not wait -> barrier -> body -> "
                      f"all-wave drain -> barrier -> thread 0: {between}")
    types = plan.get("types", {})
    pairs = []
    for name, (cid, block) in sorted(phase1.items()):
        m = re.fullmatch(r"o_proj_(\d+)", name)
        if not m or not block[-2].endswith(f"// {CONSUMER_TILE}"):
            continue
        layer = m.group(1)
        prod = f"reduce_{layer}"
        c = match_block(CONSUMER_BLOCK, block, f"source {name} (phase 1)", errors)
        if prod not in phase3 or prod not in phase1:
            errors.append(f"source: {name} reads {CROSS} but {prod} has no phase 1 and phase 3 case")
            continue
        if not phase1[prod][1][-2].endswith(f"// {PRODUCER_TILE}"):
            errors.append(f"source: {prod} runs {phase1[prod][1][-2]!r}, not {PRODUCER_TILE}")
            continue
        p = match_block(PRODUCER_BLOCK, phase3[prod][1], f"source {prod} (phase 3)", errors)
        if c is None or p is None:
            continue
        if (c["event"], c["word"]) != (p["event"], p["word"]):
            errors.append(f"source: {prod} arrives on {p['event']} word {p['word']}, "
                          f"{name} waits on {c['event']} word {c['word']}")
            continue
        others = [n for n, (_, b) in phase3.items() if n != prod
                  and any(re.search(rf"\(p, {p['word']} \+ \(idx\)|ETX_EVW\({p['word']} \+ \(idx\)\)", s) for s in b)]
        if others:
            errors.append(f"source: {others} also arrive on word {p['word']} ({p['event']})")
        for n, k in ((name, c["id"]), (prod, phase3[prod][0])):
            t = types.get(n)
            if t is None or t.get("id") != int(k):
                errors.append(f"plan: type {n} id {t and t.get('id')} differs from source case {k}")
        pairs.append(dict(layer=int(layer), producer=prod, consumer=name, event=p["event"], word=int(p["word"]),
                          share=int(p["share"]), consumer_grid=int(c["grid"])))
    if not pairs:
        errors.append(f"source: no {PRODUCER} -> {CONSUMER} pair (reduce_L -> o_proj_L running {CONSUMER_TILE})")
    return sorted(pairs, key=lambda p: p["layer"])


def check_plan(plan: dict, pairs: list[dict], errors) -> None:
    events = {e["name"]: e for e in plan.get("events", [])}
    types = plan.get("types", {})
    descs = plan.get("descs", [])
    task_in = plan.get("task_in", [])
    ev_inst = {e[0]: e for e in plan.get("ev_inst", [])}
    for pr in pairs:
        where = f"plan {pr['producer']} -> {pr['consumer']}"
        e = events.get(pr["event"])
        if e is None:
            errors.append(f"{where}: event {pr['event']} missing")
            continue
        tp, tc = types.get(pr["producer"]), types.get(pr["consumer"])
        if tp is None or tc is None:
            errors.append(f"{where}: task type missing from the plan")
            continue
        if tp.get("symbol") != PRODUCER_TILE or tc.get("symbol") != CONSUMER_TILE:
            errors.append(f"{where}: symbols {tp.get('symbol')}, {tc.get('symbol')}")
        if tp.get("mode") != "static" or tc.get("mode") != "static":
            errors.append(f"{where}: modes {tp.get('mode')}, {tc.get('mode')}: the contract covers static tasks only")
        prods = [d for d in descs if d["type"] == tp["id"]]
        cons = [d for d in descs if d["type"] == tc["id"]]
        if e.get("scope") != "DEVICE" or e.get("offset") != pr["word"] or e.get("shape") != [1] \
                or e.get("runtime_init") or e.get("counts") != [len(prods)]:
            errors.append(f"{where}: event {e} is not a DEVICE event on word {pr['word']} counting {len(prods)} producers")
        inst = ev_inst.get(pr["event"])
        if inst is None or sorted(inst[2]) != sorted(d["id"] for d in prods):
            errors.append(f"{where}: the producers of {pr['event']} are {inst and inst[2]}, not the {pr['producer']} tasks")
        if len(cons) != pr["consumer_grid"] or any(task_in[d["id"]] != [pr["word"]] for d in cons):
            errors.append(f"{where}: not every {pr['consumer']} task waits on word {pr['word']} alone")
        per_domain = {}
        for d in prods:
            per_domain[d["domain"]] = per_domain.get(d["domain"], 0) + 1
        if pr["share"] > 1 and set(per_domain.values()) != {pr["share"]}:
            errors.append(f"{where}: arrive share {pr['share']} differs from producers per domain {per_domain}")
        pr["producer_domains"] = sorted(per_domain)
        pr["consumer_domains"] = sorted({d["domain"] for d in cons})
        pr["crosses_xcd"] = any(dc != dp for dc in pr["consumer_domains"] for dp in pr["producer_domains"])


# ---------------------------------------------------------------- lowering, runtime, flags

def top_level_defines(text: str) -> dict[str, str]:
    out, depth = {}, 0
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"#\s*if", s):
            depth += 1
        elif re.match(r"#\s*endif", s):
            depth -= 1
        elif depth == 0:
            m = re.match(r"#define (\w+(?:\([^)]*\))?) (.*)$", s)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def check_lowering(text: str, errors) -> None:
    d = top_level_defines(text)
    if d.get("ETX_ARCH") != '"gfx942"':
        errors.append(f"lowering: ETX_ARCH is {d.get('ETX_ARCH')}, the contract is for gfx942")
    for k, v in LOWERING.items():
        if d.get(k) != v:
            errors.append(f"lowering: {k} is {d.get(k)!r}, the contract needs {v!r}")
    if "#ifdef ETX_ACQUIRE_DEVICE_RELAXED" in text:
        errors.append("lowering: a relaxed DEVICE acquire exists; the contract has no condition for it")


def function_text(text: str, start: str) -> str | None:
    k = text.find(start)
    if k < 0:
        return None
    end = text.find("\n}", k)
    return None if end < 0 else text[k:end + 2]


def check_runtime(primitives: str, imp: str, imp_tiles: str, errors) -> None:
    strip = lambda s: re.sub(r"//[^\n]*", "", s)
    got = function_text(primitives, "static __device__ __forceinline__ int32_t etx_arrive_flush_DEVICE_s(")
    if got is None or compact(strip(got)) != compact(ARRIVE_FLUSH):
        errors.append("runtime: etx_arrive_flush_DEVICE_s differs from the contract's text (re-derive the event order)")
    s = strip(primitives).replace("\\\n", "\n")
    if compact(WAIT_SCOPE) not in compact(s):
        errors.append("runtime: etx_wait_<scope> differs from the contract's text")
    got = function_text(imp, "static __device__ __forceinline__ void etx_imp_touch(")
    if got is None or compact(strip(got)) != compact(IMP_TOUCH):
        errors.append("runtime: etx_imp_touch differs from the contract's text (the pre-wait read set changed)")
    if PRE_WAIT not in [ln.strip() for ln in imp_tiles.splitlines()]:
        errors.append("imp_tiles: etx_pre_wvsplitk_y2 is not the argument-block touch the contract allows before the wait")


def check_cflags(cflags: str, errors) -> list[str]:
    defs = re.findall(r"(?:^|\s)-D\s*(\w+)", cflags)
    for flag in re.findall(r"(?:^|\s)-(U|include\b|imacros\b)", cflags):
        errors.append(f"cflags: -{flag} is not allowed (it can change the lowering macros)")
    for d in defs:
        if d not in ALLOWED_DEFINES:
            errors.append(f"cflags: -D{d} is not in the contract's allowlist")
    return defs


# ---------------------------------------------------------------- main

def run(build: Path | None, ir: Path | None, cflags: str | None, runtime: Path) -> tuple[list[str], dict]:
    if cflags is None:
        raise Refusal("missing input: --cflags. A -DETX_COHERENT_DEVICE_DATA or -DETX_NO_ALLWAVE_DRAIN on the device "
                      "compile removes a fence and leaves no trace in the build directory")
    files = {}
    for name in ("megakernel_d0.hip", "imp_tiles.hip", "etx_lowering.h", "plan.json"):
        path = (build / name) if build else None
        if path is None or not path.is_file():
            raise Refusal(f"missing input: {path or name}")
        files[name] = path.read_text()
    ir = ir or (build / "imports.bc")
    for name in ("primitives.h", "import.h"):
        if not (runtime / name).is_file():
            raise Refusal(f"missing input: {runtime / name}")
    errors: list[str] = []
    report = dict(cflags=check_cflags(cflags, errors))
    report["ir"] = check_ir(load_ir(ir), errors)
    report["bindings"] = check_bindings(errors)
    plan = json.loads(files["plan.json"])
    pairs = check_source(files["megakernel_d0.hip"], plan, errors)
    check_plan(plan, pairs, errors)
    report["pairs"] = pairs
    check_lowering(files["etx_lowering.h"], errors)
    check_runtime((runtime / "primitives.h").read_text(), (runtime / "import.h").read_text(),
                  files["imp_tiles.hip"], errors)
    return errors, report


def summary(report: dict, out) -> None:
    for body, inv in report["ir"].items():
        if body == "attn_bodies":
            continue
        export = PRODUCER if PRODUCER in body else CONSUMER
        cells = [f"{CONTRACT[export][k].name}={'+'.join(f'{n} {c}' for c, n in sorted(v.items()))}"
                 for k, v in inv.items()]
        print(f"  {body}: {', '.join(cells)}", file=out)
    pairs = report["pairs"]
    if pairs:
        x = sum(p.get("crosses_xcd", False) for p in pairs)
        p0 = pairs[0]
        print(f"  {len(pairs)} pairs (layers {pairs[0]['layer']}-{pairs[-1]['layer']}); {x} cross XCDs; "
              f"e.g. {p0['producer']} -> {p0['consumer']} on {p0['event']} (word {p0['word']}, share {p0['share']}), "
              f"producer domains {p0.get('producer_domains')}, consumer domains {p0.get('consumer_domains')}", file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m etx.tools.acquire_contract", description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, required=True, help="build directory (examples/vllm_llm/build.sh output)")
    ap.add_argument("--ir", type=Path, help="imports .ll or .bc (default: <build>/imports.bc)")
    ap.add_argument("--cflags", help="the device compile flags of the build, e.g. \"-DETX_DSL_NT -DETX_DSL_VECNORM\"")
    ap.add_argument("--runtime", type=Path, default=ROOT / "etx/runtime/include/etx")
    a = ap.parse_args(argv)
    try:
        errors, report = run(a.build.expanduser(), a.ir, a.cflags, a.runtime)
    except Refusal as e:
        print(f"REFUSED: {e}. No verdict.")
        return 2
    for e in errors:
        print(f"VIOLATION: {e}")
    if errors:
        print(f"FAIL: {len(errors)} contract violations. Keep the acquire; do not build the transformed pair.")
        return 1
    print(f"PASS: {PRODUCER} -> {CONSUMER} matches docs/core/CONTRACT-acquire-2026-10-05.md")
    summary(report, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
