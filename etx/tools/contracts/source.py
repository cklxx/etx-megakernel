"""Source: acquire contract."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from etx.tools.contracts.source_control import compact

from .schema import ALLOWED_DEFINES, ARRIVE_FLUSH, ATOMIC_SUB_DEVICE, CONSUMER, CONSUMER_TILE, CONTRACT, CROSS, CROSS_USERS, IMP_TOUCH, LOWERING, PRE_WAIT, PRODUCER, PRODUCER_TILE, Refusal, WAIT_SCOPE

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


def check_atomic_sub_device(amdgcn: str, errors) -> None:
    got = function_text(amdgcn, "static __device__ __forceinline__ int32_t etx_atomic_sub_device(")
    if got is None or compact(got) != compact(ATOMIC_SUB_DEVICE):
        errors.append("runtime: etx_atomic_sub_device differs from the contract's text "
                      "(requires relaxed agent-scope fetch-sub)")


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
