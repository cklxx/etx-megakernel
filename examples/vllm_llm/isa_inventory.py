#!/usr/bin/env python3
"""Instruction inventory of every kernel of one vLLM-kernel build, vLLM's form next to ETX's imported form.

  python3 -m examples.vllm_llm.isa_inventory <build dir> [--json out.json]

Reads the assembly examples/vllm_llm/isa_build.sh writes (orig_*.s: vLLM's HIP kernels; imp_wrap.s: each import alone in
a 1024-thread workgroup) and the Triton kernels' cached .amdgcn. Per kernel: registers, LDS, scratch, and static counts
of every operation class, for the whole kernel and for each loop (the loop header comment LLVM writes, up to the branch
back to it). Loop trip counts are not in the assembly; the kernel model (kmodel.py) supplies them from the shapes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from etx.importer.vllm_kernels import SOURCES, TRITON, TRITON_DIR

CLASSES = [
    ("barrier", r"^s_barrier\b"),
    ("vmem_load", r"^(global|buffer|flat)_load"),
    ("vmem_store", r"^(global|buffer|flat)_store"),
    ("vmem_atomic", r"^(global|buffer|flat)_atomic"),
    ("smem_load", r"^s_(load|buffer_load)"),
    ("lds_atomic", r"^ds_(add|sub|inc|dec|min|max|and|or|xor|cmpst|cmpswap|wrxchg)"),
    ("lds_permute", r"^ds_(bpermute|permute|swizzle)"),
    ("lds_read", r"^ds_(read|load)"),
    ("lds_write", r"^ds_(write|store)"),
    ("mfma", r"^v_mfma"),
    ("dpp", r"_dpp\b|row_|quad_perm|row_bcast"),
    ("readlane", r"^v_(readlane|readfirstlane|writelane)"),
    ("wait_vm", r"^s_waitcnt.*vmcnt"),
    ("wait_lgkm", r"^s_waitcnt.*lgkmcnt"),
    ("sleep", r"^s_sleep"),
    ("branch", r"^s_(cbranch|branch)"),
    ("valu", r"^v_"),
    ("salu", r"^s_"),
]
_CLS = [(n, re.compile(p)) for n, p in CLASSES]
_WIDTH = re.compile(r"_(dwordx4|dwordx3|dwordx2|dword|b128|b96|b64|b32|ubyte|ushort|short|byte|d16)")
WBYTES = {"dwordx4": 16, "b128": 16, "dwordx3": 12, "b96": 12, "dwordx2": 8, "b64": 8, "dword": 4, "b32": 4, "ushort": 2,
          "short": 2, "d16": 2, "ubyte": 1, "byte": 1}


def classify(op: str, line: str) -> str:
    for n, rx in _CLS:
        if n == "dpp":
            if rx.search(line):
                return n
            continue
        if rx.search(op):
            return n
    return "other"


def functions(asm: str):
    """(name, body lines, metadata) of every function with a kernel descriptor, and of device functions."""
    out = {}
    for m in re.finditer(r"^(\S+):\s*(?:;.*)?$", asm, flags=re.M):
        name = m.group(1)
        if name.startswith(".") or f".type\t{name},@function" not in asm and f".type {name},@function" not in asm:
            continue
        end = asm.find(".Lfunc_end", m.end())
        body = asm[m.end(): end if end > 0 else len(asm)]
        meta = {}
        kd = re.search(rf"\.amdhsa_kernel {re.escape(name)}\n(.*?)\.end_amdhsa_kernel", asm, flags=re.S)
        if kd:
            for k in ("next_free_vgpr", "accum_offset", "next_free_sgpr", "group_segment_fixed_size", "private_segment_fixed_size"):
                v = re.search(rf"\.amdhsa_{k} (\d+)", kd.group(1))
                if v:
                    meta[k] = int(v.group(1))
        out[name] = (body.split("\n"), meta)
    return out


def count(lines):
    c = {n: 0 for n, _ in CLASSES}
    c["other"] = 0
    c["load_bytes_lane"] = 0
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith((";", ".", "//")) or s.endswith(":"):
            continue
        op = s.split()[0]
        k = classify(op, s)
        c[k] = c.get(k, 0) + 1
        if k == "vmem_load":
            w = _WIDTH.search(op)
            c["load_bytes_lane"] += WBYTES.get(w.group(1), 4) if w else 4
    c["instructions"] = sum(v for k, v in c.items() if k not in ("load_bytes_lane", "instructions"))
    return c


def loops(lines):
    """Innermost-first list of loops: header label, depth, instruction counts of the body (header .. back edge)."""
    heads = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^(\.LBB\S+):.*Loop Header: Depth=(\d+)", ln.strip())
        if m:
            heads[m.group(1)] = (i, int(m.group(2)))
    out = []
    for lab, (i, depth) in heads.items():
        back = max((j for j, ln in enumerate(lines) if j > i and re.search(rf"s_c?branch\w*\s+{re.escape(lab)}\b", ln)), default=None)
        if back is None:
            continue
        out.append({"header": lab, "depth": depth, **count(lines[i:back + 1])})
    return sorted(out, key=lambda d: -d["depth"])


def inventory(build: Path) -> dict:
    res = {}
    orig_asm = {p.stem[5:]: p.read_text() for p in build.glob("orig_*.s")}
    for s in SOURCES:
        base = Path(s.path).stem
        if base not in orig_asm:
            continue
        fns = functions(orig_asm[base])
        for k in s.kernels:
            hit = [n for n in fns if k.match in n]
            if hit:
                lines, meta = fns[hit[0]]
                res.setdefault(k.export, {})["vllm"] = {"symbol": hit[0], "meta": meta, "total": count(lines), "loops": loops(lines)}
    for e, (kname, _) in TRITON.items():
        p = TRITON_DIR / e / f"{kname}.amdgcn"
        if p.exists():
            fns = functions(p.read_text())
            if kname in fns:
                lines, meta = fns[kname]
                res.setdefault(e, {})["vllm"] = {"symbol": kname, "meta": meta, "total": count(lines), "loops": loops(lines)}
    wrap = build / "imp_wrap.s"
    if wrap.exists():
        fns = functions(wrap.read_text())
        for n, (lines, meta) in fns.items():
            m = re.match(r"wrap_(.+)$", n)
            if m:
                res.setdefault(m.group(1), {})["etx"] = {"symbol": n, "meta": meta, "total": count(lines), "loops": loops(lines)}
    return res


def fmt(d, keys, div=1):
    return " ".join(f"{round(d.get(k, 0) / div):5d}" for k in keys)


def is_spin(lp) -> bool:
    """etx_imp_slot_barrier's wait loop: an LDS read of the generation word and an s_sleep, nothing else of weight."""
    return lp["sleep"] >= 1 and lp["lds_read"] >= 1 and lp["instructions"] <= 8


def summarise(inv: dict, slots: dict) -> dict:
    """Per kernel and form: one block's code (an import holds one copy of the kernel per slot: counts / slots), its
    loops grouped by signature, and the software barriers (spin loops) per block."""
    out = {}
    for e, forms in inv.items():
        for f, d in forms.items():
            k = slots.get(e, 1) if f == "etx" else 1
            groups = {}
            spins = 0
            for lp in d["loops"]:
                if is_spin(lp):
                    spins += 1
                    continue
                sig = tuple(lp.get(c, 0) for c in ("instructions", "vmem_load", "load_bytes_lane", "lds_read", "lds_write", "mfma", "barrier"))
                groups.setdefault(sig, [lp, 0])[1] += 1
            m = d["meta"]
            out.setdefault(e, {})[f] = {
                "vgpr": m.get("accum_offset", m.get("next_free_vgpr", 0)),
                "agpr": m.get("next_free_vgpr", 0) - m.get("accum_offset", m.get("next_free_vgpr", 0)),
                "lds": m.get("group_segment_fixed_size", 0), "scratch": m.get("private_segment_fixed_size", 0),
                "per_block": {c: d["total"].get(c, 0) / k for c in d["total"]},
                "hw_barriers": d["total"].get("barrier", 0) / k, "sw_barriers": spins / k,
                "loops": [dict(lp, copies=n / k) for lp, n in groups.values()]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    inv = inventory(Path(a.build))
    slots = {e: i["slots"] for e, i in json.loads((Path(a.build) / "imp/imports.json").read_text()).items()}
    sm = summarise(inv, slots)
    keys = ["instructions", "vmem_load", "load_bytes_lane", "vmem_store", "smem_load", "lds_read", "lds_write", "lds_atomic",
            "lds_permute", "dpp", "mfma", "wait_vm"]
    hdr = " ".join(f"{k[:5]:>5s}" for k in keys)
    print(f"{'kernel':20s} {'form':4s} {'vgpr':>4s} {'agpr':>4s} {'lds':>6s} {'scr':>3s} {'hwbar':>5s} {'swbar':>5s}  {hdr}")
    for e, forms in sm.items():
        for f in ("vllm", "etx"):
            if f not in forms:
                continue
            d = forms[f]
            print(f"{e:20s} {f:4s} {d['vgpr']:4d} {d['agpr']:4d} {d['lds']:6d} {d['scratch']:3d} {d['hw_barriers']:5.0f} {d['sw_barriers']:5.0f}  "
                  f"{fmt(d['per_block'], keys)}")
            for lp in d["loops"]:
                print(f"{'':20s} {'loop':4s} {'x' + format(lp['copies'], 'g'):>4s} {'':4s} {'':6s} {'':3s} {lp['barrier']:5d} {'':5s}  {fmt(lp, keys)}")
    print("per block (an import's slot copies divided out); hwbar = s_barrier, swbar = slot-barrier spin loops; loops: body counts")
    if a.json:
        Path(a.json).write_text(json.dumps(sm, indent=1))


if __name__ == "__main__":
    main()
