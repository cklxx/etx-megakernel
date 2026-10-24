"""Import a compiled GPU kernel into an ETX megakernel at the LLVM IR level.

A HIP or Triton kernel is an `amdgpu_kernel` function in AMDGPU LLVM IR. This turns one into an ordinary
device function the megakernel's tile can call, without touching the kernel's code:

  * the kernel's parameters are loaded from an argument block (a device buffer the host fills with the
    same values it would pass to the launch, laid out like a C struct of the parameter types, followed by
    the grid size); the exported entry is  void EXPORT(ptr args, i32 block, i32 tid, i32 slot)
  * block and thread ids come from the ETX task: __ockl_get_group_id / llvm.amdgcn.workgroup.id.* read the
    block id the adapter passes, __ockl_get_local_id / llvm.amdgcn.workitem.id.* are recomputed from the
    thread's index in the kernel's original block shape, sizes are constants
  * the kernel's static LDS moves into one arena shared by all imported tiles (they run one after another
    on a workgroup, so their LDS can overlap); allocations above the caller's limit are rejected, never truncated
  * when the megakernel's workgroup is k times the kernel's block, k blocks run side by side ("slots"),
    each on its own LDS copy, and the kernel's s_barrier becomes a barrier over the slot's waves only

The input is the IR before the device libraries are linked (hipcc -nogpulib -emit-llvm), so thread and
block ids are still __ockl_* calls; the intrinsic forms (Triton) are handled too. Anything the importer
does not understand (implicit-argument or dispatch pointers, dynamic LDS, ids read inside helper
functions that stayed out of line) is refused with the reason, never guessed.

  python -m etx.importer.ir_import in.ll --kernel NAME --export etx_imp_x --block 64,16,1 \
      --threads 1024 [--lds-cap 24576] -o out.ll --info out.json
"""
from __future__ import annotations

from etx.ir.llvm import (
    json,
    re,
    dataclass,
    field,
    ARENA,
    SLOT_BARRIER,
    _SIZES,
    ImportError_,
    Param,
    ImportInfo,
    _split_top,
    _matching_paren,
    _functions,
    _param_type,
    _size_align,
    _ID_CALL,
    _BARRIER,
    _REFUSE,
    _lds_globals,
    _type_bytes,
)

def import_kernel(text: str, kernel: str, export: str, block: tuple[int, int, int], threads: int,
                  lds_cap: int | None = None, inline: bool = True, max_slots: int | None = None) -> tuple[str, ImportInfo]:
    """inline=False keeps the entry out of line: the kernel then gets its own register allocation instead of
    sharing the megakernel's (for register-heavy tiles off the critical path; a call costs the callee's
    saved registers per task)."""
    bx, by, bz = block
    nthr = bx * by * bz
    slot_thr = (nthr + 63) // 64 * 64                 # a slot is whole waves; a partial wave runs with its lanes masked, as on hardware
    if slot_thr > threads:
        raise ImportError_(f"block {block} ({nthr} threads) is larger than the megakernel workgroup ({threads})")
    slots = threads // slot_thr
    if max_slots:                                      # fewer blocks per workgroup: the grid spreads over more CUs
        slots = max(1, min(slots, max_slots))
    info = ImportInfo(export=export, kernel="", block=block, threads=threads, slots=slots)
    info.slot_threads = slot_thr

    funcs = _functions(text)
    cand = [f for f in funcs if kernel in f[2] and "amdgpu_kernel" in text[f[0]: text.find("\n", f[0])]]
    if len(cand) != 1:
        raise ImportError_(f"kernel {kernel!r}: {len(cand)} amdgpu_kernel matches ({[c[2] for c in cand]})")
    k0, k1, kname = cand[0]
    info.kernel = kname
    body = text[k0:k1]
    if _REFUSE.search(body):
        raise ImportError_(f"{kname} uses {_REFUSE.search(body).group(0)}: not importable yet")
    for f0, f1, fname in funcs:                        # helpers that read ids out of line would need the ids too
        if (f0, f1) != (k0, k1) and "amdgpu_kernel" not in text[f0: text.find("\n", f0)] and _ID_CALL.search(text[f0:f1]):
            raise ImportError_(f"helper {fname} reads thread/block ids out of line (compile with -O3 so it inlines)")

    # ---- header: parameters, calling convention, linkage, attributes
    hdr_end = body.find("\n")
    header = body[:hdr_end]
    lp = header.find("@" + kname) + len(kname) + 1
    if header[lp] == '"':
        lp += 1
    lp = header.find("(", lp)
    rp = _matching_paren(header, lp)
    params = []
    off = 0
    for p in _split_top(header[lp + 1: rp]):
        ty, attrs, name = _param_type(p)
        if "byref" in attrs or "byval" in attrs:
            raise ImportError_(f"aggregate parameter {p.strip()!r} is not supported yet")
        size, align = _size_align(ty)
        off = (off + align - 1) // align * align
        params.append(Param(ty=ty, attrs=attrs, name=name, size=size, align=align, offset=off))
        off += size
    info.params = params
    info.grid_offset = (off + 3) // 4 * 4
    info.arg_bytes = info.grid_offset + 16
    tail = header[rp + 1:]
    attr_ref = re.search(r"#(\d+)", tail)
    extra = ["i32 %etx.bx", "i32 %etx.by", "i32 %etx.bz", "i32 %etx.gx", "i32 %etx.gy", "i32 %etx.gz", "i32 %etx.tid", "i32 %etx.slot"]
    new_params = header[lp + 1: rp] + (", " if params else "") + ", ".join(extra)
    tail = re.sub(r"\bcomdat(\([^)]*\))?", "", tail)
    tail = re.sub(r"!\w+ !\d+", "", tail)                 # kernel metadata attachments (reqd_work_group_size ...)
    ret_ty = header[:header.find("@" + kname)].split()[-1]
    if ret_ty != "void":
        raise ImportError_("kernel does not return void")

    # ---- body: ids, sizes, barriers
    rest = body[hdr_end:]
    for bad in re.finditer(r"@__ockl_get_\w+\((?!i32 (?:noundef )?\d)", rest):
        raise ImportError_(f"{kname}: a thread/block id with a non-constant dimension")
    dims = {"x": 0, "y": 1, "z": 2}

    def id_repl(m: re.Match) -> str:
        ind, dst, rty, fn, dim = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if fn.startswith("llvm.amdgcn."):
            d = dims[fn[-1]]
            base = ("%etx.bx", "%etx.by", "%etx.bz")[d] if "workgroup" in fn else ("%etx.lx", "%etx.ly", "%etx.lz")[d]
            return f"{ind}{dst} = add i32 {base}, 0"
        d = int(dim)
        if d > 2:
            raise ImportError_(f"dimension {d}")
        kind = fn[len("__ockl_get_"):]
        val = {"group_id": ("%etx.bx", "%etx.by", "%etx.bz")[d],
               "local_id": ("%etx.lx", "%etx.ly", "%etx.lz")[d],
               "num_groups": ("%etx.gx", "%etx.gy", "%etx.gz")[d],
               "local_size": str(block[d]),
               "global_offset": "0"}.get(kind)
        if kind == "global_id":
            return (f"{ind}{dst}.m = mul i32 {('%etx.bx', '%etx.by', '%etx.bz')[d]}, {block[d]}\n"
                    f"{ind}{dst}.a = add i32 {dst}.m, {('%etx.lx', '%etx.ly', '%etx.lz')[d]}\n"
                    f"{ind}{dst} = zext i32 {dst}.a to i64")
        if kind == "global_size":
            return (f"{ind}{dst}.m = mul i32 {('%etx.gx', '%etx.gy', '%etx.gz')[d]}, {block[d]}\n"
                    f"{ind}{dst} = zext i32 {dst}.m to i64")
        if val.lstrip("-").isdigit():
            return f"{ind}{dst} = add {rty} {val}, 0"
        return f"{ind}{dst} = zext i32 {val} to i64" if rty == "i64" else f"{ind}{dst} = add i32 {val}, 0"

    rest = _ID_CALL.sub(id_repl, rest)
    info.barriers = len(_BARRIER.findall(rest))
    if slots > 1 or slot_thr != threads:             # other waves of the workgroup are not in this block
        rest = _BARRIER.sub(lambda m: f"{m.group(1)}call void @{SLOT_BARRIER}(i32 %etx.slot, i32 {slot_thr // 64})", rest)
        info.slot_barrier = True
    if re.search(r"@(__ockl_get_\w+|llvm\.amdgcn\.work(group|item)\.id)", rest):
        raise ImportError_(f"{kname}: an id read the importer did not recognise remains")
    # thread coordinates in the kernel's own block shape, at the top of the entry block
    lz = f"  %etx.lx = urem i32 %etx.tid, {bx}\n  %etx.t1 = udiv i32 %etx.tid, {bx}\n" \
         f"  %etx.ly = urem i32 %etx.t1, {by}\n  %etx.lz = udiv i32 %etx.t1, {by}\n"
    if not header.rstrip().endswith("{"):
        raise ImportError_("unexpected function header layout")
    nl = 0                                               # rest starts with the newline that ends the header
    first = rest[1: rest.find("\n", 1)]
    if re.match(r"^[\w.$-]+:", first):                   # a named entry block: insert after its label
        nl = rest.find("\n", nl + 1)
    rest = rest[: nl + 1] + lz + rest[nl + 1:]

    # ---- LDS into the arena
    lds = [g for g in _lds_globals(text) if re.search(r"@" + re.escape(g[0]) + r"\b", body)]
    all_lds = set(re.findall(r'^@("[^"]+"|[\w.$]+) = [^\n]*addrspace\(3\) global', text, flags=re.M))
    missed = [n for n in all_lds if n.strip('"') not in {g[0] for g in lds} and re.search(r"@" + re.escape(n) + r"(?![\w.$])", body)]
    if missed:
        raise ImportError_(f"{kname}: LDS variables the importer could not size: {missed}")
    offs, cur = [], 0
    for name, nbytes, align, _ in lds:
        cur = (cur + max(align, 4) - 1) // max(align, 4) * max(align, 4)
        offs.append(cur); cur += nbytes
    per_slot = (cur + 15) // 16 * 16
    # A single LDS variable may be over-dimensioned by the kernel template.
    if lds_cap is not None and len(lds) == 1:
        per_slot = (lds_cap + 15) // 16 * 16
    if lds_cap is not None and len(lds) > 1 and per_slot > lds_cap:
        raise ImportError_(f"{kname}: static LDS needs {per_slot} B per slot, exceeding the {lds_cap} B limit")
    info.lds_bytes_per_slot = per_slot
    info.lds_bytes = per_slot * slots

    # ---- attributes of the body: drop kernel-only hints, force inlining
    attrs_text = ""
    if attr_ref:
        g = re.search(r"^attributes #" + attr_ref.group(1) + r" = \{([^\n]*)\}", text, flags=re.M)
        if not g:
            raise ImportError_("kernel attribute group not found")
        keep = [a for a in re.findall(r'"[^"]*"(?:="[^"]*")?|\S+', g.group(1))
                if not re.match(r'"amdgpu-(no-|flat-work-group-size|waves-per-eu|max-num-workgroups|cluster)|"uniform-work-group-size"|noinline|optnone', a)]
        n_attr = max(int(x) for x in re.findall(r"^attributes #(\d+)", text, flags=re.M)) + 1
        attrs_text = f"attributes #{n_attr} = {{ alwaysinline {' '.join(keep)} }}\n"
        tail = tail.replace(attr_ref.group(0), f"#{n_attr}")

    # ---- one body per slot (the LDS offsets are constants), then the entry point
    out_bodies = []
    for s in range(slots):
        b = f"define internal void @{export}.body.s{s}({new_params}){tail}{rest}"
        for (name, nbytes, align, _), o in zip(lds, offs):
            gep = f"getelementptr inbounds (i8, ptr addrspace(3) @{ARENA}, i32 {s * per_slot + o})"
            b = re.sub(r"@" + re.escape(name) + r"\b", gep, b)
        out_bodies.append(b)

    loads, argv = [], []
    for i, p in enumerate(params):
        loads.append(f"  %a{i}.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {p.offset}\n"
                     f"  %a{i} = load {p.ty}, ptr addrspace(4) %a{i}.p, align {p.align}")
        argv.append(f"{p.ty} %a{i}")
    g = info.grid_offset
    entry = [f"define void @{export}(ptr noundef %args, i32 noundef %blk, i32 noundef %tid, i32 noundef %slot) #{'ENTRY'} {{",
             "  %args4 = addrspacecast ptr %args to ptr addrspace(4)"] + loads + [
             f"  %gx.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {g}",
             "  %gx = load i32, ptr addrspace(4) %gx.p, align 4",
             f"  %gy.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {g + 4}",
             "  %gy = load i32, ptr addrspace(4) %gy.p, align 4",
             f"  %gz.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {g + 8}",
             "  %gz = load i32, ptr addrspace(4) %gz.p, align 4",
             "  %bx = urem i32 %blk, %gx", "  %b1 = udiv i32 %blk, %gx", "  %by = urem i32 %b1, %gy", "  %bz = udiv i32 %b1, %gy"]
    ids = "i32 %bx, i32 %by, i32 %bz, i32 %gx, i32 %gy, i32 %gz, i32 %tid, i32 %slot"
    if slot_thr != nthr:
        entry += [f"  %inblk = icmp ult i32 %tid, {nthr}", "  br i1 %inblk, label %run, label %skip", "skip:", "  ret void", "run:"]
    call_args = ", ".join(argv + [ids]) if argv else ids
    if slots == 1:
        entry += [f"  call void @{export}.body.s0({call_args})", "  ret void", "}"]
    else:
        entry += [f"  switch i32 %slot, label %s0 [" + " ".join(f"i32 {s}, label %s{s}" for s in range(1, slots)) + " ]"]
        for s in range(slots):
            entry += [f"s{s}:", f"  call void @{export}.body.s{s}({call_args})", "  br label %done"]
        entry += ["done:", "  ret void", "}"]
    n_entry = (max(int(x) for x in re.findall(r"^attributes #(\d+)", text + attrs_text, flags=re.M)) + 1)
    entry_txt = "\n".join(entry).replace("#ENTRY", f"#{n_entry}") + "\n"
    attrs_text += f'attributes #{n_entry} = {{ {"alwaysinline" if inline else "noinline"} convergent nounwind "target-cpu"="gfx942" }}\n'

    # ---- assemble: the module minus every kernel and the imported LDS variables, plus the new functions
    pieces, last = [], 0
    for f0, f1, fname in funcs:
        if "amdgpu_kernel" in text[f0: text.find("\n", f0)]:
            pieces.append(text[last:f0]); last = f1
    pieces.append(text[last:])
    mod = "".join(pieces)
    for name, _, _, line in lds:
        mod = mod.replace(line + "\n", "")
    mod = re.sub(r"^@llvm\.(compiler\.)?used = [^\n]*\n", "", mod, flags=re.M)
    mod = re.sub(r"^@__hip_cuid_\w+ = [^\n]*\n", "", mod, flags=re.M)    # per-TU id for the HIP runtime; clashes when imports are linked together
    decls = [f"@{ARENA} = external addrspace(3) global [0 x i8], align 16"]
    if getattr(info, "slot_barrier", False) and info.barriers:
        decls.append(f"declare void @{SLOT_BARRIER}(i32, i32) #{n_entry}")
    mod = mod.rstrip() + "\n\n" + "\n".join(decls) + "\n\n" + "\n".join(out_bodies) + "\n" + entry_txt + attrs_text
    return mod, info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ll"); ap.add_argument("--kernel", required=True); ap.add_argument("--export", required=True)
    ap.add_argument("--block", required=True, help="bx,by,bz of the original launch")
    ap.add_argument("--threads", type=int, required=True, help="megakernel workgroup size")
    ap.add_argument("--lds-cap", type=int, default=None)
    ap.add_argument("-o", required=True); ap.add_argument("--info", default=None)
    a = ap.parse_args()
    block = tuple(int(x) for x in a.block.split(","))
    block = block + (1,) * (3 - len(block))
    out, info = import_kernel(open(a.ll).read(), a.kernel, a.export, block, a.threads, a.lds_cap)
    open(a.o, "w").write(out)
    if a.info:
        json.dump(info.to_json(), open(a.info, "w"), indent=1)
    print(f"{info.kernel} -> {a.export}: {len(info.params)} params, {info.arg_bytes} B args, "
          f"{info.slots} slot(s), {info.lds_bytes} B LDS, {info.barriers} barrier(s)")


if __name__ == "__main__":
    main()
