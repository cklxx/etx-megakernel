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
    on a workgroup, so their LDS can overlap), optionally capped to what the instance really uses
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
    if lds_cap is not None and lds_cap < per_slot:
        if len(lds) != 1:
            raise ImportError_("--lds-cap needs exactly one LDS variable")
        per_slot = (lds_cap + 15) // 16 * 16
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
