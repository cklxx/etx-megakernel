"""Import NVPTX kernels as ETX device functions, without a GPU.

The export ABI is void(ptr args, i32 linear_block, i32 local_tid, i32 slot).
The argument block contains naturally aligned parameters followed by u32 grid[3]
(and one reserved u32). Slots consist of complete warps; barrier IDs 1..15 are
reserved for imports. Callers must pass a valid slot and a slot-local thread ID.
"""
from __future__ import annotations

import re
from pathlib import Path

from etx.importer.ir_import import (
    ImportError_, ImportInfo, Param, _functions, _lds_globals,
    _matching_paren, _param_type, _size_align, _split_top,
)

_IDS = r'llvm\.nvvm\.read\.ptx\.sreg\.(tid|ctaid|ntid|nctaid)\.([xyz])'
_ID = re.compile(r'^(\s*)(%[\w.]+) = (?:tail )?call i32 @' + _IDS + r'\(\)[^\n]*$', re.M)
_BARRIER = re.compile(r'^(\s*)(?:tail )?call void @llvm\.nvvm\.(barrier0|barrier\.cta\.sync\.aligned\.all)\(([^)]*)\)[^\n]*$', re.M)


def import_kernel(text: str, kernel: str, export: str, block: tuple[int, int, int],
                  threads: int, *, target_cpu: str, lds_cap: int | None = None) -> tuple[str, ImportInfo]:
    """Transform one kernel; target_cpu is supplied by the target YAML, never inferred."""
    if not re.fullmatch(r'sm_[0-9]+[a-z]?', target_cpu):
        raise ImportError_('invalid target-cpu')
    if not re.fullmatch(r'[A-Za-z_$][\w.$]*', export):
        raise ImportError_('invalid export name')
    if (len(block) != 3 or any(x <= 0 for x in block)
            or not 0 < threads <= 1024 or threads % 32):
        raise ImportError_('invalid block/threads')
    n = block[0] * block[1] * block[2]
    slot_threads = (n + 31) // 32 * 32
    slots = threads // slot_threads
    if not slots or slots > 15:
        raise ImportError_('block does not fit or exceeds 15 named barrier slots')
    # Refuse module-wide: helpers can hide synchronization and implicit state.
    if re.search(r'cluster|grid[._]sync|griddepcontrol|cooperative_groups|barrier\.sync|barrier\.arrive|mbarrier', text, re.I):
        raise ImportError_('clusters, grid sync or unsupported named barriers')
    if re.search(r'addrspace\(3\) (?:externally_initialized )?global \[0 x|external[^\n]*addrspace\(3\)', text):
        raise ImportError_('dynamic shared memory is unsupported')
    if re.search(r'asm[^\n]*(?:bar\.|%tid|%ctaid|%ntid|%nctaid)', text):
        raise ImportError_('inline assembly id reads or barriers are unsupported')
    funcs = _functions(text)
    annotated = set(re.findall(r'!\d+ = !\{ptr @([\w.$]+), !"kernel", i32 1\}', text))
    kernels = [f for f in funcs if 'ptx_kernel' in text[f[0]:text.find('\n', f[0])] or f[2] in annotated]
    cand = [f for f in kernels if f[2] == kernel]
    if len(cand) != 1:
        raise ImportError_(f'kernel {kernel!r}: expected one kernel definition')
    start, end, _ = cand[0]
    for a, b, name in funcs:
        if (a, b) != (start, end) and re.search(_IDS, text[a:b]):
            raise ImportError_(f'helper {name} reads ids out of line')
        if (a, b) != (start, end) and _BARRIER.search(text[a:b]):
            raise ImportError_(f'helper {name} has an out-of-line barrier')
    body = text[start:end]
    header, rest = body.split('\n', 1)
    if header[:header.index('@')].split()[-1] != 'void':
        raise ImportError_('kernel must return void')
    lp = header.index('(', header.index('@'))
    rp = _matching_paren(header, lp)
    params, off = [], 0
    for raw in _split_top(header[lp + 1:rp]):
        ty, attrs, name = _param_type(raw)
        if not name or re.search(r'byval|byref|sret', attrs):
            raise ImportError_('unsupported aggregate or unnamed parameter')
        size, align = _size_align(ty)
        off = (off + align - 1) // align * align
        params.append(Param(ty, attrs, name, size, align, off))
        off += size
    info = ImportInfo(export, kernel, block, threads, slots, params)
    info.slot_threads = slot_threads
    info.grid_offset = (off + 3) // 4 * 4
    info.arg_bytes = info.grid_offset + 16
    def ids(m):
        ind, dst, kind, dim = m.groups()
        d = 'xyz'.index(dim)
        val = {'tid': ('%etx.lx', '%etx.ly', '%etx.lz'),
               'ctaid': ('%etx.bx', '%etx.by', '%etx.bz'),
               'ntid': tuple(map(str, block)),
               'nctaid': ('%etx.gx', '%etx.gy', '%etx.gz')}[kind][d]
        return f'{ind}{dst} = add i32 {val}, 0'
    rest = _ID.sub(ids, rest)
    def barrier(m):
        if m[2] != 'barrier0' and not re.fullmatch(r'i32(?: noundef)? 0', m[3].strip()):
            raise ImportError_('non-zero named barrier is unsupported')
        info.barriers += 1
        return (f'{m[1]}%etx.bar{info.barriers} = add i32 %etx.slot, 1\n'
                f'{m[1]}call void asm sideeffect "bar.sync $0, $1;", "r,r,~{{memory}}"'
                f'(i32 %etx.bar{info.barriers}, i32 {slot_threads})')
    rest = _BARRIER.sub(barrier, rest)
    info.slot_barrier = bool(info.barriers)
    if re.search(r'@llvm\.nvvm\.(?:read\.ptx\.sreg|barrier)', rest):
        raise ImportError_('unrecognised id read or barrier')
    # Partial warps cannot supply the named barrier participant count.
    if info.barriers and n != slot_threads:
        raise ImportError_('barriers require whole-warp blocks')
    coords = (f'  %etx.lx = urem i32 %etx.tid, {block[0]}\n'
              f'  %etx.t1 = udiv i32 %etx.tid, {block[0]}\n'
              f'  %etx.ly = urem i32 %etx.t1, {block[1]}\n'
              f'  %etx.lz = udiv i32 %etx.t1, {block[1]}\n')
    first = rest.split('\n', 1)[0]
    pos = len(first) + 1 if re.match(r'^[\w.$-]+:', first) else 0
    rest = rest[:pos] + coords + rest[pos:]
    lds = [g for g in _lds_globals(text) if re.search('@' + re.escape(g[0]) + r'\b', body)]
    offset, offsets = 0, []
    for name, size, align, line in lds:
        offset = (offset + align - 1) // align * align
        offsets.append(offset)
        offset += size
    for a, b, name in funcs:
        if (a, b) != (start, end) and any(
            re.search('@' + re.escape(g[0]) + r'\b', text[a:b]) for g in _lds_globals(text)
        ):
            raise ImportError_(f'helper {name} uses shared memory out of line')
    per_slot = (offset + 15) // 16 * 16
    if lds_cap is not None and per_slot > lds_cap:
        raise ImportError_('static shared memory exceeds cap')
    info.lds_bytes_per_slot, info.lds_bytes = per_slot, per_slot * slots
    # Metadata references to the former kernel must not keep it a kernel.
    mod = text[:start] + text[end:]
    mod = re.sub(r'^!nvvm.annotations = .*\n?', '', mod, flags=re.M)
    mod = re.sub(r'^!\d+ = !\{ptr @' + re.escape(kernel) + r',.*\n?', '', mod, flags=re.M)
    mod = re.sub(r'^@llvm\.(?:compiler\.)?used = .*\n?', '', mod, flags=re.M)
    for _, _, _, line in lds:
        mod = mod.replace(line + '\n', '')
    # Drop stale kernel tuning attributes: YAML is authoritative for both functions.
    attr = f'alwaysinline convergent nounwind "target-cpu"="{target_cpu}"'
    extras = ', '.join('i32 %etx.' + x for x in ('bx','by','bz','gx','gy','gz','tid','slot'))
    original = header[lp+1:rp]
    bodies = []
    for s in range(slots):
        r = rest
        for (name, _, _, _), o in zip(lds, offsets):
            r = re.sub('@' + re.escape(name) + r'\b', f'getelementptr (i8, ptr addrspace(3) @etx_lds_arena, i32 {s*per_slot+o})', r)
        bodies.append(f'define internal void @{export}.body.s{s}({original}{", " if params else ""}{extras}) {attr} {{\n{r}')
    lines = [f'define void @{export}(ptr %args, i32 %blk, i32 %tid, i32 %slot) {attr} {{']
    argv = []
    for i, p in enumerate(params):
        lines += [f'  %p{i} = getelementptr i8, ptr %args, i64 {p.offset}', f'  %a{i} = load {p.ty}, ptr %p{i}, align {p.align}']
        argv.append(f'{p.ty} %a{i}')
    for i, d in enumerate('xyz'):
        lines += [f'  %g{d}.p = getelementptr i8, ptr %args, i64 {info.grid_offset+4*i}', f'  %g{d} = load i32, ptr %g{d}.p, align 4']
    lines += ['  %bx = urem i32 %blk, %gx', '  %b1 = udiv i32 %blk, %gx', '  %by = urem i32 %b1, %gy', '  %bz = udiv i32 %b1, %gy']
    argv += ['i32 %' + x for x in ('bx','by','bz','gx','gy','gz','tid','slot')]
    if n != slot_threads:
        lines += [f'  %active = icmp ult i32 %tid, {n}', '  br i1 %active, label %dispatch, label %done', 'dispatch:']
    lines += ['  switch i32 %slot, label %done [' + ' '.join(f'i32 {s}, label %s{s}' for s in range(slots)) + ' ]']
    for s in range(slots):
        lines += [f's{s}:', f'  call void @{export}.body.s{s}({", ".join(argv)})', '  br label %done']
    lines += ['done:', '  ret void', '}']
    arena = '@etx_lds_arena = external addrspace(3) global [0 x i8], align 16\n' if lds else ''
    return mod + '\n' + arena + '\n'.join(bodies + lines) + '\n', info


def import_from_yaml(text: str, config: str | Path) -> tuple[str, ImportInfo]:
    """Read kernel/export/block/threads and target-cpu from a YAML mapping."""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    return import_kernel(text, cfg['kernel'], cfg['export'], tuple(cfg['block']),
                         cfg['threads'], target_cpu=cfg['target-cpu'], lds_cap=cfg.get('lds-cap'))
