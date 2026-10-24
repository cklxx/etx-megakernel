"""Syntax: hops."""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


CLASSES = ('coherent', 'plain', 'scalar', 'scratch', 'store', 'lds')


_REG = re.compile(r'^(v|s|a|ttmp)(\d+)$|^(v|s|a|ttmp)\[(\d+):(\d+)\]$')


_ALIASES = {'vcc_lo': 'vcc', 'vcc_hi': 'vcc', 'exec_lo': 'exec', 'exec_hi': 'exec', 'vcc': 'vcc', 'exec': 'exec',
            'm0': 'm0', 'scc': 'scc', 'flat_scratch': 'flat_scratch', 'flat_scratch_lo': 'flat_scratch',
            'flat_scratch_hi': 'flat_scratch'}


_BRANCH = re.compile(r'^s_(branch|cbranch_\w+|setpc_b64|endpgm\w*)$')


_NODEF = re.compile(r'^(global_store|flat_store|buffer_store|scratch_store|ds_write|ds_store|s_store|s_buffer_store|'
                    r's_waitcnt|s_barrier|s_branch|s_cbranch|s_endpgm|s_nop|s_sleep|s_setprio|s_sendmsg|s_dcache|'
                    r'buffer_inv|buffer_wbl2|buffer_wbinvl1|s_trap|s_icache|s_set_inst_prefetch|s_inst_prefetch|'
                    r's_ttrace|s_setpc|v_nop|s_cmp|s_bitcmp|exp|s_wakeup|s_sethalt|s_denorm|s_round|s_code_end)')


_CARRY_OUT = re.compile(r'_co_|div_scale|^v_mad_(u64_u32|i64_i32)')


_SALU_SCC = re.compile(r'^s_(add|sub|addc|subb|and|or|xor|andn2|orn2|nand|nor|xnor|lshl|lshr|ashr|bfe|min|max|not|'
                       r'abs|absdiff|bcnt|lshl\d_add|mul_hi|quadmask|wqm|bitreplicate|cselect)')


def regs(op: str) -> list[str]:
    """Register names of one operand. v[4:5] gives v4 and v5. vcc_lo gives vcc."""
    op = op.strip().lstrip('-|').rstrip('|')
    op = re.sub(r'^(abs|neg|sext)\((.*)\)$', r'\2', op)
    if op in _ALIASES:
        return [_ALIASES[op]]
    m = _REG.match(op)
    if not m:
        return []
    if m.group(1):
        return [f'{m.group(1)}{m.group(2)}']
    k, a, b = m.group(3), int(m.group(4)), int(m.group(5))
    return [f'{k}{i}' for i in range(a, b + 1)]


@dataclass
class Insn:
    line: int
    mnem: str
    ops: list[str]
    mods: str
    text: str

    @property
    def kind(self) -> str:
        m = self.mnem
        # LDS DMA (`buffer_load_* ... lds`, `global_load_lds_*`): no VGPR destination, every operand is an address
        if m.startswith(('global_load_lds', 'scratch_load_lds', 'flat_load_lds')) or (
                m.startswith('buffer_load') and re.search(r'\blds\b', self.mods)):
            return 'lds_dma'
        if m.startswith('scratch_load'):
            return 'scratch_load'
        if m.startswith('scratch_store'):
            return 'scratch_store'
        if m.startswith(('s_load', 's_buffer_load', 's_scratch_load')):
            return 'scalar_load'
        if m.startswith(('global_load', 'flat_load', 'buffer_load')):
            return 'vm_load'
        if m.startswith(('global_store', 'flat_store', 'buffer_store')):
            return 'vm_store'
        if m.startswith(('global_atomic', 'flat_atomic', 'buffer_atomic')):
            return 'vm_atomic_ret' if self.returns else 'vm_store'
        if m.startswith('ds_'):
            if m.startswith(('ds_write', 'ds_store')) or (not self.returns and re.match(r'ds_(add|sub|max|min|and|or|xor|inc|dec|cmpst|cmpswap)', m) and '_rtn' not in m):
                return 'lds_store'
            return 'lds_load'
        return 'alu'

    @property
    def returns(self) -> bool:
        m = self.mnem
        if m.startswith('ds_'):
            return '_rtn' in m or m.startswith(('ds_read', 'ds_load', 'ds_swizzle', 'ds_permute', 'ds_bpermute', 'ds_consume', 'ds_append'))
        return bool(re.search(r'\b(sc0|glc)\b', self.mods))

    @property
    def coherent(self) -> bool:
        return bool(re.search(r'\bsc1\b', self.mods))


@dataclass
class Block:
    label: str | None
    insns: list[Insn] = field(default_factory=list)
    succ: list[int] = field(default_factory=list)
    start: int = 0
    exit: bool = False          # control can leave the region from this block


_IMPDEF = re.compile(r'^\s*;\s*implicit-def:\s*(.*)$')


def _impdef_regs(text: str) -> list[str]:
    """'$vgpr6_vgpr7, $sgpr0' gives ['v6', 'v7', 's0']."""
    out = []
    for m in re.finditer(r'(vgpr|sgpr|agpr)(\d+)|\b(vcc|exec)\b', text):
        out.append(m.group(3) if m.group(3) else {'vgpr': 'v', 'sgpr': 's', 'agpr': 'a'}[m.group(1)] + m.group(2))
    return out


def parse(text: str, first_line: int = 1) -> list[Insn]:
    out = []
    for n, raw in enumerate(text.splitlines(), first_line):
        s = raw.split(';', 1)[0].strip()
        if not s:
            mi = _IMPDEF.match(raw)
            if mi:                       # the compiler marks a register as undefined here (an undef phi input)
                out.append(Insn(n, 'implicit_def', _impdef_regs(mi.group(1)), '', raw))
                continue
        if not s or s.startswith('.') or s.endswith(':'):
            out.append(Insn(n, '', [], '', raw) if s.endswith(':') else None)
            continue
        parts = s.split(None, 1)
        mnem, rest = parts[0], parts[1] if len(parts) > 1 else ''
        if mnem == 's_waitcnt':
            out.append(Insn(n, mnem, [rest], '', raw))
            continue
        ops = [o.strip() for o in rest.split(',')] if rest else []
        mods = ''
        if ops:
            last = ops[-1].split()
            ops[-1] = last[0] if last else ''
            mods = ' '.join(last[1:])
        out.append(Insn(n, mnem, ops, mods, raw))
    return [i for i in out if i is not None]


def function_lines(lines: list[str], name: str) -> tuple[int, int]:
    """1-based first and last line of function `name` in the .s text."""
    start = end = None
    for i, l in enumerate(lines, 1):
        if start is None and l.split(';', 1)[0].strip() == f'{name}:':
            start = i
        elif start is not None and re.match(r'^\.Lfunc_end\d+:', l.strip()):
            end = i
            break
    if start is None or end is None:
        raise KeyError(f'function {name} not found')
    return start, end


def label_line(lines: list[str], label: str) -> int:
    for i, l in enumerate(lines, 1):
        if l.split(';', 1)[0].strip() == f'{label}:':
            return i
    raise KeyError(f'label {label} not found')


def blocks(insns: list[Insn], text_lines: dict[int, str]) -> list[Block]:
    out: list[Block] = []
    cur = Block(None)
    for ins in insns:
        if ins.mnem == '':
            if cur.insns or cur.label:
                out.append(cur)
            cur = Block(text_lines.get(ins.line, ins.text).split(';', 1)[0].strip()[:-1], start=ins.line)
            continue
        if not cur.insns and not cur.start:
            cur.start = ins.line
        cur.insns.append(ins)
        if _BRANCH.match(ins.mnem):
            out.append(cur)
            cur = Block(None, start=0)
    if cur.insns or cur.label:
        out.append(cur)
    idx = {b.label: k for k, b in enumerate(out) if b.label}
    for k, b in enumerate(out):
        last = b.insns[-1] if b.insns else None
        nxt = [k + 1] if k + 1 < len(out) else []
        if last and last.mnem == 's_branch':
            b.succ = [idx[last.ops[0]]] if last.ops[0] in idx else []
            b.exit = last.ops[0] not in idx
        elif last and last.mnem.startswith('s_cbranch'):
            b.succ = ([idx[last.ops[0]]] if last.ops[0] in idx else []) + nxt
            b.exit = last.ops[0] not in idx or not nxt
        elif last and (last.mnem.startswith('s_endpgm') or last.mnem == 's_setpc_b64'):
            b.succ, b.exit = [], True
        else:
            b.succ, b.exit = nxt, not nxt
    return out
