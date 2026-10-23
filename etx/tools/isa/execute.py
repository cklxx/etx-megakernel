"""Execute: hops."""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .syntax import Insn, _CARRY_OUT, _NODEF, _SALU_SCC, regs
from .state import BOT, Prices, State, ZERO, _f_or, vmax

def hop(v, ins: Insn, cls: str, prices: Prices, coherent=True):
    return (v[0] + prices.of(cls, coherent), v[1] + ((ins.line, ins.mnem, cls),))


def _slot_key(ins: Insn, base_ops: list[str]) -> tuple:
    off = re.search(r'offset:(-?\w+)', ins.mods)
    return (tuple(base_ops), int(off.group(1), 0) if off else 0)


_CMP = {'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b, 'lg': lambda a, b: a != b, 'lt': lambda a, b: a < b,
        'le': lambda a, b: a <= b, 'gt': lambda a, b: a > b, 'ge': lambda a, b: a >= b}


def _mkey(op: str) -> str | None:
    r = regs(op)
    return r[0] if r and (r[0][0] == 's' or r[0] in ('vcc', 'exec')) else None


def _mval(st: State, op: str) -> str | None:
    if re.fullmatch(r'-1|0xffffffffffffffff', op):
        return 'E'
    if op == '0':
        return 'Z'
    k = _mkey(op)
    return st.mask.get(k) if k else None


def _nz(v):
    return v == 'E' or (isinstance(v, str) and v.startswith('X'))


def _m_and(a, b):
    if 'Z' in (a, b):
        return 'Z'
    if a == 'E':
        return b
    if b == 'E':
        return a
    return a if a == b and a is not None else None


def _m_or(a, b):
    if 'E' in (a, b):
        return 'E'
    if a == 'Z':
        return b
    if b == 'Z':
        return a
    return a if a == b and a is not None else None


def _m_xor(a, b):
    if a == 'Z':
        return b
    if b == 'Z':
        return a
    return 'Z' if a == b and a is not None else None


def _m_andn2(a, b):
    if b == 'E' or a == 'Z' or (a == b and a is not None):
        return 'Z'
    return a if b == 'Z' else None


def _exec(st: State, ins: Insn) -> str:
    """The exec value. An unknown exec gets an identity symbol X<line>. The wave executes, so it is not zero."""
    v = st.mask.get('exec')
    if v is None:
        v = st.mask['exec'] = f'X{ins.line}'
    return v


def _scc(v):
    return '0' if v == 'Z' else ('1' if _nz(v) else None)


LANE_MODELS = ('phi', 'kill')




def _forget(st: State, atom):
    """A new value of a branch literal: facts about its old value end."""
    st.fm = {k: (frozenset(l for l in f if l[0] != atom) if k == 'exec' else f) for k, f in st.fm.items()
             if k == 'exec' or all(l[0] != atom for l in f)}
    st.vprev = {k: ((None, False, p) if w is not None and any(l[0] == atom for l in w) else (w, e, p))
                for k, (w, e, p) in st.vprev.items()}


def _forms(st: State, ins: Insn):
    """Lane formulas of exec and the 64-bit masks. A formula is a conjunction of literals, a superset of the
    lanes. The literals come from compares and from each s_and_saveexec (atom ('k', line)), so the else lanes
    of a structured branch are disjoint from its then lanes."""
    m, ops = ins.mnem, ins.ops
    if not ops or m == 's_waitcnt' or (_NODEF.match(m) and not m.startswith('v_cmp')):
        return
    E = frozenset()
    fm = st.fm
    def f(o):
        if o in ('exec',):
            return fm.get('exec', E)
        if o == '0':
            return BOT
        k = _mkey(o)
        return fm.get(k, E) if k else E
    if m.startswith('v_cmp'):
        dst = 'vcc' if len(ops) == 2 else _mkey(ops[0])
        _forget(st, ('c', ins.line))
        lit = st.fm.get('exec', E) | {(('c', ins.line), True)}   # inactive lanes give 0: result within exec
        if dst:
            st.fm[dst] = lit
        if m.startswith('v_cmpx'):
            st.fm['exec'] = lit
        return
    d = _mkey(ops[0])
    dr = regs(ops[0])
    if not (m.startswith('s_') and m.endswith('_b64')) or not d:
        for r in dr:
            if r == 'exec':
                st.fm['exec'] = E
            elif r[0] == 's' and r[1:].isdigit():
                st.fm.pop(r, None); st.fm.pop('s%d' % (int(r[1:]) - 1), None)
            elif r == 'vcc':
                st.fm.pop('vcc', None)
        return
    ex = fm.get('exec', E)
    if 'saveexec' in m:
        src = ops[1] if len(ops) > 1 else ''
        if m == 's_and_saveexec_b64':
            _forget(st, ('k', ins.line))
            sk = _mkey(src)
            g = st.fm.get(sk, E) if sk else E
            nex = ex | g | {(('k', ins.line), True)}
        elif m == 's_or_saveexec_b64':
            nex = _f_or(ex, f(src))
        elif m == 's_andn2_saveexec_b64':
            nex = f(src)
        else:
            nex = E
        if d != 'exec':
            st.fm[d] = ex
        st.fm['exec'] = frozenset(nex)
        return
    a = f(ops[1]) if len(ops) > 1 else E
    b = f(ops[2]) if len(ops) > 2 else E
    new = E
    if m == 's_mov_b64':
        new = a
    elif m == 's_and_b64':
        new = a | b
    elif m == 's_or_b64':
        new = _f_or(a, b)
    elif m == 's_andn2_b64':
        new = a
    elif m == 's_xor_b64' and len(ops) > 2 and 'exec' in ops[1:]:
        saved = b if ops[1] == 'exec' else a
        ks = [l for l in ex - saved if l[0][0] == 'k' and l[1]]
        if len(ks) == 1 and saved <= ex:
            new = saved | {(ks[0][0], False)}
    if d == 'exec':
        st.fm['exec'] = frozenset(new)
    else:
        st.fm[d] = frozenset(new)
        if d[0] == 's' and d[1:].isdigit():
            st.fm.pop('s%d' % (int(d[1:]) - 1), None)


def _masks(st: State, ins: Insn):
    """Update the mask lattice and the pins for one instruction."""
    m, ops = ins.mnem, ins.ops
    if not ops or m == 's_waitcnt':
        return
    defs = regs(ops[0]) if (not _NODEF.match(m) or m.startswith('v_cmp')) and ins.kind != 'lds_dma' else []
    for r in defs:
        st.pin.pop(r, None)
    if ins.line in st.pin_at:
        reg, val = st.pin_at[ins.line]
        st.pin[reg] = val
    mm = re.match(r'v_cmpx?_(eq|ne|lg|lt|le|gt|ge)_[iu]32', m)
    if mm:
        a_ops = ops if len(ops) == 2 else ops[1:]
        dst = 'vcc' if len(ops) == 2 else _mkey(ops[0])
        def ival(o):
            if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', o):
                return int(o, 0)
            r = regs(o)
            return st.pin.get(r[0]) if len(r) == 1 else None
        x, y = ival(a_ops[0]), ival(a_ops[1])
        res = None
        if x is not None and y is not None:
            res = _exec(st, ins) if _CMP[mm.group(1)](x, y) else 'Z'
        elif st.mask.get('exec') == 'Z':
            res = 'Z'
        if dst:
            if res is None:
                st.mask.pop(dst, None)
            else:
                st.mask[dst] = res
        if m.startswith('v_cmpx'):
            if res is None:
                st.mask.pop('exec', None)
            else:
                st.mask['exec'] = res
        return
    if not m.startswith('s_') or not m.endswith('_b64'):
        for r in defs:
            st.mask.pop(r, None)
        if defs and m.startswith('s_'):
            st.mask.pop('scc', None)
        return
    d = _mkey(ops[0])
    srcv = [(_exec(st, ins) if o in ('exec', 'exec_lo') else _mval(st, o)) for o in ops[1:]]
    ex = _exec(st, ins)
    new = None
    if 'saveexec' in m:
        src = srcv[0] if srcv else None
        f = {'s_and_saveexec_b64': _m_and, 's_or_saveexec_b64': _m_or, 's_xor_saveexec_b64': _m_xor}.get(m)
        nex = f(src, ex) if f else (_m_andn2(src, ex) if m == 's_andn2_saveexec_b64' else None)
        if d:
            if ex is None:
                st.mask.pop(d, None)
            else:
                st.mask[d] = ex
        if nex is None:
            st.mask.pop('exec', None); st.mask.pop('scc', None)
        else:
            st.mask['exec'] = nex
            if _scc(nex) is None:
                st.mask.pop('scc', None)
            else:
                st.mask['scc'] = _scc(nex)
        return
    if m == 's_mov_b64':
        new = srcv[0]
    elif m == 's_and_b64':
        new = _m_and(*srcv[:2])
    elif m == 's_or_b64':
        new = _m_or(*srcv[:2])
    elif m == 's_xor_b64':
        new = _m_xor(*srcv[:2])
    elif m == 's_andn2_b64':
        new = _m_andn2(*srcv[:2])
    elif m == 's_cselect_b64':
        sc = st.mask.get('scc')
        new = srcv[0] if sc == '1' else (srcv[1] if sc == '0' else (srcv[0] if srcv[0] == srcv[1] else None))
    if d:
        if new is None:
            st.mask.pop(d, None)
        else:
            st.mask[d] = new
    if m != 's_mov_b64' and m != 's_cselect_b64':
        if _scc(new) is None:
            st.mask.pop('scc', None)
        else:
            st.mask['scc'] = _scc(new)


def feasible(st: State, last: Insn | None, target: int | None, fall: int | None) -> list[int]:
    """Successors that the mask lattice allows after the last instruction of a block."""
    both = [x for x in (target, fall) if x is not None]
    if last is None or not last.mnem.startswith('s_cbranch'):
        return both
    c = last.mnem[len('s_cbranch_'):]
    v = {'execz': st.mask.get('exec'), 'execnz': st.mask.get('exec'), 'vccz': st.mask.get('vcc'),
         'vccnz': st.mask.get('vcc'), 'scc0': st.mask.get('scc'), 'scc1': st.mask.get('scc')}.get(c)
    if v is None or not (v in ('Z', '0', '1') or _nz(v)):
        return both
    zero = v in ('Z', '0')
    taken = zero if c in ('execz', 'vccz', 'scc0') else not zero
    pick = target if taken else fall
    return [pick] if pick is not None else []


def step(st: State, ins: Insn, prices: Prices, scratch_rsrc: str | None = None, barrier_drain: bool = False,
         block: int = -1):
    m, ops, kind = ins.mnem, ins.ops, ins.kind
    if m == 'implicit_def':              # undefined in every lane: no dependence, and no old value to keep
        for r in ops:
            st.r[r] = ZERO
            st.vprev.pop(r, None)
        return
    _masks(st, ins)
    if m.startswith('v_') and not m.startswith('v_cmp'):
        _forms(st, ins)                    # VALU: the exec formula is the one before this instruction
        fpost = None
    else:
        fpost = ins
    _step(st, ins, prices, scratch_rsrc, barrier_drain, block)
    if fpost is not None:
        _forms(st, fpost)
        if ops and ('saveexec' in m or m.startswith('v_cmpx') or regs(ops[0])[:1] == ['exec']):
            f = st.fm.get('exec', frozenset())
            if m == 's_or_b64' and 'exec' in ops[1:] and f in st.fval:
                # restore at the end of a branch: the mask is an earlier one again, so is its value
                st.r['exec'] = st.fval[f]
            else:
                st.fval[f] = st.r.get('exec', ZERO)


def _step(st: State, ins: Insn, prices: Prices, scratch_rsrc, barrier_drain, block):
    m, ops, kind = ins.mnem, ins.ops, ins.kind
    if m == 's_waitcnt':
        mm = re.search(r'vmcnt\((\d+)\)', ops[0] if ops else '')
        if mm:
            keep = int(mm.group(1))
            done, st.pending = (st.pending[:-keep], st.pending[-keep:]) if keep else (st.pending, [])
            for k, v in done:
                if k == 'store':
                    st.drained = vmax(st.drained, v)
        return
    if m == 's_barrier':
        if barrier_drain and st.drained[0] > 0:   # opt-in: a store drain before a barrier gates later loads
            st.cdep[-2 - ins.line] = st.drained
        return
    if m.startswith('s_cbranch'):
        cond = {'scc': ['scc'], 'vcc': ['vcc'], 'exe': ['exec']}.get(m[len('s_cbranch_'):][:3], [])
        v = vmax(st.get(cond), st.sctrl)
        if v[0] > 0 or v[1]:
            st.cdep[block] = vmax(st.cdep.get(block, ZERO), v)
        return
    if m in ('s_branch', 's_endpgm', 's_nop', 's_setpc_b64') or m.startswith(('s_set_inst', 's_inst_prefetch')):
        return
    if kind == 'lds_dma':                   # a global hop whose result lands in LDS, not in a register
        v = hop(vmax(st.get([r for o in ops for r in regs(o)]), st.ctrl), ins, 'coherent' if ins.coherent else 'plain', prices)
        st.lds = vmax(st.lds, v)
        st.pending.append(('load', v))
        return
    is_scratch_buf = scratch_rsrc is not None and m.startswith(('buffer_load', 'buffer_store')) and len(ops) >= 3 and ops[2] == scratch_rsrc
    if kind == 'scratch_store' or (is_scratch_buf and m.startswith('buffer_store')):
        data = regs(ops[1]) if kind == 'scratch_store' else regs(ops[0])
        base = [o for o in (ops[0], ops[2]) if o != 'off'] if kind == 'scratch_store' else [o for o in ops[1:4] if o != 'off']
        st.slot[_slot_key(ins, base)] = st.get(data)
        st.pending.append(('scratch', ZERO))
        return
    if kind == 'scratch_load' or (is_scratch_buf and m.startswith('buffer_load')):
        dst = regs(ops[0])
        base = [o for o in ops[1:3] if o != 'off'] if kind == 'scratch_load' else [o for o in ops[1:4] if o != 'off']
        srcs = [r for o in base for r in regs(o)]
        key = _slot_key(ins, base)
        v = hop(vmax(st.slot.get(key, ZERO), st.get(srcs), st.ctrl), ins, 'scratch', prices)
        for r in dst:
            st.setv(r, v)
        st.pending.append(('load', v))
        return
    if kind in ('vm_load', 'vm_atomic_ret', 'scalar_load'):
        dst = regs(ops[0])
        srcs = [r for o in ops[1:] for r in regs(o)]
        if ins.line in st.watch:
            st.seen = vmax(st.seen, st.get(srcs), st.ctrl)
        if kind == 'vm_atomic_ret':
            srcs = [r for o in ops[1:] for r in regs(o)]
        cls = 'scalar' if kind == 'scalar_load' else ('coherent' if ins.coherent else 'plain')
        v = hop(vmax(st.get(srcs), st.sctrl if kind == 'scalar_load' else st.ctrl), ins, cls, prices)
        for r in dst:
            st.setv(r, v)
        if kind != 'scalar_load':
            st.pending.append(('load', v))
        return
    if kind == 'vm_store':
        srcs = [r for o in ops for r in regs(o)]
        src = vmax(st.get(srcs), st.ctrl)
        st.issued = vmax(st.issued, src)
        v = hop(src, ins, 'store', prices, ins.coherent)
        st.pending.append(('store', v))
        st.stores = vmax(st.stores, v)
        return
    if kind == 'lds_store':
        srcs = [r for o in ops for r in regs(o)]
        st.lds = vmax(st.lds, st.get(srcs), st.ctrl)
        return
    if kind == 'lds_load':
        dst = regs(ops[0])
        srcs = [r for o in ops[1:] for r in regs(o)]
        base = vmax(st.lds, st.get(srcs), st.ctrl)
        v = (base[0], base[1] + ((ins.line, m, 'lds'),))
        for r in dst:
            st.setv(r, v)
        return
    # ALU
    if m.startswith('v_writelane'):
        val = st.get(regs(ops[1]))
        st.lane[(ops[0], ops[2])] = val
        for r in regs(ops[0]):
            st.r[r] = vmax(st.r.get(r, ZERO), val)
        return
    if m.startswith('v_readlane'):
        val = st.lane.get((ops[1], ops[2]), st.get(regs(ops[1])))
        for r in regs(ops[0]):
            st.r[r] = vmax(val, st.get(regs(ops[2])))
        return
    if _NODEF.match(m):
        if m.startswith(('s_cmp', 's_bitcmp')):
            st.r['scc'] = st.get([r for o in ops for r in regs(o)])
        return
    defs, uses = [], ops[1:]
    if ops:
        defs = regs(ops[0])
        if m.startswith('v_cmp') and len(ops) == 2:          # VOPC e32: implicit vcc
            defs, uses = (['vcc'] + (['exec'] if m.startswith('v_cmpx') else [])), ops
        elif m.startswith('v_cmpx'):
            defs = defs + ['exec']
        elif _CARRY_OUT.search(m) and len(ops) > 1 and regs(ops[1]) and regs(ops[1])[0][0] in 'sv' and regs(ops[1])[0][0] != 'v':
            defs, uses = defs + regs(ops[1]), ops[2:]
        elif _CARRY_OUT.search(m) and len(ops) > 1 and ops[1] == 'vcc':
            defs, uses = defs + ['vcc'], ops[2:]
    src = [r for o in uses for r in regs(o)]
    if m.startswith(('v_cndmask', 'v_addc', 'v_subb', 'v_div_fmas')):
        src.append('vcc')
    if m.startswith(('s_cselect', 's_addc', 's_subb', 's_cmov', 's_cbranch_scc')):
        src.append('scc')
    if 'saveexec' in m:                    # sdst = old exec, exec = f(old exec, src)
        old = st.get(['exec'])
        new = st.get(src + ['exec'])
        for r in defs:
            st.r[r] = old
        st.r['exec'] = new
        st.r['scc'] = new
        return
    val = st.get(src)
    if m.startswith(('v_mov', 's_mov', 'v_accvgpr')) and not src:
        val = ZERO
    if defs == ['exec'] and m.startswith(('s_or_b64', 's_or_b32')) and 'exec' in ops[1:]:
        other = [o for o in ops[1:] if o != 'exec']
        val = st.get([r for o in other for r in regs(o)])      # restore of a saved mask
    for r in defs:
        st.setv(r, val)
    if m.startswith('s_') and _SALU_SCC.match(m):
        st.r['scc'] = val
