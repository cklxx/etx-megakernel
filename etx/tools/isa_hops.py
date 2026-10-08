"""Dependent global-load chains in AMDGPU assembly (gfx9 syntax).

Rule, agreed by the review on 2026-10-08:
* A hop is one wait on the data chain. Follow each load result register, VGPR or SGPR, to the address or the
  control of the next load, or to the event publish. Each wait on that chain is one hop, for any count value.
* Loads in flight together are one hop: a load costs one hop above the deepest value that it reads.
* Infeasible paths are pruned. A lattice tracks each 64-bit mask as Z, no lane, or E, the entry exec mask.
  A pinned register is a wave-uniform register with a known value, for example the slot index of an
  imported kernel. A v_cmp of a pinned register with an immediate gives Z or exec. A branch on a known
  mask takes one edge only. Copies of a body for different slots then do not chain.
* Control is control dependence. A vector memory or LDS operation reads exec. A scalar branch adds its
  condition until the immediate post-dominator of the branch block. `s_or_b64 exec, exec, sN` and
  `s_mov_b64 exec, sN` restore a saved mask: exec then has the value of sN only. Code after the
  reconvergence point is not control dependent on the branch. This is the agreed rule for data dependency.
* Spills continue the chain. A scratch store keeps the value of its data register in a slot keyed by the
  base register and the offset. A scratch reload reads that slot and costs one hop of class 'scratch'.
  SGPR spills to VGPR lanes, v_writelane and v_readlane, and AGPR copies keep the value without a hop.
* LDS is not a global hop. A ds_read reads the LDS value, which is the deepest value that any ds_write stored.
  It counts as an 'lds' step with no cost.
* A global store completes one hop after its sources. A store enters the chain only through the drain at the
  end of the region, which is the release before the publish. Stores with sc1 are coherent. Other stores
  drain at the cached price. A drain before s_barrier is a vmcnt(0) wait off the data chain. It does not
  count by default. barrier_drain=True counts it, for a global-memory exchange between waves.

Classes: 'coherent' vector load with sc1, 'plain' vector load without sc1, 'scalar' s_load, 'scratch' reload,
'store' drain, 'lds' step. Prices, in us, come from a preset:
* 'warm': coherent = RTT, plain = cached, scalar = cached, scratch = cached, store = RTT with sc1, else cached.
* 'acquired': as 'warm', but plain = RTT. A task after `buffer_inv sc1` misses on each new line.

Loops: layout order is structured. A back edge goes to a block at or before the branch. The tool runs the
region with each loop at one trip, then with one and two extra trips of that loop. The difference gives the
hops per extra trip. A trip count is 'constant N at line L' when the loop exit compares with an immediate.
"""
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
# VOP3b: the second operand is an SGPR carry-out (a definition), not a source
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


# ---- values -------------------------------------------------------------------------------------------------
# A value is (cost, chain). chain is a tuple of hops (line, mnemonic, class). Merge keeps the larger cost.
ZERO = (0.0, ())


def _key(v):
    n_lds = sum(1 for h in v[1] if h[2] == 'lds')
    return (round(v[0], 9), len(v[1]) - n_lds, -n_lds)


def vmax(*vals):
    """The deeper value: larger cost, then more memory hops, then fewer LDS steps."""
    best = ZERO
    for v in vals:
        if v is not best and _key(v) > _key(best):
            best = v
    return best


@dataclass
class Prices:
    rtt: float = 0.50
    cached: float = 0.104
    preset: str = 'warm'

    def of(self, cls: str, coherent: bool = True) -> float:
        if cls == 'coherent':
            return self.rtt
        if cls == 'plain':
            return self.rtt if self.preset == 'acquired' else self.cached
        if cls == 'store':
            return self.rtt if coherent else self.cached
        if cls in ('scalar', 'scratch'):
            return self.cached
        return 0.0


class State:
    __slots__ = ('r', 'slot', 'lane', 'lds', 'cdep', 'pending', 'drained', 'stores', 'issued', 'mask', 'pin', 'pin_at',
                 'watch', 'seen', 'fm', 'vprev', 'kill', 'fval')

    def __init__(self):
        self.r: dict[str, tuple] = {}
        self.slot: dict[tuple, tuple] = {}
        self.lane: dict[tuple, tuple] = {}
        self.lds = ZERO
        self.cdep: dict[int, tuple] = {}   # branch block -> condition value, until its post-dominator
        self.pending: list[tuple] = []   # vm ops in issue order: ('load'|'store', value at completion)
        self.drained = ZERO              # deepest store retired by a vmcnt wait since the last barrier
        self.stores = ZERO               # deepest store completion seen, for the release at the region end
        self.issued = ZERO               # deepest store source, the chain at the store issue
        self.mask: dict[str, str] = {'exec': 'E'}   # 64-bit masks and scc: 'Z', 'E', or '0'/'1' for scc
        self.pin: dict[str, int] = {}               # wave-uniform VGPRs with a known value
        self.pin_at: dict[int, tuple] = {}          # line -> (register, value): pin set by the def at that line
        self.watch: frozenset = frozenset()         # lines whose address dependence is recorded
        self.seen = ZERO                            # deepest address dependence at a watched line
        self.fm: dict[str, frozenset] = {'exec': frozenset()}   # lane formulas: conjunctions of branch literals
        self.kill = False                           # lane model 'kill': an ambiguous read sees the last write only
        self.fval: dict[frozenset, tuple] = {frozenset(): ZERO}   # exec formula -> exec value when exec had it
        self.vprev: dict[str, tuple] = {}           # VGPR -> (lanes of the last write: superset formula or None,
                                                    #          formula is exact, value of the other lanes)

    def copy(self):
        s = State()
        s.r = dict(self.r); s.slot = dict(self.slot); s.lane = dict(self.lane)
        s.lds, s.cdep, s.pending, s.drained, s.stores = self.lds, dict(self.cdep), list(self.pending), self.drained, self.stores
        s.issued = self.issued
        s.mask = dict(self.mask); s.pin = dict(self.pin); s.pin_at = self.pin_at
        s.watch, s.seen = self.watch, self.seen
        s.fm = dict(self.fm); s.vprev = dict(self.vprev); s.kill = self.kill; s.fval = dict(self.fval)
        return s

    def merge(self, o: 'State'):
        r_self = {k: self.r.get(k, ZERO) for k in set(self.vprev) | set(o.vprev)}   # before the merge below
        for d, od in ((self.r, o.r), (self.slot, o.slot), (self.lane, o.lane), (self.cdep, o.cdep)):
            for k, v in od.items():
                d[k] = vmax(d.get(k, ZERO), v)
        self.lds = vmax(self.lds, o.lds)
        self.drained = vmax(self.drained, o.drained); self.stores = vmax(self.stores, o.stores)
        self.issued = vmax(self.issued, o.issued)
        self.mask = {k: v for k, v in self.mask.items() if o.mask.get(k) == v}
        self.pin = {k: v for k, v in self.pin.items() if o.pin.get(k) == v}
        self.seen = vmax(self.seen, o.seen)
        ex = _f_or(self.fm.get('exec', frozenset()), o.fm.get('exec', frozenset()))
        self.fm = {k: _f_or(v, o.fm[k]) for k, v in self.fm.items() if k != 'exec' and k in o.fm}
        self.fm['exec'] = ex
        for k, v in o.fval.items():
            self.fval[k] = vmax(self.fval.get(k, ZERO), v)
        vp = {}
        for k in set(self.vprev) | set(o.vprev):
            a, b = self.vprev.get(k), o.vprev.get(k)
            if a and b and a[0] is not None and b[0] is not None and a[0] != b[0] and (a[0] <= b[0] or b[0] <= a[0]):
                # one write covers the lanes of the other: keep the narrower lanes, whose outside holds
                # the old values of both paths and the wider write
                narrow, wide, wr = (b, a, r_self) if a[0] <= b[0] else (a, b, o.r)
                vp[k] = (narrow[0], narrow[1], vmax(narrow[2], wide[2], wr.get(k, ZERO)))
            elif a and b:
                same = a[0] == b[0]
                w = a[0] if same else (a[0] & b[0] if a[0] is not None and b[0] is not None else None)
                vp[k] = (w, same and a[1] and b[1], vmax(a[2], b[2]))
            else:
                w, exact, prev = a or b
                vp[k] = (w, exact, vmax(prev, o.r.get(k, ZERO) if a else r_self[k]))
        self.vprev = vp
        if len(o.pending) > len(self.pending):
            self.pending = list(o.pending)

    def get(self, names):
        return vmax(*(self.read(n) for n in names)) if names else ZERO

    def read(self, n):
        """One register under the current exec. A VGPR written under a partial exec keeps its old value in
        the other lanes. A read under a disjoint exec sees the old value only, a read inside the written
        lanes sees the new value only, other reads see both."""
        v = self.r.get(n, ZERO)
        p = self.vprev.get(n)
        if p is None:
            return v
        w, exact, prev = p
        ex = self.fm.get('exec', frozenset())
        if w is not None and _disjoint(w, ex):
            return prev
        if self.kill or (exact and w is not None and w <= ex):
            return v
        return vmax(v, prev)

    def setv(self, n, v):
        """Write one register. A VGPR write under a partial exec records the old value of the other lanes."""
        if n[0] in 'va' and n != 'vcc' and n[1:].isdigit():
            ex = self.fm.get('exec', frozenset())
            if not ex:
                self.vprev.pop(n, None)
            else:
                p = self.vprev.get(n)
                if p is not None and p[0] == ex and p[1]:
                    pass                                   # same lanes again: the other lanes keep their value
                else:
                    old = self.read(n) if p is None else vmax(self.r.get(n, ZERO), p[2])
                    self.vprev[n] = (ex, True, old)
        self.r[n] = v

    @property
    def ctrl(self):
        """Control of a vector memory operation: the scoped branch conditions and exec."""
        return vmax(self.r.get('exec', ZERO), *self.cdep.values())

    @property
    def sctrl(self):
        """Control of a scalar operation: the scoped branch conditions only."""
        return vmax(*self.cdep.values()) if self.cdep else ZERO


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


# 'phi': a read in lanes that are neither inside nor outside the last partial write sees both values (sound).
# 'kill': it sees the last write only (the compiler's phi inputs from other lanes are often undefined).
LANE_MODELS = ('phi', 'kill')
BOT = frozenset({(('false',), True), (('false',), False)})   # no lanes: a contradiction, disjoint from all


def _f_or(a: frozenset, b: frozenset) -> frozenset:
    """Superset formula of the union of two lane sets."""
    return b if a == BOT else (a if b == BOT else a & b)


def _disjoint(a: frozenset, b: frozenset) -> bool:
    return any((atom, not pol) in b for atom, pol in a)


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


@dataclass
class Region:
    blocks: list[Block]
    first_line: int
    last_line: int
    entry: int = 0


def region(path_or_text: str | Path, func: str | None = None, start: str | None = None,
           stop: tuple[str, ...] = (), lines: tuple[int, int] | None = None) -> Region:
    """Blocks of a function, of a label range or of a line range. The region runs from `start` to the
    first label in `stop`, in layout order."""
    text = Path(path_or_text).read_text() if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and '\n' not in path_or_text) else path_or_text
    all_lines = text.splitlines()
    a, b = 1, len(all_lines)
    if func:
        a, b = function_lines(all_lines, func)
    if start:
        a = label_line(all_lines, start)
    if stop:
        b = min(label_line(all_lines, s) for s in stop) - 1
    if lines:
        a, b = lines
    sub = '\n'.join(all_lines[a - 1:b])
    ins = parse(sub, a)
    tl = {i: all_lines[i - 1] for i in range(a, b + 1)}
    return Region(blocks(ins, tl), a, b)


def _cfg(reg: Region):
    """Reverse postorder from the entry, the back edges of the DFS, and the natural loop of each back edge."""
    bl = reg.blocks
    post, back, state = [], set(), {}
    stack = [(reg.entry, iter(bl[reg.entry].succ))]
    state[reg.entry] = 1
    while stack:
        u, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop(); state[u] = 2; post.append(u)
            continue
        if state.get(nxt) == 1:
            back.add((u, nxt))
        elif nxt not in state:
            state[nxt] = 1
            stack.append((nxt, iter(bl[nxt].succ)))
    rpo = post[::-1]
    preds: dict[int, list[int]] = {k: [] for k in rpo}
    for u in rpo:
        for v in bl[u].succ:
            if v in preds:
                preds[v].append(u)
    loops = {}
    for (l, h) in sorted(back):
        body, todo = {h, l}, [l]
        while todo:
            x = todo.pop()
            if x == h:
                continue
            for p in preds[x]:
                if p not in body:
                    body.add(p); todo.append(p)
        loops[(h, l)] = body
    return _loop_order(bl, rpo, back, loops), back, loops


def _ipdom(reg: Region, nodes: list[int]) -> dict[int, int | None]:
    """Immediate post-dominator of each block. Exit blocks lead to one virtual exit, -1."""
    bl = reg.blocks
    ns = set(nodes)
    succ = {k: [v for v in bl[k].succ if v in ns] + ([-1] if bl[k].exit or not bl[k].succ else []) for k in nodes}
    full = set(nodes) | {-1}
    pdom = {k: set(full) for k in nodes}
    pdom[-1] = {-1}
    changed = True
    order = list(reversed(nodes))
    while changed:
        changed = False
        for k in order:
            ss = succ[k]
            new = set.intersection(*(pdom[v] for v in ss)) if ss else set()
            new = new | {k}
            if new != pdom[k]:
                pdom[k] = new
                changed = True
    out = {}
    for k in nodes:
        cands = pdom[k] - {k}
        # the immediate one is post-dominated by all the others
        imm = None
        for c in cands:
            if all(c == d or d in pdom.get(c, {-1}) for d in cands):
                imm = c
                break
        out[k] = imm
    return out


def _loop_order(bl, rpo, back, loops):
    """Topological order of the forward edges. Ready blocks of the most recently opened loop go first, so
    each loop body is complete before its exits."""
    nodes = set(rpo)
    indeg = {k: 0 for k in rpo}
    for u in rpo:
        for v in bl[u].succ:
            if v in nodes and (u, v) not in back:
                indeg[v] += 1
    rank = {k: n for n, k in enumerate(rpo)}
    heads = {}
    for (h, l), body in loops.items():
        heads.setdefault(h, set()).update(body)
    ready, out, open_loops = [rpo[0]], [], []
    done = set()
    while ready:
        pick = None
        for body in reversed(open_loops):
            c = [k for k in ready if k in body]
            if c:
                pick = min(c, key=rank.get)
                break
        if pick is None:
            pick = min(ready, key=rank.get)
        ready.remove(pick)
        out.append(pick); done.add(pick)
        if pick in heads:
            open_loops.append(heads[pick])
        open_loops = [b for b in open_loops if not b <= done]
        for v in bl[pick].succ:
            if v in nodes and (pick, v) not in back:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
    out += [k for k in rpo if k not in done]      # irreducible parts keep reverse postorder
    return out


def run(reg: Region, prices: Prices, extra: dict[tuple[int, int], int] | None = None,
        scratch_rsrc: str | None = None, entry: State | None = None, barrier_drain: bool = False,
        pins: dict[str, int] | None = None, watch=(), lanes: str = 'phi') -> State:
    """Forward pass in reverse postorder over the forward edges. A back edge (h, l) with extra[(h, l)] = k
    runs the natural loop of that edge k more times, with the latch state fed back into the header."""
    bl = reg.blocks
    rpo, back, loops = _cfg(reg)
    order = {k: n for n, k in enumerate(rpo)}
    ipd = _ipdom(reg, rpo)
    extra = dict(extra or {})
    first = entry.copy() if entry else State()
    first.watch = frozenset(watch)
    if lanes not in LANE_MODELS:
        raise ValueError(f'lanes must be one of {LANE_MODELS}')
    first.kill = lanes == 'kill'
    for k, v in (pins or {}).items():          # 'v6' pins at entry, 'v6@103144' pins at the def on that line
        if '@' in k:
            r, ln = k.split('@')
            first.pin_at[int(ln)] = (r, v)
        else:
            first.pin[k] = v
    ins: dict[int, State] = {reg.entry: first}
    end = State()

    def do_block(k, st, inside=None, sink=None):
        for b in [b for b in st.cdep if b >= 0 and ipd.get(b) == k]:
            del st.cdep[b]
        for i in bl[k].insns:
            step(st, i, prices, scratch_rsrc, barrier_drain, k)
        last = bl[k].insns[-1] if bl[k].insns else None
        tgt_idx = fall_idx = None
        if last is not None and last.mnem.startswith('s_cbranch'):
            fall_idx = k + 1 if (k + 1) in bl[k].succ else None
            tgt_idx = next((v for v in bl[k].succ if v != fall_idx), None) if len(bl[k].succ) > (1 if fall_idx is not None else 0) else None
            ok = set(feasible(st, last, tgt_idx if tgt_idx is not None else -1, fall_idx if fall_idx is not None else -2))
            succ = [v for v in bl[k].succ if v in ok]
            leaves = (-1 in ok) or (-2 in ok) or bl[k].exit and not succ
        else:
            succ = bl[k].succ
            leaves = bl[k].exit or not bl[k].succ
        if leaves:
            end.merge(st)
        for v in succ:
            if (k, v) in back:
                continue
            tgt = sink if (inside is not None and v in inside) else ins
            if v in tgt:
                tgt[v].merge(st)
            else:
                tgt[v] = st.copy()
        return st

    for k in rpo:
        st = ins.pop(k, None)
        if st is None:
            continue
        out = do_block(k, st)
        for (h, l), n in extra.items():
            if l != k or (k, h) not in back:
                continue
            body = sorted(loops[(h, l)], key=order.get)
            carry = out
            for _ in range(n):
                local: dict[int, State] = {h: carry.copy()}
                for b in body:
                    stb = local.pop(b, None)
                    if stb is None:
                        continue
                    o = do_block(b, stb, inside=set(body), sink=local)
                    if b == l:
                        carry = o
    return end


def _loops(reg: Region):
    return sorted((h, l) for (l, h) in _cfg(reg)[1])


def chain_end(st: State, with_drain: bool = True, end: str = 'publish'):
    """'publish': the event publish waits for the stores and for the control path. 'regs': the deepest register."""
    if end == 'regs':
        return vmax(st.ctrl, st.lds, *st.r.values(), *([st.stores] if with_drain else []))
    return vmax(st.ctrl, st.stores if with_drain else st.issued)


def counts(v) -> dict[str, int]:
    c = {k: 0 for k in CLASSES}
    for _, _, cls in v[1]:
        c[cls] += 1
    return c


def trip_source(reg: Region, loop: tuple[int, int]) -> str:
    """The compare that sets the condition of the latch branch, searched back through the loop in layout order."""
    h, l = loop
    bl = reg.blocks
    last = bl[l].insns[-1] if bl[l].insns else None
    cands = [i for k in range(min(h, l), l + 1) for i in bl[k].insns] if h <= l else list(bl[l].insns)
    for ins in reversed(cands[:-1]):
        if ins.mnem.startswith(('s_cmp', 'v_cmp')):
            imms = [o for o in ins.ops if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', o)]
            if imms and not any(regs(o) and regs(o)[0][0] in 'v' for o in ins.ops):
                return f'constant {int(imms[0], 0)} at line {ins.line}'
            return f'runtime, compare at line {ins.line}'
    return f'unknown, branch at line {last.line if last else "?"}'


def analyse(reg: Region, prices: Prices = Prices(), scratch_rsrc: str | None = None, with_drain: bool = True,
            end: str = 'publish', entry: State | None = None, barrier_drain: bool = False,
            pins: dict[str, int] | None = None, watch=(), lanes: str = 'phi') -> dict:
    """`watch`: lines of the first loads of a stream. The result gives the chain before those loads.
    `lanes`: the lane model, see LANE_MODELS."""
    base_end = run(reg, prices, scratch_rsrc=scratch_rsrc, entry=entry, barrier_drain=barrier_drain, pins=pins,
                   watch=watch, lanes=lanes)
    v0 = chain_end(base_end, with_drain, end)
    loops = []
    for lp in _loops(reg):
        e1 = chain_end(run(reg, prices, {lp: 1}, scratch_rsrc, entry, barrier_drain, pins, lanes=lanes), with_drain, end)
        e2 = chain_end(run(reg, prices, {lp: 2}, scratch_rsrc, entry, barrier_drain, pins, lanes=lanes), with_drain, end)
        g1, g2 = counts(e1), counts(e2)
        c0 = counts(v0)
        first = {k: g1[k] - c0[k] for k in CLASSES}
        per = {k: g2[k] - g1[k] for k in CLASSES}      # steady state: the growth from the 2nd to the 3rd trip
        dcost = round(e2[0] - e1[0], 4)
        loops.append({'header_line': reg.blocks[lp[0]].insns[0].line if reg.blocks[lp[0]].insns else reg.blocks[lp[0]].start,
                      'latch_line': reg.blocks[lp[1]].insns[-1].line,
                      'header': reg.blocks[lp[0]].label, 'hops_per_extra_trip': per, 'cost_per_extra_trip_us': dcost, 'linear': per == first,
                      'first_extra_trip': {'hops': first, 'cost_us': round(e1[0] - v0[0], 4)},
                      'trip_count': trip_source(reg, lp)})
    return {'lines': [reg.first_line, reg.last_line], 'hops': counts(v0), 'cost_us': round(v0[0], 4),
            'chain': [{'line': l, 'insn': m, 'class': c} for l, m, c in v0[1]],
            'with_drain': with_drain, 'end': end, 'barrier_drain': barrier_drain, 'pins': dict(pins or {}),
            'loops': [x for x in loops if x['cost_per_extra_trip_us'] > 1e-9],
            'loops_without_carried_hops': sum(1 for x in loops if x['cost_per_extra_trip_us'] <= 1e-9),
            'scratch_reloads_on_chain': counts(v0)['scratch'],
            'before_watch': ({'hops': counts(base_end.seen), 'cost_us': round(base_end.seen[0], 4),
                              'chain': [{'line': l, 'insn': m, 'class': c} for l, m, c in base_end.seen[1]]}
                             if watch else None),
            'lanes': lanes,
            'prices': {'rtt_us': prices.rtt, 'cached_us': prices.cached, 'preset': prices.preset}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    ap.add_argument('asm')
    ap.add_argument('--func')
    ap.add_argument('--start')
    ap.add_argument('--stop', action='append', default=[])
    ap.add_argument('--lines', help='A:B, 1-based and inclusive')
    ap.add_argument('--rtt', type=float, default=0.50)
    ap.add_argument('--preset', choices=('warm', 'acquired'), default='warm')
    ap.add_argument('--no-drain', action='store_true')
    ap.add_argument('--end', choices=('publish', 'regs'), default='publish')
    a = ap.parse_args(argv)
    ln = tuple(int(x) for x in a.lines.split(':')) if a.lines else None
    reg = region(Path(a.asm), a.func, a.start, tuple(a.stop), ln)
    print(json.dumps(analyse(reg, Prices(a.rtt, preset=a.preset), with_drain=not a.no_drain, end=a.end), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ---- regions of an inlined megakernel ------------------------------------------------------------------------
def owner_of(frames: list, stop_after: int = 8758) -> str:
    """Owner of one instruction from its inline chain, outermost frame last. A tile body owns its instructions.
    Dispatcher code after the body call, the release and the publish, is 'post'."""
    for name, loc in reversed(frames):
        if name.startswith(('etx_tile_', 'vl_prologue', 'vl_argmax')):
            return name
        if name.startswith(('etx_touch_', 'etx_pre_')):
            return 'touch'
    for name, loc in frames:
        if name == 'etx_run_task':
            try:
                ln = int(loc.split(':')[1])
            except (IndexError, ValueError):
                return 'dispatch'
            return 'post' if ln > stop_after else 'dispatch'
    return 'dispatch'


def owned_region(path: Path, func: str, owners: dict[int, str], tile: str, after_line: int = 0) -> Region:
    """Blocks of `func` that one inlined tile body reaches. The walk starts at the first block with an
    instruction of `tile` after `after_line`. It stops at blocks with instructions of another owner,
    except plain dispatch code such as the switch on the body symbol."""
    full = region(path, func=func)
    bl = full.blocks
    def own(b):
        return {owners.get(i.line, 'dispatch') for i in b.insns}
    entry = next(k for k, b in enumerate(bl) if b.insns and b.insns[0].line > after_line and tile in own(b))
    seen, todo = {entry}, [entry]
    while todo:
        k = todo.pop()
        for s in bl[k].succ:
            if s in seen:
                continue
            o = own(bl[s])
            if o - {tile, 'dispatch'}:
                continue
            seen.add(s); todo.append(s)
    keep = sorted(seen)
    pos = {k: n for n, k in enumerate(keep)}
    sub = []
    for k in keep:
        b = bl[k]
        succ = [pos[s] for s in b.succ if s in pos]
        sub.append(Block(b.label, b.insns, succ, b.start, b.exit or len(succ) < len(b.succ) or not b.succ))
    lines = [i.line for b in sub for i in b.insns]
    return Region(sub, min(lines), max(lines), pos[entry])
