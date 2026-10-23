"""State: hops."""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

BOT = frozenset({(('false',), True), (('false',), False)})  # no lanes

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


def _f_or(a: frozenset, b: frozenset) -> frozenset:
    """Superset formula of the union of two lane sets."""
    return b if a == BOT else (a if b == BOT else a & b)


def _disjoint(a: frozenset, b: frozenset) -> bool:
    return any((atom, not pol) in b for atom, pol in a)
