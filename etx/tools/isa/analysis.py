"""Analysis: hops."""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .syntax import Block, CLASSES, blocks, function_lines, label_line, parse, regs
from .execute import LANE_MODELS, feasible, step
from .state import Prices, State, vmax

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
