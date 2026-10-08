"""Exact hop counts per decode phase from the gfx942 ISA of the inlined megakernel.

The megakernel inlines every tile body into `etx_megakernel_d0`. Thus a phase is an inlined region, not a
function. Steps:

1. `map`: compile the same source again with `-gline-tables-only` (the instruction stream must be identical),
   assemble it with llvm-mc, disassemble it with llvm-objdump, and give each address to llvm-symbolizer
   `--inlining`. The result maps each instruction line of the original .s to its inline frames.
2. `phases`: for each phase, take the blocks that the tile body owns (`isa_hops.owned_region`), and run
   `isa_hops.analyse` at each round trip. Imported vLLM kernels with slot copies get pins on the slot register.

See docs/FLOORS-2026-10-08.md, section "ISA hop table", for the commands and the results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from etx.tools import isa_hops as ih

FUNC = 'etx_megakernel_d0'
# phase -> (tile symbol, price preset). 'acquired': the task starts after `buffer_inv sc1`, so plain loads miss.
PHASES = {
    'prologue': ('vl_prologue', 'warm'),
    'in_norm': ('etx_tile_rep_rms_norm_2d', 'acquired'),
    'qkv_0': ('etx_tile_x_dsl_gemv', 'warm'),
    'fold': ('etx_tile_dsl_gemv_fold', 'warm'),          # qkv of layers 1..47 and the router
    'prep': ('etx_tile_dsl_attnprep', 'warm'),
    'attn': ('etx_tile_attn_mfma16_g8', 'acquired'),
    'reduce': ('etx_tile_attn_reduce', 'acquired'),
    'o_proj': ('etx_tile_dsl_gemv', 'warm'),
    'topk': ('etx_tile_rep_dsl_topk', 'warm'),
    'moe_up': ('etx_tile_x_dsl_moe_up_fold', 'warm'),
    'moe_down': ('etx_tile_x_dsl_moe_down_fold', 'warm'),
    'argmax': ('vl_argmax', 'acquired'),
}
# slot register of the imported kernels: (source location of its definition, shift, values to try)
# imp_tiles.hip: slot = tx / 128 (reduce, 8 slots), tx / 256 (in_norm, 4 slots), tx / 256 (attn, 1 slot)
PIN_DEFS = {
    'reduce': ('imp_tiles.hip:165', 7, list(range(8))),
    'in_norm': ('imp_tiles.hip:234', 8, list(range(4))),
    'attn': ('imp_tiles.hip:112', 8, [0]),
}
# the first loads of the weight stream: dsl_gemv_rows, `w[u] = base[...]`
STREAM_LOCS = ('dsl_tiles.hip:102', 'dsl_tiles.hip:106')
RTTS = (0.28, 0.50, 0.69)


def build_map(s_path: Path, dbg_obj: Path, objdump: Path, symbolizer: str) -> dict[int, list]:
    """Inline frames per instruction line of `s_path`, from the debug copy `dbg_obj` and its objdump."""
    lines = s_path.read_text().splitlines()
    start, end = ih.function_lines(lines, FUNC)
    ins_lines = []
    for i in range(start + 1, end):
        s = lines[i - 1].split(';', 1)[0].strip()
        if s and not s.startswith('.') and not s.endswith(':'):
            ins_lines.append((i, s.split()[0]))
    addrs, mn, cur = [], [], None
    for line in objdump.read_text().splitlines():
        m = re.match(r'^[0-9a-f]+ <(\w+)>:', line)
        if m:
            cur = m.group(1)
            continue
        if cur == FUNC:
            m = re.search(r'//\s*([0-9A-F]{12}):', line)
            if m:
                addrs.append(int(m.group(1), 16))
                mn.append(line.split()[0])
    extra = mn[len(ins_lines):]
    if not all(x == 's_nop' for x in extra):
        raise ValueError(f'the debug copy has extra instructions: {extra[:5]}')
    addrs, mn = addrs[:len(ins_lines)], mn[:len(ins_lines)]
    norm = lambda x: re.sub(r'_e(32|64)$', '', x)
    bad = [(i, a, b) for (i, a), b in zip(ins_lines, mn) if norm(a) != norm(b)]
    if bad:
        raise ValueError(f'instruction streams differ, first at line {bad[0]}')
    r = subprocess.run([symbolizer, f'--obj={dbg_obj}', '--inlining', '--functions=short'],
                       input=''.join(f'{a:#x}\n' for a in addrs), capture_output=True, text=True, check=True)
    groups = r.stdout.strip('\n').split('\n\n')
    if len(groups) != len(addrs):
        raise ValueError('symbolizer output does not match the addresses')
    out = {}
    for (ln, _), g in zip(ins_lines, groups):
        fr = g.split('\n')
        out[ln] = [[fr[k], fr[k + 1].rsplit('/', 1)[-1]] for k in range(0, len(fr) - 1, 2)]
    return out


def load_map(path: Path) -> dict[int, list]:
    return {int(k): [[n, loc.rsplit('/', 1)[-1]] for n, loc in v] for k, v in json.loads(path.read_text()).items()}


def find_pins(lines: list[str], frames: dict[int, list]) -> dict[str, tuple[str, list[int]]]:
    """phase -> ('vN@line', values): the `v_ashrrev_i32 vN, shift, ...` that defines the slot register."""
    out = {}
    for ph, (loc, shift, vals) in PIN_DEFS.items():
        tile = PHASES[ph][0]
        hits = []
        for ln, fr in frames.items():
            if fr and fr[0][1].startswith(loc + ':') and any(n == tile or n.startswith('etx_all_blocks') for n, _ in fr):
                m = re.match(rf'v_ashrrev_i32_e32 (v\d+), {shift}, ', lines[ln - 1].strip())
                if m:
                    hits.append((ln, m.group(1)))
        if len(hits) != 1:
            raise ValueError(f'{ph}: expected one slot definition at {loc}, found {hits}')
        ln, reg = hits[0]
        out[ph] = (f'{reg}@{ln}', vals)
    return out


def phases(s_path: Path, frames: dict[int, list], only=None, after_line: int = 0) -> dict:
    lines = s_path.read_text().splitlines()
    owners = {k: ih.owner_of(v) for k, v in frames.items()}
    pins = find_pins(lines, frames)
    res = {}
    for ph, (tile, preset) in PHASES.items():
        if only and ph not in only:
            continue
        reg = ih.owned_region(s_path, FUNC, owners, tile, after_line=after_line)
        watch = {ln for ln, fr in frames.items() if owners.get(ln) == tile and fr and fr[0][0].startswith('dsl_gemv_rows')
                 and fr[0][1].rsplit(':', 1)[0] in STREAM_LOCS and ih.Insn(ln, lines[ln - 1].split()[0], [], '', '').kind == 'vm_load'}
        models = {}
        for lanes in ih.LANE_MODELS:
            by_rtt = {}
            for rtt in RTTS:
                p = ih.Prices(rtt, preset=preset)
                if ph in pins:
                    reg_name, vals = pins[ph]
                    cands = [ih.analyse(reg, p, pins={reg_name: v}, watch=watch, lanes=lanes) for v in vals]
                    best = max(cands, key=lambda r: (r['cost_us'], sum(r['hops'].values())))
                    best['per_pin'] = {str(v): [c['hops'], c['cost_us']] for v, c in zip(vals, cands)}
                    by_rtt[f'{rtt:.2f}'] = best
                else:
                    by_rtt[f'{rtt:.2f}'] = ih.analyse(reg, p, watch=watch, lanes=lanes)
            r = by_rtt['0.50']
            for h in r['chain'] + ((r.get('before_watch') or {}).get('chain') or []):
                fr = frames.get(h['line'], [])
                h['src'] = fr[0][1] if fr else '?'
                h['in'] = [n.split('(')[0] for n, _ in fr[:3]]
            for lp in r['loops']:
                fr = frames.get(lp['header_line'], [])
                lp['header_src'] = [f"{n.split('(')[0]}@{loc}" for n, loc in fr[:3]]
            models[lanes] = by_rtt
        # the floor takes the lower chain: a floor must not exceed the true chain
        low = min(ih.LANE_MODELS, key=lambda m: (models[m]['0.50']['cost_us'], sum(models[m]['0.50']['hops'].values())))
        res[ph] = {'tile': tile, 'preset': preset, 'pin': pins.get(ph), 'region_blocks': len(reg.blocks),
                   'stream_loads': len(watch), 'floor_model': low, 'by_rtt': models[low], 'models': models}
    return res


def summary(res: dict) -> dict:
    """Compact table: counts, cost per round trip, prefix before the stream, chain lines, loops."""
    out = {}
    for ph, r in res.items():
        r5 = r['by_rtt']['0.50']
        bw = r5.get('before_watch')
        out[ph] = {
            'tile': r['tile'], 'preset': r['preset'], 'pin': r['pin'], 'floor_model': r['floor_model'],
            'models': {m: {'hops': v['0.50']['hops'], 'cost_us': v['0.50']['cost_us']} for m, v in r['models'].items()},
            'exact': len({json.dumps(v['0.50']['hops'], sort_keys=True) for v in r['models'].values()}) == 1,
            'hops': r5['hops'], 'cost_us': {k: v['cost_us'] for k, v in r['by_rtt'].items()},
            'scratch_reloads_on_chain': r5['scratch_reloads_on_chain'],
            'chain': [f"{h['line']} {h['class']} {h['insn']} {h['src']}" for h in r5['chain']],
            'before_stream': None if not bw or not r['stream_loads'] else {
                'hops': bw['hops'], 'cost_us': {k: v['before_watch']['cost_us'] for k, v in r['by_rtt'].items()},
                'chain': [f"{h['line']} {h['class']} {h['insn']} {h['src']}" for h in bw['chain']]},
            'loops': [{'header_line': lp['header_line'], 'src': lp['header_src'][:1], 'hops_per_extra_trip': lp['hops_per_extra_trip'],
                       'cost_per_extra_trip_us': lp['cost_per_extra_trip_us'], 'trip_count': lp['trip_count']} for lp in r5['loops']],
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('map', help='inline frames per .s line from a -gline-tables-only copy')
    m.add_argument('s'); m.add_argument('dbg_obj'); m.add_argument('objdump'); m.add_argument('out')
    m.add_argument('--symbolizer', default='llvm-symbolizer')
    p = sub.add_parser('phases', help='exact hop counts per phase')
    p.add_argument('s'); p.add_argument('map'); p.add_argument('out')
    p.add_argument('--only', default='')
    p.add_argument('--after-line', type=int, default=0, help='skip tile code above this line (the touch hooks)')
    p.add_argument('--summary', action='store_true', help='write the compact table only')
    a = ap.parse_args(argv)
    if a.cmd == 'map':
        Path(a.out).write_text(json.dumps(build_map(Path(a.s), Path(a.dbg_obj), Path(a.objdump), a.symbolizer)))
        return 0
    res = phases(Path(a.s), load_map(Path(a.map)), [x for x in a.only.split(',') if x], a.after_line)
    Path(a.out).write_text(json.dumps(summary(res) if a.summary else res, indent=1))
    for ph, r in summary(res).items():
        print(ph, r['floor_model'], r['hops'], r['cost_us'], 'scratch', r['scratch_reloads_on_chain'],
              'before stream', r['before_stream'] and r['before_stream']['hops'], 'exact' if r['exact'] else r['models'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
