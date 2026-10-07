"""CPU-only overlap and serial floors for one Qwen3-30B-A3B decode step on MI300X.

All times are microseconds. This is NOT a measurement. It splits the measured layer period
into four parts and gives the floor that each lever can reach.

Model of one layer, phase by phase:
* stream  = phase bytes / device bandwidth.
* skeleton = max(0, critical body - achievable stream time). The BUDGET table gives both
  columns. It is the part of the critical task that the stream does not explain.
* serial floor of a phase = max(stream, skeleton). It assumes no prefetch across a barrier.
* overlap floor of the step = max(total stream, chain of skeletons + handoffs). It assumes that
  every stream hides under earlier phases, with unlimited LDS, L2 and MALL capacity.

Assumptions:
* The BUDGET table (2026-10-01 trace, 5.082 ms) describes the bodies. The shipped build is
  newer and its bodies are 2.80 us per layer shorter (BUDGET shipped-body table).
* The 9 handoffs per layer stay at the measured 1.00 us each.
* prologue, in_norm and argmax keep their measured round11 device time. lm_head streams at
  device bandwidth.
* The skeleton is a lower bound. In-tile ablation (gemv.py NOVMEM) gives larger MoE skeletons.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 'docs/BUDGET-2026-10-01.md'
LAT = 'docs/LATENCY-2026-10-07.md'
GEMV = 'etx/ops/gemv.py'
PHASES = ('qkv', 'prep', 'attn', 'reduce', 'o_proj', 'router', 'topk', 'moe_up', 'moe_down')
LAYERS = 48
MEASURED_SPAN_US = 5081.5
BANDWIDTHS_TBS = (4.3, 3.5)


def source(path: str, text: str, after: str | None = None) -> tuple[str, str]:
    """Return (location, line) for an exact repository line; fail closed if it disappears.

    With `after`, the search starts below the first line that contains `after`.
    """
    seen = after is None
    for line, value in enumerate((ROOT / path).read_text().splitlines(), 1):
        if not seen:
            seen = after in value
        elif text in value:
            return f'{path}:{line}', value
    raise ValueError(f'missing source: {path}: {text}')


def cells(line: str) -> list[str]:
    return [c.strip().strip('*') for c in line.strip().strip('|').split('|')]


@dataclass(frozen=True)
class Phase:
    name: str
    mb: float
    mean_body: float
    crit_body: float
    achievable: float | None
    where: str


def budget_phases() -> dict[str, Phase]:
    """Read the per-layer BUDGET table. 'latency' in the achievable column means zero stream time."""
    out = {}
    for name in PHASES:
        where, line = source(BUDGET, f'| {name} | ', after='## The table (one layer')
        row = cells(line)
        if len(row) != 10:
            raise ValueError(f'unexpected BUDGET row: {where}')
        ach = None if row[7] == 'latency' else float(row[7])
        out[name] = Phase(name, float(row[3]), float(row[4]), float(row[5]), ach, where)
    return out


def measured_us(phase: str) -> tuple[float, str]:
    where, line = source(LAT, f'| {phase} | ')
    return float(cells(line)[2]), where


def handoff_us_per_layer() -> tuple[float, str]:
    where, line = source(BUDGET, '| handoff, 9 edges at 1.00 us |')
    return float(cells(line)[1]), where


def start_delay_us_per_layer() -> tuple[float, str]:
    where, line = source(BUDGET, '| start delay of the critical task |')
    return float(cells(line)[1]), where


def novmem_skeleton() -> tuple[dict[str, float], str]:
    where, line = source(GEMV, 'NOVMEM  (one weight vector)')
    up, down = [float(x) for x in line.split(')')[1].split()[:2]]
    return {'moe_up': up, 'moe_down': down}, where


def floors(bw_tbs: float = 4.3, use_novmem: bool = False) -> dict:
    ph = budget_phases()
    hand, hand_src = handoff_us_per_layer()
    start, start_src = start_delay_us_per_layer()
    lm_bytes = float(cells(source(LAT, '| lm_head | ')[1])[1])
    small = {p: measured_us(p)[0] for p in ('prologue', 'in_norm', 'argmax')}
    nov, nov_src = novmem_skeleton()
    rows = {}
    for p in ph.values():
        stream = p.mb / bw_tbs
        skel = max(0.0, p.crit_body - (p.achievable or 0.0))
        if use_novmem and p.name in nov:
            skel = max(skel, nov[p.name])
        floor = max(stream, skel)
        rows[p.name] = {'stream': stream, 'skeleton': skel, 'serial': floor,
                        'crit_body': p.crit_body, 'body_excess': p.crit_body - floor}
    tail_stream = lm_bytes / 1e6 / bw_tbs
    tail = tail_stream + sum(small.values())
    stream_total = LAYERS * sum(r['stream'] for r in rows.values()) + tail_stream
    chain = LAYERS * (sum(r['skeleton'] for r in rows.values()) + hand) + sum(small.values())
    serial = LAYERS * (sum(r['serial'] for r in rows.values()) + hand) + tail
    overlap = max(stream_total, chain)
    layer_measured = sum(p.crit_body for p in ph.values()) + start + hand
    body_excess = LAYERS * sum(r['body_excess'] for r in rows.values())
    return {
        'bandwidth_tbs': bw_tbs,
        'skeleton_source': 'crit body - achievable' + (' and NOVMEM' if use_novmem else ''),
        'bandwidth_floor_us': stream_total,
        'skeleton_chain_us': chain,
        'overlap_floor_us': overlap,
        'serial_floor_us': serial,
        'measured_span_us': MEASURED_SPAN_US,
        'prefetch_gain_max_us': serial - overlap,
        'non_prefetch_gain_max_us': MEASURED_SPAN_US - serial,
        'split_of_non_prefetch_gain_us': {
            'body_excess': body_excess,
            'start_delay': LAYERS * start,
            'rest': MEASURED_SPAN_US - serial - body_excess - LAYERS * start,
        },
        'body_excess_by_phase_us': {k: LAYERS * v['body_excess'] for k, v in rows.items()},
        'utilization_ceiling': {'serial': stream_total / serial, 'overlap': stream_total / overlap},
        'layer_period_check_us': {'budget_parts': layer_measured, 'budget_table': 101.83},
        'per_layer': rows,
        'sources': {'phases': {k: v.where for k, v in ph.items()}, 'handoff': hand_src,
                    'start_delay': start_src, 'novmem': nov_src},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--novmem', action='store_true', help='use the larger in-tile MoE skeletons')
    args = parser.parse_args()
    print(json.dumps([floors(bw, args.novmem) for bw in BANDWIDTHS_TBS], indent=2))


if __name__ == '__main__':
    main()
