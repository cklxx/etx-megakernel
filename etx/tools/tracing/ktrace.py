"""Unadjusted rocprof device durations; never subtract profiler overhead.

Boundary report mode measures clock disagreement across ranges, never vendor per-kernel durations.
Plan text parsing is shared with compare; timeline reads a different JSON/raw format.
Sampling specification: /Users/ckl/.hermes/cache/scratch/o55-kernel-compare.json:1.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from examples.vllm_llm.compare import read_vl_plan
from examples.vllm_llm.vllm_profile import bursts
from etx.tools.costs.floors import PHASES


def read_csv(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('Start_Timestamp', 'End_Timestamp'):
            r[k] = int(r[k])
        if r['End_Timestamp'] < r['Start_Timestamp']:
            raise ValueError('negative duration')
    return sorted(rows, key=lambda r: r['Start_Timestamp'])


def duration(r):
    return (r['End_Timestamp'] - r['Start_Timestamp']) / 1000


def summarize(groups):
    return {k: {'median_us': median(v), 'samples': len(v)} for k, v in groups.items() if v}


def etx(path, plan, steps=8):
    inst = read_vl_plan(plan)
    if [r['n'] for r in inst] != list(range(len(inst))):
        raise ValueError('plan instances are not contiguous')
    rows = [r for r in read_csv(path) if r['Kernel_Name'] == 'etx_unfused_d0']
    # Launch order, not overlapping device start order (host.hip:747-750).
    rows.sort(key=lambda r: int(r['Dispatch_Id']))
    per = len(inst) + 2
    if len(rows) != steps * per:
        raise ValueError(f'incomplete trace: {len(rows)} dispatches; expected {steps * per}')
    if len({r['Agent_Id'] for r in rows}) != 1:
        raise ValueError('expected one GPU')
    groups = defaultdict(list)
    for step in range(1, steps):
        for i, r in zip(inst, rows[step * per + 1:(step + 1) * per - 1]):
            if 1 <= i['layer'] <= 45 or i['role'] == 'lm_head':
                groups[i['role']].append(duration(r))
    out = summarize(groups)
    return {p: out[p] for p in (*PHASES, 'lm_head') if p in out and p not in ('attn', 'reduce')}


def vllm(path):
    rows = read_csv(path)
    last = bursts([(r['Start_Timestamp'], r['End_Timestamp'], r['Kernel_Name']) for r in rows])[-1]
    lo = last[0][0]
    rows = [r for r in rows if r['Start_Timestamp'] >= lo]
    tensile = [i for i, r in enumerate(rows) if 'Cijk_' in r['Kernel_Name']]
    if not tensile:
        raise ValueError('cannot locate prefill/decode boundary')
    rows = rows[tensile[-1] + 1:]
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        n = r['Kernel_Name']
        prev = rows[i - 1]['Kernel_Name'] if i else ''
        nxt = rows[i + 1]['Kernel_Name'] if i + 1 < len(rows) else ''
        role = None
        if 'wvSplitK' in n:
            if ', 64, 1,' in n:
                role = 'router'
            else:
                groups['wvSplitK_all'].append(duration(r))
                if 'paged_attention' in prev:
                    role = 'o_proj'
                elif 'bfloat16tofloat32' in nxt:
                    role = 'lm_head'
                elif 'rms_norm' in prev:
                    role = 'qkv'
        elif 'fused_moe_kernel' in n:
            role = 'moe_down' if 'act_and_mul' in prev else 'moe_up'
        elif 'topkGating' in n:
            role = 'topk'
        elif 'paged_attention' in n:
            role = 'reduce' if 'reduce' in n else 'attn'
        if role:
            groups[role].append(duration(r))
    return summarize(groups)


def _vendor_ranges(trace_dir):
    kernels, markers = [], []
    for p in Path(trace_dir).rglob('*kernel_trace.csv'):
        kernels.extend(read_csv(p))
    for p in Path(trace_dir).rglob('*marker_api_trace.csv'):
        markers.extend(read_csv(p))
    kernels.sort(key=lambda k: k['Start_Timestamp'])
    starts = [k['Start_Timestamp'] for k in kernels]
    ranges = []
    for r in markers:
        label = r.get('Function', r.get('Name', ''))
        if label.startswith('kcmp:'):
            left = bisect_left(starts, r['Start_Timestamp'])
            right = bisect_right(starts, r['End_Timestamp'])
            ranges.append((label, r, kernels[left:right]))
    return ranges


def _boundary_crossing(label, marker, inside):
    worst = max(inside, key=lambda k: k['End_Timestamp'], default=None)
    if worst is None or worst['End_Timestamp'] <= marker['End_Timestamp']:
        return None
    return dict(label=label, in_range=len(inside),
                overhang_us=(worst['End_Timestamp'] - marker['End_Timestamp']) / 1000,
                kernel=worst['Kernel_Name'], kernel_end_us=worst['End_Timestamp'] / 1000,
                marker_end_us=marker['End_Timestamp'] / 1000)


def _boundary_summary(lib, ranges, crossings):
    overhang = max((c['overhang_us'] for c in crossings), default=0)
    print(f'BOUNDARY-SUMMARY lib={lib} ranges={len(ranges)} '
          f'crossings={len(crossings)} max_overhang_us={overhang:.1f}', flush=True)


def vendor(trace_dir, metadata, lib=''):
    """Sum device durations inside synchronized ROCTx row ranges, excluding gaps."""
    ranges, crossings = [], []
    try:
        ranges = _vendor_ranges(trace_dir)
        crossings = [c for args in ranges if (c := _boundary_crossing(*args)) is not None]
        return _vendor_measure(ranges, metadata)
    finally:
        _boundary_summary(lib, ranges, crossings)


def _vendor_measure(ranges, metadata):
    groups, names = defaultdict(list), defaultdict(set)
    seen = set()
    for label, r, inside in ranges:
        _, shape, it, layer = label.split(':')
        key = (shape, int(it), int(layer))
        if key in seen:
            raise ValueError('duplicate vendor range')
        seen.add(key)
        crossing = _boundary_crossing(label, r, inside)
        if crossing is not None:
            fields = ' '.join(f'{key}={value:.1f}' if key.endswith('_us') else f'{key}={value}'
                              for key, value in crossing.items())
            print(f'BOUNDARY-CROSS {fields}', flush=True)
            raise ValueError(f'kernel crosses vendor range boundary: {label}')
        if not inside:
            raise ValueError(f'empty vendor range: {label}')
        groups[shape].append(sum(duration(k) for k in inside))
        names[shape].update(k['Kernel_Name'] for k in inside)
    out = {}
    for shape, m in metadata.items():
        if not m.get('check_ok'):
            out[shape] = {'refused': m.get('reason', 'correctness not established')}
            continue
        expected = {(shape, it, layer) for it in range(m['iters']) for layer in range(m['copies'])}
        actual = {k for k in seen if k[0] == shape}
        if expected != actual:
            out[shape] = {'refused': 'incomplete timed ranges'}
            continue
        us = median(groups[shape])
        if us < m['bytes'] / 5.3e6:
            out[shape] = {'refused': 'cache-contaminated: below bytes / 5.3 TB/s'}
            continue
        out[shape] = {'median_us': us, 'samples': len(groups[shape]), 'clock': 'rocprof',
                      'check_ok': True, 'kernel_names': sorted(names[shape]), 'bytes': m['bytes'],
                      'check_basis': m.get('check_basis')}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', nargs='?')
    ap.add_argument('--trace-dir')
    ap.add_argument('--boundaries', action='store_true')
    ap.add_argument('--plan')
    a = ap.parse_args(argv)
    if a.boundaries:
        if not a.trace_dir or a.csv or a.plan:
            ap.error('--boundaries requires --trace-dir and cannot be used with csv or --plan')
        ranges = _vendor_ranges(a.trace_dir)
        crossings = [c for args in ranges if (c := _boundary_crossing(*args)) is not None]
        crossings.sort(key=lambda c: c['overhang_us'], reverse=True)
        print(json.dumps([{k: round(v, 1) if k.endswith('_us') else v for k, v in c.items()}
                          for c in crossings], indent=2))
        _boundary_summary('', ranges, crossings)
        return
    if not a.csv or a.trace_dir:
        ap.error('csv is required unless using --trace-dir with --boundaries')
    print(json.dumps(etx(a.csv, a.plan) if a.plan else vllm(a.csv), indent=2))


if __name__ == '__main__':
    main()
