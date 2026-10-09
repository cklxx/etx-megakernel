"""Report only complete comparable kernel measurements; retain all refusals."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

from etx.tools.ktrace import etx, vllm
from examples.sessions import kcmp
from examples.vllm_llm.vendor_gemv import SHAPES, LIBS
from examples.vllm_llm import nvref_report
from examples.vllm_llm.pairs import Refusal

HISTORICAL = {
    'A': 'results_2026-09-27/vlnext/moe/prof/8f2e8af04e27/7_kernel_trace.csv',
    'C': 'results_2026-09-28b/vprof_qknr/1b7d92084d3f/7_kernel_trace.csv',
}


def decide(gains, phase):
    if len(gains) != 3:
        return 'NOT MEASURED: incomplete pairs'
    med, spread = median(gains), max(gains) - min(gains)
    close = 30 if phase == 'lm_head' else 0.63
    if min(gains) < 0 < max(gains):
        return 'CLOSE: mixed signs; more pairs require owner request'
    if phase == 'lm_head' and med < close and all(g >= 0.63 for g in gains) and med > 2 * spread:
        return 'AMBIGUOUS: lm_head close and vendor-win thresholds both hold'
    if med < close:
        return 'CLOSE: below row threshold'
    if all(g >= 0.63 for g in gains) and med > 2 * spread:
        return 'VENDOR WINS: record only; no tuning authorization'
    return 'INCONCLUSIVE: spread or pair threshold'


def nvref_evidence(root):
    session = nvref_report.Session(root)
    for name in ('gate_all', 'gate_unf'):
        text = session.read(name)
        nvref_report.correctness(text, name)
        if not re.search(r'^GATE-OK$', text, re.M):
            raise ValueError(f'{name}: missing GATE-OK')
        if name == 'gate_all' and not re.search(r'^CHECK-DSL OK\b', text, re.M):
            raise ValueError('missing CHECK-DSL OK')
    text = session.read('check_imp1')
    if not re.search(r'^CHECK OK: 0 launch\(es\) differ$', text, re.M) or 'CHECK FAILED' in text:
        raise ValueError('import correctness evidence missing or failed')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session', type=Path)
    ap.add_argument('--historical-root', type=Path, default=Path.home() / 'code/etx')
    a = ap.parse_args(argv)
    print(kcmp.__doc__)
    print('CONDITIONS: Qwen3-30B-A3B bf16, batch 1, one MI300X required; '
          'ETX prompt positions 1..7, layers 1..45; lm_head once per step. '
          'Clock: rocprof kernel device duration; no in-situ or host-wall comparison.')
    print('LIMITS: qkv/router fold norm; MoE proxy omits gather, activation, routing weight and sum. '
          'attn/reduce excluded for context mismatch. No end-to-end fusion verdict. '
          'qkv is refused: gate dump cannot check it; o_proj uses gate dump, '
          'other vendor rows use CPU float64 linear references, not model-stage checks.')
    print('IMPORT: NOT ESTABLISHED for every row; source availability and measured import cost unknown. '
          'Listed body bounds are estimates on another clock, not standalone comparisons.')
    print('DESIGN CONFLICTS: lm_head close limit is 30 us but vendor-win rule says 0.63 us; '
          'its import ceiling 106.7 is not the listed body 156.65 minus 1.04. '
          'Preserved verbatim; conflicting classifications remain ambiguous.')
    evidence_error = None
    try:
        nvref_evidence(a.session)
    except (ValueError, OSError, Refusal) as exc:
        evidence_error = str(exc)
        print('REFUSED ETX', evidence_error)
    session = nvref_report.Session(a.session)

    def current_job(name):
        state = session.status.get(name)
        if state is None or state.get('status') != 'ok' or state.get('gpu') != 0:
            raise ValueError(f'{name}: missing, failed, incomplete, or wrong GPU in current status.json')
        session.read(name)

    pairs = []
    for rnd in range(1, 4):
        e, v = {}, {}
        try:
            if evidence_error:
                raise ValueError(evidence_error)
            current_job(f'kc_etx_{rnd}')
            files = list((a.session / f'kc_etx_{rnd}').rglob('*kernel_trace.csv'))
            if len(files) != 1:
                raise ValueError('expected exactly one ETX kernel CSV')
            e = etx(files[0], a.session / 'b_all/vl_plan.txt')
        except (OSError, ValueError, KeyError, Refusal) as exc:
            print('REFUSED ETX', rnd, str(exc))
        try:
            current_job(f'kc_vnd_{rnd}')
            payload = json.loads((a.session / f'kc_vnd_{rnd}/result.json').read_text())
            if not payload.get('complete'):
                raise ValueError('vendor run incomplete')
            v = payload['arms']
        except (OSError, ValueError, KeyError, Refusal) as exc:
            print('REFUSED VENDOR', rnd, str(exc))
        pairs.append((e, v))
        for phase, row in e.items():
            print('ETX', rnd, phase, json.dumps(row))
    for phase in SHAPES:
        gains = []
        for rnd, (e, v) in enumerate(pairs, 1):
            candidates = []
            for lib, arm in v.items():
                row = arm.get(phase, {})
                if row.get('clock') == 'rocprof' and row.get('check_ok') and 'median_us' in row:
                    candidates.append((row['median_us'], lib))
                else:
                    print('REFUSED', rnd, lib, phase, row.get('refused', 'clock or correctness mismatch'))
            if phase in e and len(candidates) == len(LIBS):
                best, lib = min(candidates)
                gain = e[phase]['median_us'] - best
                gains.append(gain)
                print(f'PAIR {rnd} {phase}: etx={e[phase]["median_us"]:.6f} vendor={best:.6f} '
                      f'lib={lib} g={gain:.6f} us')
        print('DECISION', phase, decide(gains, phase),
              json.dumps({'gains_us': gains, 'median_us': median(gains) if gains else None,
                          'spread_us': max(gains) - min(gains) if gains else None,
                          'row_limit_us': 30 if phase == 'lm_head' else 0.63,
                          'vendor_pair_limit_us': 0.63, 'spread_factor': 2,
                          'link_gain_limit_us': 1.04,
                          'link_gain_pass': len(gains) == 3 and all(g >= 1.04 for g in gains),
                          'source_available': None, 'import_cost_us': None, 'import': 'unknown'}))
    for label, path in HISTORICAL.items():
        try:
            print('HISTORICAL', label, 'different session/context; descriptive only',
                  json.dumps(vllm(a.historical_root / path)))
        except (OSError, ValueError) as exc:
            print('HISTORICAL', label, 'NOT AVAILABLE', str(exc))
    print('SYSTEM BASELINE: retained nvref measurements, separate clock and experiment')
    nvref_report.main([str(a.session)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
