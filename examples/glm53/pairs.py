"""Interleaved GLM reference/candidate pairs from bench certificates, with the T10 keep/drop verdict.

  python -m examples.glm53.pairs --reference-label pp2 cert_r1.json cert_c1.json cert_r2.json ... [--pairs 5]

Certificates (examples/glm53/bench.py bench_result.json) are given in run order and pair up
(1, 2), (3, 4), ...; each pair holds one reference run (its build-options SHA-256 equals the
first certificate's) and one candidate run, in either order (counterbalance the order across
pairs). Gain of a pair: 100 * (reference - candidate) / reference on median_decode_wall_ms, the
host-wall batch step (docs/glm/PARALLEL-MEASUREMENT-PLAN-2026-10-22.md).

Verdict (the plan's frozen rule): KEEP only if every gain is positive, the median gain is above
2% and above twice the spread (max gain - min gain, percentage points); otherwise
NO-ESTABLISHED-GAIN. Exit 0 KEEP, 1 NO-ESTABLISHED-GAIN, 2 refused.

Refused (NOT-COMPLETE, no number): a certificate that is not PASS or not TOKENS-EQUAL, not every
request 32/32, a missing or differing GPU model / GPU count / batch / context / steps basis between
runs, a pair that is not one run of each build, more than two distinct builds, or a pair count
other than --pairs. A partial set never yields a median.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path


class Refusal(Exception):
    pass


BASIS = ('batch', 'steps')
CONDITIONS = ('gpus', 'gpu', 'gpu_arch', 'context', 'generated', 'layers')


def load(path):
    cert = json.loads(Path(path).read_text())
    if cert.get('status') != 'PASS':
        raise Refusal(f'{path}: certificate status {cert.get("status")!r}')
    if cert.get('qualification', {}).get('status') != 'TOKENS-EQUAL':
        raise Refusal(f'{path}: correctness receipt is not TOKENS-EQUAL')
    if cert.get('tokens_matched') != cert.get('tokens_expected') or not cert.get('tokens_expected'):
        raise Refusal(f'{path}: {cert.get("tokens_matched")}/{cert.get("tokens_expected")} tokens')
    if any(cert.get('conditions', {}).get(k) in (None, '') for k in ('gpus', 'gpu', 'mode')):
        raise Refusal(f'{path}: no captured device evidence (GPU count/model/mode)')
    build = cert.get('build_identity', {}).get('options_sha256')
    if not build:
        raise Refusal(f'{path}: no build options identity')
    ms = cert.get('median_decode_wall_ms')
    if not isinstance(ms, (int, float)) or not ms > 0:
        raise Refusal(f'{path}: invalid median_decode_wall_ms {ms!r}')
    return dict(path=str(path), build=build, mode=cert['conditions']['mode'], ms=float(ms),
                basis={k: cert.get(k) for k in BASIS}, conditions={k: cert['conditions'].get(k) for k in CONDITIONS},
                binary=cert.get('build_identity', {}).get('binary_sha256'))


def pair(runs, expected):
    if len(runs) != 2 * expected:
        raise Refusal(f'{len(runs)} runs; {expected} pairs need exactly {2 * expected}')
    builds = {r['build'] for r in runs}
    if len(builds) != 2:
        raise Refusal(f'{len(builds)} distinct builds; a pair set compares exactly two')
    first = runs[0]
    for r in runs[1:]:
        if r['basis'] != first['basis'] or r['conditions'] != first['conditions']:
            raise Refusal(f'{r["path"]}: basis/conditions {r["basis"]} {r["conditions"]} differ from '
                          f'{first["basis"]} {first["conditions"]}')
    reference = first['build']
    out = []
    for a, b in zip(runs[0::2], runs[1::2]):
        if a['build'] == b['build']:
            raise Refusal(f'pair {a["path"]} / {b["path"]} is not one run of each build')
        ref, cand = (a, b) if a['build'] == reference else (b, a)
        out.append((ref, cand))
    return out


def verdict(gains):
    m = statistics.median(gains); s = max(gains) - min(gains)
    keep = all(g > 0 for g in gains) and m > 2 and m > 2 * s
    return dict(verdict='KEEP' if keep else 'NO-ESTABLISHED-GAIN', median_gain_pct=m, spread_pct=s,
                all_positive=all(g > 0 for g in gains))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('certificates', nargs='+', type=Path)
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    try:
        ps = pair([load(p) for p in a.certificates], a.pairs)
    except (Refusal, OSError, ValueError, KeyError) as exc:
        print(f'GLM-PAIRS NOT-COMPLETE {exc}', flush=True)
        if a.out:
            a.out.write_text(json.dumps(dict(schema=1, status='NOT-COMPLETE', reason=str(exc)), indent=2) + '\n')
        return 2
    gains = [100 * (r['ms'] - c['ms']) / r['ms'] for r, c in ps]
    result = dict(schema=1, status='COMPLETE', reference_mode=ps[0][0]['mode'], candidate_mode=ps[0][1]['mode'],
                  reference_build=ps[0][0]['build'], candidate_build=ps[0][1]['build'],
                  conditions=ps[0][0]['conditions'], basis=ps[0][0]['basis'],
                  measured='median host-wall batch decode step per run (bench.py), five interleaved pairs',
                  pairs=[dict(reference=r['path'], candidate=c['path'], reference_ms=r['ms'], candidate_ms=c['ms'],
                              delta_ms=c['ms'] - r['ms'], gain_pct=g) for (r, c), g in zip(ps, gains)],
                  **verdict(gains))
    for i, p in enumerate(result['pairs']):
        print(f'GLM-PAIR {i} reference_ms={p["reference_ms"]:.6f} candidate_ms={p["candidate_ms"]:.6f} '
              f'gain_pct={p["gain_pct"]:.3f}', flush=True)
    print(f'GLM-PAIRS {result["verdict"]} {result["candidate_mode"]} vs {result["reference_mode"]} '
          f'median_gain_pct={result["median_gain_pct"]:.3f} spread_pct={result["spread_pct"]:.3f} '
          f'gpu={result["conditions"]["gpu"]!r} gpus={result["conditions"]["gpus"]} batch={result["basis"]["batch"]}', flush=True)
    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['verdict'] == 'KEEP' else 1


if __name__ == '__main__':
    sys.exit(main())
