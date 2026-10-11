#!/usr/bin/env python3
"""Offline occurrence audit. Repository paths never fall back to results copies.

Run from any directory: python3 docs/citation-audit-2026-10-11/audit.py
Review the inventory with --show START END (inclusive occurrence IDs).
Verdicts/assertions live in decisions.json; no resolution implies semantic support.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/Users/ckl/code/etx')
FLEET = Path('/Users/ckl/code/fleet-mi300x')
NAMES = ['DEP-GAP-LEDGER', 'XCD-TIMELINE', 'XCD-PREFETCH',
         'MOE-BODY-ANATOMY', 'REVIEW-POOL-ANALYSIS']
RX = re.compile(r'(?<![\w/:-])(?P<path>[~A-Za-z_][\w./~-]*):(?P<spec>\d+(?:-\d+)?(?:(?:,|\s*/\s*)\d+(?:-\d+)?)*)')
IMPLICIT = re.compile(r'(?<![\w/:-]):(?P<spec>\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)')
COMMON = {
    'kernel.py': 'etx/codegen/kernel.py',
    'gemv.py': 'etx/ops/gemv.py', 'G': 'etx/ops/gemv.py',
    'gfx942.yaml': 'etx/machine/arch/gfx942.yaml',
    'primitives.h': 'etx/runtime/include/etx/primitives.h',
    'timeline.py': 'etx/tools/timeline.py',
    'xcd_account.py': 'examples/vllm_llm/xcd_account.py',
    'model.py': 'examples/vllm_llm/model.py',
    'qwen300.py': 'examples/sessions/qwen300.py',
    'MODEL': 'docs/MODEL-vllm-kernels.md',
    'PLAN': 'docs/PLAN-300TOKS-2026-10-11.md',
    'PLAN-300TOKS': 'docs/PLAN-300TOKS-2026-10-11.md',
    'FLOORS': 'docs/FLOORS-2026-10-08.md',
    'DEP': 'docs/DEP-GAP-LEDGER-2026-10-11.md',
    'XT': 'docs/XCD-TIMELINE-2026-10-11.md',
    'PF': 'docs/XCD-PREFETCH-2026-10-11.md',
    'MOE': 'docs/MOE-BODY-ANATOMY-2026-10-11.md',
    'RB': 'docs/review-pool-analysis-2026-10-11/rebase.txt',
    'ACC': 'docs/review-pool-analysis-2026-10-11/account_on.txt',
    'host.hip': 'examples/vllm_llm/host.hip',
    'coh.py': 'etx/ops/coh.py',
    'pairs.py': 'examples/vllm_llm/pairs.py',
    'rocm_build_check.sh': 'examples/vllm_llm/rocm_build_check.sh',
    'impbar_hw.log': 'results_2026-09-28b/impbar_hw.log',
    'round21/streamw.log': 'results_2026-10-01/round21/streamw.log',
    'overlap.log': 'results_2026-09-27/vlnext/overlap.log',
    'ntc.txt': 'results_2026-10-01/round15/ntc.txt',
    'FINDING.md': 'results_2026-10-01/round15/FINDING.md',
    'STATUS.md': str(FLEET / 'docs/STATUS.md'),
    'ab2_all_x.log': 'results_tiles/ab2_all_x.log',
    'ab3_all_x.log': 'results_tiles/ab3_all_x.log',
}
for name in ['source.py', 'dispatcher.py', 'attribution.py', 'build.py', 'README.md']:
    COMMON[name] = 'examples/vllm_llm/bf16_300/' + name
for name in ['tr_off.log', 'off_compile.log', 'job2_rerun.log']:
    COMMON[name] = 'results_qwen300_2026-10-11/' + name

def expand(path, name, line):
    if path.startswith('R/'):
        return 'results_qwen300_2026-10-11/' + path[2:]
    if path.startswith('bf16_300/'):
        return 'examples/vllm_llm/' + path
    if path.startswith('attempt1/'):
        return 'results_qwen300_2026-10-11/' + path
    if path in ['account.txt', 'latency.txt', 'bandwidth.txt', 'report.txt']:
        directory = 'dep-gap-ledger' if name == NAMES[0] and path == 'account.txt' else 'xcd-timeline'
        if name == NAMES[4] and path == 'account.txt':
            directory = 'dep-gap-ledger'
        return f'docs/{directory}-2026-10-11/{path}'
    if path in ['trace-sums.json', 'isa-audit.txt']:
        return 'docs/moe-body-anatomy-2026-10-11/' + path
    if name == NAMES[2] and path == 'docs/STATUS.md' and 'fleet' in line:
        return str(FLEET / path)
    if path in COMMON:
        return COMMON[path]
    return path

def location(path):
    if path.startswith('results_'):
        return ARCHIVE / path
    if path.startswith('~/'):
        return Path(path).expanduser()
    if path.startswith('/'):
        return Path(path)
    return ROOT / path

def indices(spec):
    result = []
    for part in re.split(r'[,/]', spec):
        part = part.strip()
        a, _, b = part.partition('-')
        result.extend(range(int(a), int(b or a) + 1))
    return result

def extract():
    rows, inputs = [], {}
    for name in NAMES:
        doc = 'docs/' + name + '-2026-10-11.md'
        raw = (ROOT / doc).read_text()
        inputs[doc] = hashlib.sha256(raw.encode()).hexdigest()
        lines = raw.splitlines()
        last_path = None
        for n, line in enumerate(lines, 1):
            matches = [(m, False) for m in RX.finditer(line)]
            matches += [(m, True) for m in IMPLICIT.finditer(line)]
            for m, implicit in sorted(matches, key=lambda x: x[0].start()):
                path = last_path if implicit else expand(m['path'], name, line)
                # These explicitly named revision references name a file in prose.
                if not implicit and m['path'] == 'a312354':
                    path = 'docs/XCD-TIMELINE-2026-10-11.md' if 'XCD report' in line else 'docs/PLAN-300TOKS-2026-10-11.md'
                # MOE paragraph 13 switches between archived and canonical observations.
                if name == NAMES[3] and n == 13 and implicit:
                    before = line[:m.start()]
                    if 'current top-level log' in before:
                        path = 'results_qwen300_2026-10-11/tr_off.log'
                    elif 'archive reports' in before and 'four flags' not in before:
                        path = 'results_qwen300_2026-10-11/attempt1/tr_off.log'
                # Explicit historical continuation in MOE section 3.2.
                if name == NAMES[3] and 'Upstream update' in line and implicit:
                    path = 'docs/PLAN-300TOKS-2026-10-11.md'
                if name == NAMES[4] and 'The 0.254 ms sum' in line and implicit:
                    path = 'docs/PLAN-300TOKS-2026-10-11.md'
                last_path = path
                start, end = n-1, n
                while start and lines[start-1].strip():
                    start -= 1
                while end < len(lines) and lines[end].strip():
                    end += 1
                row = dict(id=len(rows)+1, document=doc, document_line=n,
                           citation=m[0], implicit=implicit, path=path,
                           spec=m['spec'], sentence=line, context='\n'.join(lines[start:end]))
                p = location(path) if path else None
                row['absolute_path'] = str(p)
                row['source_lines'] = []
                row['missing'] = []
                if not p or not p.is_file():
                    row['missing'].append('file missing')
                else:
                    data = p.read_bytes()
                    inputs[str(p)] = hashlib.sha256(data).hexdigest()
                    contents = data.decode('utf-8', errors='replace').splitlines()
                    for i in indices(m['spec']):
                        if 1 <= i <= len(contents):
                            row['source_lines'].append([i, contents[i-1]])
                        else:
                            row['missing'].append(f'line {i} missing (file has {len(contents)} lines)')
                rows.append(row)
    return rows, inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', nargs=2, type=int)
    parser.add_argument('--new-lines', action='store_true', help='Display each source line once; TSV always retains all lines')
    args = parser.parse_args()
    rows, inputs = extract()
    decisions = json.loads((OUT / 'decisions.json').read_text()) if (OUT / 'decisions.json').exists() else {}
    for row in rows:
        d = decisions.get(str(row['id']), {})
        row['verdict'] = 'MISSING' if row['missing'] else d.get('verdict', 'UNREVIEWED')
        row['review'] = d
        text = '\n'.join(s for _, s in row['source_lines'])
        for assertion in d.get('contains', []):
            if assertion not in text:
                raise AssertionError(f"{row['id']}: missing assertion {assertion!r}")
        for pattern in d.get('regex', []):
            if not re.search(pattern, text):
                raise AssertionError(f"{row['id']}: failed regex {pattern!r}")
    if args.show:
        seen = {(r['path'], i) for r in rows if r['id'] < args.show[0] for i, _ in r['source_lines']} if args.new_lines else set()
        claims_seen = set()
        for row in rows:
            if args.show[0] <= row['id'] <= args.show[1]:
                print(f"\n[{row['id']}] {row['document']}:{row['document_line']} {row['citation']} => {row['path']}:{row['spec']}")
                if (row['document'], row['document_line']) not in claims_seen:
                    print('CLAIM: ' + row['sentence'])
                    claims_seen.add((row['document'], row['document_line']))
                for i, s in row['source_lines']:
                    if not args.new_lines or (row['path'], i) not in seen:
                        print(f'{i}: {s}')
                    seen.add((row['path'], i))
                for error in row['missing']:
                    print('MISSING: ' + error)
        return
    (OUT / 'inventory.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'inputs.sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')
    with (OUT / 'citations.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['document', 'citation', 'verdict', 'evidence'])
        for row in rows:
            writer.writerow([row['document'], row['citation'], row['verdict'],
                             json.dumps({k: row[k] for k in ['id', 'document_line', 'implicit', 'path', 'absolute_path', 'sentence', 'source_lines', 'missing', 'review']}, ensure_ascii=False)])
    summary = {'total': len(rows), 'explicit': sum(not r['implicit'] for r in rows),
               'implicit': sum(r['implicit'] for r in rows), 'resolved': sum(not r['missing'] for r in rows),
               'verdicts': dict(Counter(r['verdict'] for r in rows)),
               'documents': {n: dict(Counter(r['verdict'] for r in rows if n in r['document'])) for n in NAMES}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
