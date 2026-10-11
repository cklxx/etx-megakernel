#!/usr/bin/env python3
"""Offline integrity and independently re-derived numerical audit checks."""
import ast
import csv
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import audit

HERE = Path(__file__).resolve().parent
ROOT = audit.ROOT
rows = json.loads((HERE / 'inventory.json').read_text())
summary = json.loads((HERE / 'summary.json').read_text())
results = {}

# Verify the snapshot before rerunning any producer.
inputs = json.loads((HERE / 'inputs.sha256.json').read_text())
for path, expected in inputs.items():
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, path
results['hashed_text_inputs_unchanged'] = len(inputs)

# Exact agreement with baseline, including all files the brief prohibits editing.
changed = subprocess.check_output(['git', 'diff', '--name-only', 'ed1b1f4', '--'], cwd=ROOT, text=True).splitlines()
allowed = lambda s: s == 'docs/CITATION-AUDIT-2026-10-11.md' or s.startswith('docs/citation-audit-2026-10-11/')
assert all(allowed(s) for s in changed), changed
results['tracked_changes_confined_to_audit_deliverables'] = True

csv.field_size_limit(sys.maxsize)
with (HERE / 'citations.tsv').open(newline='') as f:
    exported = list(csv.DictReader(f, delimiter='\t'))
assert len(exported) == len(rows) == 486
assert all(r['verdict'] in ('PASS', 'MISSING', 'STALE', 'MISMATCH', 'BASIS') for r in rows)
for row, written in zip(rows, exported):
    assert written['document'] == row['document'] and written['citation'] == row['citation']
    assert written['verdict'] == row['verdict']
    evidence = json.loads(written['evidence'])
    assert evidence['source_lines'] == row['source_lines']
    assert evidence['review'] == row['review']
    assert evidence['absolute_path'] == row['absolute_path']
    live = Path(row['absolute_path']).read_text().splitlines()
    for n, line in row['source_lines']:
        assert live[n-1] == line
    # No search-copy substitution for repository citations.
    if not row['path'].startswith(('results_', '/', '~/')):
        assert Path(row['absolute_path']) == ROOT / row['path']
printed = (HERE / 'resolved-lines.txt').read_text()
for row in rows:
    assert f'[{row["id"]}] {row["document"]}:{row["document_line"]}' in printed
    for n, line in row['source_lines']:
        assert (f'{n}:' + (f' {line}' if line else '') + '\n') in printed
results['tsv_rows_and_full_source_lines_roundtrip'] = len(exported)
results['selected_lines'] = sum(len(r['source_lines']) for r in rows)
results['unique_path_line_locations'] = len({(r['path'], n) for r in rows for n, _ in r['source_lines']})
results['unique_path_selections'] = len({(r['path'], r['spec']) for r in rows})
assert (results['selected_lines'], results['unique_path_line_locations'], results['unique_path_selections']) == (2926, 1669, 393)

report = (ROOT / 'docs/CITATION-AUDIT-2026-10-11.md').read_text()
assert len(report.splitlines()) <= 250
failures = [r for r in rows if r['verdict'] != 'PASS']
for r in failures:
    assert f'### F{r["id"]:03d}' in report
    quote = html.escape('\n'.join(f'{n}: {s}' for n, s in r['source_lines'])).replace('\n', '&#10;')
    assert '<pre>' + quote + '</pre>' in report
    for correction in r['review'].get('corrected', []):
        path, spec = correction.rsplit(':', 1)
        lines = audit.location(path).read_text().splitlines()
        assert all(1 <= n <= len(lines) for n in audit.indices(spec)), correction
results['failure_sections_with_complete_quotations'] = len(failures)
results['report_lines'] = len(report.splitlines())

# Reproduce the existing rebase table from the raw trace binaries, without writing
# any audited artifact or running a GPU command.
rebase = ROOT / 'docs/review-pool-analysis-2026-10-11/rebase.py'
completed = subprocess.run([sys.executable, str(rebase)], cwd=ROOT, text=True,
                           capture_output=True, timeout=60, check=True)
expected = rebase.with_suffix('.txt').read_text()
assert completed.stdout == expected, 'rebase output differs from the audited artifact'
(HERE / 'rebase-reproduced.txt').write_text(completed.stdout)
results['raw_trace_rebase_byte_identical'] = True
results['raw_trace_plan_rows_matched'] = {'tr_off': 60362, 'tr_on': 60362}
raw_paths = ['results_tiles/tr_all.bin', 'results_qwen300_2026-10-11/tr_off.bin',
             'results_qwen300_2026-10-11/tr_on.bin', 'results_vlbest2_2026-10-10/b_all/plan.json']
results['raw_trace_and_plan_sha256'] = {s: hashlib.sha256((audit.ARCHIVE / s).read_bytes()).hexdigest() for s in raw_paths}

# Independently check all 864 serialized body rows, their cited log lines and sums.
body_artifact = json.loads((ROOT / 'docs/moe-body-anatomy-2026-10-11/trace-sums.json').read_text())
body_sums = {}
for entry in body_artifact:
    source = audit.location(entry['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
    log = source.read_text().splitlines()
    sums = defaultdict(float)
    assert len(entry['rows']) == 48 * 9
    seen = set()
    for row in entry['rows']:
        fields = log[row['line']-1].split()
        assert fields[0] == f'{row["role"]}_{row["layer"]}' or fields[0] == f'{row["role"]}_{row["layer"]}_x'
        assert int(fields[1]) == row['tasks']
        assert float(fields[3]) == row['body_us']
        sums[row['role']] += float(fields[3])
        seen.add((row['role'], row['layer']))
    assert len(seen) == 432
    for role, total in sums.items():
        assert math.isclose(total, entry['sums_us'][role], abs_tol=1e-8)
    assert math.isclose(sum(sums.values()), entry['sum_layer_bodies_us'], abs_tol=1e-8)
    body_sums[entry['path']] = {k: round(v, 2) for k, v in sums.items()}
results['serialized_body_rows_rederived'] = 864
results['body_sums_us'] = body_sums

# Reprice the prefetched byte service using time fractions, not rate increases.
overlap = audit.location('results_2026-09-27/vlnext/overlap.log').read_text().splitlines()
reprice = {}
for n in (6, 8):
    m = re.search(r'cold ([\d.]+) TB/s.*mall ([\d.]+) TB/s', overlap[n-1])
    cold, mall = map(float, m.groups())
    reprice[str(n)] = {'cold_TB_s': cold, 'mall_TB_s': mall,
                      'rate_gain': mall / cold - 1, 'time_fraction_saved': 1-cold/mall,
                      '702.3_us_repriced_saving_us': 702.3*(1-cold/mall)}
assert round(reprice['6']['702.3_us_repriced_saving_us'], 2) == 58.21
assert round(reprice['8']['702.3_us_repriced_saving_us'], 2) == 102.94
results['prefetch_elapsed_time_repricing'] = reprice

# Pair-level timing basis and spread, from the completed observation.
run4 = audit.location('results_qwen300_2026-10-11/job2_run4.log').read_text().splitlines()
key = run4[137]
savings = ast.literal_eval(re.search(r'savings_ms=(\[.*?\])', key).group(1))
median = statistics.median(savings)
spread = max(savings)-min(savings)
assert math.isclose(median, float(re.search(r'median=([^ ]+)', key).group(1)))
assert math.isclose(spread, float(re.search(r'spread=([^ ]+)', key).group(1)))
assert len(savings) == 5 and all(s > 0 for s in savings)
results['KEY32_completed_five_pair_wall_ms'] = {'savings': savings, 'median': median, 'spread': spread}

# Snapshot the package inventory bounding the negative claim in F372.
package = ROOT / 'examples/vllm_llm/bf16_300'
results['bf16_300_tracked_package_inventory'] = subprocess.check_output(
    ['git', 'ls-files', 'examples/vllm_llm/bf16_300'], cwd=ROOT, text=True).splitlines()
results['unverified'] = ['fresh GPU correctness/performance', 'floor attainability',
                         'unseen external tools and measurements', 'complete historical semantic search']
(HERE / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'OK: {len(rows)} TSV rows; {len(failures)} full failure quotations; byte-identical rebase; 864 body rows; prefetch and five-pair arithmetic; {len(report.splitlines())} report lines')
