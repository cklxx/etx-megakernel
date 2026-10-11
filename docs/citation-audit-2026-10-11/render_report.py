#!/usr/bin/env python3
"""Render every failure without clipping any selected source line."""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
rows = json.loads((HERE / 'inventory.json').read_text())
summary = json.loads((HERE / 'summary.json').read_text())
failures = [r for r in rows if r['verdict'] != 'PASS']


def pre(lines):
    # Character references keep full multiline quotations within the physical line cap.
    return '<pre>' + html.escape('\n'.join(f'{n}: {s}' for n, s in lines)).replace('\n', '&#10;') + '</pre>'


def prose(text):
    return ' '.join(text.splitlines())


out = [
    '# Citation audit — 2026-10-11', '',
    'All five documents were audited at live baseline `ed1b1f4` on `wf/citeaudit`. **486 occurrences resolve; 452 PASS and 34 fail: 23 STALE, 9 BASIS, 2 MISMATCH, 0 MISSING.** No audited document was edited. CPU only; no GPU, provisioning, network access or spend.', '',
    '## Coverage and method', '',
    'The complete occurrence table is [citations.tsv](citation-audit-2026-10-11/citations.tsv); its evidence column is JSON containing occurrence ID, document line, expanded path, attributed sentence, every selected source line, and the verdict/assertions. [inventory.json](citation-audit-2026-10-11/inventory.json) additionally preserves the surrounding paragraph. The plain-text [resolved-lines.txt](citation-audit-2026-10-11/resolved-lines.txt) prints every occurrence and its selected lines whole. Repeated citations count separately.', '',
    'The extractor found **445 named file/alias tokens**, **2 revision references naming their files in prose**, and **39 leading-colon continuations**: 486 checks. Ranges/comma lists/slash continuations are one occurrence per named token, with every selected line expanded. This covers 2,926 selected line occurrences, 1,669 distinct path/line locations and 393 distinct path/selection pairs. Eleven occurrences address the external fleet repository and are included in the totals.', '',
    'Repository paths resolve only against this live worktree, never against a results/archive copy. Explicit `results_*/` paths resolve under `/Users/ckl/code/etx`; external fleet paths resolve under `/Users/ckl/code/fleet-mi300x`. Document shorthands are explicitly mapped in [audit.py](citation-audit-2026-10-11/audit.py). Historical revision references are checked against the required live file; preserved historical content is named as a correction where available.', '',
    'Extraction, resolution, recorded literal/regex assertions, hashes and CPU arithmetic are machine checked. Every occurrence also received a contextual claim-to-source assessment recorded in [decisions.json](citation-audit-2026-10-11/decisions.json); semantic mapping is audit-authored, not inferred from a keyword hit. Derived numbers may PASS when the selected inputs and the stated arithmetic support them. Explicitly conditional cross-build estimates are not failed merely for being estimates. BASIS marks an attribution that changes the measured condition or denominator.', '',
    'STALE means the attributed content was located elsewhere; MISMATCH means no source for that meaning was found in the relevant live sources and named results. Partial support is insufficient for an additional unsupported attribution. Corrected citations repair evidence location only; historical corrections do not turn an old number into the current ledger number.', '',
    'Reproduce: `python3 docs/citation-audit-2026-10-11/audit.py`, then `python3 docs/citation-audit-2026-10-11/audit.py --show 1 486 > docs/citation-audit-2026-10-11/resolved-lines.txt`, then `python3 docs/citation-audit-2026-10-11/render_report.py`, then `/Users/ckl/code/etx/.venv/bin/python docs/citation-audit-2026-10-11/verify.py`. The archived/external inputs must remain present. [inputs.sha256.json](citation-audit-2026-10-11/inputs.sha256.json) records the audited text snapshot; [verification.json](citation-audit-2026-10-11/verification.json) records the independent CPU checks.', '',
    '## Counts', '',
    '| Document | Found/resolved | PASS | MISSING | STALE | MISMATCH | BASIS | Failures |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for name, counts in summary['documents'].items():
    total = sum(counts.values())
    out.append('| ' + name + ' | ' + str(total) + '/' + str(total) + ' | ' + ' | '.join(str(counts.get(c, 0)) for c in ['PASS', 'MISSING', 'STALE', 'MISMATCH', 'BASIS']) + f' | {total-counts["PASS"]} |')
out += [
    '| **Total** | **486/486** | **452** | **0** | **23** | **2** | **9** | **34** |', '',
    '**Cleanest by failure rate:** MOE-BODY-ANATOMY, 4/150 (2.67%). REVIEW has fewer absolute failures, 3/104 (2.88%). **Weakest by failure rate:** XCD-PREFETCH, 6/39 (15.38%). DEP has the largest absolute failure count, 14/102 (13.73%). No document is entirely clean.', '',
    '## Every failure', '',
    'IDs refer to TSV occurrences. Each Claim reproduces the document paragraph containing the failed attribution, joining wrapped lines with spaces. Each quotation includes **all** selected source lines, including blanks and the complete second half of long log lines. Line numbers below are source line numbers. Expanded paths remove shorthand ambiguity.', '',
]
for r in failures:
    d = r['review']
    out += [
        f'### F{r["id"]:03d} — {r["verdict"]} — {Path(r["document"]).stem}:{r["document_line"]}',
        f'Citation: `{r["citation"]}` → `{r["path"]}:{r["spec"]}`. **Claim:** {prose(d.get("claim", r["context"]))}',
        pre(r['source_lines']),
        '**Finding:** ' + d['note'],
    ]
    corrections = d.get('corrected', [])
    detail = '**Correction/location:** ' + (', '.join(f'`{c}`' for c in corrections) if corrections else 'No supporting corrected citation located.')
    if d.get('correction_limit'):
        detail += ' ' + d['correction_limit']
    if d.get('bases'):
        b = d['bases']
        detail += f' **Bases:** cited = {b["cited"]}; attributed = {b["claimed"]}. **Other document numbers:** {b["other_numbers"]}'
    out += [detail, '']
out += [
    '## Limits and decision-sensitive citations', '',
    'This verifies printed evidence, live implementation locations and reproducible CPU derivations, not the quality of a proposed optimization. No GPU measurements, correctness certificates, byte equivalence, full-model output or attainable speedup were independently rerun. A PASS certifies support for the attribution, not that its upstream measurement is physically valid. Negative claims about absent capabilities are bounded by the inspected source/package inventory; unseen tools or measurements cannot be excluded.', '',
    'MISMATCH searches covered the pertinent live implementation/documents and explicitly named archived evidence, not every file ever produced or every Git revision. Untracked archives and the fleet tree remain external inputs, so rerunning elsewhere needs those exact files. Whole-file references, ordinary links and prose without line selections are outside the counted grammar. The brief’s simple grammar omits aliases, implicit `:line` continuations and revision-in-prose references; these were included rather than silently dropped. No substantive error in its CPU-only/live-path requirement was found.', '',
    'Three load-bearing supports whose failure would most weaken the build/stop decision (a sensitivity ranking, not an argument verdict):', '',
    '1. **Measured KEY32 step saving:** `results_qwen300_2026-10-11/job2_run4.log:138` (PASS). The paired 0.231 ms saving sets the remaining gap; confusing completed and aborted observations would undermine the common starting point (F007/F013).',
    '2. **Attention/reduce envelope:** `docs/review-pool-analysis-2026-10-11/rebase.txt:7-8` (PASS). The 0.236 ms ceiling depends on the 2.92/2.81 us floor inputs and tr_on critical bodies. CPU reproduction succeeds; attainability and comparison to a vendor dispatch remain unverified, and F467 identifies the proposed harness’s incompatible basis.',
    '3. **Remaining target amount:** `docs/PLAN-300TOKS-2026-10-11.md:136,146` (F073/F249, STALE). The live gap is 0.238 ms, while the attributed 0.324 ms is historical and survives at 138/148. Using the historical pursuit threshold is possible only when explicitly named as such; it cannot certify the live remaining-gap attribution.', '',
]
report = '\n'.join(out)
assert len(report.splitlines()) <= 250, len(report.splitlines())
(ROOT / 'docs/CITATION-AUDIT-2026-10-11.md').write_text(report)
print(f'{len(failures)} failure sections; {len(report.splitlines())} report lines')
