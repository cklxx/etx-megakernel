#!/usr/bin/env python3
"""Strict offline check of the receipts a two-GPU session record must contain.

The session files already end with this tool in their `final` lane, so a
record whose qualification jobs exited 0 but wrote weaker evidence fails the
session. On a fetched copy the same command re-checks the sealed record.

  python3 tools/two_gpu_verify.py <record> --qwen-only [--require-seal]
  python3 tools/two_gpu_verify.py <record> --glm-only  [--require-seal]

Qwen: `qualify_b{2,4,8}/correctness.json` must be PASS with the fixed recipe,
every planned imported instance identical at positions 0 and 1040, 97
instances per request per position, and 32/32 tokens in both repetitions of
the coverage, fused and unfused runs.

GLM: `qualification/qualification.json` must be TOKENS-EQUAL over two
independent runs, 32/32 tokens per request in both, with finite full logits
compared to HF and a build/reference identity for both.

Failures print `VERIFY FAIL <item>: <reason>`; exit 2.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVENTORY = ROOT / 'docs/evidence/qwen-batch-glm53/qwen-plan-inventory.json'
HEX64 = re.compile('[0-9a-f]{64}')


class Fail(Exception):
    """A receipt is missing or does not prove what the session claims."""


def receipt(path):
    if not path.is_file():
        raise Fail(f'missing {path}')
    return json.loads(path.read_text())


def qwen(record):
    from examples.vllm_llm.batch_correctness import BATCHES, CONTEXT, GEN, REPEAT, golden_names, recipe

    counts = {e['batch']: e['imported_instances'] for e in json.loads(INVENTORY.read_text())}
    summary = {}
    for batch in BATCHES:
        item = f'qualify_b{batch}/correctness.json'
        r = receipt(record / item)
        if r.get('status') != 'PASS':
            raise Fail(f'{item}: status {r.get("status")!r} (error: {r.get("error")})')
        if r.get('performance') != 'unmeasured':
            raise Fail(f'{item}: a correctness receipt must not report performance')
        if r.get('batch') != batch:
            raise Fail(f'{item}: batch {r.get("batch")!r}, expected {batch}')
        inputs = r.get('inputs') or {}
        files = inputs.get('files') or {}
        # batch_correctness.inputs binds exactly the assets this batch reads: 4 + 2 per request.
        names = golden_names(batch)
        assets = {'hf/config.json', 'hf/REVISION', 'vl.manifest', 'vl.bin', *names,
                  *(n.replace('golden', 'prefill', 1) for n in names)}
        if (inputs.get('model') != 'Qwen/Qwen3-30B-A3B' or inputs.get('batch') != batch
                or inputs.get('context') != CONTEXT or inputs.get('gen') != GEN
                or not re.fullmatch('[0-9a-f]{40}', str(inputs.get('revision')))
                or set(files) != assets or not all(HEX64.fullmatch(str(v)) for v in files.values())):
            raise Fail(f'{item}: incomplete staged-asset identity')
        identity = r.get('build_identity') or {}
        overrides = identity.get('overrides') or {}
        if identity.get('check_enabled') is not True:
            raise Fail(f'{item}: build did not enable imported-launch checks')
        for key, value in recipe(batch).items():
            if overrides.get(key) != value:
                raise Fail(f'{item}: build override {key}={overrides.get(key)!r}, expected {value!r}')
        if r.get('positions') != [0, 1040]:
            raise Fail(f'{item}: positions {r.get("positions")!r}, expected [0, 1040]')
        planned = counts[batch]
        if planned % batch:
            raise Fail(f'{item}: plan inventory {planned} is not divisible by batch {batch}')
        if r.get('imported_per_position') != planned:
            raise Fail(f'{item}: imported_per_position {r.get("imported_per_position")!r}, expected {planned}')
        per = r.get('byte_identity_per_request') or {}
        for position in (0, 1040):
            gate = per.get(str(position)) or {}
            if (gate.get('position') != position or gate.get('requests') != batch
                    or gate.get('instances_per_request') != planned // batch):
                raise Fail(f'{item}: position {position} does not cover all {batch} requests '
                           f'with {planned // batch} instances each')
        if r.get('greedy_tokens_per_mode') != batch * GEN * REPEAT:
            raise Fail(f'{item}: greedy_tokens_per_mode {r.get("greedy_tokens_per_mode")!r}, '
                       f'expected {batch * GEN * REPEAT}')
        if r.get('modes') != ['fused', 'unfused']:
            raise Fail(f'{item}: modes {r.get("modes")!r}, expected fused and unfused')
        summary[str(batch)] = dict(imported=planned, instances_per_request=planned // batch,
                                   tokens=batch * GEN * REPEAT)
    return summary


def glm(record, gate='production'):
    from examples.glm53.prepare import gate_identity
    _, model = gate_identity(gate)
    item = 'qualification/qualification.json'
    r = receipt(record / item)
    if r.get('status') != 'TOKENS-EQUAL':
        raise Fail(f'{item}: status {r.get("status")!r} (error: {r.get("error")})')
    # Receipts written before the module gate existed carry no gate field: they are production.
    if r.get('gate', 'production') != gate:
        raise Fail(f'{item}: a {r.get("gate")!r}-gate receipt cannot pass the {gate} gate')
    if r.get('model') != model or not re.fullmatch('[0-9a-f]{40}', str(r.get('revision'))):
        raise Fail(f'{item}: missing pinned model/revision identity for the {gate} gate')
    if r.get('performance') != 'unmeasured':
        raise Fail(f'{item}: a correctness receipt must not report performance')
    batch = r.get('batch')
    if batch not in (1, 2, 4, 8):
        raise Fail(f'{item}: batch {batch!r}')
    if r.get('generated_each') != 32 or r.get('repetitions') != 2:
        raise Fail(f'{item}: expected 32 tokens and two independent repetitions')
    if r.get('matched_each') != [32] * batch:
        raise Fail(f'{item}: matched_each {r.get("matched_each")!r}, expected {[32] * batch}')
    if r.get('model_certified') is not False:
        raise Fail(f'{item}: this session does not certify the model; model_certified must be false')
    verdicts = r.get('verdicts') or []
    if len(verdicts) != 2 or [v.get('repetition') for v in verdicts] != [0, 1]:
        raise Fail(f'{item}: expected verdicts for repetitions 0 and 1')
    for v in verdicts:
        rep = v.get('repetition')
        if v.get('status') != 'TOKENS-EQUAL' or v.get('matched_each') != [32] * batch:
            raise Fail(f'{item}: repetition {rep} is not 32/32 for every request')
        if v.get('checkpoint_execution') != 'real two-GPU ETX model':
            raise Fail(f'{item}: repetition {rep} did not execute the real two-GPU model')
        if v.get('logit_gate') not in ('REPORTED', 'ENFORCED'):
            raise Fail(f'{item}: repetition {rep} has no full-logit comparison')
        error = v.get('logit_max_abs')
        if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
            raise Fail(f'{item}: repetition {rep} has no finite full-logit error')
        steps = v.get('logit_max_abs_per_step')
        if (not isinstance(steps, list) or len(steps) != 32
                or not all(type(x) in (int, float) and math.isfinite(x) for x in steps)):
            raise Fail(f'{item}: repetition {rep} does not report 32 finite per-step logit errors')
        if v.get('logit_argmax_equal') is not True:
            raise Fail(f'{item}: repetition {rep} logit argmax differs from HF')
        if gate == 'module':
            from examples.glm53.module_gate import MEDIAN_LOGIT_BOUND, module_rule
            ok, median = module_rule(v)
            if not ok:
                raise Fail(f'{item}: repetition {rep} median per-step logit error {median:.4g} exceeds the '
                           f'pre-registered module-gate bound {MEDIAN_LOGIT_BOUND}')
    if r.get('logit_gate') not in ('REPORTED', 'ENFORCED'):
        raise Fail(f'{item}: no receipt-level full-logit gate')
    build_hashes = r.get('build_hashes') or {}
    required = {'run', 'build-spec.json', 'plan.json', 'plan_data.h', 'glm53_spec.h',
                'glm53_contract.h', 'load.tsv', 'megakernel_d0.hip', 'megakernel_d1.hip'}
    if set(build_hashes) != required or not all(HEX64.fullmatch(str(v)) for v in build_hashes.values()):
        raise Fail(f'{item}: missing build file hashes')
    reference_hashes = r.get('reference_hashes') or {}
    if len(reference_hashes) != 2 or not all(HEX64.fullmatch(str(v)) for v in reference_hashes.values()):
        raise Fail(f'{item}: missing reference/logits hashes')
    out = dict(gate=gate, batch=batch, repetitions=2, matched_each=[32] * batch,
               logit_gate=r.get('logit_gate'),
               logit_max_abs=max(v['logit_max_abs'] for v in verdicts))
    if gate == 'module':
        import statistics
        out['median_logit_max_abs'] = [statistics.median(v['logit_max_abs_per_step']) for v in verdicts]
    return out


def sealed(record):
    from etx.session import record as record_module
    code, lines = record_module.verify(record)
    if code != 0:
        raise Fail('; '.join(lines[-2:]))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('record', type=Path, help='session record directory')
    which = ap.add_mutually_exclusive_group()
    which.add_argument('--qwen-only', action='store_true', help='check only qualify_b*/correctness.json')
    which.add_argument('--glm-only', action='store_true', help='check only qualification/qualification.json')
    ap.add_argument('--require-seal', action='store_true', help='the record must also verify against its SEAL')
    ap.add_argument('--gate', choices=['production', 'module'], default='production',
                    help='GLM identity to require; module also applies the pre-registered median-logit rule')
    ap.add_argument('--out', type=Path, help='write the verification record here')
    args = ap.parse_args(argv)
    record = args.record.resolve()
    if not record.is_dir():
        print(f'VERIFY FAIL record: {record} is not a directory')
        return 2
    result, failed = dict(schema=1, record=str(record)), 0
    if args.require_seal:
        try:
            result['seal'] = sealed(record)
            print('VERIFY OK seal: every byte matches the MANIFEST')
        except Exception as exc:
            failed = 1
            result['seal'] = dict(error=str(exc))
            print(f'VERIFY FAIL seal: {exc}')
    if not args.glm_only:
        try:
            result['qwen'] = qwen(record)
            print('VERIFY OK qwen: ' + ', '.join(
                f'B{b} {v["imported"]} instances ({v["instances_per_request"]}/request) {v["tokens"]} tokens'
                for b, v in result['qwen'].items()))
        except Exception as exc:
            failed = 1
            result['qwen'] = dict(error=str(exc))
            print(f'VERIFY FAIL qwen: {exc}')
    if not args.qwen_only:
        try:
            result['glm'] = glm(record, args.gate)
            print(f'VERIFY OK glm{" (module gate)" if args.gate == "module" else ""}: batch {result["glm"]["batch"]}, 2 repetitions, '
                  f'32 tokens/request, logit gate {result["glm"]["logit_gate"]}, '
                  f'max |logit-HF| {result["glm"]["logit_max_abs"]:.3g}'
                  + (f', median {max(result["glm"]["median_logit_max_abs"]):.3g}' if args.gate == 'module' else ''))
        except Exception as exc:
            failed = 1
            result['glm'] = dict(error=str(exc))
            print(f'VERIFY FAIL glm: {exc}')
    result['status'] = 'FAIL' if failed else 'OK'
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    if failed:
        return 2
    print('VERIFY OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
