"""Per-batch GLM timing certificate (PP, TP or EP): one build, one qualification, one number.

This is the contract of section 9 of docs/glm/GLM-EXECUTION-DESIGN-2026-10-19.md,
mirroring examples/vllm_llm/bench.py: each batch B in {1,2,4,8} has its own
build, its own correctness qualification and its own certificate; no gate or
number is shared between batches. The timed quantity is the host wall between
the rank launches and every rank's stream retiring (examples/glm53/model_host.hip
calls wait_stream on every rank, then takes the timestamp, then runs after_wait,
which does synchronous D2H copies), one sample per position, 1024-1054.

Fail-closed: a missing ms= field, an identity change during the run, a
correctness receipt that is not TOKENS-EQUAL, or a non-finite/non-positive
sample refuses the certificate. Nothing is estimated or filled in. GPU count,
GPU model and parallel mode come from the timed log's GLM-VARIANT/GLM-DEVICE
lines and must agree with the build options; they are never assumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from examples.glm53 import model_build, stage
from examples.glm53 import options as build_options
from examples.glm53 import run as runner
from examples.glm53.parallel_variants import MODES

BATCHES = (1, 2, 4, 8)
CONTEXT = 1024
GEN = 32
FIRST, LAST = 1024, 1054  # 31 post-first-token decode samples, the Qwen window


class Refusal(Exception):
    pass


def step_samples(text, batch=1):
    """The 31 decode-step host-wall samples of one model execution log.

    Refuses a log without ms= (the host must print it), an incomplete or
    duplicated step sequence, or a non-finite/non-positive sample.
    """
    loose = re.findall(r'^GLM-MODEL-STEP pos=(\d+) batch=(\d+)(?: ms=(\S+))?$', text, re.M)
    if not loose:
        raise Refusal('no GLM-MODEL-STEP lines in the model log')
    missing = [pos for pos, _, ms in loose if not ms]
    if missing:
        raise Refusal(f'step pos={missing[0]} carries no ms=; model_host.hip must print the step time')
    want = [(p, batch) for p in range(CONTEXT + GEN - 1)]
    got = [(int(p), int(b)) for p, b, _ in loose]
    if got != want:
        raise Refusal(f'incomplete/duplicate decode steps: {len(got)} rows, expected positions 0..{CONTEXT + GEN - 2} for batch={batch}')
    xs = [float(ms) for pos, _, ms in loose if FIRST <= int(pos) <= LAST]
    if len(xs) != LAST - FIRST + 1:
        raise Refusal(f'requires {LAST - FIRST + 1} post-first-token timing samples')
    if any(not math.isfinite(x) or x <= 0 for x in xs):
        raise Refusal('invalid decode timing sample')
    return xs


def summarise(xs, batch):
    if batch not in BATCHES:
        raise Refusal(f'batch must be one of {BATCHES}')
    if len(xs) != LAST - FIRST + 1 or any(not math.isfinite(x) or x <= 0 for x in xs):
        raise Refusal('invalid timing samples')
    ms = statistics.median(xs)
    return dict(median_decode_wall_ms=ms, tokens_per_s=batch * 1000 / ms,
                ms_per_token=ms / batch, spread_ms=max(xs) - min(xs), samples_ms=list(xs))


def device_evidence(text, options):
    """GPU count, model and parallel mode as the timed host printed them; refuse disagreement."""
    variants = re.findall(r'^GLM-VARIANT devices=(\d+) parallel=(\d+) attention_tp=(\d+) '
                          r'up_tail_balance=(\d+) up_pairs=(\d+)$', text, re.M)
    if len(variants) != 1:
        raise Refusal(f'the model log carries {len(variants)} GLM-VARIANT lines, expected exactly one')
    keys = ('devices', 'parallel', 'attention_tp', 'up_tail_balance', 'up_pairs')
    got = dict(zip(keys, map(int, variants[0])))
    config = build_options.from_dict(options)
    want = dict(devices=config.devices, parallel=config.parallel, attention_tp=config.attention_tp,
                up_tail_balance=config.up_tail_balance, up_pairs=config.effective_up_pairs)
    if got != want:
        raise Refusal(f'GLM-VARIANT {got} differs from the build options {want}')
    devices = re.findall(r'^GLM-DEVICE rank=(\d+) name=(.*?) arch=(\S+) ', text, re.M)
    if [int(rank) for rank, _, _ in devices] != list(range(got['devices'])):
        raise Refusal(f'GLM-DEVICE ranks {[rank for rank, _, _ in devices]} differ from devices={got["devices"]}')
    names = {name.strip() for _, name, _ in devices}
    arches = {arch for _, _, arch in devices}
    if len(names) != 1 or len(arches) != 1 or not next(iter(names)):
        raise Refusal(f'GPU model not uniform or not reported: names={sorted(names)} arches={sorted(arches)}')
    mode = next(name for name, parallel, attention in MODES
                if (parallel, attention) == (got['parallel'], got['attention_tp']))
    return dict(gpus=got['devices'], gpu=names.pop(), gpu_arch=arches.pop(), mode=mode,
                parallel=got['parallel'], attention_tp=got['attention_tp'],
                up_tail_balance=got['up_tail_balance'], effective_up_pairs=got['up_pairs'],
                basis='GLM-VARIANT and GLM-DEVICE lines of the timed model log')


def digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def certificate(staged, build, identity, batch, xs, receipt, source, gate='production', *, evidence=None):
    """The bench_result.json payload. Conditions are recorded, never inferred later.
    Without device evidence the GPU/mode conditions are unrecorded (None), never assumed."""
    t = staged['config']
    device = evidence or dict.fromkeys(('gpus', 'gpu', 'gpu_arch', 'mode'))
    return dict(schema=1, status='PASS', batch=batch, gate=gate,
                slice=staged.get('slice'), build=str(build), build_sha256=digest(identity),
                build_identity=identity, source_identity=source,
                conditions=dict(**device, layers=t['num_hidden_layers'],
                                hidden=t['hidden_size'], context=CONTEXT, generated=GEN,
                                precision='native FP8 weights, BF16 activations',
                                maxlen=staged['dimensions']['MAXLEN']),
                measured='host wall from the rank launches to every rank stream retiring, before any completion check or D2H copy, steps 1024-1054',
                steps=LAST - FIRST + 1, **summarise(xs, batch),
                tokens_matched=sum(receipt.get('matched_each') or []),
                tokens_expected=GEN * batch,
                qualification=dict(status=receipt.get('status'), repetitions=receipt.get('repetitions'),
                                   logit_gate=receipt.get('logit_gate'), logit_atol=receipt.get('logit_atol'),
                                   numerical_certification=receipt.get('numerical_certification')))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=Path, required=True, help='staged model (--gate slice stages carry their window)')
    ap.add_argument('--build', type=Path, required=True, help='GPU-LINKED build of exactly this staged model')
    ap.add_argument('--reference', type=Path, required=True, help='HF reference record for the same batch/window')
    ap.add_argument('--out', type=Path, required=True, help='new directory for the qualification and certificate')
    ap.add_argument('--gate', choices=['production', 'module', 'slice'], default='production')
    ap.add_argument('--seconds', type=float, default=7200.0, help='per-execution timeout')
    a = ap.parse_args(argv)
    try:
        staged = stage.validate(a.stage, gate=a.gate)
        identity = model_build.verified_build(a.build)
        if json.loads((a.build / 'build-spec.json').read_text()) != staged:
            raise Refusal('build/staged model identity differs')
        batch = json.loads((a.build / 'plan.json').read_text())['shape'][0]
        if batch not in BATCHES:
            raise Refusal(f'unsupported compiled batch {batch}')
        ref = json.loads(a.reference.read_text())
        if ref.get('batch') != batch:
            raise Refusal(f'reference batch {ref.get("batch")} differs from compiled batch {batch}')
        if stage.validate(a.stage, gate=a.gate) != staged:
            raise Refusal('staged model changed before the run')
        a.out.mkdir(parents=True, exist_ok=False)
        receipt = runner.qualify(a.stage, a.build, a.reference, a.out / 'qualification', a.seconds, None, a.gate)
        if receipt.get('status') != 'TOKENS-EQUAL':
            raise Refusal(f'correctness receipt is {receipt.get("status")!r}')
        timed = (a.out / 'qualification' / 'run1.log').read_text()
        xs = step_samples(timed, batch)
        evidence = device_evidence(timed, identity['options'])
        if model_build.verified_build(a.build) != identity:
            raise Refusal('build identity changed during the benchmark')
        if stage.validate(a.stage, gate=a.gate) != staged:
            raise Refusal('staged model changed during the benchmark')
        source = model_build.source_identity(a.build)
        result = certificate(staged, a.build, identity, batch, xs, receipt, source, a.gate, evidence=evidence)
        (a.out / 'bench_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'BUILD-IDENTITY PASS batch={batch} build_sha256={result["build_sha256"]} '
              f'layers={result["conditions"]["layers"]} slice={result["slice"]}', flush=True)
        print(f'CORRECTNESS PASS tokens={result["tokens_matched"]}/{result["tokens_expected"]} '
              f'receipt={receipt.get("status")}', flush=True)
        print(f'DEVICES gpus={evidence["gpus"]} gpu={evidence["gpu"]!r} arch={evidence["gpu_arch"]} '
              f'mode={evidence["mode"]}', flush=True)
        print(f'TIMING batch={batch} median_decode_wall_ms={result["median_decode_wall_ms"]:.6f} '
              f'tokens_per_s={result["tokens_per_s"]:.2f} ms_per_token={result["ms_per_token"]:.6f} '
              f'spread_ms={result["spread_ms"]:.6f} steps={result["steps"]}', flush=True)
        print(f'BENCH PASS {result["median_decode_wall_ms"]:.6f} {result["tokens_per_s"]:.2f} batch={batch}', flush=True)
        return 0
    except (Refusal, ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        reason = ' '.join(str(exc).split())
        print('BUILD-IDENTITY FAIL unavailable: ' + reason, flush=True)
        print(f'CORRECTNESS FAIL reason={reason}', flush=True)
        try:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / 'bench_result.json').write_text(json.dumps(
                dict(schema=1, status='FAIL', reason=reason, timing=None), indent=2) + '\n')
        except OSError:
            pass
        print('BENCH FAIL ' + reason, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
