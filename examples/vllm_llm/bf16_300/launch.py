"""Validated sparse launcher and full raw-record reader. No GPU use on dry run."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import time

from .pack import sha
from .source import PAIRS

RECORD = struct.Struct('<QQIIII')
POINTERS = (0, 8, 16, 24, 32, 64, 72, 80, 88, 112)


def validate_pack(pack, reference, arguments, layer, task, wave, row):
    if pack.get('version') != 1 or pack.get('precision') != 'bf16' or pack.get('batch') != 1:
        raise ValueError('invalid pack identity')
    if pack['layer'] != layer or not pack['first'] <= task < pack['first']+pack['count'] or not 0 <= wave < 16 or not 0 <= row < 64:
        raise ValueError('selector does not match captured original arguments')
    if sha(arguments) != pack['arguments_sha256'] or Path(arguments).stat().st_size != 160:
        raise ValueError('original argument block changed')
    if sha(reference) != pack['reference']['sha256'] or Path(reference).stat().st_size != pack['reference']['bytes']:
        raise ValueError('independent original reference changed')
    block = Path(arguments).read_bytes()
    if any(struct.unpack_from('<Q', block, i)[0] for i in POINTERS[5:]):
        raise ValueError('folded/optional pointers unsupported')
    if struct.unpack_from('<i', block, 104)[0] or struct.unpack_from('<i', block, 120)[0] != 1:
        raise ValueError('only unfolded batch-one MoE supported')
    if struct.unpack_from('<i', block, 156)[0] != pack['total_tasks'] or len(pack['buffers']) != 5:
        raise ValueError('captured grid/buffers mismatch')
    for b in pack['buffers']:
        if b['offset'] < 0 or b['bytes'] <= 0 or Path(b['file']).stat().st_size < b['offset']+b['bytes']:
            raise ValueError('truncated original buffer')
        if 'sha256' in b and sha(b['file']) != b['sha256']:
            raise ValueError('original buffer changed')
    return block


def read_records(path, pack, iterations, task, wave, row):
    data = Path(path).read_bytes()
    n = 1 + pack['count']*16
    if len(data) != iterations*n*RECORD.size:
        raise ValueError('incomplete raw probe records')
    rows = [dict(zip(('begin', 'end', 'task', 'wave', 'xcd', 'row'), v)) for v in RECORD.iter_unpack(data)]
    evidence = []
    for it in range(iterations):
        rs = rows[it*n:(it+1)*n]
        r = rs[0]
        if (r['task'], r['wave'], r['row']) != (task, wave, row) or not 0 < r['begin'] <= r['end']:
            raise ValueError('sparse sample missing/selector mismatch')
        for j, r in enumerate(rs[1:]):
            if (r['task'], r['wave'], r['row']) != (pack['first']+j//16, j%16, row) or not 0 < r['begin'] <= r['end'] or not 0 <= r['xcd'] < 8:
                raise ValueError('missing/duplicate/invalid task-wave completion')
        last = max(r['end'] for r in rs[1:])
        owners = [(r['task'], r['wave']) for r in rs[1:] if r['end'] == last]
        evidence.append(dict(iteration=it, completed_tasks=pack['count'], completed_waves=pack['count']*16,
                             last_completed=owners, sample_is_last=(task, wave) in owners))
    return dict(records=rows, completion=evidence, tick_ns=10, layout='QQIIII (32 bytes), sample then all task/waves')


def run(a):
    pack = json.loads(a.pack.read_text())
    identity = json.loads((a.build/'identity.json').read_text())
    validate_pack(pack, a.reference, a.arguments, a.layer, a.task, a.wave, a.row)
    if identity['pair'] != a.pair or identity['sampling'] != dict(task=a.task, wave=a.wave, row=a.row):
        raise ValueError('probe binary selector mismatch')
    for name in ('original.hsaco', 'candidate.hsaco', 'launcher'):
        if sha(a.build/name) != identity['hashes'][name]:
            raise ValueError('probe/launcher identity changed')
    if identity['capture_math'] != 'bf16_unfolded_batch1':
        raise ValueError('unsupported math identity')
    quoted = lambda v: json.dumps(str(v))
    prefix = a.output.resolve()
    fields = [a.build/'original.hsaco', a.build/'candidate.hsaco', 'dsl_'+pack['role'], prefix, a.reference, a.arguments]
    config = ' '.join(quoted(v) for v in fields) + f' {pack["first"]} {pack["count"]} {a.iterations} {a.deadline} {int(a.selector == "candidate")} {pack["reference"]["output_offset"]} {pack["reference"]["bytes"]}\n'
    config += ''.join(f'{quoted(b["file"])} {b["offset"]} {b["bytes"]}\n' for b in pack['buffers'])
    command = [str((a.build/'launcher').resolve()), str(prefix)+'.config.txt']
    if a.dry_run:
        return dict(command=command, config=config, selector=a.selector, gpu_run=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(prefix)+'.json').exists() or Path(str(prefix)+'.bin').exists():
        raise ValueError('output already exists; preserve prior raw evidence')
    Path(str(prefix)+'.config.txt').write_text(config)
    log = Path(str(prefix)+'.log')
    # OS-level outer deadline bounds module load/copies as well as stream waits.
    with log.open('w') as f:
        p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while p.poll() is None:
                remaining = a.deadline-(time.monotonic()-start)
                if remaining <= 0:
                    raise TimeoutError('NOT-COMPLETE launcher deadline')
                try: p.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    print(f'PROGRESS probe pid={p.pid} elapsed={time.monotonic()-start:.0f}s log={log}', flush=True)
            if p.returncode:
                raise RuntimeError(f'probe rc={p.returncode}; full log={log}')
        finally:
            if p.poll() is None:
                import os, signal
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    raw = read_records(str(prefix)+'.bin', pack, a.iterations, a.task, a.wave, a.row)
    timings = [json.loads(l) for l in Path(str(prefix)+'.timing.jsonl').read_text().splitlines()]
    if len(timings) != a.iterations or any(t['iteration'] != i for i,t in enumerate(timings)):
        raise ValueError('incomplete timing pairs')
    result = dict(raw=raw, resource_identity=identity, selector=a.selector, record_arm='candidate_diagnostic_companion',
        original_reference=pack['reference'], checks=dict(output_byte_equal=True, imports='see original capture check log',
        full_model_tokens='NOT_RUN_BY_SPARSE_LAUNCHER'), pairs=timings,
        body_perturbation_us=statistics.median(t['body_perturbation_us'] for t in timings),
        perturbation_basis='whole standalone constituent grid device span, not per-task body or full-model wall',
        exclusive_removable_us=None, timing_eligible=False)
    Path(str(prefix)+'.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector', choices=('original','candidate'), required=True)
    p.add_argument('--pair', choices=PAIRS, required=True)
    for key in ('pack','build','arguments','reference','output'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('layer','task','wave','row','iterations'):
        p.add_argument('--'+key, type=int, required=True)
    p.add_argument('--deadline', type=float, required=True, help='seconds from start; includes all copies and launches')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.iterations <= 0 or a.deadline <= 0: p.error('positive iterations/deadline required')
    result = run(a)
    print(json.dumps(result if a.dry_run else {k:v for k,v in result.items() if k not in ('raw','resource_identity','pairs')}, indent=2))


if __name__ == '__main__':
    main()
