"""Emit explicit GLM PP/TP/EP variant options, never build or launch them.

Run as ``python -m examples.glm53.parallel_variants --out NEW_DIRECTORY``.
The directory must not exist. All entries remain uncompiled and untested;
writing options is not a correctness, capacity or performance qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path

from examples.glm53.options import Options


WORLDS = (2, 4, 8)
BATCHES = (1, 2, 4, 8)
MODES = (
    ('pp', 0, 0),
    ('ffn-tp', 1, 0),
    ('ep', 2, 0),
    ('full-tp', 1, 1),
    ('head-tp-ep', 2, 1),
)
REQUIRED_CHECKS = (
    'native compile and compiler VGPR/spill/LDS/occupancy resource reports',
    'shard byte identity, replicated rows, rank ownership and peak capacity including full-stride state',
    'preserve full (32,B,VOCAB) logits and numerical/state contracts; every request HF greedy 32/32',
    'request-isolated state/scratch and per-WG LDS versus batch-scaled capacity',
    'paired batch host-wall latency and aggregate tok/s against same-batch/world PP, same-kernel replay and strongest legal reference',
    'measured overlap, communication traffic and root/straggler critical path',
    'interleaved repeats with correctness for every arm and spread reported',
)


def variants():
    """Fixed, independently named arms; no implicit balanced or attention default."""
    result = []
    for world in WORLDS:
        for mode, parallel, attention_tp in MODES:
            for balance in (0, 1):
                name = f'{mode}{world}' + ('-up-balanced' if balance else '')
                config = Options(devices=world, parallel=parallel,
                                 attention_tp=attention_tp, up_tail_balance=balance)
                numerical = (
                    'TP numerical qualification: changed FP summation in partial projection/output reduction'
                    if parallel == 1 or attention_tp else
                    'EP preservation: expert selection, expert-local arithmetic and ordered root combination'
                    if parallel == 2 else
                    'PP preservation: unchanged per-row arithmetic and recurrent state; verify changed tile ownership'
                    if balance else
                    'PP reference qualification; only PP2 uses the measured default configuration'
                )
                result.append(dict(name=name, mode=mode, degree=world,
                                   options_file=name+'.json', options=config.to_dict(),
                                   options_sha256=config.digest(),
                                   measured_default_configuration=(world == 2 and parallel == 0 and balance == 0),
                                   effective_up_pairs=config.effective_up_pairs,
                                   numerical_contract=numerical,
                                   status='uncompiled-untested',
                                   checks_still_required=list(REQUIRED_CHECKS)))
    return result


def case_digest(options_sha256, batch):
    """Planning identity only, not compiled-source or runtime qualification."""
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('batch must be 1, 2, 4 or 8')
    value = dict(options_sha256=options_sha256, batch=batch)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cases(entries=None):
    """Cross the 30 compute options with B without changing Options identity."""
    entries = variants() if entries is None else entries
    return [dict(name=f'{entry["name"]}-b{batch}', variant=entry['name'],
                 mode=entry['mode'], degree=entry['degree'], batch=batch, bindings={'B':batch},
                 options_file=entry['options_file'], options_sha256=entry['options_sha256'],
                 case_sha256=case_digest(entry['options_sha256'], batch),
                 measured_default_case=entry['measured_default_configuration'] and batch == 1,
                 numerical_contract=entry['numerical_contract'], status='uncompiled-untested',
                 checks_still_required=list(REQUIRED_CHECKS))
            for entry in entries for batch in BATCHES]


def generate(out):
    """Write only fresh options and a manifest, refusing any existing output path."""
    out = Path(out)
    entries = variants()
    out.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        with (out/entry['options_file']).open('x') as stream:
            stream.write(json.dumps(entry['options'], sort_keys=True, indent=2)+'\n')
    manifest = dict(schema=2, purpose='options-generation-only', status='uncompiled-untested',
                    launches_performed=0, worlds=list(WORLDS), batches=list(BATCHES),
                    hash_basis='Options.digest(): SHA256 of sorted compact options JSON, not file bytes',
                    case_hash_basis='SHA256 of sorted compact {options_sha256,batch} JSON; NOT a build certificate',
                    batch_binding='B is a compile binding, not an Options field or option hash input',
                    measured_default_options=Options().to_dict(),
                    attention_state_capacity='full global stride per private rank and request; only local head interval accessed',
                    batch_capacity='per-WG LDS unchanged; per-request scratch and state scale with B',
                    correctness=dict(logits_shape=[32, 'B', 'VOCAB'], preserve_all_logits=True,
                                     hf_greedy_tokens_required_each=32, request_isolation_required=True),
                    variants=entries, cases=cases(entries))
    with (out/'manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new directory; existing paths are refused')
    args = parser.parse_args()
    try:
        manifest = generate(args.out)
    except FileExistsError:
        parser.error(f'output already exists: {args.out}')
    print(f'GLM-VARIANTS options={len(manifest["variants"])} cases={len(manifest["cases"])} launches=0 status=uncompiled-untested out={args.out}')


if __name__ == '__main__':
    main()
