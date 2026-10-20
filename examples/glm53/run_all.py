"""One command for the GLM real-weight oracle: slice, HF reference, PP2 candidate, certificate.

This is section 6 of docs/GLM-EXECUTION-DESIGN-2026-10-19.md as one entry point
instead of a shell heredoc. Every parameter is a flag or the environment
variable named beside it; nothing about a particular machine is baked in.

Before any work the runner checks every input it needs (pinned checkpoint,
layer window, new output root, interpreter and modules, ROCm compiler, visible
GPUs) and refuses -- exit 2, output root not created -- naming the one command
that supplies each missing input. `--dry-run` runs nothing, prints the exact
step commands and the same refusal list, and exits 0 when the run is ready or 3
when it is not.

usage: python -m examples.glm53.run_all --model-root <pinned release dir> \
           --out <new output root> [--layers 22-25] [--batch 4] [--gpus 2]
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.glm53.prepare import (EVIDENCE, MODEL, check_slice, metadata,
                                    parse_layers, sha)

CONTEXT = 1024          # the certificate's timing window starts at 1024
MAXLEN = 1072           # context + 32 generated + 16 margin
GENERATED = 32
SEED = 20261019
DEFAULT_LAYERS = '22-25'
DEFAULT_MAX_MEMORY = '{"0":"80GiB","1":"80GiB","cpu":"32GiB"}'
MODULES = ('numpy', 'torch', 'transformers', 'safetensors', 'accelerate')
# The dependency/GPU probe runs in the interpreter the steps will use, so a
# broken environment is found here rather than an hour into the run.
PROBE = (
    'import importlib.util, json\n'
    f'need = {list(MODULES)!r}\n'
    'missing = [m for m in need if importlib.util.find_spec(m) is None]\n'
    'count = None\n'
    "if 'torch' not in missing:\n"
    '    try:\n'
    '        import torch\n'
    '        count = int(torch.cuda.device_count())\n'
    '    except Exception as exc:\n'
    "        count = 'error: ' + str(exc)\n"
    'print(json.dumps(dict(missing=missing, gpus=count)))\n'
)
TIMEOUTS = dict(prompts=600, stage=10800, prepare=600, reference=21600,
                build=2100, qualify=15000, oracle=300, bench=15300)
BENCH_SECONDS = 7200    # per model execution inside bench (it runs two)


class Failure(Exception):
    pass


def env_default(name, default=None):
    value = os.environ.get(name)
    return value if value else default


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-root', default=env_default('GLM_MODEL_ROOT'),
                    help='pinned GLM-5.3 release directory (env GLM_MODEL_ROOT)')
    ap.add_argument('--out', default=env_default('GLM_OUT_ROOT'),
                    help='new output root; every stage inside it refuses to overwrite (env GLM_OUT_ROOT)')
    ap.add_argument('--layers', default=env_default('GLM_LAYERS', DEFAULT_LAYERS),
                    help='contiguous source layers that straddle the PP2 split (env GLM_LAYERS)')
    ap.add_argument('--batch', type=int, choices=(1, 2, 4, 8), default=env_default('GLM_BATCH', '4'),
                    help='requests per step; one build and one certificate per batch (env GLM_BATCH)')
    ap.add_argument('--gpus', type=int, default=env_default('GLM_GPUS', '2'),
                    help='visible GPU count; the PP2 model host requires exactly 2 (env GLM_GPUS)')
    ap.add_argument('--max-memory', default=env_default('GLM_MAX_MEMORY', DEFAULT_MAX_MEMORY),
                    help='JSON device map for the offloaded HF reference (env GLM_MAX_MEMORY)')
    ap.add_argument('--python', default=env_default('GLM_PYTHON', sys.executable),
                    help='interpreter for every step (env GLM_PYTHON)')
    ap.add_argument('--hipcc', default=env_default('GLM_HIPCC', 'hipcc'),
                    help='ROCm compiler for the tile and host build (env GLM_HIPCC)')
    ap.add_argument('--seed', type=int, default=env_default('GLM_SEED', str(SEED)),
                    help='deterministic prompt seed (env GLM_SEED)')
    ap.add_argument('--seconds', type=int, default=env_default('GLM_SECONDS'),
                    help='override every step timeout (env GLM_SECONDS)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and the refusal list; run nothing')
    return ap.parse_args(argv)


def executable(name):
    """An interpreter/compiler given as a path or as a PATH name, or None."""
    if os.sep in name:
        return name if os.path.isfile(name) and os.access(name, os.X_OK) else None
    return shutil.which(name)


def preflight(args):
    """Every missing input, with the one command that supplies it. Touches nothing."""
    problems = []
    supply_seen = set()

    def problem(what, supply):
        if (what, supply) not in supply_seen:
            supply_seen.add((what, supply))
            problems.append((what, supply))

    try:
        cfg, api = metadata()
    except (ValueError, OSError, KeyError) as exc:
        cfg = api = None
        problem(f'pinned evidence is unreadable: {exc}', 'git checkout -- docs/evidence/qwen-batch-glm53')

    model = Path(args.model_root).expanduser() if args.model_root else None
    revision = api['sha'] if api else '<pinned revision>'
    download = (f'hf download {MODEL} --revision {revision} '
                f'--local-dir {shlex.quote(str(model or "<model-root>"))}')
    if model is None:
        problem('no model root (--model-root or GLM_MODEL_ROOT)',
                f'hf download {MODEL} --revision {revision} --local-dir <model-root>')
    elif not model.is_dir():
        problem(f'model root {model} is not a directory', download)
    else:
        config = model / 'config.json'
        if not config.is_file():
            problem(f'{config} is missing', download)
        elif cfg is not None:
            try:
                same = sha(config) == sha(EVIDENCE / 'glm-config.json')
            except OSError as exc:
                problem(f'{config} is unreadable: {exc}', download)
            else:
                if not same:
                    problem(f'{config} is not the pinned release config', download)
        stated = model / 'REVISION'
        if not stated.is_file():
            problem(f'{stated} is missing', download)
        elif api is not None:
            try:
                revision_text = stated.read_text().strip()
            except OSError as exc:
                problem(f'{stated} is unreadable: {exc}', download)
            else:
                if revision_text != api['sha']:
                    problem(f'{stated} names a different revision than the pin', download)
        index = model / 'model.safetensors.index.json'
        if not index.is_file():
            problem(f'{index} is missing', download)
        else:
            try:
                shards = list(json.loads(index.read_text())['weight_map'].values())
            except (ValueError, KeyError, TypeError) as exc:
                problem(f'{index} is unreadable: {exc}', download)
            else:
                unsafe = [shard for shard in shards if Path(shard).name != shard]
                if unsafe:
                    problem(f'{index} names a shard outside the snapshot: {unsafe[0]!r}', download)
                missing = sorted({shard for shard in shards
                                  if Path(shard).name == shard and not (model / shard).is_file()})
                if missing:
                    problem(f'{len(missing)} checkpoint shard(s) are missing, e.g. {", ".join(missing[:3])}',
                            download)

    if cfg is not None:
        try:
            check_slice(cfg['text_config'], parse_layers(args.layers))
        except ValueError as exc:
            problem(f'layer window {args.layers!r} is unusable: {exc}',
                    '--layers 22-25 (a contiguous window straddling the PP2 split at layers 23/24)')

    if args.gpus != 2:
        problem(f'--gpus {args.gpus}: examples/glm53/model_host.hip requires exactly two GPUs',
                '--gpus 2 (provision with tools/hotaisle_grab.sh 2x)')

    out = Path(args.out).expanduser() if args.out else None
    if out is None:
        problem('no output root (--out or GLM_OUT_ROOT)', '--out <new directory>')
    elif out.exists():
        problem(f'output root {out} already exists; every stage inside it refuses to overwrite',
                f'--out {out}-<label>')

    try:
        memory = json.loads(args.max_memory)
        if (not isinstance(memory, dict) or not memory
                or any(not isinstance(k, str) or not isinstance(v, (str, int)) for k, v in memory.items())):
            raise ValueError('expected a JSON object of device -> capacity')
    except (ValueError, TypeError) as exc:
        memory = None
        problem(f'--max-memory is not usable: {exc}',
                '--max-memory ' + shlex.quote(DEFAULT_MAX_MEMORY))

    python = executable(args.python)
    if python is None:
        problem(f'interpreter {args.python} not found',
                "python3 -m venv .venv && .venv/bin/pip install -e '.[reference]' accelerate")
    else:
        try:
            probe = subprocess.run([python, '-c', PROBE], capture_output=True, text=True, timeout=300)
            lines = [line for line in probe.stdout.splitlines() if line.strip()]
            if probe.returncode != 0 or not lines:
                raise ValueError((probe.stderr.strip().splitlines() or ['no output'])[-1])
            report = json.loads(lines[-1])
            if not isinstance(report, dict):
                raise ValueError(f'probe did not return a JSON object: {report!r}')
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report = None
            problem(f'dependency probe through {python} failed: {exc}',
                    f"{shlex.quote(python)} -m pip install -e '.[reference]' accelerate")
        if report is not None:
            missing = [name for name in MODULES if name in report.get('missing', [])]
            if missing:
                problem(f'{python} cannot import {", ".join(missing)}',
                        f"{shlex.quote(python)} -m pip install -e '.[reference]' accelerate")
            count = report.get('gpus')
            if not isinstance(count, int):
                problem(f'could not query the GPU count through {python}: {count}',
                        'tools/hotaisle_grab.sh 2x')
            elif count != args.gpus:
                problem(f'{count} visible GPU(s), --gpus {args.gpus}',
                        'HIP_VISIBLE_DEVICES=0,1 for two MI300X GPUs; provision with tools/hotaisle_grab.sh 2x')

    hipcc = executable(args.hipcc)
    if hipcc is None:
        problem(f'hipcc not found ({args.hipcc}); the tile and host build needs the ROCm compiler',
                'tools/hotaisle_grab.sh 2x (the ROCm two-GPU image), or --hipcc <path to hipcc>')

    ready = dict(model=model, out=out, python=python, hipcc=hipcc)
    return problems, ready


def plan(args, ready):
    """The ordered steps; every path derives from --out and --model-root."""
    python = ready['python'] or args.python
    out = ready['out'] or Path('<out>')
    model = ready['model'] or Path('<model-root>')
    stage = out / 'stage'
    reference = out / 'reference.json'
    build = out / 'build'
    qualification = out / 'qualification'
    certificate = out / 'certificate'
    bench_seconds = args.seconds or BENCH_SECONDS

    def seconds(name):
        return args.seconds or TIMEOUTS[name]

    steps = [
        dict(name='prompts', argv=None, log=None, seconds=seconds('prompts'),
             note=f'write {out / "prompts.json"} (batch={args.batch}, context={CONTEXT}, seed={args.seed})'),
        dict(name='stage', log=out / 'logs' / 'stage.log', seconds=seconds('stage'),
             argv=[python, '-u', '-m', 'examples.glm53.stage', '--src', str(model), '--out', str(stage),
                   '--maxlen', str(MAXLEN), '--gate', 'slice', '--layers', args.layers]),
        dict(name='prepare', log=out / 'logs' / 'prepare.log', seconds=seconds('prepare'),
             argv=[python, '-m', 'examples.glm53.prepare', '--require-model', '--stage', str(stage),
                   '--gate', 'slice', '--batch', str(args.batch), '--context', str(CONTEXT),
                   '--gpu-gib', '192', '--out', str(out / 'prepare.json')]),
        dict(name='reference', log=out / 'logs' / 'reference.log', seconds=seconds('reference'),
             argv=[python, '-u', '-m', 'examples.glm53.reference', '--src', str(model),
                   '--prompts', str(out / 'prompts.json'), '--out', str(reference),
                   '--offload', str(out / 'offload'), '--max-memory', args.max_memory,
                   '--layers', args.layers, '--gate', 'slice']),
        dict(name='build', log=out / 'build-log' / 'build.log', seconds=seconds('build'),
             argv=['bash', 'examples/glm53/build_model.sh', str(stage), str(build)],
             note=f'ETX_GLM_BATCH={args.batch} ETX_GLM_GATE=slice ETX_PYTHON={python}'),
        dict(name='qualify', log=out / 'logs' / 'qualify.log', seconds=seconds('qualify'),
             argv=[python, '-u', '-m', 'examples.glm53.run', 'qualify', '--stage', str(stage),
                   '--build', str(build), '--reference', str(reference),
                   '--out', str(qualification), '--batch', str(args.batch), '--gate', 'slice']),
    ]
    for repetition in (0, 1):
        steps.append(dict(name=f'oracle-run{repetition}', log=out / 'logs' / f'oracle-run{repetition}.log',
                          seconds=seconds('oracle'),
                          argv=[python, '-m', 'examples.glm53.oracle', '--reference', str(reference),
                                '--candidate', str(qualification / f'run{repetition}.json'),
                                '--out', str(out / f'oracle-run{repetition}.json'), '--gate', 'slice']))
    steps.append(dict(name='bench', log=out / 'logs' / 'bench.log', seconds=seconds('bench'),
                      argv=[python, '-u', '-m', 'examples.glm53.bench', '--stage', str(stage),
                            '--build', str(build), '--reference', str(reference),
                            '--out', str(certificate), '--gate', 'slice', '--seconds', str(bench_seconds)]))
    return steps


def print_plan(args, steps):
    out = Path(args.out).expanduser() if args.out else '<out>'
    model = Path(args.model_root).expanduser() if args.model_root else '<model-root>'
    print(f'PLAN model={model} out={out} layers={args.layers} batch={args.batch} gpus={args.gpus} '
          f'context={CONTEXT} maxlen={MAXLEN} generated={GENERATED}', flush=True)
    for index, step in enumerate(steps):
        if step['argv'] is None:
            print(f'PLAN {index} {step["name"]}: {step["note"]}', flush=True)
            continue
        note = f', {step["note"]}' if step.get('note') else ''
        print(f'PLAN {index} {step["name"]} [{step["seconds"]}s{note}]: {shlex.join(step["argv"])}', flush=True)
    print(f'CERTIFICATE {Path(out, "certificate", "bench_result.json")}', flush=True)


def print_refusal(problems, dry_run):
    state = "dry run: nothing was executed" if dry_run else "no work was done; the output root was not created"
    print(f'REFUSED before any work ({len(problems)} input(s) missing); {state}:', flush=True)
    for what, supply in problems:
        print(f'  - {what}', flush=True)
        print(f'    supply: {supply}', flush=True)


def step_env(args, ready):
    env = dict(os.environ)
    inherited = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + inherited if inherited else '')
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['ETX_GLM_BATCH'] = str(args.batch)
    env['ETX_GLM_GATE'] = 'slice'
    env['ETX_PYTHON'] = ready['python']
    if ready['hipcc'] and os.sep in ready['hipcc']:
        env['PATH'] = str(Path(ready['hipcc']).parent) + os.pathsep + env.get('PATH', '')
    return env


def execute(argv, log, seconds, env):
    """Stream a step to the console and its own log; stop it at the deadline."""
    print(f'RUN {shlex.join(argv)}', flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log.open('w') as sink:
        # A new session lets the deadline kill the whole step (bash and the
        # python/hipcc children it started), not just the shell.
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)

        def pump():
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        def stop():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop()
            code = process.wait()
        except BaseException:
            stop()
            process.wait()
            raise
        reader.join(timeout=60)
    if timed_out:
        raise Failure(f'timed out after {seconds}s; log {log}')
    if code != 0:
        raise Failure(f'exit {code}; log {log}')


def prompts(path, batch, seed, vocab):
    """Batch distinct equal-length rows, deterministic for a seed; the reference refuses anything else."""
    rows = [random.Random(seed + r).sample(range(vocab), CONTEXT) for r in range(batch)]
    if len({tuple(row) for row in rows}) != batch:
        raise Failure('generated duplicate prompts')
    path.write_text(json.dumps(rows) + '\n')
    print(f'PROMPTS {path} batch={batch} context={CONTEXT} vocab={vocab} seed={seed}', flush=True)


def run(args, ready, steps):
    out = ready['out']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'offload').mkdir()
    cfg, _ = metadata()
    env = step_env(args, ready)
    for step in steps:
        if step['argv'] is None:
            prompts(out / 'prompts.json', args.batch, args.seed, cfg['text_config']['vocab_size'])
            continue
        execute(step['argv'], step['log'], step['seconds'], env)
    certificate = out / 'certificate' / 'bench_result.json'
    result = json.loads(certificate.read_text())
    median = result.get('median_decode_wall_ms')
    shown = f'{median:.6f}' if isinstance(median, float) else median
    print(f'RESULT batch={args.batch} tokens={result.get("tokens_matched")}/{result.get("tokens_expected")} '
          f'median_decode_wall_ms={shown} certificate={certificate}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    problems, ready = preflight(args)
    steps = plan(args, ready)
    if args.dry_run:
        print_plan(args, steps)
        if problems:
            print_refusal(problems, dry_run=True)
            print(f'DRY-RUN NOT READY: {len(problems)} input(s) missing', flush=True)
            return 3
        print(f'DRY-RUN READY: {len(steps)} steps would run into {ready["out"]}', flush=True)
        return 0
    if problems:
        print_refusal(problems, dry_run=False)
        return 2
    try:
        run(args, ready, steps)
    except Failure as exc:
        print(f'FAILED {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
