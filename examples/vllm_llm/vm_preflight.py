"""Refuse an unstaged MI300X box before building: python3 -B -m examples.vllm_llm.vm_preflight.

Read-only by default. --fix runs the printed supply commands, except GPU
provisioning. ETX_STAGE_SOURCE is a local directory or user@host:/directory
containing llm/, rocm-headers/inc/ and cpu-gates.json from CPU-tested staging.
It copies existing inputs; it never generates gates or correctness seals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from examples.vllm_llm.qwen300_config import IMAGE

ROOT = Path(__file__).resolve().parents[2]
KFD = Path('/dev/kfd')
VLLM_TAG = 'v0.27.1'  # docs/DATA.md: pinned runtime's release tag
GOLDEN_SHA256 = 'bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484'
GATES = ('pytest', 'gfx942', 'dry_run', 'shell', 'local_matrix_16', 'rocm_build_check')


def command(argv):
    """Bound every read-only probe; suppress Python bytecode writes too."""
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), timeout=10)
    if result.returncode:
        raise ValueError((result.stderr or result.stdout).strip() or f'exit {result.returncode}')
    return result.stdout.strip()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)
    return detail


def stage(relative, destination, directory=False):
    source = '${ETX_STAGE_SOURCE:?set ETX_STAGE_SOURCE to the CPU-tested staging directory}'
    suffix = '/' if directory else ''
    return (f'mkdir -p {shlex.quote(str(destination if directory else destination.parent))} && '
            f'rsync -a "{source}/"{shlex.quote(relative+suffix)} {shlex.quote(str(destination)+suffix)}')


def check(model, arms, gates=None, *, fix=False):
    """Print exactly eight item verdicts and a summary; return the CLI exit code."""
    model, arms = Path(model).expanduser().resolve(), Path(arms).expanduser().resolve()
    gates = Path(gates).expanduser().resolve() if gates else arms/'cpu-gates.json'
    vllm = Path(os.environ.get('VLLM', str(Path.home()/'vllm'))).expanduser().resolve()
    headers = Path(os.environ.get('ETX_ROCM_HEADERS', str(Path.home()/'rocm-headers/inc'))).expanduser().resolve()
    py = ROOT/'.venv/bin/python'
    q = shlex.quote

    def interpreter():
        require(py.is_file(), f'missing {py}')
        command([py, '-I', '-B', '-c',
                 'import sys,yaml,numpy; from pathlib import Path; '
                 'assert sys.prefix != sys.base_prefix, "not a venv"; '
                 'assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) '
                 'for m in (yaml,numpy)), "dependencies outside .venv"'])
        return f'{py}; yaml, numpy import from .venv'

    def source():
        require((vllm/'.git').exists(), f'not a git checkout: {vllm}')
        top = command(['git', '-C', vllm, 'rev-parse', '--show-toplevel'])
        require(Path(top).resolve() == vllm, f'not a checkout root: {vllm}')
        commit = command(['git', '-C', vllm, 'rev-parse', 'HEAD'])
        require((vllm/'csrc/rocm/attention.cu').is_file(), f'missing {vllm}/csrc/rocm/attention.cu')
        return f'{vllm}; commit={commit}; runtime_tag={VLLM_TAG}'

    def packed():
        sizes, missing = [], []
        for name in ('vl.bin', 'vl.manifest', 'hf/config.json', 'golden_c1024.txt'):
            path = model/name
            if path.is_file():
                sizes.append(f'{name}={path.stat().st_size} bytes')
            else:
                missing.append(name)
        actual = (hashlib.sha256((model/'golden_c1024.txt').read_bytes()).hexdigest()
                  if (model/'golden_c1024.txt').is_file() else 'missing')
        detail = (f'{model}; '+', '.join(sizes)+f'; golden sha256={actual} expected={GOLDEN_SHA256}'
                  + ('; missing='+','.join(missing) if missing else ''))
        return require(not missing and actual == GOLDEN_SHA256, detail)

    def build_directory():
        return require(arms.is_dir() and bool(arms.stat().st_mode & 0o222)
                       and os.access(arms, os.W_OK | os.X_OK), f'arms={arms}; must exist and be writable')

    def gpu():
        # Check both, so a missing device cannot be mistaken for a software fix.
        product = command(['rocm-smi', '--showproductname'])
        return require(KFD.exists() and bool(re.search(r'\bMI300X\b', product)),
                       f'{KFD} exists={KFD.exists()}; rocm-smi={product}')

    def cpu_gates():
        values, error = {}, ''
        try:
            values = json.loads(gates.read_text())
            if not isinstance(values, dict):
                raise ValueError('expected a JSON object')
        except (OSError, ValueError) as exc:
            values, error = {}, f'; {exc}'
        detail = f'{gates}; '+', '.join(f'{k}={values.get(k)!r}' for k in GATES)+error
        return require(all(values.get(k) is True for k in GATES), detail)

    # get-pip works even when the runtime image has neither ensurepip nor pip.
    install = (f'cd {q(str(ROOT))} && python3 -m venv --without-pip .venv && '
               'curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/etx-get-pip.py && '
               '.venv/bin/python /tmp/etx-get-pip.py && '
               '.venv/bin/python -m pip install -e ".[dev]"')
    clone = f'git clone --depth 1 --branch {VLLM_TAG} https://github.com/vllm-project/vllm.git {q(str(vllm))}'
    if vllm.exists():  # rsynced csrc alone is not a checkout; preserve it on repair.
        backup = q(str(vllm)+'.preflight-backup')
        clone = f'test ! -e {backup} && mv {q(str(vllm))} {backup} && '+clone
    # Golden bytes are registered in the repo; use them rather than regenerating.
    golden = ROOT/'examples/llm/results/golden_c1024_qwen3-30b-a3b.txt'
    supply_model = stage(f'llm/{model.name}', model, directory=True)
    if all((model/n).is_file() for n in ('vl.bin', 'vl.manifest', 'hf/config.json')):
        supply_model = f'cp {q(str(golden))} {q(str(model/"golden_c1024.txt"))}'
    items = [
        ('interpreter', interpreter, install),
        ('vllm_source', source, clone),
        ('packed_model', packed, supply_model),
        ('arms', build_directory, f'mkdir -p {q(str(arms))} && chmod u+wx {q(str(arms))}'),
        ('rocm_headers', lambda: require(headers.is_dir() and (headers/'hip/hip_runtime.h').is_file(),
                                       f'ETX_ROCM_HEADERS={headers}; requires hip/hip_runtime.h'),
         stage('rocm-headers/inc', headers, directory=True)),
        ('runtime_image', lambda: f'{IMAGE}; '+command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']),
         f'docker pull {IMAGE}'),
        ('gpu', gpu, f'bash {q(str(ROOT/"tools/hotaisle_grab.sh"))} 1x'),
        ('gates', cpu_gates, stage('cpu-gates.json', gates)),
    ]
    failures = 0
    for name, probe, supply in items:
        try:
            try:
                detail = probe()
            except (OSError, ValueError, subprocess.SubprocessError) as missing:
                if not fix or name == 'gpu':
                    raise
                print(f'FIX {name}: {supply}', flush=True)
                try:
                    subprocess.run(['bash', '-c', supply], check=True, cwd=ROOT)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ValueError(f'{missing}; fix failed: {exc}') from exc
                detail = probe()
            print(f'PASS {name}: {" ".join(detail.splitlines())}', flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures += 1
            print(f'FAIL {name}: {" ".join(str(exc).splitlines())}; supply: {supply}', flush=True)
    print('PREFLIGHT PASS' if not failures else f'PREFLIGHT FAIL {failures} item(s)', flush=True)
    return int(bool(failures))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', nargs='?', type=Path, default=Path.home()/'llm/qwen3-30b-a3b')
    ap.add_argument('arms', nargs='?', type=Path, default=Path.home()/'arms')
    ap.add_argument('--gates', type=Path, help='default: <arms>/cpu-gates.json')
    ap.add_argument('--fix', action='store_true', help='run printed supply commands (never provision a GPU)')
    args = ap.parse_args(argv)
    return check(args.model, args.arms, args.gates, fix=args.fix)


if __name__ == '__main__':
    raise SystemExit(main())
