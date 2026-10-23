"""Offline compile/link of `etx compile` output directories; never executes a device binary.

Replaces the body of tools/cpu_device_check.sh. Container paths (/src, /out), working
directory, environment and compiler command lines are the legacy ones, so outputs are
byte-identical; each step now runs through the shared executor with a full log,
a deadline and a named container that is removed before its result is admitted.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from etx.api.execution import ProcessRequest
from etx.api.types import Configuration
from etx.contracts.execution import execute


ROOT = Path(__file__).resolve().parents[2]
# Graph directory : tile/host name, in the legacy order.
GRAPHS = (('splitk_sum', 'splitk'), ('moe_layer', 'moe'), ('gemm_rs', 'gemm_rs'))
# Same exports as the legacy shell, evaluated inside the toolchain's own environment.
PRELUDE = ('export ROCM_PATH="$1" HIP_PATH="$1" HIP_CLANG_PATH="$1/llvm/bin" HCC_AMDGPU_TARGET=gfx942; '
           'export LD_LIBRARY_PATH="$1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; shift; ')


def check_codegen(codegen, *, toolchain, docker='docker', logs=None, deadline_s=600):
    """Link every imported IR file and compile/link the three example graphs in `codegen`."""
    out = Path(codegen).resolve()
    if not out.is_dir():
        raise ValueError(f'codegen directory not found: {codegen}')
    if type(toolchain) is not str or not toolchain or toolchain.startswith('-'):
        raise ValueError('toolchain must be an explicit ROCm directory or local image')
    logs = Path(logs).absolute() if logs is not None else out.parent / (out.name + '-device-check')
    logs.mkdir(parents=True, exist_ok=False)
    local = Path(toolchain).is_dir()
    steps = []

    def invoke(argv, name, environment, deadline=deadline_s, echo=True):
        request = ProcessRequest(tuple(argv), str(ROOT if local else logs), deadline,
                                 environment=environment, silence_s=120)
        result = execute(request, output=logs / name)
        if echo:
            # The legacy console carried each compiler command's output; keep it on ours.
            sys.stdout.buffer.write((logs / name / 'output.log').read_bytes())
            sys.stdout.flush()
        return result

    if local:
        # Logical absolute path, as the legacy `cd && pwd`.
        rocm, src, target = os.path.abspath(toolchain), str(ROOT), str(out)
        environment = {key: value for key, value in os.environ.items()
                       if key in ('PATH', 'LD_LIBRARY_PATH', 'HOME', 'TMPDIR')}

        def run(script, name):
            result = invoke(['/bin/bash', '-c', PRELUDE + script, 'etx-device-check', rocm], name, environment)
            steps.append(dict(step=name, status=result.status, returncode=result.returncode))
            return result
    else:
        docker = shutil.which(docker) or docker
        environment = {'PATH': '/usr/bin:/bin'}
        inspected = invoke([docker, 'image', 'inspect', toolchain, '--format', '{{.Id}} {{json .RepoDigests}}'],
                           'image', environment, echo=False)
        steps.append(dict(step='image', status=inspected.status, returncode=inspected.returncode))
        if inspected.status != 'complete':
            return _receipt(out, logs, toolchain, steps)
        (out / 'compiler-image.txt').write_bytes((logs / 'image/output.log').read_bytes())
        image_id = (logs / 'image/output.log').read_text().split()[0]
        rocm, src, target = '/opt/rocm', '/src', '/out'

        def run(script, name):
            container = 'etx-check-' + hashlib.sha256(os.urandom(16)).hexdigest()[:32]
            try:
                result = invoke([docker, 'run', '--name', container, '--pull=never', '--network', 'none',
                                 '--user', f'{os.getuid()}:{os.getgid()}',
                                 '--mount', f'type=bind,src={ROOT},dst=/src,readonly',
                                 '--mount', f'type=bind,src={out},dst=/out', '--workdir', '/src', image_id,
                                 'bash', '-c', PRELUDE + script, 'etx-device-check', rocm], name, environment)
            finally:
                removed = invoke([docker, 'rm', '--force', container], name + '-cleanup', environment, 20,
                                 echo=False)
                if removed.status != 'complete':
                    raise RuntimeError(f'compiler container cleanup not confirmed; see {logs}')
            steps.append(dict(step=name, status=result.status, returncode=result.returncode))
            return result

    if run(f'"$ROCM_PATH/bin/hipcc" --version > {target}/compiler-version.txt; '
           f'sha256sum "$ROCM_PATH/llvm/bin/clang" > {target}/compiler-sha256.txt', 'compiler').status != 'complete':
        return _receipt(out, logs, toolchain, steps)
    for ll in sorted((out / 'imports').glob('*.ll')):
        ir = f'{target}/imports/{ll.name}'
        print(f'Verify imported IR {ir}', flush=True)
        if run(f'"$ROCM_PATH/llvm/bin/llvm-link" {ir} -o {ir[:-3]}.bc && '
               f'"$ROCM_PATH/llvm/bin/opt" -passes=verify -disable-output {ir[:-3]}.bc',
               'import-' + ll.stem).status != 'complete':
            return _receipt(out, logs, toolchain, steps)
    for graph, tile in GRAPHS:
        print(f'Compile/link {graph} (no execution)', flush=True)
        if run(f'cd {src} && "$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx942 -fgpu-rdc '
               f'-I etx/runtime/include -I "{target}/{graph}" "{target}/{graph}"/megakernel_d*.hip '
               f'"examples/tiles/{tile}.hip" "examples/hosts/{tile}.hip" -o "{target}/{graph}/run"',
               'link-' + graph).status != 'complete':
            return _receipt(out, logs, toolchain, steps)
    return _receipt(out, logs, toolchain, steps)


def _receipt(out, logs, toolchain, steps):
    links = [step for step in steps if step['step'].startswith('link-')]
    linked = len(links) == len(GRAPHS) and all(step['status'] == 'complete' for step in steps)
    outputs = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and (path.suffix == '.bc' or path.name in ('run', 'compiler-image.txt',
                                                                       'compiler-version.txt', 'compiler-sha256.txt')):
            outputs[str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    failed = next((step for step in steps if step['status'] != 'complete'), None)
    record = dict(schema_version=1, kind='codegen-device-check', toolchain=toolchain, codegen=str(out),
                  logs=str(logs), steps=steps, outputs=outputs, status='LINKED' if linked else 'FAILED',
                  returncode=0 if linked else (failed['returncode'] if failed and failed['returncode'] else 1),
                  device_execution=False)
    with (logs / 'device-check.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return Configuration(record)
