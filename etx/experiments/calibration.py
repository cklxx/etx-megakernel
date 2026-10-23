"""Phase-0 calibration (bench/calib): lowering header, then compile and run each bench, as bench/calib/run_calib.sh did.

Each compile and run is a step on the shared executor with its full log; a bench's stdout is also kept in
<output>/<bench>.txt (the historical `| tee`). Runs touch the GPU; nothing here provisions one.
"""
import os
from pathlib import Path
import sys

BENCHES = ('atomic_pingpong', 'flag_latency', 'queue_contention')


def run(root, *, arch='gfx942', output=None, compiler='hipcc', deadline_s=900.0):
    from etx.api.execution import ProcessRequest
    from etx.codegen import emit_lowering_header
    from etx.contracts.execution import execute
    from etx.contracts.models import compiler_command
    from etx.machine import load_machine
    import uuid
    root = Path(root)
    out = Path(output) if output is not None else root / f'build/calib_{arch}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'etx_lowering.h').write_text(emit_lowering_header(load_machine(arch)))
    logs = out / 'calib-logs' / uuid.uuid4().hex

    def step(name, command, environment, container=None):
        request = ProcessRequest(tuple(command), str(root), deadline_s, environment=environment, silence_s=deadline_s)
        try:
            result = execute(request, output=logs / name)
        finally:
            if container is not None:
                removed = execute(ProcessRequest((command[0], 'rm', '--force', container), str(root), 30,
                                                 environment=environment), output=logs / (name + '-cleanup'))
                if removed.status != 'complete':
                    raise RuntimeError(f'compiler container cleanup not confirmed; see {logs}')
        sys.stdout.buffer.write((logs / name / 'output.log').read_bytes())
        sys.stdout.flush()
        return 0 if result.status == 'complete' else (result.returncode or 1)

    for bench in BENCHES:
        argv = ['hipcc', '-O2', f'--offload-arch={arch}', '-I', f'{root}/etx/runtime/include', '-I', str(out),
                f'{root}/bench/calib/{bench}.hip', '-o', f'{out}/{bench}']
        command, environment, container = compiler_command(compiler, argv, root, out)
        code = step('compile-' + bench, command, environment, container)
        if code:
            return code
        print(f'== {bench}', flush=True)
        # stdout to the console and <bench>.txt, stderr to the console only, like `"$OUT/$b" | tee "$OUT/$b.txt"`.
        code = step('run-' + bench, ('/bin/bash', '-c', 'set -o pipefail; "$1" | tee "$2"', 'etx-calibration',
                                     f'{out}/{bench}', f'{out}/{bench}.txt'),
                    {key: value for key, value in os.environ.items()
                     if key in ('PATH', 'HOME', 'LD_LIBRARY_PATH') or key.startswith(('HIP_', 'ROCR_', 'HSA_'))})
        if code:
            return code
    return 0
