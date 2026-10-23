"""GPU-ready GLM boundary transport build (historically examples/glm53/build_boundary.sh); no download or provisioning.

  build_boundary.sh <new-output-dir> [--local]

Plans examples/glm53/boundary.py (B from ETX_GLM_BATCH, default 1), then either links the two rank kernels
with the boundary host (hipcc, default) or, with --local, checks device IR/assembly and host syntax with a
local LLVM (ETX_LLVM) and HIP headers (ETX_ROCM_HEADERS). Same relative commands from the repository root,
each a step on the shared executor with its full log.
"""
import os
from pathlib import Path
import shutil
import sys

from . import paths


def main(argv=None):
    from etx.api.execution import ProcessRequest
    from etx.contracts.execution import execute
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('build_boundary.sh: line 5: 1: new build output directory', file=sys.stderr)
        return 1
    out, local = args[0], (args[1:2] == ['--local'])
    root = str(paths.ROOT)
    environment = {key: value for key, value in os.environ.items()
                   if key in ('PATH', 'HOME', 'TMPDIR', 'LD_LIBRARY_PATH') or key.startswith(('ETX_', 'ROCM_', 'HIP', 'HCC_'))}
    import uuid
    logs = Path(root) / 'build' / 'boundary-logs' / uuid.uuid4().hex
    count = iter(range(100))

    def step(name, argv):
        tool = argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0]) or argv[0]
        directory = logs / f'{next(count):02d}-{name}'
        result = execute(ProcessRequest((tool, *argv[1:]), root, 1800.0, environment=environment, silence_s=1800.0),
                         output=directory)
        sys.stdout.buffer.write((directory / 'output.log').read_bytes())
        sys.stdout.flush()
        return 0 if result.status == 'complete' else (result.returncode or 1)

    batch = os.environ.get('ETX_GLM_BATCH') or '1'
    code = step('compile', [sys.executable, '-m', 'etx', 'compile', 'examples/glm53/boundary.py', '--arch', 'gfx942',
                            '--out', out, '--inline-tiles', '--bind', f'B={batch}'])
    if code:
        return code
    llvm = os.environ.get('ETX_LLVM') or '/opt/homebrew/opt/llvm/bin'
    if local:
        for d in (0, 1):
            code = step(f'ir-d{d}', ['bash', 'etx/importer/hip_ir_local.sh', f'{out}/megakernel_d{d}.hip', f'{out}/mk{d}.ll',
                                     '-Ietx/runtime/include', '-I', out, '-I', '.'])
            code = code or step(f'llc-d{d}', [f'{llvm}/llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3',
                                              f'{out}/mk{d}.ll', '-o', f'{out}/mk{d}.s'])
            if code:
                return code
        headers = os.environ.get('ETX_ROCM_HEADERS') or os.path.expanduser('~/code/rocm-headers')
        code = step('host-syntax', [f'{llvm}/clang++', '-x', 'hip', '--cuda-host-only', '-fsyntax-only', '-nogpuinc',
                                    '-nogpulib', '--offload-arch=gfx942', '-std=c++17', '-isystem', f'{headers}/clang',
                                    '-isystem', f'{headers}/inc', '-include', '__clang_hip_runtime_wrapper.h',
                                    '-isystem', 'etx/importer/shim/mac', '-D__HIP_PLATFORM_AMD__', '-Ietx/runtime/include',
                                    '-I', out, '-I', '.', 'examples/glm53/boundary_host.hip'])
    else:
        code = step('link', ['hipcc', '-O2', '-std=c++17', '--offload-arch=gfx942', '-Ietx/runtime/include', '-I', out,
                             '-I', '.', f'{out}/megakernel_d0.hip', f'{out}/megakernel_d1.hip',
                             'examples/glm53/boundary_host.hip', '-pthread', '-o', f'{out}/run'])
    if code:
        return code
    print('TRANSPORT-BUILD OK; GPU result unmeasured')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
