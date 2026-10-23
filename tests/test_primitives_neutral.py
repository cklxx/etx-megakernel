"""CPU-only vendor compiler acceptance checks for the shared runtime."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from etx.codegen import emit_kernel, emit_lowering_header
from etx.passes import compile_graph
from examples import splitk_sum

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / 'etx/runtime/include'
HEADERS = INCLUDE / 'etx'
ROCM_HEADERS = Path('/data/rocm-headers')


def _compiler(env, fallback):
    command = os.environ.get(env) or shutil.which(fallback) or fallback
    if not Path(command).is_file():
        pytest.skip(f'{env} compiler unavailable: {command}')
    return command


def _source(directory, arch):
    directory.mkdir(parents=True, exist_ok=True)
    plan = compile_graph(splitk_sum.build(), arch, splitk_sum.bindings(), {})
    (directory / 'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
    source = directory / 'megakernel.hip'
    source.write_text(emit_kernel(plan))
    return source


def _run(args):
    return subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=120)


def test_shared_primitives_are_vendor_neutral():
    source = (HEADERS / 'primitives.h').read_text()
    assert '__hip_' not in source
    assert 'amdgcn' not in source.replace('#include "etx/primitives_amdgcn.h"', '')
    assert source.count('etx_atomic_sub_device(') == 4


def test_gfx942_isa_unchanged(tmp_path):
    compiler = _compiler('ROCM', str(ROCM_HEADERS / 'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/clang++'))
    source = _source(tmp_path, 'gfx942')
    baseline = tmp_path / 'baseline'
    shutil.copytree(INCLUDE, baseline)
    # Reconstruct the original four inline HIP atomics in the shared header.
    header = baseline / 'etx/primitives.h'
    header.write_text(header.read_text().replace(
        'etx_atomic_sub_device(p.events + ETX_EVW(idx), share)',
        '__hip_atomic_fetch_sub(p.events + ETX_EVW(idx), share, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)'))
    # Keep the HIP compilation-unit symbol stable for a literal byte comparison.
    flags = [compiler, '-x', 'hip', '--cuda-device-only', '--offload-arch=gfx942',
             '-O3', '-std=c++17', '-nogpuinc', '-cuid=etx_primitives_accept',
             f'--rocm-device-lib-path={ROCM_HEADERS}/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode',
             '-isystem', ROCM_HEADERS / 'inc', '-include', '__clang_hip_runtime_wrapper.h',
             '-D__HIP_PLATFORM_AMD__', f'-I{tmp_path}', '-S', source]
    outputs = []
    for name, include in [('before', baseline), ('after', INCLUDE)]:
        output = tmp_path / f'{name}.s'
        result = _run([*flags, f'-I{include}', '-o', output])
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_sm90_generated_sources_compile_and_reject_cu_slots(tmp_path):
    compiler = _compiler('CUDA', '/opt/cuda/bin/nvcc')
    source = _source(tmp_path, 'sm_90')
    # Explicitly instantiate every runtime lowering helper, including ones that
    # a static split-K plan can otherwise leave unused.
    with source.open('a') as stream:
        stream.write('''
__global__ void etx_primitive_probe(int32_t* p) {
  int32_t v = etx_ld_acquire_gpu(p) + etx_ld_acquire_sys(p);
  v += etx_atomic_sub_sys(p) + etx_atomic_sub_device(p, 2);
  etx_store_relaxed_device(p, v);
}
''')
    # Link-mode tile declarations require relocatable device code.
    flags = [compiler, '-arch=sm_90', '-x', 'cu', '-c', '-rdc=true', f'-I{INCLUDE}',
             f'-I{tmp_path}', source, '-o', tmp_path / 'kernel.o']
    result = _run(flags)
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run([*flags, '-DETX_SLOT_BY_CU=1'])
    assert result.returncode != 0
    assert 'ETX_SLOT_BY_CU is supported only on AMD' in result.stderr
