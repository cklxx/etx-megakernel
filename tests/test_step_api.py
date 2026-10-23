"""C ABI/decode contracts via ctypes and a CPU backend; never execute GPU code."""
import ctypes as C
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
from etx.runtime.python import Backend, Descriptor, RankHook, Runtime, StepHook, descriptor

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / 'etx/runtime/host'


def run(cmd):
    result = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    lib = tmp_path_factory.mktemp('step') / 'libetx_rt.so'
    run([shutil.which('c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
         '-shared', '-fPIC', HOST/'step.cc', '-o', lib])
    return Runtime(lib)


def stub(*, ranks=2, fail=None):
    events = []

    def rank(phase):
        @RankHook
        def callback(context, r, timeout):
            events.append((phase, r))
            assert timeout == 5.0
            return int(fail == (phase, r))
        return callback

    @StepHook
    def prepare(context, d):
        events.append(('prepare', d.contents.position, d.contents.token))
        return int(fail == 'prepare')

    @StepHook
    def finish(context, d):
        events.append(('finish', d.contents.position))
        d.contents.next_token = d.contents.token + 1
        return int(fail == 'finish')

    backend = Backend(*(rank(p) for p in ('reset', 'launch', 'wait')))
    d = descriptor(backend, ranks=ranks, token=7, prepare=prepare, finish=finish)
    return d, events


def test_step_and_decode_feedback(runtime):
    d, events = stub()
    assert runtime.step(d) == 8
    assert d.position == 1
    assert events == [('prepare', 0, 7), ('reset', 0), ('reset', 1),
                      ('launch', 0), ('launch', 1), ('wait', 0), ('wait', 1),
                      ('finish', 0)]
    assert runtime.decode(d, 5, eos=10) == [9, 10]
    assert (d.position, d.token) == (3, 10)
    assert ('prepare', 2, 9) in events


@pytest.mark.parametrize('failure', ['prepare', ('reset', 1), ('launch', 1),
                                    ('wait', 0), 'finish'])
def test_failures_do_not_advance_and_drain(runtime, failure):
    d, events = stub(fail=failure)
    with pytest.raises(RuntimeError, match='backend failure'):
        runtime.step(d)
    assert (d.position, d.token, d.next_token) == (0, 7, 7)
    if failure == ('launch', 1):
        assert events[-1] == ('wait', 0)
    if failure == ('wait', 0):
        assert events[-1] == ('wait', 1)
    if failure != 'finish':
        assert not any(e[0] == 'finish' for e in events)


@pytest.mark.parametrize('field,value', [('abi_version', 99), ('struct_size', 0),
    ('reserved', 1), ('ranks', 0), ('timeout_s', 0), ('timeout_s', float('nan')),
    ('timeout_s', float('inf')), ('position', -1), ('position', 2**63-1)])
def test_bad_descriptor_has_no_side_effects(runtime, field, value):
    d, events = stub()
    setattr(d, field, value)
    assert runtime.library.etx_step(C.byref(d)) == 1
    assert events == []


def test_null_contracts(runtime):
    assert runtime.library.etx_step(None) == 1
    d, events = stub()
    d.backend = C.POINTER(Backend)()
    assert runtime.library.etx_step(C.byref(d)) == 1
    assert events == []
    d, _ = stub()
    completed = C.c_uint64(99)
    assert runtime.library.etx_decode(C.byref(d), 1, -1, None, C.byref(completed)) == 1
    assert completed.value == 0
    assert runtime.library.etx_decode(C.byref(d), 0, -1, None, None) == 1
    assert runtime.decode(d, 0) == []
    with pytest.raises(ValueError):
        runtime.decode(d, -1)


def test_decode_retains_successful_prefix(runtime):
    d, events = stub()

    @StepHook
    def finish(ctx, state):
        state.contents.next_token = state.contents.token + 1
        return int(state.contents.position == 2)

    d.finish = finish
    with pytest.raises(RuntimeError) as error:
        runtime.decode(d, 5)
    assert error.value.tokens == [8, 9]
    assert (d.position, d.token) == (2, 9)


def test_c_header(tmp_path):
    source = tmp_path/'header.c'
    source.write_text('#include "step.h"\nint main(void) { etx_step_descriptor d = {0}; return d.reserved; }\n')
    run([shutil.which('cc'), '-std=c11', '-Wall', '-Werror', '-I'+str(HOST),
         '-c', source, '-o', tmp_path/'header.o'])


@pytest.mark.parametrize('backend,arch', [('CUDA', 'sm_90'), ('ROCM', 'gfx942')])
def test_generated_run_links(tmp_path, backend, arch):
    # SDK roots ($CUDA/$ROCM) or compiler commands; no device execution.
    value = os.environ.get(backend)
    name = 'nvcc' if backend == 'CUDA' else 'hipcc'
    compiler = (str(Path(value)/'bin'/name) if value and Path(value).is_dir() else value)
    compiler = compiler or shutil.which(name)
    if not compiler:
        pytest.skip(f'{backend} SDK unavailable; set ${backend} to enable link acceptance')
    run([sys.executable, '-m', 'etx', 'compile', 'examples/splitk_sum.py',
         '--arch', arch, '--inline-tiles', '--out', tmp_path])
    if backend == 'CUDA':
        # The example's reference tiles are HIP source. Translate only SDK
        # spellings in a temporary copy; the graph and arithmetic are unchanged.
        tile = tmp_path/'splitk.cu'
        tile.write_text((ROOT/'examples/tiles/splitk.hip').read_text()
                        .replace('<hip/hip_runtime.h>', '<cuda_runtime.h>')
                        .replace('<hip/hip_bf16.h>', '<cuda_bf16.h>')
                        .replace('__hip_bfloat16', '__nv_bfloat16'))
        kernel = tmp_path/'megakernel_d0.cu'
        kernel.write_text(kernel.read_text().replace('"examples/tiles/splitk.hip"',
                                                      '"'+str(tile)+'"'))
    flags = shlex.split(os.environ.get(f'ETX_HOST_{backend}_FLAGS', ''))
    common = [compiler, '-std=c++17', *flags, '-I'+str(HOST),
              '-I'+str(ROOT/'etx/runtime/include'), '-I'+str(tmp_path), '-I'+str(ROOT)]
    if backend == 'CUDA':
        common += ['-DETX_HOST_CUDA']
        device_flags = ['-x', 'cu', '-arch='+arch]
    else:
        device_flags = ['-x', 'hip', '--offload-arch='+arch]
    lib = tmp_path/'libetx_rt.so'
    run([shutil.which('c++'), '-std=c++17', '-shared', '-fPIC',
         HOST/'step.cc', '-o', lib])
    run([*common, *device_flags, tmp_path/('megakernel_d0.cu' if backend == 'CUDA' else 'megakernel_d0.hip'), HOST/'run.cc',
         HOST/'loader.cc', '-L'+str(tmp_path), '-letx_rt', '-o', tmp_path/'run'])
