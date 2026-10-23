from pathlib import Path
import shutil
import subprocess

import pytest

from etx.backends.nvptx.ir_import import ImportError_, import_kernel, import_from_yaml

FIX = Path(__file__).parent / 'fixtures/nvptx'
BASE = '''target triple = "nvptx64-nvidia-cuda"
define ptx_kernel void @tile(ptr %out) {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  call void @llvm.nvvm.barrier0()
  store i32 %t, ptr %out
  ret void
}
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
'''


def imp(text=BASE, **kw):
    return import_kernel(text, 'tile', 'etx_tile', (32, 1, 1), 64, target_cpu='sm_90', **kw)


def test_ids_and_barriers():
    text = BASE
    for kind in ('tid', 'ctaid', 'ntid', 'nctaid'):
        for dim in 'xyz':
            text = text.replace('  ret void', f'  %{kind}{dim} = call i32 @llvm.nvvm.read.ptx.sreg.{kind}.{dim}()\n  ret void')
    out, info = imp(text)
    assert '@llvm.nvvm.read.ptx.sreg' not in out[out.index('define internal'):]
    assert 'bar.sync $0, $1;' in out
    assert 'i32 32)' in out
    assert info.slots == 2 and info.barriers == 1 and info.grid_offset == 8


def test_annotations_and_yaml(tmp_path):
    text = BASE.replace('ptx_kernel ', '') + '!nvvm.annotations = !{!0}\n!0 = !{ptr @tile, !"kernel", i32 1}\n'
    cfg = tmp_path / 'target.yaml'
    cfg.write_text('kernel: tile\nexport: etx_tile\nblock: [32, 1, 1]\nthreads: 64\ntarget-cpu: sm_90\n')
    out, _ = import_from_yaml(text, cfg)
    assert '!nvvm.annotations' not in out
    assert '"target-cpu"="sm_90"' in out


@pytest.mark.parametrize('extra', [
    'declare void @llvm.nvvm.grid.sync()',
    'declare i32 @llvm.nvvm.read.ptx.sreg.clusterid.x()',
    '@dyn = external addrspace(3) global [0 x i8]',
    '''define i32 @helper() {
      %x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      ret i32 %x
    }
''',
])
def test_refusals(extra):
    with pytest.raises(ImportError_):
        imp(BASE + extra)


@pytest.mark.parametrize('arg', ['1', '%dynamic'])
def test_named_barrier_refusal(arg):
    with pytest.raises(ImportError_, match='non-zero'):
        imp(BASE.replace('barrier0()', f'barrier.cta.sync.aligned.all(i32 {arg})'))


def test_named_zero_barrier():
    out, info = imp(BASE.replace('barrier0()', 'barrier.cta.sync.aligned.all(i32 0)'))
    assert info.barriers == 1 and 'bar.sync' in out


def test_offline_cuda_pipeline(tmp_path):
    # Acceptance is deliberately mandatory: missing tools fail rather than skip.
    clang = shutil.which('clang')
    link = shutil.which('llvm-link')
    llc = shutil.which('llc')
    ptxas = shutil.which('ptxas') or '/opt/cuda/bin/ptxas'
    assert clang and link and llc
    def run(*args):
        subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True)
    source = tmp_path / 'tile.ll'
    run(clang, '-x', 'cuda', '--cuda-device-only', '--cuda-gpu-arch=sm_90',
        '--cuda-path=/opt/cuda', '-O3', '-S', '-emit-llvm', FIX / 'tile.cu', '-o', source)
    out, info = imp(source.read_text())
    assert info.lds_bytes_per_slot == 128
    imported = tmp_path / 'imported.ll'
    imported.write_text(out)
    caller = tmp_path / 'caller.ll'
    caller.write_text('''target triple = "nvptx64-nvidia-cuda"
@etx_lds_arena = addrspace(3) global [256 x i8] undef, align 16
declare void @etx_tile(ptr, i32, i32, i32)
define ptx_kernel void @caller(ptr %args, i32 %tid, i32 %slot) {
  call void @etx_tile(ptr %args, i32 0, i32 %tid, i32 %slot)
  ret void
}
''')
    bc, ptx = tmp_path / 'linked.bc', tmp_path / 'tile.ptx'
    run(link, imported, caller, '-o', bc)
    run(llc, '-mcpu=sm_90', bc, '-o', ptx)
    run(ptxas, '-arch=sm_90', '-v', ptx, '-o', tmp_path / 'tile.cubin')
    assert 'bar.sync' in ptx.read_text()


@pytest.mark.parametrize('block,threads', [((0, 1, 1), 64), ((64, 1, 1), 32), ((32, 1, 1), 1024), ((32, 1, 1), 63)])
def test_invalid_slot_layout(block, threads):
    with pytest.raises(ImportError_):
        import_kernel(BASE, 'tile', 'imp', block, threads, target_cpu='sm_90')


def test_shared_cap():
    text = '@s = addrspace(3) global [32 x i32] undef, align 16\n' + BASE.replace(
        '  ret void', '  store i32 %t, ptr addrspace(3) @s\n  ret void')
    with pytest.raises(ImportError_, match='cap'):
        imp(text, lds_cap=64)


def test_partial_warp_barrier_refused():
    with pytest.raises(ImportError_, match='whole-warp'):
        import_kernel(BASE, 'tile', 'imp', (17, 1, 1), 64, target_cpu='sm_90')
