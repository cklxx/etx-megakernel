"""WP-18 offline acceptance: Qwen DSL tiles, warp32 capacity and AMD stability."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from etx.codegen import emit_lowering_header
from etx.machine.model import load_machine
from etx.ops import attnprep, coh, gemv, last_router, local_fold, prims, topk, topk_key32

ROOT = Path(__file__).resolve().parents[1]
# SHA256 of the exact pre-WP-18 emitted gfx942 sources, including optional tiles.
BASELINES = {
    'topk': 'e47def2aa7c2f21e4f218dd5fee91c8cc534d7354c6bbe59e28e70eebbfb7a53',
    'attnprep': '61f7a62daa891dfc73175288af4bf9dd3b25071d1d8fef002471814d7132ae02',
    'coh': '2e5a9a6b04a8ab70eaea82dbceeb4aab681b10c7c345cfd6e5fbd433b0c30b27',
    'local_fold': '6e65e10b95919b0c2974f646ed1a39331198042a1313ddec1061f9460df05a59',
    'last_router': '0291f77457d997517db909e09815bb1eeb3f54ea5c4945983ba620948e49ff50',
}


def _fragment(name, machine):
    if name == 'local_fold':
        return local_fold.emit(gemv.STREAM, gemv.BODIES['plain'], machine=machine)
    if name == 'last_router':
        return last_router.emit(machine=machine)
    return globals()[name].emit_tiles(machine=machine)


@pytest.mark.parametrize('name', BASELINES)
def test_gfx942_source_byte_identical(name):
    source = _fragment(name, 'gfx942')
    assert hashlib.sha256(source.encode()).hexdigest() == BASELINES[name]


def _nvcc():
    value = os.environ.get('CUDA', '/opt/cuda')
    compiler = Path(value) / 'bin/nvcc' if Path(value).is_dir() else Path(value)
    compiler = str(compiler) if compiler.is_file() else shutil.which('nvcc')
    # Acceptance must fail, not silently pass via skipped vendor compilation.
    assert compiler, 'CUDA compiler unavailable; set CUDA to SDK root or nvcc'
    return compiler


def _compile(source, tmp_path, flags):
    path = tmp_path / 'tiles.cu'
    path.write_text(emit_lowering_header(load_machine('sm_90')) + source)
    result = subprocess.run([_nvcc(), '-arch=sm_90', '-std=c++17', '-dc',
                             '-I', str(ROOT / 'etx/runtime/include'), *flags,
                             str(path), '-o', str(tmp_path / 'tiles.o')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'tiles.o').stat().st_size


@pytest.mark.parametrize('name', ['topk', 'attnprep', 'coh'])
@pytest.mark.parametrize('flags', [(), ('-DETX_TOPK_KEY32', '-DETX_TOPK_PAIRED_LOADS',
                                      '-DETX_DSL_COH_L1', '-DETX_DSL_UNIFORM')])
def test_sm90_standalone_tiles_compile(name, flags, tmp_path):
    source = _fragment(name, 'sm_90')
    assert '__hip_atomic' not in source and '__builtin_amdgcn' not in source
    if name == 'topk':
        assert 'float p[4]' in source and 'k < 4' in source
    if name == 'attnprep':
        assert 'HD == 128 && ETX_WAVE_SIZE >= 16' in source
    _compile(source, tmp_path, flags)


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('verify', [False, True])
def test_sm90_local_fold_compiles(vector, verify, tmp_path):
    # Assemble legacy GEMV with the target-specific optional fragment, then
    # lower the translation unit once, as the combined DSL emitter does.
    source = gemv.emit_tiles() + local_fold.emit(gemv.STREAM, gemv.BODIES['plain'], machine='sm_90')
    source = source.replace(prims.lower('gfx942', 'uniform'), prims.lower('sm_90', 'uniform'))
    source = source.replace(prims.lower('gfx942', 'wave_sum0'), prims.lower('sm_90', 'wave_sum0'))
    source = prims.emit_source(source, 'sm_90', lds_bytes=gemv.machine_lds_bytes('sm_90'))
    flags = ['-DETX_DSL_LOCAL_FOLD=1', '-DETX_DSL_COH', '-DETX_TOPK_KEY32']
    if vector:
        flags.append('-DETX_DSL_VECNORM')
    if verify:
        flags.append('-DETX_DSL_LOCAL_FOLD_VERIFY=1')
    _compile(source, tmp_path, flags)


def test_sm90_last_router_compiles(tmp_path):
    # Narrow router forms are selected at module import; isolate that setting.
    script = '''from etx.ops import gemv, topk, last_router
s = gemv.emit_tiles(machine="sm_90")
for n in ("etx_tile_dsl_gemv_router_fold", "etx_tile_x_dsl_gemv_router_fold"):
    s = s.replace("void " + n + "(", "void " + n + "_original_unused(")
# Add only tile body: the shared CUDA support is already in GEMV's source.
print(s)
print(topk.TILE)
print(last_router.emit(machine="sm_90"))
'''
    import sys
    env = dict(os.environ, ETX_VL_ROUTER_NARROW='1', ETX_VL_LAST_ROUTER='0')
    result = subprocess.run([sys.executable, '-c', script], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    # Lower the legacy top-k body independently before joining the already
    # lowered GEMV support (no duplicate prelude/primitive definitions).
    source = result.stdout
    start = source.index(topk.TILE)
    lowered = topk.emit_tiles(machine='sm_90')
    body = lowered[lowered.index('// ---- dsl_topk:'):]
    source = source[:start] + body + source[start + len(topk.TILE):]
    _compile(source, tmp_path, ['-DETX_TOPK_KEY32', '-DETX_DSL_COH'])


@pytest.mark.parametrize('wave_size', [32, 64])
@pytest.mark.parametrize('renorm', [False, True])
def test_topk_all_128_experts_and_ties(wave_size, renorm):
    rng = np.random.default_rng(18)
    rows = [np.zeros(128, np.float32), np.arange(128, dtype=np.float32) / 128]
    rows += [rng.normal(size=128).astype(np.float32) for _ in range(30)]
    for row in rows:
        bits = topk_key32.bf16_bits(row)
        logits = topk_key32.bf16_to_f32(bits)
        expected = sorted(range(128), key=lambda i: (-float(logits[i]), i))[:8]
        ids, weights = topk_key32.key_topk(bits, 8, renorm, wave_size=wave_size)
        ref_ids, ref_weights = topk_key32.ref_topk(bits, 8, renorm, wave_size=wave_size)
        assert ids == ref_ids == expected
        np.testing.assert_array_equal(np.asarray(weights).view(np.uint32),
                                      np.asarray(ref_weights).view(np.uint32))
