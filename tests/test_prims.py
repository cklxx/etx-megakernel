"""Tile primitives (etx/ops/prims.py, PLAN section 6.4, Friday item 7): the wave_sum lowerings for gfx942 and sm_90."""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from etx.machine import load_machine
from etx.ops import gemv, prims

ROOT = Path(__file__).resolve().parent.parent
BEFORE = (Path(__file__).parent / "data" / "wave_sum0_gfx942_before.txt").read_text()   # gemv.py's inline text at 5a017f7


def test_gfx942_gemv_source_is_unchanged():
    """The emitted gemv source for gfx942 equals the source with the inline text from before the change."""
    before = gemv.STREAM.replace(prims.MARK.format("wave_sum0") + "\n", BEFORE)
    assert before != gemv.STREAM
    expected = gemv.emit_tiles().replace(prims.expand(gemv.STREAM, None), prims.expand(before, None))
    assert gemv.emit_tiles() == expected
    assert prims.lower("gfx942", "wave_sum0") + "\n" == BEFORE
    assert gemv.emit_tiles() == gemv.emit_tiles(machine="gfx942") == gemv.emit_tiles(machine=load_machine("gfx942"))


def test_gfx942_wave_sum_is_the_runtime_header_text():
    header = (ROOT / "etx/runtime/include/etx/tile_util.h").read_text()
    assert prims.lower("gfx942", "wave_sum") + "\n" in header


@pytest.mark.parametrize("arch", ["gfx942", "sm_90"])
def test_each_lowering_defines_its_function(arch):
    for name, fn in prims.PRIMITIVES.items():
        text = prims.lower(arch, name)
        ty = "int" if name == "uniform" else "float"
        assert re.search(rf"\b{ty} {fn}\({ty} v\)", text), (arch, name)
        size = load_machine(arch).wave_size
        assert f"static_assert(ETX_WAVE_SIZE == {size}" in text or name == "wave_sum" and arch == "gfx942"


def test_sm_90_source_has_no_amd_code():
    src = gemv.emit_tiles(machine="sm_90")
    assert "__builtin_amdgcn" not in src
    blk = src[src.index("#ifdef ETX_DSL_WSUM"):src.index("#endif", src.index("#ifdef ETX_DSL_WSUM"))]
    assert "__builtin_amdgcn" not in blk and "__shfl_xor_sync(0xffffffffu, v, 16)" in blk
    for name in prims.PRIMITIVES:
        assert "amdgcn" not in prims.lower("sm_90", name) and "__shfl_xor(" not in prims.lower("sm_90", name)


def test_sm_90_lowerings_compile(tmp_path):
    clang = "/opt/homebrew/opt/llvm/bin/clang"
    if not Path(clang).exists():
        clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang with nvptx64 is unavailable")
    targets = subprocess.run([clang, "--print-targets"], capture_output=True, text=True, check=True)
    if "nvptx64" not in targets.stdout:
        pytest.skip("clang lacks nvptx64")
    # Four-line shim: compile the primitive bodies without CUDA headers.
    shim = """#define __device__ __attribute__((device))
#define __forceinline__ inline __attribute__((always_inline))
#define __shfl_xor_sync(mask, v, lane) __nvvm_shfl_sync_bfly_f32(mask, v, lane, 31)
#define __shfl_sync(mask, v, lane) __nvvm_shfl_sync_idx_i32(mask, v, lane, 31)
"""
    source = tmp_path / "primitives.cu"
    source.write_text(shim + "#define ETX_WAVE_SIZE 32\n" +
                      "\n".join(prims.lower("sm_90", name) for name in prims.PRIMITIVES) +
                      '\nextern "C" __attribute__((global)) void probe(float* f, int* i) {\n'
                      '  f[0] = etx_wave_sum(f[0]); f[1] = dsl_wave_sum0(f[1]); i[0] = etx_uniform(i[0]);\n}\n')
    output = tmp_path / "primitives.ptx"
    proc = subprocess.run([clang, "-x", "cuda", "--cuda-gpu-arch=sm_90", "--cuda-device-only",
                           "-nocudainc", "-nocudalib", "-Xclang", "-target-feature", "-Xclang", "+ptx80",
                           "-S", str(source), "-o", str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ptx = output.read_text()
    assert ".target sm_90" in ptx and "shfl.sync.bfly" in ptx and "shfl.sync.idx" in ptx


def _warp32_butterfly(v):
    lanes = np.arange(32)
    for off in (16, 8, 4, 2, 1):
        v = (v + v[lanes ^ off]).astype(np.float32)
    return v


def test_sm_90_wave_sum0_lane0_is_bit_identical_to_the_butterfly():
    """Simulate the emitted __shfl_xor_sync steps. Lane 0 must equal the etx_wave_sum loop order bit for bit."""
    offs = [int(o) for o in re.findall(r"__shfl_xor_sync\(0xffffffffu, v, (\d+)\)", prims.lower("sm_90", "wave_sum0"))]
    assert offs == [16, 8, 4, 2, 1]
    rng = np.random.default_rng(11)
    lanes = np.arange(32)
    for t in range(2000):
        v = (rng.standard_normal(32) * np.float32(10.0) ** rng.integers(-6, 6, 32)).astype(np.float32)
        w = v.copy()
        for o in offs:
            w = (w + w[lanes ^ o]).astype(np.float32)
        a = _warp32_butterfly(v)
        assert (w.view(np.uint32) == a.view(np.uint32)).all(), t     # every lane, not only lane 0
        assert np.isclose(w[0], v.astype(np.float64).sum(), rtol=1e-5, atol=1e-30 + 1e-6 * np.abs(v).sum())


def test_missing_lowering_is_a_clear_error():
    with pytest.raises(KeyError, match="no lowering for tile primitive 'wave_sum0'"):
        prims.lower("gfx90a", "wave_sum0")
    with pytest.raises(KeyError, match="unknown tile primitive"):
        prims.lower("gfx942", "wave_max")
    with pytest.raises(KeyError, match="unknown primitive mark"):
        prims.expand("@@ETX_PRIM:nope@@", "gfx942")


def test_prims_module_has_no_arch_branch():
    """Only the default machine name may appear in the code. The docstring may name machines."""
    import ast
    tree = ast.parse((ROOT / "etx/ops/prims.py").read_text())
    doc = tree.body[0].value if isinstance(tree.body[0], ast.Expr) else None
    names = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n is not doc and re.search(r"\b(gfx9\d\d|sm_\d+)\b", n.value)]
    assert names == [prims.DEFAULT_MACHINE], names
