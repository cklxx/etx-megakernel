"""Tile primitives (etx/ops/prims.py, PLAN section 6.4, Friday item 7): the wave_sum lowerings for gfx942 and sm_90."""
import re
import os
import sys
import hashlib
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
    expected = gemv.emit_tiles().replace(prims.expand(gemv.STREAM, "gfx942"), prims.expand(before, "gfx942"))
    assert gemv.emit_tiles() == expected
    # Full output captured before WP-17, including coherent accessors and hooks.
    assert hashlib.sha256(gemv.emit_tiles(machine="gfx942").encode()).hexdigest() == "b9f4bee576ceedaac49649e21bda9d3c15fadc1643a3375a38918459d342ce0a"
    assert prims.lower("gfx942", "wave_sum0") + "\n" == BEFORE
    assert gemv.emit_tiles() == gemv.emit_tiles(machine="gfx942") == gemv.emit_tiles(machine=load_machine("gfx942"))


def test_gfx942_wave_sum_is_the_runtime_header_text():
    header = (ROOT / "etx/runtime/include/etx/tile_util.h").read_text()
    assert prims.lower("gfx942", "wave_sum") + "\n" in header


@pytest.mark.parametrize("arch", ["gfx942", "sm_90"])
def test_each_lowering_defines_its_function(arch):
    for name, fn in prims.PRIMITIVES.items():
        if name not in load_machine(arch).primitives:
            continue  # legacy functions remain in the HIP runtime header
        text = prims.lower(arch, name)
        if name.startswith("block_"):
            assert f"float {fn}(float v, float* red)" in text
            continue
        ty = "int" if name == "uniform" else "unsigned" if name == "wave_max_u32" else "float"
        assert re.search(rf"\b{ty} {fn}\({ty} v\)", text), (arch, name)
        size = load_machine(arch).wave_size
        assert f"static_assert(ETX_WAVE_SIZE == {size}" in text or name == "wave_sum" and arch == "gfx942"


def test_sm_90_source_has_no_amd_code():
    src = gemv.emit_tiles(machine="sm_90")
    assert "__builtin_amdgcn" not in src
    assert prims.lower("sm_90", "wave_sum0") in src
    assert "__hip_atomic" not in src and "sc0 sc1" not in src
    for name in prims.PRIMITIVES:
        assert "amdgcn" not in prims.lower("sm_90", name) and "__shfl_xor(" not in prims.lower("sm_90", name)


def _nvcc():
    cuda = os.environ.get("CUDA", os.environ.get("CUDA_HOME", "/opt/cuda"))
    nvcc = shutil.which("nvcc") or str(Path(cuda) / "bin/nvcc")
    if not Path(nvcc).exists():
        pytest.skip("CUDA nvcc is unavailable; sm_90 compilation not checked")
    return nvcc


def _compile(source, tmp_path, flags=()):
    path = tmp_path / "tiles.cu"
    path.write_text(source)
    proc = subprocess.run([_nvcc(), "-arch=sm_90", "-std=c++17", "-dc",
                           "-I", str(ROOT / "etx/runtime/include"),
                           *flags, str(path), "-o", str(tmp_path / "tiles.o")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "tiles.o").stat().st_size > 0


@pytest.mark.parametrize("flags", [(), ("-DETX_DSL_WSUM", "-DETX_DSL_SCALAR", "-DETX_DSL_COH",
                                      "-DETX_DSL_UNIFORM", "-DETX_TOPK_KEY32"),
    ("-DETX_DSL_COH_L1", "-DETX_DSL_NT", "-DETX_DSL_NORM_HOIST", "-DETX_DSL_GU1",
     "-DETX_EMIT_DSL_BIGK=1", "-DETX_EMIT_DSL_EXT=1", "-DETX_DSL_RBATCH"),
    ("-DDSL_ABLATE_NORED", "-DDSL_ABLATE_NOGEMV", "-DDSL_ABLATE_MOEUP_NOGEMV")])
def test_sm_90_gemv_compiles_with_nvcc(tmp_path, flags):
    source = gemv.emit_tiles(machine="sm_90")
    # Exercise every core primitive, including those not used by default GEMV.
    source += r'''
__global__ void primitive_probe(float* f, unsigned* u, float* scratch) {
      float v = etx_wave_sum(f[threadIdx.x]);
      v += dsl_wave_sum0(v);
      v = etx_wave_max(v);
      v = etx_block_sum(v, scratch);
      f[threadIdx.x] = etx_block_max(v, scratch);
      u[threadIdx.x] = etx_wave_max_u32(u[threadIdx.x]) + etx_uniform(1);
    }
'''
    _compile(source, tmp_path, flags)


@pytest.mark.parametrize("key32", [False, True])
def test_sm_90_combined_cli_compiles(tmp_path, key32):
    imports = tmp_path / "imports.json"
    imports.write_text("{}")
    tiles = tmp_path / "tiles.h"
    subprocess.run([sys.executable, "-m", "etx.ops.emit", str(imports), "--tiles", str(tiles),
                    "--ops", "gemv,topk,attnprep", "--machine", "sm_90"],
                   cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    source = tiles.read_text()
    assert "float p[4]" in source and "k < 4" in source
    assert "__builtin_amdgcn" not in source and "__hip_atomic" not in source
    flags = ["-DETX_DSL_WSUM", "-DETX_DSL_COH_L1", "-DETX_DSL_UNIFORM", "-DETX_TOPK_PAIRED_LOADS"]
    if key32:
        flags.append("-DETX_TOPK_KEY32")
    _compile(source, tmp_path, flags)


def test_wave32_partial_storage_and_argument_layout():
    old, new = gemv.import_entries(machine="gfx942"), gemv.import_entries(machine="sm_90")
    for name in old:
        assert old[name]["params"] == new[name]["params"]
        assert old[name]["arg_bytes"] == new[name]["arg_bytes"]
        expected = 4 * (gemv.DSL_NB * gemv.K_MAX + 2 * gemv.RCH +
                        gemv.DSL_NB * (gemv.THREADS // 32) * gemv.RCH + 32)
        assert new[name]["lds_bytes"] == new[name]["lds_bytes_per_slot"] == expected
    assert f"#define ETX_IMPORT_LDS {expected}" in gemv.emit_tiles(machine="sm_90")


def test_primitive_target_must_be_explicit():
    with pytest.raises(ValueError, match="explicit machine"):
        prims.lower(None, "uniform")
    with pytest.raises(ValueError, match="explicit machine"):
        prims.expand("", None)


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
        prims.lower("gfx942", "not_a_primitive")
    with pytest.raises(KeyError, match="unknown primitive mark"):
        prims.expand("@@ETX_PRIM:nope@@", "gfx942")


def test_prims_module_has_no_arch_branch():
    """No architecture name may appear in the code. The docstring may name machines."""
    import ast
    tree = ast.parse((ROOT / "etx/ops/prims.py").read_text())
    doc = tree.body[0].value if isinstance(tree.body[0], ast.Expr) else None
    names = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n is not doc and re.search(r"\b(gfx9\d\d|sm_\d+)\b", n.value)]
    assert names == [], names
