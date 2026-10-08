"""Tile primitives (etx/ops/prims.py, PLAN section 6.4, Friday item 7): the wave_sum lowerings for gfx942 and sm_90."""
import re
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
    expected = gemv.emit_tiles().replace(prims.expand(gemv.STREAM, None), before)
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
        assert re.search(rf"\bfloat {fn}\(float v\)", text), (arch, name)
        size = load_machine(arch).wave_size
        assert f"static_assert(ETX_WAVE_SIZE == {size}" in text or name == "wave_sum" and arch == "gfx942"


def test_sm_90_source_has_no_amd_code():
    src = gemv.emit_tiles(machine="sm_90")
    blk = src[src.index("#ifdef ETX_DSL_WSUM"):src.index("#endif", src.index("#ifdef ETX_DSL_WSUM"))]
    assert "__builtin_amdgcn" not in blk and "__shfl_xor_sync(0xffffffffu, v, 16)" in blk
    for name in prims.PRIMITIVES:
        assert "amdgcn" not in prims.lower("sm_90", name) and "__shfl_xor(" not in prims.lower("sm_90", name)


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
