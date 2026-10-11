"""CPU numerical/mapping evidence; does NOT execute gfx942 or replace CHECK-DSL."""
import os
import subprocess
import sys

import numpy as np
import pytest

from etx.ops import gemv


def bf(v):
    u = np.asarray(v, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def wave(v):
    v = v.copy()
    for off in (32, 16, 8, 4, 2, 1):
        v = (v + v[..., np.arange(64) ^ off]).astype(np.float32)
    return v[..., 0]


def router_dot(W, x, narrow):
    # Original nrt=1, spr=4, nw=16 -> per=1, four contributing waves.
    lanes = (W * x).reshape(-1, 4, 64, 8)
    acc = np.zeros(lanes.shape[:-1], np.float32)
    for j in range(8):
        acc = (acc + lanes[..., j]).astype(np.float32)
    reduced = wave(acc)
    part = np.zeros((len(W), 16), np.float32)
    part[:, :4] = np.float32(0) + reduced
    out = np.zeros(len(W), np.float32)
    for w in range(16):
        out = (out + (part[:, w] if not narrow or w < 4 else np.float32(0))).astype(np.float32)
    return bf(out)


@pytest.mark.parametrize('seed', range(8))
def test_router_same_partition_and_bf16_reference_gate(seed):
    rng = np.random.default_rng(seed)
    W = bf(rng.normal(size=(128, 2048)))
    # Use finite bf16 inputs, include cancellation and explicit zero rows.
    x = bf(rng.normal(size=2048))
    W[0] = 0; W[1, 1024:] = -W[1, :1024]; x[1024:] = x[:1024]
    original, narrow = router_dot(W, x, False), router_dot(W, x, True)
    assert np.array_equal(original.view(np.uint32), narrow.view(np.uint32))
    ref = (W.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    # host.hip cmp: round reference, 2 ulps, vector-scale cancellation floor.
    rounded = bf(ref)
    tol = np.maximum(2 * np.abs(rounded) / 128, np.max(np.abs(ref)) / 4096)
    assert np.all(np.abs(narrow - rounded) <= tol)


def test_router_opt_in_and_shape_refusal():
    code = '''
from examples.llm.model import load_dims
from examples.vllm_llm import model as M
from etx.ops import gemv
from examples.vllm_llm.plan_exports import exports, build_settings
cfg='examples/llm/configs/qwen3-30b-a3b.json'
d, names, phases=exports(cfg)
assert names['dsl_gemv_router_fold']=='dsl'
assert 'gemv' in build_settings(d,names,phases)['DSLOPS']
assert all(i['export']=='dsl_gemv_router_fold' for i in phases if i['role']=='router')
assert all(i['export']!='dsl_gemv_router_fold' for i in phases if i['role']!='router')
assert 'dsl_gemv_router_fold' in gemv.import_entries()
assert 'etx_touch_x_dsl_gemv_router_fold' in gemv.emit_tiles()
try: M.instances(load_dims('examples/llm/configs/mixtral-8x7b.json'))
except SystemExit as e: assert 'H=2048 E=128' in str(e)
else: raise AssertionError('wrong shape accepted')
'''
    r = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'ETX_VL_ROUTER_NARROW':'1'}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'dsl_gemv_router_fold' not in gemv.import_entries()


def norm_vector(rows, residual, weights, hoist):
    K = len(residual)
    # Mirrors VECNORM: 8 consecutive elements per thread, same row/add rounds.
    saved = weights.copy() if hoist else None
    z = np.zeros(K, np.float32)
    for row in rows:
        z = (z + row).astype(np.float32)
    if len(rows) > 1:
        z = bf(z)
    z = bf(z + residual)
    ss = np.zeros(1024, np.float32)
    for i in range((K // 8 + 1023) // 1024):
        for j in range(8):
            ix = (i * 1024 + np.arange(1024)) * 8 + j
            active = ix < K
            ss[active] = (ss[active] + z[ix[active]] * z[ix[active]]).astype(np.float32)
    red = wave(ss.reshape(16, 64))
    total = np.float32(0)
    for v in red:
        total = np.float32(total + v)
    sv = np.float32(1) / np.sqrt(np.float32(total / K + np.float32(1e-6)))
    # Same multiply order and bf16 rounding; only weight read time differs.
    return bf((z * sv).astype(np.float32) * (saved if hoist else weights))


@pytest.mark.parametrize('K,R', [(2048,1), (2048,8), (4096,8), (16384,1)])
def test_norm_hoist_same_bits_and_reference_tolerance(K, R):
    # Include the later-trip fallback at K > 8*1024; not a decoder shape claim.
    rng = np.random.default_rng(K + R)
    rows, rin, w = bf(rng.normal(size=(R,K))), bf(rng.normal(size=K)), bf(rng.uniform(.1,2,size=K))
    before, after = norm_vector(rows,rin,w,False), norm_vector(rows,rin,w,True)
    assert np.array_equal(before.view(np.uint32),after.view(np.uint32))
    z = np.zeros(K,np.float32)
    for row in rows: z = (z + row).astype(np.float32)
    z = bf((bf(z) if R > 1 else z) + rin)
    sv = np.float32(1 / np.sqrt(np.float32(np.mean(z.astype(np.float64)**2) + 1e-6)))
    ref = ((z * sv).astype(np.float32) * w).astype(np.float32)
    rounded = bf(ref)
    tol = np.maximum(2 * np.abs(rounded) / 128, np.max(np.abs(ref)) / 4096)
    assert np.all(np.abs(after - rounded) <= tol)
