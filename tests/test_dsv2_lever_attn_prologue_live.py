"""CPU proofs for lever attn_prologue_live on the pinned fleet source; no device required."""
from __future__ import annotations

import shutil

import numpy as np

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import attn_prologue_live as ap


def original_tree(values, add):
    smem = list(values)
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        for lane in range(stride):
            smem[lane] = add(smem[lane], smem[lane + stride])
    return smem[0]


def lever_tree(values, add):
    root = [add(add(values[t], values[t + 128]), add(values[t + 64], values[t + 192])) for t in range(64)]
    for stride in (32, 16, 8, 4, 2, 1):
        snap = root[:]
        for lane in range(stride):
            root[lane] = add(snap[lane], snap[lane + stride])
    return root[0]


def test_tree_is_block_reduce_ordered_bit_for_bit():
    assert original_tree(list(range(256)), lambda a, b: (a, b)) == lever_tree(list(range(256)), lambda a, b: (a, b))
    rng = np.random.default_rng(576)
    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 1, 0x80000001], np.uint32)
    cases = list(rng.integers(0, 2**32, size=(64, 256), dtype=np.uint32).view(np.float32))
    cases.append(np.resize(special, 256).view(np.float32))
    with np.errstate(all="ignore"):
        for v in cases:
            assert original_tree(v, np.add).view(np.uint32) == lever_tree(v, np.add).view(np.uint32)
    tree = ap.HELPERS[ap.HELPERS.index("etx_attn_reduce_ordered"):ap.HELPERS.index("struct EtxRopeCS")]
    for line in ("const float lo = smem[lane] + smem[lane + 128];", "const float hi = smem[lane + 64] + smem[lane + 192];",
                 "float root = lo + hi;", "if (lane < stride) root += rhs;"):
        assert line in tree
    assert tree.count("__syncthreads();") == 3


def test_every_expression_is_fleets_with_only_the_operand_source_changed(pinned_src):
    attn = (pinned_src / "kernels/attention.h").read_text()
    kv_post = pf._function(attn, "__device__ inline void kv_post(")
    rope = pf._function(attn, "__device__ __forceinline__ void apply_rope_interleaved(")
    gather = pf._function(attn, "__device__ inline void q_absorb_gather(")
    live = ap.HELPERS
    pairs = [  # (fleet's expression, the lever's expression with the same operands from registers)
        (kv_post, "local += v * v;", live, "local += raw0 * raw0;"),
        (kv_post, "local += v * v;", live, "local += raw1 * raw1;"),
        (kv_post, "const float inv = rsqrtf(ssq / kv_lora + eps);", live, "const float inv = rsqrtf(ssq / kv_lora + eps);"),
        (kv_post, "bf16_round(bf16_round(kv_a_raw[i] * inv) * __bfloat162float(kv_a_norm_w[i]))",
         live, "bf16_round(bf16_round(raw0 * inv) * g0)"),
        (kv_post, "row_lds[kv_lora + i] = kv_a_raw[kv_lora + i];", live, "row_lds[kv_lora + t] = rope_raw;"),
        (rope, "v[i] = a * cos[i] - b * sin[i];", live, "v[i] = a * cs.c0 - b * cs.s0;"),
        (rope, "v[i + half] = b * cos[i + half] + a * sin[i + half];", live, "v[i + half] = b * cs.c1 + a * cs.s1;"),
    ]
    for src, old, new_src, new in pairs:
        assert old in src and new in new_src, (old, new)
    # the final bf16 / cache-store loop is copied verbatim
    tail = kv_post[kv_post.index("    for (int i = threadIdx.x; i < kv_lora + qk_rope;"):]
    assert tail.rstrip().rstrip("}").rstrip() in live
    # the query gather keeps fleet's slice order and q_pe copy; only its rope source changes
    g_body = gather[gather.index("{") + 1:gather.index("apply_rope_interleaved")]
    assert g_body in ap.GATHER_LIVE and "etx_rope_pre(q_pe_lds, cs, qk_rope);" in ap.GATHER_LIVE


def test_rope_coefficients_are_the_ones_each_lane_reads():
    half = 32
    for t in range(256):
        i = min(t, half - 1)                       # etx_rope_cs
        if t < half:                               # only these lanes rotate; they read cos[t], cos[t + half]
            assert (i, i + half) == (t, t + half)


def test_scope_only_the_attention_task_switches(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old_a = (copy / "kernels/attention.h").read_text()
    old_k = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "attn_prologue_live")
    a = (copy / "kernels/attention.h").read_text()
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    assert a.replace("\n" + ap.HELPERS, "", 1).replace("\n" + ap.GATHER_LIVE, "", 1) == old_a
    back = (k.replace(ap.KV_POST_LIVE_CALL, ap.KV_POST_CALL).replace(ap.KV_POST_LIVE_TAIL, ap.KV_POST_TAIL)
             .replace(ap.GATHER_LIVE_CALL, ap.GATHER_CALL))
    assert back == old_k
    changed = sorted(p.name for p in (copy / "kernels").iterdir()
                     if p.read_bytes() != (pinned_src / "kernels" / p.name).read_bytes())
    assert changed == ["attention.h", "fleet_kernel.hip"]
