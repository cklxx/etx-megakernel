#!/usr/bin/env python3
"""NumPy reference for one absorbed-MLA batch-1 decode step (DeepseekV2).

This is the arithmetic the HIP tasks must reproduce, written once in fp32 so
every Fleet task has something to be compared against (docs/design.md §6).
It follows HF's `modeling_deepseek.py` exactly where exactness matters:

  * RoPE uses DeepseekV2's interleave trick — `view(d/2, 2).transpose()` before
    the standard `rotate_half`. Getting this wrong is silent and catastrophic.
  * The router runs in fp32: `softmax(x @ W_gate.T)`, then top-k with
    `sorted=False`; with `norm_topk_prob=false` the weights are used unnormalised
    and scaled by `routed_scaling_factor`.
  * `softmax_scale = q_head_dim**-0.5 * mscale**2`, mscale from `yarn_get_mscale`.

Absorbed form: instead of materialising K/V per head per token, the cache holds
one row `c ‖ k_pe` (576) shared by all heads, and W_UK/W_UV are folded into the
query and output projections:

    q_c[h] = q_nope[h] @ W_UK[h]          # [512]
    s[h,t] = scale * (q_c[h]·c[t] + q_pe[h]·k_pe[t])
    o_c[h] = Σ_t softmax(s)[h,t] · c[t]   # [512]
    o[h]   = o_c[h] @ W_UV[h].T           # [128]

which is algebraically identical to the non-absorbed path but reads 576 floats
per position instead of 16×(128+128).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- primitives

def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round fp32 to the nearest bf16 (ties to even), returned as fp32.

    HuggingFace runs the model in bf16, so every tensor a bf16 module emits is
    rounded like this. With ModelConfig.bf16 the reference applies it at the
    same points HF does, and the HIP kernel applies it at the same points
    (bf16_round in gemv.h). Without it the reference is the fp32 ideal.
    """
    x = np.array(x, dtype=np.float32)          # contiguous copy; keeps 0-d scalars 0-d
    u = x.view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.astype(np.uint32).view(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float,
             bf16: bool = False) -> np.ndarray:
    """DeepseekV2RMSNorm: normalise in fp32, scale by weight.

    In bf16 HF casts the normalised value to bf16 *before* multiplying by the
    bf16 weight, and the product is bf16 again: two roundings, both applied
    when bf16=True.
    """
    x32 = x.astype(np.float32)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * (1.0 / np.sqrt(var + eps))
    if bf16:
        y = bf16_round(y)
    out = y * weight.astype(np.float32)
    return bf16_round(out) if bf16 else out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """HF yarn_get_mscale — identity below scale 1."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(beta_fast, beta_slow, dim, base, max_pos):
    low = math.floor(yarn_find_correction_dim(beta_fast, dim, base, max_pos))
    high = math.ceil(yarn_find_correction_dim(beta_slow, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001  # HF's guard against a zero-width ramp
    linear = (np.arange(dim, dtype=np.float32) - low) / (high - low)
    return np.clip(linear, 0, 1)


def yarn_cos_sin(dim: int, seq_len: int, base: float, scale: float,
                 original_max: int, beta_fast: float, beta_slow: float,
                 mscale: float, mscale_all_dim: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces DeepseekV2YarnRotaryEmbedding._set_cos_sin_cache."""
    freq_extra = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freq_inter = 1.0 / (scale * base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max)
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)                      # [S, dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)      # [S, dim]
    _mscale = yarn_get_mscale(scale, mscale) / yarn_get_mscale(scale, mscale_all_dim)
    return np.cos(emb) * _mscale, np.sin(emb) * _mscale


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """DeepseekV2's RoPE: interleave-transpose, then the standard rotation.

    HF does `x.view(..., d//2, 2).transpose(-1, -2).reshape(..., d)` first, which
    turns [x0,x1,x2,...] into [x0,x2,x4,...,x1,x3,x5,...]. Without this the
    result is subtly wrong rather than obviously broken.
    """
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).swapaxes(-1, -2).reshape(*x.shape[:-1], d)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------- weights

@dataclass
class LayerWeights:
    """One decoder layer, fp32. Names mirror HF module names."""
    input_layernorm: np.ndarray
    q_proj: np.ndarray                # [heads*q_head_dim, hidden]
    kv_a_proj_with_mqa: np.ndarray    # [kv_lora+qk_rope, hidden]
    kv_a_layernorm: np.ndarray        # [kv_lora]
    kv_b_proj: np.ndarray             # [heads*(qk_nope+v_head), kv_lora]
    o_proj: np.ndarray                # [hidden, heads*v_head]
    post_attention_layernorm: np.ndarray
    # dense MLP (layer < first_k_dense) or MoE (layer >= first_k_dense)
    mlp_gate: np.ndarray | None = None
    mlp_up: np.ndarray | None = None
    mlp_down: np.ndarray | None = None
    gate_weight: np.ndarray | None = None      # [n_routed, hidden]
    experts_gate: np.ndarray | None = None     # [n_routed, moe_inter, hidden]
    experts_up: np.ndarray | None = None
    experts_down: np.ndarray | None = None     # [n_routed, hidden, moe_inter]
    shared_gate: np.ndarray | None = None      # [moe_inter*n_shared, hidden]
    shared_up: np.ndarray | None = None
    shared_down: np.ndarray | None = None      # [hidden, moe_inter*n_shared]

    @property
    def is_moe(self) -> bool:
        return self.gate_weight is not None


@dataclass
class ModelConfig:
    hidden: int
    heads: int
    kv_lora_rank: int
    qk_nope: int
    qk_rope: int
    v_head: int
    moe_inter: int
    n_routed: int
    top_k: int
    rms_eps: float = 1e-6
    rope_theta: float = 1e4
    rope_factor: float = 40.0
    rope_original_max: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 0.707
    mscale_all_dim: float = 0.707
    routed_scaling: float = 1.0
    norm_topk_prob: bool = False
    bf16: bool = False   # apply HF's bf16 rounding boundaries

    def r(self, x: np.ndarray) -> np.ndarray:
        """Round to bf16 where HF's bf16 modules would."""
        return bf16_round(x) if self.bf16 else x

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def softmax_scale(self) -> float:
        s = self.q_head_dim ** -0.5
        if self.mscale_all_dim:
            m = yarn_get_mscale(self.rope_factor, self.mscale_all_dim)
            s = s * m * m
        return s


# ---------------------------------------------------------------- tasks
# Each function below corresponds to one Fleet task in docs/design.md §3, so a
# HIP task can be diffed against exactly one of them.

def attention_absorbed(cfg: ModelConfig, w: LayerWeights, x: np.ndarray,
                       cache: np.ndarray, pos: int,
                       cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """One MLA decode step in absorbed form; appends this token's cache row.

    cache: [max_pos, kv_lora+qk_rope] fp32, rows 0..pos-1 already filled.
    Returns the attention block output [hidden] (before the residual add).
    """
    H, Dk, Dr, Dv = cfg.heads, cfg.qk_nope, cfg.qk_rope, cfg.v_head
    L = cfg.kv_lora_rank

    R = cfg.r
    x_n = rms_norm(x, w.input_layernorm, cfg.rms_eps, cfg.bf16)

    # q_proj then split per head into the nope/rope halves
    q = R(w.q_proj @ x_n).reshape(H, cfg.q_head_dim)
    q_nope, q_pe = q[:, :Dk], q[:, Dk:]

    # kv_a: compressed KV plus the single shared rope key
    kv_a = R(w.kv_a_proj_with_mqa @ x_n)
    c_new, k_pe_new = kv_a[:L], kv_a[L:]
    c_new = rms_norm(c_new, w.kv_a_layernorm, cfg.rms_eps, cfg.bf16)

    # RoPE on this position, then append the row shared by every head
    k_pe_new = R(apply_rope(k_pe_new[None, :], cos[pos][None, :], sin[pos][None, :])[0])
    q_pe = R(apply_rope(q_pe, cos[pos][None, :], sin[pos][None, :]))
    cache[pos, :L] = c_new
    cache[pos, L:] = k_pe_new

    c = cache[: pos + 1, :L]        # [S, 512]
    k_pe = cache[: pos + 1, L:]     # [S, 64]

    # kv_b_proj rows: head h owns [h*(Dk+Dv), h*(Dk+Dv)+Dk) = W_UK,
    #                              [.. +Dk, .. +Dk+Dv)       = W_UV
    out = np.empty(H * Dv, dtype=np.float32)
    for h in range(H):
        base = h * (Dk + Dv)
        w_uk = w.kv_b_proj[base: base + Dk]          # [Dk, L]
        w_uv = w.kv_b_proj[base + Dk: base + Dk + Dv]  # [Dv, L]

        q_c = q_nope[h] @ w_uk                        # [L] — absorb W_UK into q
        s = (c @ q_c + k_pe @ q_pe[h]) * cfg.softmax_scale
        p = softmax(s)
        o_c = p @ c                                   # [L]
        out[h * Dv:(h + 1) * Dv] = w_uv @ o_c         # absorb W_UV into the output

    return R(w.o_proj @ R(out))                       # attn output and o_proj are bf16


def mlp_unit(cfg: ModelConfig, gate: np.ndarray, up: np.ndarray, down: np.ndarray,
             x_n: np.ndarray) -> np.ndarray:
    """down(SiLU(gate x) * (up x)) with HF's bf16 roundings: gate_proj,
    act_fn, up_proj, the product and down_proj each emit a bf16 tensor."""
    R = cfg.r
    h = R(R(silu(R(gate @ x_n))) * R(up @ x_n))
    return R(down @ h)


def moe(cfg: ModelConfig, w: LayerWeights, x_n: np.ndarray) -> np.ndarray:
    """Router + top-k routed experts + shared experts, matching HF's MoE path."""
    R = cfg.r
    logits = w.gate_weight @ x_n                      # fp32 router, as in HF
    scores = softmax(logits)
    idx = np.argpartition(-scores, cfg.top_k - 1)[: cfg.top_k]  # topk, unsorted
    weights = scores[idx]
    if cfg.top_k > 1 and cfg.norm_topk_prob:
        weights = weights / (weights.sum() + 1e-20)
    else:
        weights = weights * cfg.routed_scaling

    y = np.zeros_like(x_n)
    for k, e in enumerate(idx):
        # expert_out.mul_(weight) in HF is a bf16 in-place multiply
        y += R(weights[k] * mlp_unit(cfg, w.experts_gate[e], w.experts_up[e],
                                     w.experts_down[e], x_n))

    shared = mlp_unit(cfg, w.shared_gate, w.shared_up, w.shared_down, x_n)
    return R(R(y) + shared)


def dense_mlp(cfg: ModelConfig, w: LayerWeights, x_n: np.ndarray) -> np.ndarray:
    return mlp_unit(cfg, w.mlp_gate, w.mlp_up, w.mlp_down, x_n)


def decoder_layer(cfg: ModelConfig, w: LayerWeights, x: np.ndarray,
                  cache: np.ndarray, pos: int,
                  cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """One full decoder layer: attention + residual, MLP/MoE + residual.
    The residual adds are bf16 adds in HF."""
    R = cfg.r
    x = R(x + attention_absorbed(cfg, w, x, cache, pos, cos, sin))
    x_n2 = rms_norm(x, w.post_attention_layernorm, cfg.rms_eps, cfg.bf16)
    return R(x + (moe(cfg, w, x_n2) if w.is_moe else dense_mlp(cfg, w, x_n2)))
