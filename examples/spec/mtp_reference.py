"""Independent float32 NumPy MTP oracle for the module8 NoPE/dense fixture.

Transcribes /data/refs/vllm-v0.30.0/glm5next/nvidia/mtp.py. This is a
mathematical reference: weights are dequantized; activation/cache FP8 and BF16
rounding are not simulated. It does not call target-model code or GPU kernels.
Sequence axes represent ONE request; batch requests need separate calls.
"""
from __future__ import annotations

import numpy as np


def rms(x, weight, eps):
    x = np.asarray(x, dtype=np.float32)
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * weight


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def dequantize(tensors):
    """Convert fixture storage to NumPy, including 128x128 block FP8 scales."""
    out = {}
    for name, tensor in tensors.items():
        if name.endswith('.weight_scale_inv'):
            continue
        w = tensor.float().numpy().copy()
        scale = tensors.get(name.removesuffix('.weight') + '.weight_scale_inv')
        if scale is not None:
            s = scale.float().numpy()
            w *= np.repeat(np.repeat(s, 128, 0), 128, 1)[:w.shape[0], :w.shape[1]]
        out[name] = w
    return out


class MTPReference:
    def __init__(self, config, weights):
        self.c = config.get('text_config', config)
        self.w = weights
        self.prefix = f"model.language_model.layers.{self.c['num_hidden_layers']}."
        if self.c['qk_rope_head_dim'] != 0 or self.c['mlp_layer_types'][-1] != 'dense':
            raise ValueError('reference supports module8 NoPE/dense MTP only')

    def weight(self, name):
        return self.w[self.prefix + name]

    def linear(self, x, name):
        return x @ self.weight(name + '.weight').T

    def norm(self, x, name):
        return rms(x, self.weight(name + '.weight'), self.c['rms_norm_eps'])

    def selected(self, x, qlatent):
        """Causal kpool selection, attention.py:309-397, ops/kpool_compress.py:164-208.

        Orthogonal Hadamard rotation cancels in dot products in this float32
        oracle. Production quantization can change near-tied pool rankings.
        """
        c = self.c
        n, d, heads = len(x), c['index_head_dim'], c['index_n_heads']
        q = self.linear(qlatent, 'self_attn.indexer.wq_b').reshape(n, heads, d)
        k = self.linear(x, 'self_attn.indexer.wk')
        k = (k-k.mean(-1, keepdims=True)) / np.sqrt(k.var(-1, keepdims=True)+1e-6)
        k = k*self.weight('self_attn.indexer.k_norm.weight') + self.weight('self_attn.indexer.k_norm.bias')
        gates = x @ self.weight('self_attn.indexer.index_kpool_compress_gate').T
        ape = self.weight('self_attn.indexer.index_kpool_compress_ape')
        weights = self.linear(x, 'self_attn.indexer.weights_proj') / np.sqrt(d*heads)
        pool, topk = c['index_kpool'], c['index_topk']
        selected = []
        for t in range(n):
            complete = (t+1)//pool
            if t+1 <= topk:
                selected.append(np.arange(t+1))
                continue
            pooled = np.stack([(softmax(gates[j*pool:(j+1)*pool]+ape, 0)*
                                k[j*pool:(j+1)*pool]).sum(0) for j in range(complete)])
            scores = (np.maximum(q[t] @ pooled.T, 0)*weights[t, :, None]).sum(0)
            winners = np.argsort(-scores, kind='stable')[:topk//pool]
            selected.append(np.sort(np.concatenate([*(np.arange(j*pool, (j+1)*pool) for j in winners),
                                                     np.arange(complete*pool, t+1)])))
        return selected

    def forward(self, next_tokens, target_hidden, positions=None):
        """Return (logits, recycled hidden) for aligned next tokens and target states.

        Row t consumes the next token supplied by the caller and target hidden
        row t; it never reads rows >t. Positions must be a contiguous prefix.
        """
        ids = np.asarray(next_tokens)
        previous = np.asarray(target_hidden, dtype=np.float32)
        if ids.ndim != 1 or previous.shape != (len(ids), self.c['hidden_size']) or not len(ids):
            raise ValueError('expected nonempty [time] ids and [time, hidden] states')
        positions = np.arange(len(ids)) if positions is None else np.asarray(positions)
        if not np.array_equal(positions, np.arange(len(ids))):
            raise ValueError('provide a complete contiguous prefix starting at zero')
        # mtp.py:158-178; ops/fused_eh_norm.py:40-53 (embedding first).
        e = self.w['model.language_model.embed_tokens.weight'][ids].copy()
        e[positions == 0] = 0
        x = self.linear(np.concatenate([self.norm(e, 'enorm'), self.norm(previous, 'hnorm')], -1), 'eh_proj')
        # mtp.py:106-109; model.py:470-494, no mHC on MTP.
        a = self.norm(x, 'input_layernorm')
        qlatent = self.norm(self.linear(a, 'self_attn.q_a_proj'), 'self_attn.q_a_layernorm')
        c = self.c
        n, heads, dim = len(ids), c['num_attention_heads'], c['qk_nope_head_dim']
        q = self.linear(qlatent, 'self_attn.q_b_proj').reshape(n, heads, dim)
        latent = self.norm(self.linear(a, 'self_attn.kv_a_proj_with_mqa'), 'self_attn.kv_a_layernorm')
        kv = self.linear(latent, 'self_attn.kv_b_proj').reshape(n, heads, 2*dim)
        k, v = np.split(kv, 2, -1)
        attended = np.empty_like(q)
        for t, rows in enumerate(self.selected(a, qlatent)):
            scores = np.einsum('hd,shd->hs', q[t], k[rows]) / np.sqrt(dim)
            attended[t] = np.einsum('hs,shd->hd', softmax(scores), v[rows])
        residual = x + self.linear(attended.reshape(n, -1), 'self_attn.o_proj')
        f = self.norm(residual, 'post_attention_layernorm')
        # model.py:116-148: clamped SwiGLU followed by down projection.
        gate, up = self.linear(f, 'mlp.gate_proj'), self.linear(f, 'mlp.up_proj')
        limit = c.get('swiglu_limit')
        if limit is not None:
            gate, up = np.minimum(gate, limit), np.clip(up, -limit, limit)
        f = self.linear(gate / (1 + np.exp(-gate)) * up, 'mlp.down_proj')
        hidden = self.norm(residual + f, 'shared_head.norm')
        # mtp.py:109-110,180-190: recycle normalized state; no second norm.
        return self.linear(hidden, 'shared_head.head'), hidden

    def chain(self, next_tokens, target_hidden, depth):
        """Greedy depth k, reusing the single MTP layer (mtp.py:171,185).

        Each seed row owns an independent branch containing its original
        prefix. Later depths append the greedy token and recycled last hidden
        state, advancing the position and preserving the branch's KV history.
        Outputs have [depth,time,...].
        """
        if not isinstance(depth, int) or depth < 1:
            raise ValueError('depth must be positive')
        ids = np.asarray(next_tokens)
        previous = np.asarray(target_hidden, dtype=np.float32)
        tokens, logits, recycled = [], [], []
        for t in range(len(ids)):
            branch_ids, branch_hidden = ids[:t+1].copy(), previous[:t+1].copy()
            branch_tokens, branch_logits = [], []
            for _ in range(depth):
                out, hidden = self.forward(branch_ids, branch_hidden)
                token = int(np.argmax(out[-1]))
                branch_tokens.append(token)
                branch_logits.append(out[-1])
                branch_ids = np.append(branch_ids, token)
                branch_hidden = np.concatenate([branch_hidden, hidden[-1:]], 0)
            tokens.append(branch_tokens)
            logits.append(branch_logits)
            recycled.append(hidden[-1])
        return np.asarray(tokens).T, np.swapaxes(np.asarray(logits), 0, 1), np.stack(recycled)
