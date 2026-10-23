"""Offline module8 MTP draft execution, independently implemented with CPU Torch.

This is the draft dataflow's CPU implementation, not GPU qualification. Storage
FP8 is dequantized to float32; activation/cache quantization is not simulated.
The independent comparison oracle lives in examples/spec/mtp_reference.py.
"""
from __future__ import annotations

import json
import torch
import torch.nn.functional as F

from examples.glm53 import module_gate


class MTPUnavailable(RuntimeError):
    """The local gate lacks the tensors needed to execute MTP."""


def prerequisites(config=None, tensors=None):
    config = module_gate.module_config('module8') if config is None else config
    c = config.get('text_config', config)
    required = {n for n, _, _, _ in module_gate.mtp_tensors(c)}
    required.add('model.language_model.embed_tokens.weight')
    names = set(module_gate.tensors(config)) if tensors is None else set(tensors)
    missing = sorted(required - names)
    blockers = ['missing MTP tensors: ' + ', '.join(missing)] if missing else []
    return dict(gate='module8', main_layers=c['num_hidden_layers'],
                status='BLOCKED' if blockers else 'PREREQUISITES-ONLY',
                blockers=blockers, qualified=False,
                required_comparison=dict(positions=32, k=5, reference='vLLM v0.30.0 NumPy MTP'))


def require_prerequisites(config=None, tensors=None):
    report = prerequisites(config, tensors)
    if report['blockers']:
        raise MTPUnavailable('; '.join(report['blockers']))
    return report


class MTPGraphCPU:
    """Single-layer NoPE/dense draft dataflow with branch-local KV history.

    Each forward accepts a complete causal prefix. Chain appends one greedy
    token and the normalized last hidden state per iteration. With index
    sharing enabled, later iterations reuse the first iteration's index rows.
    Shared-index chains currently require the all-visible (<= topk) prefix;
    longer sparse chains are refused rather than approximating cache behavior.
    """

    def __init__(self, config, tensors):
        require_prerequisites(config, tensors)
        self.c = config.get('text_config', config)
        c = self.c
        if c['qk_rope_head_dim'] or c['mlp_layer_types'][-1] != 'dense':
            raise ValueError('only module8 NoPE/dense MTP is supported')
        self.prefix = f"model.language_model.layers.{c['num_hidden_layers']}."
        self.w = {}
        for name, value in tensors.items():
            if name.endswith('.weight_scale_inv'):
                continue
            w = value.detach().to(device='cpu', dtype=torch.float32).clone()
            scale = tensors.get(name.removesuffix('.weight') + '.weight_scale_inv')
            if scale is not None:
                s = scale.float().cpu().repeat_interleave(128, 0).repeat_interleave(128, 1)
                w *= s[:w.shape[0], :w.shape[1]]
            self.w[name] = w

    def weight(self, name):
        return self.w[self.prefix + name]

    def linear(self, x, name):
        return F.linear(x, self.weight(name + '.weight'))

    def norm(self, x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.c['rms_norm_eps']) * self.weight(name + '.weight')

    def selected(self, x, qlatent):
        c = self.c
        n, d, heads = len(x), c['index_head_dim'], c['index_n_heads']
        q = self.linear(qlatent, 'self_attn.indexer.wq_b').reshape(n, heads, d)
        k = F.layer_norm(self.linear(x, 'self_attn.indexer.wk'), (d,),
                         self.weight('self_attn.indexer.k_norm.weight'),
                         self.weight('self_attn.indexer.k_norm.bias'), 1e-6)
        gates = F.linear(x, self.weight('self_attn.indexer.index_kpool_compress_gate'))
        ape = self.weight('self_attn.indexer.index_kpool_compress_ape')
        weights = self.linear(x, 'self_attn.indexer.weights_proj') / (d*heads)**0.5
        pool, topk = c['index_kpool'], c['index_topk']
        rows = []
        for t in range(n):
            if t+1 <= topk:
                rows.append(torch.arange(t+1))
                continue
            complete = (t+1)//pool
            pk = torch.stack([(torch.softmax(gates[j*pool:(j+1)*pool]+ape, 0)*
                               k[j*pool:(j+1)*pool]).sum(0) for j in range(complete)])
            scores = ((q[t] @ pk.T).clamp_min(0)*weights[t, :, None]).sum(0)
            winners = torch.argsort(scores, descending=True, stable=True)[:topk//pool]
            rows.append(torch.cat([*(torch.arange(j*pool, (j+1)*pool) for j in winners),
                                   torch.arange(complete*pool, t+1)]).sort().values)
        return rows

    @torch.no_grad()
    def forward(self, next_tokens, target_hidden, positions=None, *, shared_rows=None):
        ids = torch.as_tensor(next_tokens)
        previous = torch.as_tensor(target_hidden, dtype=torch.float32)
        c = self.c
        if (ids.ndim != 1 or not len(ids) or ids.dtype not in (torch.int32, torch.int64)
                or previous.shape != (len(ids), c['hidden_size'])
                or torch.any(ids < 0) or torch.any(ids >= c['vocab_size'])):
            raise ValueError('expected nonempty integer token IDs and [time, hidden] states')
        pos = torch.arange(len(ids))
        if positions is not None and not torch.equal(torch.as_tensor(positions), pos):
            raise ValueError('provide a complete contiguous prefix starting at zero')
        e = self.w['model.language_model.embed_tokens.weight'][ids].clone()
        e[0] = 0
        x = self.linear(torch.cat((self.norm(e, 'enorm'), self.norm(previous, 'hnorm')), -1), 'eh_proj')
        a = self.norm(x, 'input_layernorm')
        latent_q = self.norm(self.linear(a, 'self_attn.q_a_proj'), 'self_attn.q_a_layernorm')
        heads, dim = c['num_attention_heads'], c['qk_nope_head_dim']
        q = self.linear(latent_q, 'self_attn.q_b_proj').reshape(len(ids), heads, dim)
        latent_kv = self.norm(self.linear(a, 'self_attn.kv_a_proj_with_mqa'), 'self_attn.kv_a_layernorm')
        k, v = self.linear(latent_kv, 'self_attn.kv_b_proj').reshape(len(ids), heads, 2*dim).chunk(2, -1)
        rows = self.selected(a, latent_q) if shared_rows is None else shared_rows
        attended = torch.stack([torch.einsum('hs,shd->hd', torch.softmax(
            torch.einsum('hd,shd->hs', q[t], k[r]) / dim**0.5, -1), v[r])
            for t, r in enumerate(rows)])
        residual = x + self.linear(attended.flatten(1), 'self_attn.o_proj')
        f = self.norm(residual, 'post_attention_layernorm')
        gate, up = self.linear(f, 'mlp.gate_proj'), self.linear(f, 'mlp.up_proj')
        limit = c.get('swiglu_limit')
        if limit is not None:
            gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
        hidden = self.norm(residual + self.linear(F.silu(gate)*up, 'mlp.down_proj'), 'shared_head.norm')
        self.last_rows = rows
        return self.linear(hidden, 'shared_head.head'), hidden

    def chain(self, next_tokens, target_hidden, depth):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError('depth must be positive')
        # Validate before slicing, including empty inputs.
        self.forward(next_tokens, target_hidden)
        ids = torch.as_tensor(next_tokens)
        previous = torch.as_tensor(target_hidden, dtype=torch.float32)
        tokens, logits, recycled = [], [], []
        for t in range(len(ids)):
            branch, states = ids[:t+1].clone(), previous[:t+1].clone()
            bt, bl, rows = [], [], None
            for _ in range(depth):
                out, hidden = self.forward(branch, states, shared_rows=rows)
                token = out[-1].argmax()
                bt.append(token)
                bl.append(out[-1])
                if self.c.get('index_share_for_mtp_iteration', False):
                    # Below topk the selection is all visible rows, including
                    # the new causal row. No pool ranking needs recomputation.
                    if len(branch) + 1 > self.c['index_topk']:
                        raise ValueError('shared sparse index beyond topk requires a cache-aware reference')
                    rows = self.last_rows + [torch.arange(len(branch) + 1)]
                branch = torch.cat((branch, token.reshape(1)))
                states = torch.cat((states, hidden[-1:]))
            tokens.append(torch.stack(bt))
            logits.append(torch.stack(bl))
            recycled.append(hidden[-1])
        return (torch.stack(tokens, 1).numpy(), torch.stack(logits, 1).numpy(),
                torch.stack(recycled).numpy())


def main():
    config = module_gate.module_config('module8')
    report = prerequisites(config, module_gate.tensors(config))
    print(json.dumps(report, indent=2))
    return 2 if report['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
