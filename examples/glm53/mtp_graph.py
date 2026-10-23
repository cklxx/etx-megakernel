"""Offline module8 MTP draft: ETX graph plus the retained CPU Torch reference.

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



def build_graph(config, capacity=36, source=None):
    """Build the module8 all-visible MTP prefix as explicit ETX operations.

    This gate covers 32 seeds plus four appended drafts. Sparse prefixes beyond
    index_topk are deliberately refused. Each operation is one scalar-lane tile;
    this is a correctness graph, with no device performance qualification.
    ``source`` selects the generated tile file location (never a repo file by
    default). The same bodies are compiled for the CPU executor below.
    """
    import tempfile
    from pathlib import Path
    from etx.ir import Graph, Resource
    from etx.frontends import hip_link

    c = config.get('text_config', config)
    if c['qk_rope_head_dim'] or c['mlp_layer_types'][-1] != 'dense':
        raise ValueError('only module8 NoPE/dense MTP is supported')
    if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= c['index_topk']:
        raise ValueError('capacity must fit the all-visible index_topk prefix')
    source = Path(source) if source is not None else Path(tempfile.mkdtemp(prefix='etx-mtp-')) / 'tiles.h'
    g = Graph('glm53_mtp_dense_prefix')
    h, heads, dim = c['hidden_size'], c['num_attention_heads'], c['qk_nope_head_dim']
    prefix = f"model.language_model.layers.{c['num_hidden_layers']}."
    weight_names = {}
    for full, _, _, shape in module_gate.mtp_tensors(c):
        name = 'w' + str(len(weight_names))
        weight_names[full] = name
        g.tensor(name, shape, dtype='float32', bytes_per_elem=4, role='weight')
    weight_names['model.language_model.embed_tokens.weight'] = 'embedding_weight'
    g.tensor('embedding_weight', (c['vocab_size'], h), dtype='float32', bytes_per_elem=4, role='weight')
    g.tensor('control', (2,), dtype='int32', bytes_per_elem=4, role='runtime')  # length, reuse index
    g.tensor('ids', (capacity,), dtype='int32', bytes_per_elem=4, role='runtime')
    g.tensor('previous', (capacity, h), dtype='float32', bytes_per_elem=4, role='runtime')
    g.tensor('rows', (capacity, capacity), dtype='int32', bytes_per_elem=4, role='runtime')
    bodies, last = [], None

    def tensor(name, width):
        g.tensor(name, (capacity, width), dtype='float32', bytes_per_elem=4)
        return name

    def emit(name, inputs, output, code, integer=(), preserve=False):
        nonlocal last
        args = ['control'] + inputs + [output]
        decls = ['const int n=((const int*)ctx->args[0])[0];']
        for i, arg in enumerate(args[1:], 1):
            typ = 'int' if arg in integer else 'float'
            decls.append(f'{typ}* {arg}=({typ}*)ctx->args[{i}];')
        symbol = 'mtp_' + name
        bodies.append('extern "C" __device__ void ' + symbol + '(const etx_ctx* ctx){\n'
                      '#ifndef MTP_CPU\nif(threadIdx.x != 0) return;\n#endif\n'
                      + '\n'.join(decls) + '\n' + code + '\n}')
        event = 'done_' + name
        g.etensor(event, (1,))
        g.call_device(name, (1,), hip_link(str(source), symbol), resource=Resource(threads=64, vgpr=32),
                      args=args, reads=['control'] + inputs + ([output] if preserve else []), writes=[output],
                      in_edges={last: 'i->i'} if last else {}, out_edges={event: 'i->i'})
        last = event

    def weight(key):
        return weight_names[prefix + key]

    def norm(x, key, width, name):
        w, out = weight(key + '.weight'), tensor(name, width)
        emit(name, [x, w], out,
             f'for(int t=0;t<n;++t){{float s=0;for(int j=0;j<{width};++j)s+={x}[t*{width}+j]*{x}[t*{width}+j];'
             f'float r=1.0f/sqrtf(s/{width}+{c["rms_norm_eps"]}f);'
             f'for(int j=0;j<{width};++j){out}[t*{width}+j]={x}[t*{width}+j]*r*{w}[j];}}')
        return out

    def linear(x, key, name):
        w = weight(key + '.weight')
        rows, cols = g.tensors[w].shape
        out = tensor(name, rows)
        emit(name, [x, w], out,
             f'for(int t=0;t<n;++t)for(int r=0;r<{rows};++r){{float s=0;'
             f'for(int j=0;j<{cols};++j)s+={x}[t*{cols}+j]*{w}[r*{cols}+j];{out}[t*{rows}+r]=s;}}')
        return out

    def add(a, b, name):
        out = tensor(name, h)
        emit(name, [a, b], out, f'for(int j=0;j<n*{h};++j){out}[j]={a}[j]+{b}[j];')
        return out

    e = tensor('embedding', h)
    emit('embedding', ['ids', 'embedding_weight'], e,
         f'for(int t=0;t<n;++t)for(int j=0;j<{h};++j)embedding[t*{h}+j]=t?embedding_weight[ids[t]*{h}+j]:0;', integer=('ids',))
    en, hn = norm(e, 'enorm', h, 'en'), norm('previous', 'hnorm', h, 'hn')
    joined = tensor('joined', 2*h)
    emit('join', [en, hn], joined,
         f'for(int t=0;t<n;++t)for(int j=0;j<{h};++j){{joined[t*{2*h}+j]=en[t*{h}+j];joined[t*{2*h}+{h}+j]=hn[t*{h}+j];}}')
    x = linear(joined, 'eh_proj', 'projected')
    a = norm(x, 'input_layernorm', h, 'attn_input')
    ql = norm(linear(a, 'self_attn.q_a_proj', 'qa'), 'self_attn.q_a_layernorm', c['q_lora_rank'], 'ql')
    q = linear(ql, 'self_attn.q_b_proj', 'q')
    kvl = norm(linear(a, 'self_attn.kv_a_proj_with_mqa', 'kva'), 'self_attn.kv_a_layernorm', c['kv_lora_rank'], 'kvl')
    kv = linear(kvl, 'self_attn.kv_b_proj', 'kv')
    # Dense topk selection has no ranking. On shared iterations preserve all
    # previous rows and publish only the appended causal row.
    emit('index', [], 'rows',
         f'int begin=((const int*)ctx->args[0])[1]?n-1:0;'
         f'for(int t=begin;t<n;++t)for(int s=0;s<{capacity};++s)rows[t*{capacity}+s]=(s<=t);', integer=('rows',), preserve=True)
    out = tensor('attended', heads*dim)
    emit('attention', [q, kv, 'rows'], out,
         f'for(int t=0;t<n;++t)for(int hd=0;hd<{heads};++hd){{float scores[{capacity}],mx=-INFINITY,z=0;'
         f'for(int s=0;s<=t;++s){{float v=0;for(int j=0;j<{dim};++j)v+=q[(t*{heads}+hd)*{dim}+j]*kv[(s*{heads}+hd)*{2*dim}+j];'
         f'scores[s]=rows[t*{capacity}+s]?v/{dim**0.5}f:-INFINITY;mx=fmaxf(mx,scores[s]);}}'
         f'for(int s=0;s<=t;++s){{scores[s]=expf(scores[s]-mx);z+=scores[s];}}'
         f'for(int j=0;j<{dim};++j){{float v=0;for(int s=0;s<=t;++s)v+=scores[s]/z*kv[(s*{heads}+hd)*{2*dim}+{dim}+j];'
         f'attended[(t*{heads}+hd)*{dim}+j]=v;}}}}', integer=('rows',))
    residual = add(x, linear(out, 'self_attn.o_proj', 'attn_out'), 'residual')
    f = norm(residual, 'post_attention_layernorm', h, 'ffn_input')
    gate, up = linear(f, 'mlp.gate_proj', 'gate'), linear(f, 'mlp.up_proj', 'up')
    width = g.tensors[gate].shape[1]
    act = tensor('activation', width)
    limit = c.get('swiglu_limit')
    clamp = '' if limit is None else f'a=fminf(a,{float(limit)}f);b=fmaxf(-{float(limit)}f,fminf(b,{float(limit)}f));'
    emit('swiglu', [gate, up], act,
         f'for(int j=0;j<n*{width};++j){{float a=gate[j],b=up[j];{clamp}activation[j]=a/(1+expf(-a))*b;}}')
    hidden = norm(add(residual, linear(act, 'mlp.down_proj', 'ffn_out'), 'final_residual'), 'shared_head.norm', h, 'hidden')
    logits = linear(hidden, 'shared_head.head', 'logits')
    g.tensor('tokens', (capacity,), dtype='int32', bytes_per_elem=4)
    emit('argmax', [logits], 'tokens',
         f'for(int t=0;t<n;++t){{int best=0;for(int j=1;j<{c["vocab_size"]};++j)if(logits[t*{c["vocab_size"]}+j]>logits[t*{c["vocab_size"]}+best])best=j;tokens[t]=best;}}', integer=('tokens',))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('#include <math.h>\n#include "etx/abi.h"\n' + '\n'.join(bodies))
    # Metadata is solely for binding checkpoint names to declared graph tensors.
    g.mtp_weights = weight_names
    return g


class ETXMTPGraphCPU:
    """Execute the graph's generated C++ tile bodies, with poisoned scratch.

    Planning is exercised on both targets; CPU execution uses topological grid
    order and does not validate GPU synchronization or device compilation.
    """
    def __init__(self, config, tensors, capacity=36):
        import ctypes
        import tempfile
        import subprocess
        from pathlib import Path
        import numpy as np
        from etx.passes import compile_graph, verify_plan

        require_prerequisites(config, tensors)
        self.c = config.get('text_config', config)
        self.capacity = capacity
        self._work = tempfile.TemporaryDirectory(prefix='etx-mtp-cpu-')
        work = Path(self._work.name)
        self.graph = build_graph(config, capacity, work / 'tiles.h')
        self.plans = {arch: compile_graph(self.graph, arch, {}, {}) for arch in ('gfx942', 'sm_90')}
        assert all(not verify_plan(p) for p in self.plans.values())
        self.buffers = {}
        for name, t in self.graph.tensors.items():
            self.buffers[name] = np.full(t.shape, -1 if t.dtype == 'int32' else np.nan,
                                       dtype=np.int32 if t.dtype == 'int32' else np.float32)
        # Dequantization is storage conversion only, shared with the retained
        # Torch path; no Torch arithmetic executes the graph.
        weights = MTPGraphCPU(config, tensors).w
        for name, target in self.graph.mtp_weights.items():
            self.buffers[target][...] = weights[name].numpy()
        root = Path(__file__).resolve().parents[2]
        wrappers = []
        for i, grid in enumerate(self.graph.grids):
            wrappers.append(f'extern "C" void run_{i}(void** args){{etx_ctx c={{}};c.args=args;{grid.body.symbol}(&c);}}')
        src = work / 'cpu.cpp'
        src.write_text('#define MTP_CPU 1\n#define __device__\n#include "tiles.h"\n' + '\n'.join(wrappers))
        lib = work / 'cpu.so'
        subprocess.run(['c++', '-O3', '-ffp-contract=off', '-std=c++17', '-shared', '-fPIC',
                        '-I', str(root / 'etx/runtime/include'), str(src), '-o', str(lib)],
                       check=True, timeout=120, capture_output=True)
        self.lib = ctypes.CDLL(str(lib))
        self.calls = []
        for i, grid in enumerate(self.graph.grids):
            fn = getattr(self.lib, f'run_{i}')
            fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            args = (ctypes.c_void_p * len(grid.args))(*(self.buffers[n].ctypes.data for n in grid.args))
            self.calls.append((fn, args))
        self.index_selections = 0
        self._shared_length = None

    def forward(self, ids, previous, *, reuse_index=False):
        import numpy as np
        ids, previous = np.asarray(ids), np.asarray(previous, dtype=np.float32)
        n = len(ids) if ids.ndim == 1 else 0
        if (not n or ids.dtype.kind not in 'iu' or previous.shape != (n, self.c['hidden_size'])
                or np.any(ids < 0) or np.any(ids >= self.c['vocab_size'])):
            raise ValueError('expected nonempty integer token IDs and [time, hidden] states')
        if n > self.capacity:
            raise ValueError('prefix exceeds graph capacity / all-visible index_topk')
        if reuse_index and self._shared_length != n-1:
            raise ValueError('shared index requires the immediately preceding prefix')
        for name, t in self.graph.tensors.items():
            if t.role == 'activation':
                self.buffers[name].fill(-1 if t.dtype == 'int32' else np.nan)
        self.buffers['control'][:] = n, int(reuse_index)
        self.buffers['ids'][:n] = ids
        self.buffers['previous'][:n] = previous
        for fn, args in self.calls:
            fn(args)
        self.index_selections += int(not reuse_index)
        self._shared_length = n
        return self.buffers['logits'][:n].copy(), self.buffers['hidden'][:n].copy()

    def chain(self, ids, previous, depth):
        import numpy as np
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError('depth must be positive')
        ids, previous = np.asarray(ids), np.asarray(previous, dtype=np.float32)
        if ids.ndim != 1 or not len(ids) or len(ids)+depth-1 > self.capacity:
            raise ValueError('nonempty prefix plus drafts must fit graph capacity')
        tokens, logits, recycled = [], [], []
        for t in range(len(ids)):
            branch, states = ids[:t+1].copy(), previous[:t+1].copy()
            bt, bl = [], []
            for d in range(depth):
                out, hidden = self.forward(branch, states, reuse_index=bool(d and self.c.get('index_share_for_mtp_iteration', False)))
                token = int(self.buffers['tokens'][len(branch)-1])
                bt.append(token)
                bl.append(out[-1])
                branch = np.append(branch, token)
                states = np.concatenate((states, hidden[-1:]))
            tokens.append(bt)
            logits.append(bl)
            recycled.append(hidden[-1])
        return np.asarray(tokens).T, np.swapaxes(np.asarray(logits), 0, 1), np.stack(recycled)

def main():
    config = module_gate.module_config('module8')
    report = prerequisites(config, module_gate.tensors(config))
    print(json.dumps(report, indent=2))
    return 2 if report['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
