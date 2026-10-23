"""Phase 1 cost model (estimates only): per-phase bytes from the layouts, per-phase time from one trace.

Every output number is an ESTIMATE. Inputs:
- per-phase time: the PP2 le2 step at position 4 (docs/glm/FFN-PARALLEL-2026-10-22.md, "What the
  trace says"; results/o55-events-20261022/le2run.log), summed over all 45 layers;
- per-phase bytes: the weights-free production spec (model_build.production_compile_spec) and, per arm,
  the derived parallel layout (parallel_layout.derive), i.e. exactly the tensors each rank loads;
- cross-device hops: the planned graph (model.build_from) of each arm.

    python -m examples.glm53.cost_model            # markdown tables for docs/glm/PHASE1-COST-MODEL-*.md
"""
from __future__ import annotations

import itertools
import math

# PP2 trace, position 4, ms summed over 45 layers [measured, one step]. 'heads' is the attention-heads
# grid (KDA, and MLA where present) that the trace summary labels KDA.
TRACE = dict(up=7.4225, p1=5.3040, down=4.5374, o=2.7099, heads=1.4838, p2=1.4609, router=0.9952)
HOST_WALL = 28.353256
TRACE_RESIDUAL = HOST_WALL - sum(TRACE.values())   # hc/hcf/head grids, scheduling, host launch/retire

# Per-grid span = one tile body + scheduling. Measured on l10 (le2run.log:578-579): up body 141 of 175 us
# span, down body 82 of 114 us. The 32-34 us non-body part is applied to every phase grid (assumption).
GRID_OVERHEAD_US = 33.0
# Sharded tile bodies: 'k' = the tile keeps its rows but its reduction (K) shrinks with the shard
# (down: local experts / 1/8 intermediate; o under head TP: 1/8 of heads x dim). 'rows' = the tile
# keeps its rows and its K, only the tile count per rank falls: one tile deep, body unchanged.
SHARD_KIND = {'up': 'rows', 'down': 'k', 'p1': 'rows', 'p2': 'rows', 'heads': 'rows', 'o': 'k', 'router': 'rows'}
CONTENTION = (0.85, 1.0)   # fewer concurrent tiles: dense 342-tile up grids ran 10-15 % faster than 512-tile ones
HOP_US = (10.0, 20.0)      # one SYSTEM publication + wait across GPUs: proxies 5.46 us (intra-GPU) and 10.6 us
                           # (remote flag, 2x MI300X); never measured on the 8x MI325X fabric
PEER_GBPS = (350.0, 50.0)  # effective peer read rate, all 7 links in parallel vs one link: unknown, bounded
PHASE_SLOTS = {  # slot keys each phase reads (examples/glm53/phases.h)
    'p1': ['input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
           'self_attn.forget_gate.f_a_proj.weight', 'self_attn.b_proj.weight', 'self_attn.g_a_proj.weight',
           'self_attn.q_a_proj.weight', 'self_attn.kv_a_proj_with_mqa.weight', 'self_attn.indexer.wk.weight',
           'self_attn.indexer.weights_proj.weight', 'self_attn.indexer.index_kpool_compress_gate'],
    'p2': ['self_attn.q_a_layernorm.weight', 'self_attn.kv_a_layernorm.weight', 'self_attn.q_b_proj.weight',
           'self_attn.kv_b_proj.weight', 'self_attn.indexer.wq_b.weight',
           'self_attn.indexer.k_norm.weight', 'self_attn.indexer.k_norm.bias'],
    'heads': ['self_attn.conv1d.weight', 'self_attn.forget_gate.f_b_proj.weight', 'self_attn.forget_gate.dt_bias',
              'self_attn.forget_gate.A_log', 'self_attn.g_b_proj.weight', 'self_attn.o_norm.weight',
              'self_attn.indexer.index_kpool_compress_ape'],
    'o': ['self_attn.o_proj.weight'],
    'router': ['post_attention_layernorm.weight', 'mlp.gate.weight', 'mlp.gate.e_score_correction_bias'],
    'up': ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.shared_experts.gate_proj.weight',
           'mlp.shared_experts.up_proj.weight'],
    'down': ['mlp.down_proj.weight', 'mlp.shared_experts.down_proj.weight'],
}
ORDER = ['up', 'p1', 'down', 'o', 'heads', 'p2', 'router']
ARMS = {'EP8': dict(devices=8, parallel=2), 'headTP+EP8': dict(devices=8, parallel=2, attention_tp=1),
        'TP8': dict(devices=8, parallel=1)}
ITEM = {'torch.float8_e4m3fn': 1, 'torch.bfloat16': 2, 'torch.float32': 4}


def tensor_bytes(record):
    shape = record['shape']
    n = math.prod(shape) * ITEM[record['dtype']]
    if record['dtype'] == 'torch.float8_e4m3fn' and len(shape) == 2:
        n += math.ceil(shape[0] / 128) * math.ceil(shape[1] / 128) * 4
    return n


def phase_of(key):
    if key.startswith('mlp.experts.'):
        return 'routed_' + ('down' if key.endswith('down_proj.weight') else 'up')
    for phase, keys in PHASE_SLOTS.items():
        if key in keys:
            return phase
    return 'hc'


def ep_busiest(world=8, experts=288, topk=8, owner_extra=1):
    """E[max over ranks of selected experts held] for uniform top-k without replacement, expert e on
    rank e % world, plus the shared expert on rank 0. Exact by enumerating rank-count compositions."""
    per = experts // world
    total = math.comb(experts, topk)
    expect = 0.0
    for counts in itertools.product(range(topk + 1), repeat=world):
        if sum(counts) != topk:
            continue
        ways = math.prod(math.comb(per, c) for c in counts)
        load = max(c + (owner_extra if r == 0 else 0) for r, c in enumerate(counts))
        expect += load * ways / total
    return expect


def bytes_table(spec, layout):
    """{phase: [bytes on rank r]} per token at B1, routed experts as expected selected bytes."""
    t = spec['config']
    world = layout['devices'] if layout else 1
    out = {p: [0.0] * world for p in ORDER + ['hc', 'head', 'routed_up', 'routed_down']}
    experts, topk = t['n_routed_experts'], t['num_experts_per_tok']
    if layout is None:
        for name, w in spec['weights'].items():
            for key, rec in w['tensors'].items():
                if name in ('embedding', 'norm', 'head'):
                    # one embedding row per token; final norm and LM head read whole (head grid)
                    n = rec['shape'][1] * ITEM[rec['dtype']] if name == 'embedding' else tensor_bytes(rec)
                    out['head'][0] += n
                    continue
                p = phase_of(key)
                out[p][0] += tensor_bytes(rec) * (topk / experts if p.startswith('routed') else 1)
        return out
    for name, w in layout['weights'].items():
        r = w['rank']
        for key, rec in w['tensors'].items():
            p = phase_of(key)
            out[p][r] += tensor_bytes(rec) * (topk / experts if p.startswith('routed') else 1)
    return out


def hops(graph):
    """Cross-device hops on the longest dependency chain of one step, and peer-read bytes per request."""
    producer = {}
    depth = {}
    for g in graph.grids:
        for ev in g.out_edges:
            producer[ev] = g
    peer = 0
    writer = {}
    for g in graph.grids:
        d = 0
        for ev in g.in_edges:
            p = producer.get(ev)
            if p is not None:
                d = max(d, depth[p.name] + (p.device != g.device))
        depth[g.name] = d
        for x in g.reads:
            w = writer.get(x)
            t = graph.tensors[x]
            if w is not None and w != g.device and t.role != 'weight':
                peer += t.nbytes({'B': 1})
        for x in g.writes:
            writer[x] = g.device
    return max(depth.values()), peer


def main():
    from examples.glm53 import model
    from examples.glm53.model_build import production_compile_spec
    from examples.glm53.options import Options
    from examples.glm53.parallel_layout import derive
    spec = production_compile_spec()
    t = spec['config']
    base = bytes_table(spec, None)
    base_phase = {p: base[p][0] for p in base}
    base_phase['up'] += base_phase.pop('routed_up')
    base_phase['down'] += base_phase.pop('routed_down')
    pp2_hops, pp2_peer = hops(model.build_from(spec, 1, Options()))
    n, mla = t['num_hidden_layers'], sum(k != 'linear_attention' for k in t['layer_types'])
    grids = dict(up=n, down=n, p1=n, o=n, heads=n, p2=mla, router=n)   # PP2 grids per token per phase
    busiest = ep_busiest(experts=t['n_routed_experts'], topk=t['num_experts_per_tok'])
    print(f'EP busiest rank, expected expert-equivalents (routed + shared on rank 0): {busiest:.3f} of '
          f'{t["num_experts_per_tok"] + 1}')
    print(f'PP2 graph: cross-device hops on the critical chain {pp2_hops}, peer bytes/request {pp2_peer}')
    print(f'Trace residual (hc/hcf/head grids, scheduling, host) {TRACE_RESIDUAL:.3f} ms')
    print('\n| phase | PP2 bytes/token (GB) | PP2 trace ms |\n|---|---:|---:|')
    for p in ORDER + ['hc', 'head']:
        print(f'| {p} | {base_phase[p]/1e9:.3f} | {TRACE.get(p, float("nan")):.4f} |')
    print(f'| total | {sum(base_phase.values())/1e9:.3f} | {sum(TRACE.values()):.4f} |')
    for arm, fields in ARMS.items():
        options = Options(**fields)
        layout = derive(spec, 'tp' if options.parallel == 1 else 'ep', 8, attention_tp=bool(options.attention_tp))
        table = bytes_table(spec, layout)
        ratios = {}
        g = model.build_from(spec, 1, options, layout)
        arm_hops, arm_peer = hops(g)
        print(f'\n### {arm} {fields}\ncross-device hops on the critical chain {arm_hops}, '
              f'peer bytes/request {arm_peer} ({arm_peer/1e6:.2f} MB)')
        print('| phase | rank 0 GB (expected) | each of ranks 1-7 GB (expected) | critical rank GB | critical / PP2 bytes |\n'
              '|---|---:|---:|---:|---:|')
        for p in ORDER:
            r = list(table[p])
            crit = max(r)
            if p in ('up', 'down'):
                routed = table['routed_' + p]
                r = [r[i] + routed[i] for i in range(8)]
                if options.parallel == 2:
                    # EP: per MoE layer the straggler holds `busiest` expert-equivalents (shared counted on
                    # rank 0, same size as one routed expert); dense FFN layers stay on rank 0 in addition.
                    # per_expert: one expert's bytes summed over all MoE layers (= the shared expert's).
                    per_expert = sum(routed) / t['num_experts_per_tok']
                    dense = table[p][0] - per_expert
                    crit = per_expert * busiest + dense
                else:
                    crit = max(r)
            print(f'| {p} | {r[0]/1e9:.3f} | {max(r[1:])/1e9:.3f} | {crit/1e9:.3f} | {crit/base_phase[p]:.3f} |')
            ratios[p] = crit / base_phase[p]
        print_times(arm, ratios, grids, arm_hops - pp2_hops, arm_peer)


def print_times(arm, ratios, grids, extra_hops, peer):
    rows, lo_b, hi_b, lo_t, hi_t = [], 0, 0, 0, 0
    for p in ORDER:
        T, r = TRACE[p], ratios[p]
        over = min(grids[p] * GRID_OVERHEAD_US / 1000, T)
        body = T - over
        by = body * r + over                       # bytes-proportional: body scales with the critical rank's bytes
        if r >= 0.999:
            by, tile = T, (T, T)
        elif SHARD_KIND[p] == 'k':
            tile = (body * r + over, body * r + over)
        else:                                      # one tile deep: body unchanged, contention relief only
            tile = (body * CONTENTION[0] + over, T)
        rows.append((p, T, r, by, tile))
        lo_b += by; hi_b += by; lo_t += tile[0]; hi_t += tile[1]
    comm = (extra_hops * HOP_US[0] / 1000 + peer / PEER_GBPS[0] / 1e6,
            extra_hops * HOP_US[1] / 1000 + peer / PEER_GBPS[1] / 1e6)
    print(f'\n| phase | PP2 trace ms | critical-rank bytes / PP2 | bytes-proportional ms [E] | tile-depth ms [E] |\n|---|---:|---:|---:|---:|')
    for p, T, r, by, tile in rows:
        tt = f'{tile[0]:.2f}' if abs(tile[0]-tile[1]) < 5e-3 else f'{tile[0]:.2f}-{tile[1]:.2f}'
        print(f'| {p} | {T:.2f} | {r:.3f} | {by:.2f} | {tt} |')
    print(f'| phases | {sum(TRACE.values()):.2f} | | {lo_b:.2f} | {lo_t:.2f}-{hi_t:.2f} |')
    print(f'| residual (hc, hcf, embed, head, host) | {TRACE_RESIDUAL:.2f} | 1.000 | {TRACE_RESIDUAL:.2f} | {TRACE_RESIDUAL:.2f} |')
    print(f'| communication: {extra_hops} added critical hops, {peer/1e6:.1f} MB peer reads | 0.00 | | '
          f'{comm[0]:.2f}-{comm[1]:.2f} | {comm[0]:.2f}-{comm[1]:.2f} |')
    print(f'| **total, position 4** | **{HOST_WALL:.2f}** | | **{lo_b+TRACE_RESIDUAL+comm[0]:.1f}-{hi_b+TRACE_RESIDUAL+comm[1]:.1f}** | '
          f'**{lo_t+TRACE_RESIDUAL+comm[0]:.1f}-{hi_t+TRACE_RESIDUAL+comm[1]:.1f}** |')


if __name__ == '__main__':
    main()
