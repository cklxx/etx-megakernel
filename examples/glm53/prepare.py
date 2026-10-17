"""Offline pinned GLM architecture, ownership and implementation/asset status.

The actual numerical model path is examples.glm53.model; this command never
executes a GPU or downloads a checkpoint. --require-model validates --stage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/evidence/qwen-batch-glm53'
MODEL = 'zai-org/GLM-5.3-Flash'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(evidence=EVIDENCE):
    for source in json.loads((evidence / 'sources.json').read_text()):
        if sha(evidence / source['file']) != source['sha256']:
            raise ValueError(f'changed source evidence: {source["file"]}')
    cfg = json.loads((evidence / 'glm-config.json').read_text())
    api = json.loads((evidence / 'glm-model-api.json').read_text())
    if api['id'] != MODEL or len(api['sha']) != 40:
        raise ValueError('model identity is not pinned')
    t = cfg['text_config']
    if cfg['architectures'] != ['Glm5NextForConditionalGeneration'] or t['model_type'] != 'glm5_next_text':
        raise ValueError('unrecognized GLM architecture')
    if len(t['layer_types']) != t['num_hidden_layers'] or len(t['mlp_layer_types']) != t['num_hidden_layers']:
        raise ValueError('incomplete per-layer architecture')
    if set(t['layer_types']) != {'linear_attention', 'deepseek_sparse_attention'} or not t['mhc']:
        raise ValueError('unexpected attention or residual architecture')
    if cfg['quantization_config']['quant_method'] != 'fp8':
        raise ValueError('expected the native FP8 checkpoint')
    return cfg, api


def partition(t):
    """Own whole layers; balance routed experts, without all-to-all or tensor sharding."""
    sparse = [i for i, k in enumerate(t['mlp_layer_types']) if k == 'sparse']
    if not sparse or len(sparse) % 2:
        raise ValueError('PP2 requires an even number of sparse layers in this preparation')
    split = sparse[len(sparse) // 2]
    ranks = []
    for rank, layers in enumerate((list(range(split)), list(range(split, t['num_hidden_layers'])))):
        ranks.append(dict(rank=rank, layers=layers,
                          routed_experts=list(range(t['n_routed_experts'])),
                          moe_layers=sum(t['mlp_layer_types'][i] == 'sparse' for i in layers),
                          kda_layers=sum(t['layer_types'][i] == 'linear_attention' for i in layers),
                          mla_layers=sum(t['layer_types'][i] == 'deepseek_sparse_attention' for i in layers)))
    return ranks


def budget(cfg, api, batch, context, gpu_gib):
    if batch not in (1, 2, 4, 8) or not 1 <= context <= cfg['text_config']['max_position_embeddings'] or not math.isfinite(gpu_gib) or gpu_gib <= 0:
        raise ValueError('invalid batch/context/GPU capacity')
    t = cfg['text_config']
    types = api['safetensors']['parameters']
    sizes = dict(BF16=2, F8_E4M3=1, F32=4)
    if set(types) - sizes.keys():
        raise ValueError('unknown checkpoint dtype')
    payload = sum(n * sizes[k] for k, n in types.items())
    params = sum(types.values())
    # Arithmetic bounds from the saved dtype inventory, not an allocation trace.
    # Native experts assumed FP8; conservatively replicate all other payload.
    expert_params = sum(k == 'sparse' for k in t['mlp_layer_types']) * t['n_routed_experts'] * 3 * t['hidden_size'] * t['moe_intermediate_size']
    if expert_params > types['F8_E4M3']:
        raise ValueError('expert-byte assumption exceeds FP8 inventory')
    native_per_rank = expert_params // 2 + payload - expert_params
    bf16_per_rank_floor = params  # total BF16 bytes / two ranks
    state_bytes = batch * t['hc_mult'] * t['hidden_size'] * 2
    from examples.glm53.stage import dimensions
    dims=dimensions(t,min(context+32,t['max_position_embeddings']))
    ranks=partition(t)
    cache=[batch*sum(dims['STATE_KDA'] if t['layer_types'][i]=='linear_attention' else dims['STATE_MLA'] for i in rank['layers'])*4 for rank in ranks]
    scratch=[batch*len(rank['layers'])*dims['SCRATCH']*4 for rank in ranks]
    return dict(kind='ESTIMATE, not measured allocation', checkpoint_payload_bytes=payload,
                native_fp8_per_rank_ideal_floor=payload / 2,
                native_fp8_per_rank_conservative_payload=native_per_rank,
                bf16_per_rank_ideal_floor=bf16_per_rank_floor,
                gpu_capacity_bytes=int(gpu_gib * 2**30),
                bf16_fits_payload_floor=bf16_per_rank_floor <= gpu_gib * 2**30,
                native_payload_reserve_bytes=int(gpu_gib * 2**30) - native_per_rank,
                boundary_bytes_per_decode_step=state_bytes, token_return_bytes=4 * batch,
                boundary_bytes_for_32_steps=32 * (state_bytes + 4 * batch),
                context=context, batch=batch,native_cache_bytes_per_rank=cache,native_scratch_bytes_per_rank=scratch,
                unknown=['actual checkpoint tensor/shard allocation and FP8 conversion buffers',
                         'mHC workspace, allocator/runtime/event buffers, vision weights if retained'],
                memory_fit='UNESTABLISHED; payload reserve is not a runtime fit certificate')


def prepare(batch=1, context=1024, gpu_gib=192, evidence=EVIDENCE):
    cfg, api = metadata(evidence)
    t = cfg['text_config']
    return dict(schema=1, status='LOCAL-IMPLEMENTED', model=MODEL, revision=api['sha'],
                config_sha256=sha(evidence / 'glm-config.json'), scope='text-only PP2 preparation, MTP disabled',
                architecture=dict(wrapper=cfg['architectures'][0], hidden=t['hidden_size'],
                                  layers=t['num_hidden_layers'], kda=t['layer_types'].count('linear_attention'),
                                  sparse_mla=t['layer_types'].count('deepseek_sparse_attention'),
                                  experts=t['n_routed_experts'], topk=t['num_experts_per_tok'], hc_mult=t['hc_mult']),
                ranks=partition(t),
                ownership=dict(embedding=0, vision='excluded from first text-only experiment',
                               final_norm_and_head=1, kv_and_recurrence='owner of the complete layer',
                               mtp='disabled; not needed for 32-token greedy oracle'),
                communication=dict(strategy='PP2, complete-layer ownership, no expert all-to-all',
                                   activation='rank 0 sends [B,hc_mult,H] BF16 residual streams to rank 1',
                                   feedback='rank 1 returns B int32 greedy token IDs to rank 0',
                                   ordering='SYSTEM release/acquire events; both ranks reset before launch',
                                   transport='peer access + fine-grained buffers and shared integer counters'),
                budget=budget(cfg, api, batch, context, gpu_gib),
                implementation=dict(graph='examples/glm53/model.py',tiles='examples/glm53/tiles.hip',
                    numerics='examples/glm53/native.h',stager='examples/glm53/stage.py',
                    runner='examples/glm53/run.py',reference='examples/glm53/reference.py',
                    backend='complete-layer native correctness tiles; no performance claim'),
                hardware_gates=['pre-staged exact checkpoint and independent HF reference',
                                'real two-GPU allocation, peer visibility and full-checkpoint tokens/logits'],
                next_gate='validate staged tensors; build and execute examples/sessions/glm53_model.py on an authorized existing host',
                performance='unmeasured')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--gpu-gib', type=float, default=192)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--require-model', action='store_true')
    ap.add_argument('--stage',type=Path)
    a = ap.parse_args()
    result = prepare(a.batch, a.context, a.gpu_gib)
    code=0
    if a.require_model:
        if a.stage is None:
            result.update(status='ASSETS-REQUIRED',error='--stage must name an already staged exact model');code=2
        else:
            from examples.glm53.stage import validate
            try:
                validate(a.stage);result['status']='STAGED-MODEL-VALIDATED'
            except (ValueError,FileNotFoundError,KeyError) as exc:
                result.update(status='ASSETS-REQUIRED',error=str(exc));code=2
    text = json.dumps(result, indent=2) + '\n'
    if a.out:
        a.out.write_text(text)
    print(text, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
