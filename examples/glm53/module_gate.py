"""GLM-5.3-Flash module gate: a synthetic native-FP8 checkpoint at small dimensions.

No release checkpoint is fetched. This writes a checkpoint in the release's own
layout (native tensor names, FP8 E4M3 matrices with 128x128 FP32
`weight_scale_inv` blocks, BF16/F32 for the rest, by the release's dtype rule)
with deterministic random values. Everything after it is production code, run
with gate='module': examples.glm53.stage.stage/validate, the HF reference
(examples.glm53.reference.run, an independently instantiated Transformers
GLM5Next that dequantizes the same files itself), the model build, the two-GPU
launcher and examples.glm53.run.qualify with the same TOKENS-EQUAL rules.

  python -m examples.glm53.module_gate evidence [--write]    pinned config/identity (docs/evidence/glm53-module-gate)
  python -m examples.glm53.module_gate prepare --out DIR      checkpoint, stage, prompts, HF reference, CPU prediction

`prepare` needs torch + transformers (laptop); the GPU box needs only DIR/stage and
DIR/reference.{json,logits}. The CPU prediction runs the same native body as the
HIP tiles (examples/glm53/native.h) through the same decode loop as the launcher.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import zlib

from examples.glm53 import stage
from examples.glm53.prepare import EVIDENCE, MODEL, MODULE_EVIDENCE, MODULE_MODEL, metadata, partition, sha

SEED = 20261017
BATCH = 2
CONTEXT = 128          # > INDEX_TOPK: sparse kpool selection drops pools, unlike a 1024-token release run
# Each rank owns one KDA and one sparse-MLA layer, one dense MLP and one MoE layer: every
# (attention, MLP) pair here also occurs in the release schedule (layers 0, 3 and 4).
LAYER_TYPES = ['linear_attention', 'deepseek_sparse_attention', 'deepseek_sparse_attention', 'linear_attention']
MLP_TYPES = ['dense', 'sparse', 'sparse', 'dense']
TEXT = dict(hidden_size=512, num_hidden_layers=4, intermediate_size=1024, moe_intermediate_size=256,
            n_routed_experts=8, num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=4,
            q_lora_rank=384, kv_lora_rank=256, index_n_heads=4, index_topk=64, vocab_size=4096,
            first_k_dense_replace=1, pad_token_id=4095, eos_token_id=[4095])
LINEAR = dict(num_heads=4)
# Random MoE weights put routing on near-ties (2nd vs 3rd expert within 1e-3 in ~1% of
# decisions), which BF16-level differences flip; a random head puts greedy on near-ties
# too. The head therefore carries a permuted copy of the embedding (next token ~ pi(current))
# so every greedy decision has a margin far above those flips. Layer weights stay random.
HEAD_ALPHA = 2.0
# The module gate's added rule, registered before any GPU run (2026-10-17) from the CPU
# runs of the same native body against this reference: correct body median 0.45 (max 4.98,
# routing flips); injected bugs (shared expert dropped, KDA beta unsquashed, routed scale
# dropped) medians 5.02 / 9.13 / 5.59 with all tokens still equal. The median of the 32
# per-step max-abs logit errors must not exceed this, for every repetition.
MEDIAN_LOGIT_BOUND = 1.5


def module_rule(verdict):
    """(ok, median) for one verdict: median per-step full-logit error within MEDIAN_LOGIT_BOUND.
    Standard library only (tools/two_gpu_verify.py runs on the box's bare python3)."""
    import statistics
    steps = verdict.get('logit_max_abs_per_step')
    if not isinstance(steps, list) or len(steps) != 32:
        return False, None
    median = statistics.median(steps)
    return median <= MEDIAN_LOGIT_BOUND, median
VISION = dict(depth=1, hidden_size=64, num_heads=2, intermediate_size=128, projection_intermediate_size=128)

# The release's storage dtypes (its modules_to_not_convert and F32 inventory). With these
# rules the 45 main layers + MTP reproduce the pinned F8_E4M3 parameter count exactly
# (tests/test_glm53_module_gate.py).
F32 = {'attn_hc.base', 'attn_hc.scale', 'ffn_hc.base', 'ffn_hc.scale', 'self_attn.forget_gate.A_log',
       'self_attn.forget_gate.dt_bias', 'mlp.gate.e_score_correction_bias'}
BF16_MATRICES = {'attn_hc.fn', 'ffn_hc.fn', 'self_attn.kv_b_proj.weight', 'self_attn.indexer.wq_b.weight',
                 'self_attn.indexer.wk.weight', 'self_attn.indexer.weights_proj.weight',
                 'self_attn.indexer.index_kpool_compress_gate', 'self_attn.indexer.index_kpool_compress_ape',
                 'mlp.gate.weight'}
NATIVE = [('attn_hc.fn', 'hc_attn_fn'), ('attn_hc.base', 'hc_attn_base'), ('attn_hc.scale', 'hc_attn_scale'),
          ('ffn_hc.fn', 'hc_ffn_fn'), ('ffn_hc.base', 'hc_ffn_base'), ('ffn_hc.scale', 'hc_ffn_scale'),
          ('self_attn.forget_gate.', 'self_attn.')]


def storage(layer_type, key, shape):
    """'F8_E4M3', 'BF16' or 'F32' for one layer tensor, as the release stores it."""
    if key in F32:
        return 'F32'
    if len(shape) != 2 or key in BF16_MATRICES or (layer_type == 'linear_attention' and key.startswith('self_attn.')):
        return 'BF16'
    return 'F8_E4M3'


def module_config():
    """The release config.json with only sizes changed; every numerical constant is kept."""
    cfg, _ = metadata()
    m = copy.deepcopy(cfg)
    t = m['text_config']
    t.update(TEXT, layer_types=list(LAYER_TYPES), mlp_layer_types=list(MLP_TYPES),
             indexer_types=['full'] * len(LAYER_TYPES))
    t['linear_attn_config'].update(LINEAR, kda_layers=[i for i, k in enumerate(LAYER_TYPES) if k == 'linear_attention'],
                                   full_attn_layers=[i for i, k in enumerate(LAYER_TYPES) if k != 'linear_attention'])
    m['vision_config'].update(VISION, out_hidden_size=t['hidden_size'])
    m['quantization_config']['modules_to_not_convert'] = skip_list(cfg, t)
    return m


def skip_list(cfg, t):
    """The release skip list at module size: per-layer entries copied from a release layer of
    the same (attention, MLP) kind; MTP and vision blocks beyond the module's depth dropped."""
    import re
    rt = cfg['text_config']
    kinds = {}
    for p in range(rt['num_hidden_layers']):
        kinds.setdefault((rt['layer_types'][p], rt['mlp_layer_types'][p]), p)
    per_layer, generic = {}, []
    for name in cfg['quantization_config']['modules_to_not_convert']:
        hit = re.match(r'model\.layers\.(\d+)\.(.+)$', name)
        if hit:
            per_layer.setdefault(int(hit[1]), []).append(hit[2])
            continue
        block = re.match(r'visual\.blocks\.(\d+)\.', name)
        if block and int(block[1]) >= VISION['depth']:
            continue
        generic.append(name)
    out = list(generic)
    for i in range(t['num_hidden_layers']):
        p = kinds[(t['layer_types'][i], t['mlp_layer_types'][i])]
        out += [f'model.layers.{i}.{suffix}' for suffix in per_layer[p]]
    return out


def native_tensors(t, i):
    """[(native name, ETX slot key, slice of the slot, shape)] for layer i (conv1d is three tensors)."""
    out = []
    for key, shape in stage.shapes(t, i).items():
        name = key
        for old, new in NATIVE:
            name = name.replace(old, new)
        prefix = f'model.language_model.layers.{i}.'
        if key == 'self_attn.conv1d.weight':
            n = shape[0] // 3
            for part, which in enumerate('qkv'):
                out.append((prefix + f'self_attn.{which}_conv1d.weight', key, part, (n,) + tuple(shape[1:])))
        else:
            out.append((prefix + name, key, None, tuple(shape)))
    return out


def outer_tensors(t):
    h, v = t['hidden_size'], t['vocab_size']
    return [('model.language_model.embed_tokens.weight', (v, h)), ('model.language_model.norm.weight', (h,)),
            ('lm_head.weight', (v, h))]


def _rng(name):
    import numpy as np
    return np.random.Generator(np.random.PCG64([SEED, zlib.crc32(name.encode())]))


def values(name, key, shape):
    """Deterministic float32 values for one tensor (numpy PCG64 per name; no torch RNG)."""
    import numpy as np
    g = _rng(name)
    n = math.prod(shape)
    leaf = name.rsplit('.', 1)[-1]
    if leaf in ('A_log',):
        x = np.log(g.uniform(1, 16, n))
    elif leaf == 'dt_bias':
        dt = np.exp(g.uniform(math.log(1e-3), math.log(1e-1), n))
        x = dt + np.log(-np.expm1(-dt))                  # inverse softplus, as the HF init
    elif leaf in ('hc_attn_scale', 'hc_ffn_scale') or (len(shape) == 1 and leaf == 'weight'):
        x = 1 + 0.1 * g.standard_normal(n)               # norm gains and hc scales
    elif leaf in ('hc_attn_base', 'hc_ffn_base', 'bias', 'e_score_correction_bias', 'index_kpool_compress_ape'):
        x = 0.1 * g.standard_normal(n)
    elif 'conv1d' in name:
        x = 0.5 * g.standard_normal(n)
    elif name == 'model.language_model.embed_tokens.weight':
        x = g.standard_normal(n)
    else:                                                # every projection: unit fan-in gain
        x = g.standard_normal(n) / math.sqrt(math.prod(shape[1:]))
    x = x.astype(np.float32).reshape(shape)
    if name == 'lm_head.weight':
        embedding = values('model.language_model.embed_tokens.weight', None, shape)
        x += np.float32(HEAD_ALPHA / math.sqrt(shape[1])) * embedding[_rng('head-permutation').permutation(shape[0])]
    return x


def fp8_blocks(name, w):
    """E4M3FN weight and FP32 128x128 dequantization scales (amax/448, jittered per block)."""
    import numpy as np
    import torch
    rows, cols = w.shape
    br, bc = (rows + 127) // 128, (cols + 127) // 128
    jitter = _rng(name + '#scale').uniform(1.0, 1.25, (br, bc))
    scale = np.empty((br, bc), dtype=np.float32)
    q = np.empty_like(w)
    for r in range(br):
        for c in range(bc):
            block = w[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]
            s = np.float32(max(float(np.abs(block).max()), 1e-12) / 448 * jitter[r, c])
            scale[r, c] = s
            q[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = block / s
    return torch.from_numpy(q).to(torch.float8_e4m3fn), torch.from_numpy(scale)


def tensors(cfg):
    """{native name: torch tensor} for the whole synthetic text checkpoint, in a fixed order."""
    import torch
    t = cfg['text_config']
    out = {}
    for i in range(t['num_hidden_layers']):
        for name, key, part, shape in native_tensors(t, i):
            kind = storage(t['layer_types'][i], key, stage.shapes(t, i)[key])
            w = values(name, key, shape)
            if kind == 'F8_E4M3':
                q, s = fp8_blocks(name, w)
                out[name] = q
                out[name.removesuffix('.weight') + '.weight_scale_inv'] = s
            else:
                out[name] = torch.from_numpy(w).to(torch.float32 if kind == 'F32' else torch.bfloat16)
    for name, shape in outer_tensors(t):
        out[name] = torch.from_numpy(values(name, None, shape)).to(torch.bfloat16)
    return out


def weights_digest(ts):
    """Content hash of every tensor (name, dtype, shape, bytes), independent of file serialization."""
    import torch
    h = hashlib.sha256()
    for name in sorted(ts):
        v = ts[name].contiguous()
        h.update(f'{name}\0{v.dtype}\0{tuple(v.shape)}\0'.encode())
        h.update(v.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def evidence(write=False):
    """The pinned module-gate identity: config.json, model API record (id, revision,
    inventory), sources. --write regenerates them; tests require the committed copy to match."""
    cfg = module_config()
    config_text = json.dumps(cfg, indent=2) + '\n'
    ts = tensors(cfg)
    inventory = {}
    for v in ts.values():        # every tensor, FP8 block scales included
        k = {'torch.float8_e4m3fn': 'F8_E4M3', 'torch.bfloat16': 'BF16', 'torch.float32': 'F32'}[str(v.dtype)]
        inventory[k] = inventory.get(k, 0) + v.numel()
    digest = weights_digest(ts)
    config_sha = hashlib.sha256(config_text.encode()).hexdigest()
    _, release = metadata()
    api = dict(id=MODULE_MODEL, sha=hashlib.sha1(f'{config_sha}\0{digest}'.encode()).hexdigest(),
               kind='synthetic module-gate checkpoint; not a released model',
               generator='python -m examples.glm53.module_gate', seed=SEED, weights_sha256=digest,
               derived_from=dict(model=MODEL, revision=release['sha'], config_sha256=sha(EVIDENCE / 'glm-config.json')),
               safetensors=dict(parameters=inventory, total=sum(inventory.values())))
    api_text = json.dumps(api, indent=2) + '\n'
    files = {'glm-config.json': config_text, 'glm-model-api.json': api_text}
    sources = [dict(file=f, url='generated: python -m examples.glm53.module_gate evidence --write',
                    sha256=hashlib.sha256(text.encode()).hexdigest()) for f, text in files.items()]
    files['sources.json'] = json.dumps(sources, indent=2) + '\n'
    if write:
        MODULE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        for f, text in files.items():
            (MODULE_EVIDENCE / f).write_text(text)
    return files, ts


def checkpoint(out):
    """Write the synthetic snapshot (config.json, REVISION, safetensors + index) into a new directory."""
    from safetensors.torch import save_file
    cfg, api = metadata(MODULE_EVIDENCE)
    files, ts = evidence()
    for f, text in files.items():
        if (MODULE_EVIDENCE / f).read_text() != text:
            raise ValueError(f'generated {f} differs from the pinned module-gate evidence; '
                             'the generator changed (rerun `evidence --write` deliberately)')
    if weights_digest(ts) != api['weights_sha256']:
        raise ValueError('synthetic weights differ from the pinned module-gate revision')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text((MODULE_EVIDENCE / 'glm-config.json').read_text())
    (out / 'REVISION').write_text(api['sha'] + '\n')
    shard = 'model.safetensors'
    save_file(ts, str(out / shard), metadata=dict(format='pt'))
    index = dict(metadata=dict(total_size=sum(v.numel() * v.element_size() for v in ts.values())),
                 weight_map={name: shard for name in sorted(ts)})
    (out / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    return api


def prompts():
    g = _rng('prompts')
    vocab = TEXT['vocab_size']
    rows = [g.integers(0, vocab - 1, CONTEXT).tolist() for _ in range(BATCH)]
    if len({tuple(r) for r in rows}) != BATCH:
        raise ValueError('module-gate prompts must be distinct')
    return [[int(x) for x in r] for r in rows]


def predict(staged, ref_path, out, root=None):
    """The native body on CPU through the launcher's decode loop, judged like `qualify`.
    `root` swaps the source tree the body is compiled from (mutation tests)."""
    import tempfile
    import numpy as np
    from examples.glm53 import cpu_check, oracle, reference
    from examples.glm53.run import compare_logits
    ref = json.loads(ref_path.read_text())
    p = [r['prompt_ids'] for r in ref['requests']]
    model = cpu_check.CPUModel(staged, gate='module', work=Path(tempfile.mkdtemp(prefix='glm53-cpu-')), root=root)
    model.reset(len(p))
    context = len(p[0])
    returned = [0] * len(p)
    generated = [[] for _ in p]
    logits = []
    for pos in range(context + 31):
        got, _ = model([row[pos] if pos < context else returned[r] for r, row in enumerate(p)])
        returned = [int(x) for x in got.argmax(-1)]
        if pos >= context - 1:
            logits.append(got.astype('<f4'))
            for r, token in enumerate(returned):
                generated[r].append(token)
    candidate = dict(reference.identity('module'), backend='etx-pp2', batch=len(p),
                     requests=[dict(prompt_ids=row, generated_ids=g) for row, g in zip(p, generated)])
    verdict = oracle.compare(ref, candidate, 'module')
    expected = np.fromfile(ref_path.with_suffix('.logits'), dtype='<f4').reshape(32, len(p), -1)
    got = np.stack(logits)
    if verdict['status'] == 'TOKENS-EQUAL':
        verdict.update(compare_logits(expected, got))
        ok, median = module_rule(verdict)
        verdict.update(module_rule=dict(median_logit_max_abs=median, bound=MEDIAN_LOGIT_BOUND, ok=ok))
        if not ok:
            verdict['status'] = 'FAILED'
    verdict.update(margin(expected), checkpoint_execution='CPU native body (same C++ as the HIP tiles)',
                   gpu_correctness='unmeasured')
    out.write_text(json.dumps(verdict, indent=2) + '\n')
    return verdict


def margin(logits):
    """Smallest top-1 minus top-2 HF logit over every step and request: how close greedy came to a tie."""
    import numpy as np
    top = np.sort(logits, axis=-1)[..., -2:]
    gap = top[..., 1] - top[..., 0]
    return dict(hf_min_top2_margin=float(gap.min()), hf_median_top2_margin=float(np.median(gap)))


def prepare(out):
    """Everything the GPU session needs, made and checked offline. The CPU prediction is the
    go/no-go: if the native body does not pass the module rules on CPU, do not book a GPU."""
    from examples.glm53 import reference
    out.mkdir(parents=True, exist_ok=False)
    checkpoint(out / 'checkpoint')
    stage.stage(out / 'checkpoint', out / 'stage', gate='module')
    (out / 'prompts.json').write_text(json.dumps(prompts()) + '\n')
    reference.run(out / 'checkpoint', out / 'prompts.json', out / 'reference.json', None, None, gate='module')
    verdict = predict(out / 'stage', out / 'reference.json', out / 'cpu-prediction.json')
    if verdict['status'] != 'TOKENS-EQUAL':
        raise SystemExit(f'MODULE-GATE CPU PREDICTION FAILED: {json.dumps(verdict)}')
    print(f'MODULE-GATE CPU PREDICTION OK: 32/32 x {len(verdict["matched_each"])}, median logit error '
          f'{verdict["module_rule"]["median_logit_max_abs"]:.3f} <= {MEDIAN_LOGIT_BOUND}', flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='op', required=True)
    e = sub.add_parser('evidence')
    e.add_argument('--write', action='store_true')
    q = sub.add_parser('prepare')
    q.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.op == 'evidence':
        files, _ = evidence(a.write)
        print(files['glm-model-api.json'], end='')
    else:
        print(json.dumps(prepare(a.out.resolve()), indent=2))


if __name__ == '__main__':
    main()
