"""Model-provider seam probe: planning a declaration leaves no model modules imported."""
import sys

from etx.api.types import PlanRequest


def model_isolation_probe(workload):
    """Plan `workload` twice; return the identities and any modules the planning left behind."""
    from etx.contracts.planning import plan
    before = set(sys.modules)
    identities = (plan(PlanRequest(workload)).identity, plan(PlanRequest(workload)).identity)
    leaked = sorted(name for name in set(sys.modules) - before
                    if name.startswith(('examples', 'etx_model_declaration')))
    return dict(identities=identities, leaked=leaked)


def synthetic_llm_checkpoint(directory, root):
    """A tiny random Qwen3-MoE checkpoint (2 layers, 8 heads, 4 experts) in the HF layout prep packs; not a model."""
    import json
    from pathlib import Path
    import torch
    from safetensors.torch import save_file
    src = Path(directory)
    src.mkdir()
    config = json.loads((Path(root) / 'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    config.update(hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=16, num_hidden_layers=2, vocab_size=128)
    (src / 'config.json').write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(7)
    rand = lambda *shape: torch.randn(*shape, generator=generator).to(torch.bfloat16)
    tensors = {'model.embed_tokens.weight': rand(128, 64), 'model.norm.weight': rand(64), 'lm_head.weight': rand(128, 64)}
    for layer in range(2):
        p = f'model.layers.{layer}.'
        tensors.update({p + 'input_layernorm.weight': rand(64), p + 'post_attention_layernorm.weight': rand(64),
                        p + 'self_attn.q_proj.weight': rand(128, 64), p + 'self_attn.k_proj.weight': rand(64, 64),
                        p + 'self_attn.v_proj.weight': rand(64, 64), p + 'self_attn.q_norm.weight': rand(16),
                        p + 'self_attn.k_norm.weight': rand(16), p + 'self_attn.o_proj.weight': rand(64, 128),
                        p + 'mlp.gate.weight': rand(4, 64)})
        for expert in range(4):
            tensors.update({f'{p}mlp.experts.{expert}.gate_proj.weight': rand(32, 64),
                            f'{p}mlp.experts.{expert}.up_proj.weight': rand(32, 64),
                            f'{p}mlp.experts.{expert}.down_proj.weight': rand(64, 32)})
    save_file(tensors, str(src / 'model.safetensors'))
    return src
