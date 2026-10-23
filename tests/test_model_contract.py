"""Contract acceptance, schema parity, inference and malformed-input coverage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from etx.compiler.frontend.contract import ContractError, load_contract, validate_contract
from examples.llm.model import load_dims

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / 'examples/llm/configs').glob('*.json'))
SCHEMA = json.loads((ROOT / 'etx/compiler/frontend/schema.json').read_text())


def contract(config=CONFIGS[0]):
    return dict(name='test-model', config=str(config),
                weights={'format': 'safetensors', 'map': [dict(
                    etx_name='qkv', hf_glob='model.layers.*.self_attn.*_proj.weight',
                    concat_axis=0, dtype='bfloat16')]},
                kernels={'gemv': 'dsl', 'attention': 'vllm@abc123:attention'},
                bindings={'ctx_max': 2048}, parallel={'devices': 1, 'mode': 'none'},
                correctness={'golden': 'golden.json'})


@pytest.mark.parametrize('config', CONFIGS, ids=lambda p: p.stem)
def test_bundled_configs(config, monkeypatch):
    monkeypatch.setenv('ETX_LLM_LAYERS', '1')
    data = contract(config)
    original = deepcopy(data)
    result = validate_contract(data)
    dims = load_dims(str(config))
    c = json.loads(config.read_text())
    assert result['config_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert result['arch']['tie_embeddings'] == dims.tie
    assert result['arch']['ffn'] == ('moe' if dims.moe else 'dense')
    assert result['arch']['norm'] == ('layernorm' if dims.ln_norm else 'rmsnorm')
    assert len(result['arch']['layer_pattern']) == c['num_hidden_layers']
    assert result['arch']['rope'] == ('longrope' if c.get('model_type') == 'phimoe' else 'neox')
    if dims.moe:
        assert result['arch']['experts'] == dims.E
        assert result['arch']['topk'] == dims.TOPK
        assert result['arch']['norm_topk'] == dims.norm_topk
    assert result['bindings']['batch_sizes'] == [1, 2, 4, 8]
    assert result['correctness']['tolerance'] == {'cos': 0.999, 'rel': 0.02}
    assert data == original
    # Redundant assertions agree, but conflicting assertions cannot override HF.
    data['arch'] = result['arch']
    assert validate_contract(data)['arch'] == result['arch']
    for key, value in result['arch'].items():
        if key == 'attention':
            continue
        bad = deepcopy(data)
        bad['arch'][key] = (not value if type(value) is bool else value + 1
                            if type(value) is int else [] if isinstance(value, list)
                            else {'neox': 'llama', 'longrope': 'none', 'rmsnorm': 'layernorm',
                                  'layernorm': 'rmsnorm', 'moe': 'dense', 'dense': 'moe'}[value])
        with pytest.raises(ContractError, match='conflicts'):
            validate_contract(bad)


def required_paths(spec, prefix=()):
    paths = [prefix + (key,) for key in spec.get('required', [])]
    for key, child in spec.get('properties', {}).items():
        paths.extend(required_paths(child, prefix + (key,)))
    if 'items' in spec:
        paths.extend(required_paths(spec['items'], prefix + (0,)))
    return paths


@pytest.mark.parametrize('path', required_paths(SCHEMA))
def test_every_schema_required_field(path):
    data = contract()
    data['correctness']['tolerance'] = {'cos': 0.999, 'rel': 0.02}
    node = data
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(ContractError, match='required'):
        validate_contract(data)


@pytest.mark.parametrize('section,key,value', [
    ('bindings', 'ctx_max', True), ('bindings', 'ctx_max', 0),
    ('bindings', 'batch_sizes', []), ('bindings', 'batch_sizes', [1, 1]),
    ('bindings', 'batch_sizes', [1, False]), ('parallel', 'devices', -1),
    ('parallel', 'mode', 'magic'), ('kernels', 'gemv', 'unversioned:export'),
    ('weights', 'format', 'pickle'), ('weights', 'map', []),
    ('correctness', 'tolerance', {'cos': float('nan'), 'rel': 0.02}),
    ('correctness', 'tolerance', {'cos': 1.01, 'rel': 0.02}),
    ('correctness', 'tolerance', {'cos': 0.999, 'rel': -0.1}),
])
def test_invalid_fields(section, key, value):
    data = contract()
    data[section][key] = value
    with pytest.raises(ContractError):
        validate_contract(data)


def test_paths_and_hash(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_bytes(CONFIGS[0].read_bytes())
    data = contract(Path('config.json'))
    data['config'] = str(data['config'])
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(data))
    monkeypatch.chdir('/')
    result = load_contract(path)
    assert result['config'] == str(config)
    assert result['correctness']['golden'] == str(tmp_path / 'golden.json')
    config.write_bytes(config.read_bytes() + b'\n')
    assert load_contract(path)['config_sha256'] != result['config_sha256']


def test_attention_override_and_unknown_fields():
    data = contract()
    data['arch'] = {'attention': 'mla'}
    assert validate_contract(data)['arch']['attention'] == 'mla'
    data['typo'] = 1
    with pytest.raises(ContractError, match='unknown field'):
        validate_contract(data)


def test_duplicate_weights_and_bad_config(tmp_path):
    data = contract()
    data['weights']['map'] *= 2
    with pytest.raises(ContractError, match='duplicate'):
        validate_contract(data)
    data = contract(tmp_path / 'missing.json')
    data['config'] = str(data['config'])
    with pytest.raises(ContractError, match='config'):
        validate_contract(data)
    (tmp_path / 'missing.json').write_text('[]')
    with pytest.raises(ContractError, match='config'):
        validate_contract(data)


def test_schema_keywords_supported():
    """Keep the documented schema within the plain-Python validator's vocabulary."""
    keywords = {'type', 'properties', 'required', 'additionalProperties', 'items',
                'minItems', 'uniqueItems', 'minLength', 'pattern', 'enum',
                'minimum', 'maximum', '$schema', 'description'}
    def walk(spec):
        assert set(spec) <= keywords
        for child in spec.get('properties', {}).values():
            walk(child)
        if isinstance(spec.get('additionalProperties'), dict):
            walk(spec['additionalProperties'])
        if 'items' in spec:
            walk(spec['items'])
    walk(SCHEMA)
    assert len(CONFIGS) == 10


def test_mixed_shared_config(tmp_path):
    c = json.loads((ROOT / 'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    c.update(first_k_dense_replace=3, n_shared_experts=1, kv_lora_rank=512,
             qk_rope_head_dim=0)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(c))
    result = validate_contract(contract(path.as_posix()))
    assert result['arch']['layer_pattern'] == ['dense'] * 3 + ['moe'] * 45
    assert result['arch']['shared'] == 1
    assert result['arch']['attention'] == 'mla'
    assert result['arch']['rope'] == 'none'
    # Nested GLM-style config and explicit sparse pattern use the same inference.
    c.update(model_type='glm5_next_text', mlp_layer_types=['dense'] * 3 + ['sparse'] * 45)
    path.write_text(json.dumps({'text_config': c}))
    assert validate_contract(contract(path.as_posix()))['arch'] == result['arch']
