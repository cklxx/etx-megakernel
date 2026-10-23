"""CPU-only acceptance for WP-S4a; source contracts require the pinned local reference."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.glm53 import module_gate as gate, stage
from examples.spec.mtp_reference import MTPReference, dequantize

REF = Path('/data/refs/vllm-v0.30.0/glm5next/nvidia')


@pytest.fixture(scope='module')
def fixture():
    config = gate.module_config('module8')
    tensors = gate.tensors(config, mtp=True)
    return config, tensors, MTPReference(config, dequantize(tensors))


def test_main_blobs_unchanged(fixture, tmp_path):
    config, tensors, _ = fixture
    t = config['text_config']
    # Captured from the unmodified generator before adding MTP, using write_blob.
    expected = ['1da4811e51e585a895a6ba1314981ebe8004b60f682d0fef9edac9700527bfe0',
                'e42a48d14f2ffedfbe1ec86006a5d3cfc8c2130d225c836129076637674e45db',
                '5bcd28a43477208072e24087c78cdfb59e107d8a611cf75244f9c407b1bb9489',
                '4af1620310537c13c88c6e3763bddb0aa0682935824eb82d35ae41982b74abbb']
    main = {n: v for n, v in tensors.items() if not n.startswith('model.language_model.layers.4.')}
    assert gate.weights_digest(main) == '1564d56d4103fc3b22bdbde992fd151c03bd3087f048f3b4fdb24ba3104f0228'
    for i in range(t['num_hidden_layers']):
        entries = gate.native_tensors(t, i)
        weights = {}
        for name, key, part, _ in entries:
            if part is None:
                weights[key] = tensors[name], tensors.get(name.removesuffix('.weight')+'.weight_scale_inv')
            else:
                weights[key] = torch.cat([tensors[n] for n, k, p, s in entries if k == key]), None
        stage.write_blob(tmp_path, str(i), stage.shapes(t, i), weights.__getitem__, stage.slot_names(t))
        assert stage.file_sha(tmp_path / f'{i}.bin') == expected[i]


def constructor_shapes(filename, classname, env):
    """Read dimensions/names from actual source constructor calls, without importing vLLM."""
    tree = ast.parse((REF / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    out = {}
    def evaluate(node):
        return eval(compile(ast.Expression(node), filename, 'eval'), {'__builtins__': {}}, env)
    class ActiveBranches(ast.NodeTransformer):
        def visit_If(self, node):
            try:
                active = evaluate(node.test)
            except (NameError, AttributeError):
                return self.generic_visit(node)
            return [self.visit(n) for n in (node.body if active else node.orelse)]
    init = ActiveBranches().visit(init)
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name) or target.value.id != 'self':
            continue
        call = node.value
        kind = ast.unparse(call.func).split('.')[-1]
        if kind in ('RMSNorm', 'LayerNorm'):
            out[target.attr+'.weight'] = (evaluate(call.args[0]),)
            if kind == 'LayerNorm':
                out[target.attr+'.bias'] = out[target.attr+'.weight']
        elif kind in ('Linear', 'ReplicatedLinear', 'ColumnParallelLinear', 'RowParallelLinear', 'MergedColumnParallelLinear', 'DeepSeekV2FusedQkvAProjLinear'):
            try:
                ins, outs = map(evaluate, call.args[:2])
            except (NameError, AttributeError):
                continue  # unused no-LoRA alternate branch
            if isinstance(outs, list):
                # Source loader mappings in mtp.py determine the checkpoint split names.
                mappings = ast.parse((REF/'mtp.py').read_text())
                assignment = next(n for n in ast.walk(mappings) if isinstance(n, ast.Assign) and
                                  any(isinstance(t, ast.Name) and t.id == 'stacked_params_mapping' for t in n.targets))
                split = sorted((shard, original) for fused, original, shard in ast.literal_eval(assignment.value)
                               if fused == target.attr)
                assert len(split) == len(outs)
                for (shard, name), size in zip(split, outs):
                    out[name+'.weight'] = (size, ins)
            else:
                out[target.attr+'.weight'] = (outs, ins)
        elif kind == 'Parameter':
            inner = call.args[0]
            
            try:
                out[target.attr] = tuple(evaluate(a) for a in inner.args)
            except (NameError, AttributeError):
                pass
    return out


def test_loader_names_and_shapes_from_source(fixture):
    config, tensors, _ = fixture
    c = SimpleNamespace(**config['text_config'])
    env = dict(config=c, self=c, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
               proj_input_size=c.hidden_size)
    # Constructor self fields use these aliases, assigned earlier in attention.py.
    c.num_heads = c.num_attention_heads
    c.n_head = c.index_n_heads
    c.head_dim = c.index_head_dim
    c.qk_head_dim = c.qk_nope_head_dim+c.qk_rope_head_dim
    expected = constructor_shapes('mtp.py', 'Glm5NextMultiTokenPredictorLayer', env)
    expected.update(constructor_shapes('model.py', 'Glm5NextDecoderLayer', env))
    # HC exists only in the explicitly non-MTP branch (model.py:379).
    expected = {n: s for n, s in expected.items() if not n.startswith('hc_')}
    for file, cls, prefix in [('attention.py', 'Glm5NextMLAAttention', 'self_attn.'),
                              ('attention.py', 'Indexer', 'self_attn.indexer.'),
                              ('model.py', 'Glm5NextMLP', 'mlp.')]:
        expected.update({prefix+n: s for n, s in constructor_shapes(file, cls, env).items()})
    # The supplied snapshot imports SharedHead but omits deepseek_mtp.py.
    # Its norm/head dimensions follow hidden/vocab widths; parse those widths
    # from the actual norm and embedding constructors in mtp.py. Names come
    # from shared_head attribute accesses in forward/compute_logits.
    source = ast.parse((REF/'mtp.py').read_text())
    shared_names = {n.attr for n in ast.walk(source) if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Attribute) and n.value.attr == 'shared_head'}
    assert shared_names == {'norm', 'head'}
    embedding = next(n for n in ast.walk(source) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'VocabParallelEmbedding')
    widths = tuple(eval(compile(ast.Expression(a), 'mtp.py', 'eval'),
                        {'__builtins__': {}}, env) for a in embedding.args)
    expected.update({'shared_head.norm.weight': expected['hnorm.weight'],
                     'shared_head.head.weight': widths})
    prefix = f'model.language_model.layers.{c.num_hidden_layers}.'
    actual = {n.removeprefix(prefix): tuple(v.shape) for n, v in tensors.items()
              if n.startswith(prefix) and not n.endswith('weight_scale_inv')}
    assert actual == expected
    for n, v in tensors.items():
        if n.startswith(prefix) and v.dtype == torch.float8_e4m3fn:
            assert tuple(tensors[n.removesuffix('.weight')+'.weight_scale_inv'].shape) == tuple((d+127)//128 for d in v.shape)


def test_deterministic_and_causal(fixture):
    config, _, model = fixture
    rng = np.random.default_rng(42)
    ids = rng.integers(0, config['text_config']['vocab_size'], 7)
    hidden = rng.standard_normal((7, config['text_config']['hidden_size'])).astype(np.float32)
    tokens, logits, recycled = model.chain(ids, hidden, 3)
    again = model.chain(ids, hidden, 3)
    for a, b in zip((tokens, logits, recycled), again):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    baseline, _ = model.forward(ids, hidden)
    for t in (0, 3, 5):
        draft, state = model.forward(ids[:t+1], hidden[:t+1])
        np.testing.assert_allclose(draft[-1], logits[0, t], rtol=2e-5, atol=2e-5)
        changed_ids, changed_hidden = ids.copy(), hidden.copy()
        changed_ids[t+1:] = 0
        changed_hidden[t+1:] = 1e4
        changed, _ = model.forward(changed_ids, changed_hidden)
        np.testing.assert_array_equal(changed[:t+1], baseline[:t+1])
    assert np.array_equal(tokens, logits.argmax(-1))


def test_position_zero_masks_embedding(fixture):
    _, _, model = fixture
    hidden = np.ones((1, model.c['hidden_size']), dtype=np.float32)
    np.testing.assert_array_equal(model.forward([1], hidden)[0], model.forward([2], hidden)[0])
    with pytest.raises(ValueError, match='contiguous'):
        model.forward([1], hidden, [1])
    with pytest.raises(ValueError, match='positive'):
        model.chain([1], hidden, 0)


def test_chain_appends_token_and_recycles_normalized_hidden(fixture):
    _, _, model = fixture
    ids = np.array([7, 13])
    previous = np.ones((2, model.c['hidden_size']), dtype=np.float32)
    tokens, logits, _ = model.chain(ids, previous, 2)
    first, hidden = model.forward(ids, previous)
    second, _ = model.forward(np.append(ids, first[-1].argmax()),
                              np.concatenate([previous, hidden[-1:]], 0))
    np.testing.assert_array_equal(logits[1, -1], second[-1])
    assert tokens[1, -1] == second[-1].argmax()


def test_sparse_selection_is_causal_beyond_topk(fixture):
    _, _, model = fixture
    rng = np.random.default_rng(6)
    x = rng.standard_normal((71, model.c['hidden_size'])).astype(np.float32)
    q = rng.standard_normal((71, model.c['q_lora_rank'])).astype(np.float32)
    rows = model.selected(x, q)
    for t, selected in enumerate(rows):
        assert np.all(selected <= t)
        assert len(set(selected)) == len(selected)
    assert len(rows[-1]) < len(rows)
    prefix = model.selected(x[:69], q[:69])
    np.testing.assert_array_equal(prefix[-1], rows[68])


def test_input_order_residual_and_single_head_norm(fixture):
    """Analytic zero-decoder case, independent of the reference's helper functions."""
    config, _, original = fixture
    weights = {n: np.zeros_like(w) for n, w in original.w.items()}
    prefix = original.prefix
    h = config['text_config']['hidden_size']
    eps = config['text_config']['rms_norm_eps']
    e = np.linspace(-1, 1, h, dtype=np.float32)
    previous = np.linspace(2, 3, h, dtype=np.float32)
    weights['model.language_model.embed_tokens.weight'][9] = e
    for name in ('enorm', 'hnorm'):
        weights[prefix+name+'.weight'][:] = 1
    weights[prefix+'shared_head.norm.weight'][:] = 2
    weights[prefix+'eh_proj.weight'][:, :h] = np.eye(h)
    weights[prefix+'eh_proj.weight'][:, h:] = 3*np.eye(h)
    weights[prefix+'shared_head.head.weight'][:h] = np.eye(h)
    model = MTPReference(config, weights)
    logits, hidden = model.forward([9, 9], np.stack([previous, previous]))
    enorm = e/np.sqrt(np.mean(e*e)+eps)
    hnorm = previous/np.sqrt(np.mean(previous*previous)+eps)
    fused = np.stack([3*hnorm, enorm+3*hnorm])
    expected = 2*fused/np.sqrt(np.mean(fused*fused, -1, keepdims=True)+eps)
    np.testing.assert_allclose(hidden, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(logits[:, :h], hidden)
    np.testing.assert_array_equal(logits[:, h:], 0)


def test_default_checkpoint_retains_pinned_identity(tmp_path, fixture):
    from safetensors.torch import load_file

    config, with_mtp, _ = fixture
    evidence_dir, _ = gate.gate_identity('module8')
    pinned = {p.name: p.read_bytes() for p in evidence_dir.iterdir() if p.is_file()}
    expected = json.loads(pinned['glm-model-api.json'])
    files, default = gate.evidence(variant='module8')
    for name, text in files.items():
        assert text.encode() == pinned[name]
    assert gate.weights_digest(default) == expected['weights_sha256']
    prefix = f"model.language_model.layers.{config['text_config']['num_hidden_layers']}."
    assert not any(n.startswith(prefix) for n in default)
    main = {n: v for n, v in with_mtp.items() if not n.startswith(prefix)}
    assert gate.weights_digest(main) == expected['weights_sha256']

    base_path = tmp_path / 'base'
    assert gate.checkpoint(base_path, variant='module8') == expected
    assert gate.weights_digest(load_file(base_path / 'model.safetensors')) == expected['weights_sha256']
    assert (base_path / 'REVISION').read_text() == expected['sha'] + '\n'
    assert (base_path / 'config.json').read_bytes() == pinned['glm-config.json']

    mtp_path = tmp_path / 'mtp'
    identity = gate.checkpoint(mtp_path, variant='module8', mtp=True)
    assert identity['weights_sha256'] == gate.weights_digest(with_mtp)
    assert identity['sha'] != expected['sha']
    assert identity['safetensors']['total'] == sum(v.numel() for v in with_mtp.values())
    assert (mtp_path / 'REVISION').read_text() == identity['sha'] + '\n'
    assert (mtp_path / 'config.json').read_bytes() == pinned['glm-config.json']
    assert gate.weights_digest(load_file(mtp_path / 'model.safetensors')) == identity['weights_sha256']
    assert {p.name: p.read_bytes() for p in evidence_dir.iterdir() if p.is_file()} == pinned
    with pytest.raises(ValueError, match='module8'):
        gate.checkpoint(tmp_path / 'unsupported', mtp=True)
