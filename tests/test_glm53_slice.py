"""GLM-5.3-Flash slice gate (design section 6) and the bench certificate (section 9).

The slice is the exact release checkpoint restricted to layers 22-25, which is
the only four-layer window that both straddles the PP2 split at 23/24 and
exercises both attention kinds (K, M, K, K). Everything here is CPU-only: the
tests cover the derivation, the refusals, HF key remapping, plan generation and
the timing certificate parser.
"""
import json
import math
from pathlib import Path

import pytest

from examples.glm53 import bench, model_build, oracle, reference, stage
from examples.glm53.prepare import (EVIDENCE, check_slice, metadata, parse_layers,
                                    partition, sha, slice_config, slice_split)

ROOT = Path(__file__).resolve().parent.parent
LAYERS = (22, 23, 24, 25)


def test_parse_layers_accepts_ranges_and_lists():
    assert parse_layers('22-25') == LAYERS
    assert parse_layers('22,23,24,25') == LAYERS
    assert parse_layers(' 7-9 ') == (7, 8, 9)


@pytest.mark.parametrize('text', ['', '22', '22-', 'x-25', '25-22', '22-25,27', '24,22', '-1-2'])
def test_parse_layers_refuses_bad_windows(text):
    with pytest.raises(ValueError):
        parse_layers(text)


def test_slice_must_straddle_the_pp2_split():
    t = metadata()[0]['text_config']
    assert slice_split(t) == 24
    assert check_slice(t, LAYERS) == dict(source_num_hidden_layers=45, layers=list(LAYERS), split=24)
    for layers in ((22, 23), (24, 25), (20, 21, 22, 23), (24, 25, 26)):
        with pytest.raises(ValueError, match='straddle the PP2 boundary at layers 23/24'):
            check_slice(t, layers)
    with pytest.raises(ValueError, match='contiguous'):
        check_slice(t, (22, 24, 25))
    with pytest.raises(ValueError):
        check_slice(t, (23, 24, 25, 46))


def test_slice_config_is_the_four_layer_window_and_keeps_both_attention_kinds():
    t = metadata()[0]['text_config']
    derived = slice_config(t, LAYERS)
    assert derived['num_hidden_layers'] == 4
    assert derived['layer_types'] == ['linear_attention', 'deepseek_sparse_attention',
                                     'linear_attention', 'linear_attention']
    assert derived['mlp_layer_types'] == ['sparse'] * 4
    assert derived['layer_types'] == [t['layer_types'][i] for i in LAYERS]
    assert partition(derived)[0]['layers'] == [0, 1] and partition(derived)[1]['layers'] == [2, 3]
    assert partition(derived)[0]['moe_layers'] == 2 and partition(derived)[1]['moe_layers'] == 2
    # The source config is not mutated.
    assert t['num_hidden_layers'] == 45 and len(t['layer_types']) == 45


def test_slice_config_is_a_copy_not_an_alias():
    t = metadata()[0]['text_config']
    derived = slice_config(t, LAYERS)
    derived['layer_types'][0] = 'tampered'
    assert t['layer_types'][22] == 'linear_attention'


def test_staging_refuses_before_touching_the_checkpoint(tmp_path):
    with pytest.raises(ValueError, match='slice gate requires --layers'):
        stage.stage(tmp_path / 'missing', tmp_path / 'out', gate='slice')
    with pytest.raises(ValueError, match='only valid with --gate slice'):
        stage.stage(tmp_path / 'missing', tmp_path / 'out', gate='production', layers=LAYERS)
    with pytest.raises(ValueError, match='straddle the PP2 boundary'):
        stage.stage(tmp_path / 'missing', tmp_path / 'out', gate='slice', layers=(22, 23))


def test_reference_refuses_before_touching_the_checkpoint(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    with pytest.raises(ValueError, match='slice gate requires --layers'):
        reference.run(tmp_path / 'missing', tmp_path / 'p.json', tmp_path / 'out', tmp_path / 'off', None, 'slice')
    with pytest.raises(ValueError, match='straddle the PP2 boundary'):
        reference.run(tmp_path / 'missing', tmp_path / 'p.json', tmp_path / 'out', tmp_path / 'off', None, 'slice', (24, 25))


def test_identity_carries_the_window_and_only_for_the_slice_gate():
    full = reference.identity('production')
    assert 'slice' not in full
    sliced = reference.identity('slice', LAYERS)
    assert sliced['slice'] == dict(source_num_hidden_layers=45, layers=list(LAYERS), split=24)
    assert {k: v for k, v in sliced.items() if k != 'slice'} == full


def test_hf_layer_keys_are_reindexed_and_outer_tensors_kept():
    weight_map = {}
    for prefix in ('model.layers.', 'model.language_model.layers.', 'language_model.model.layers.'):
        for i in (21, 22, 23, 24, 25, 26):
            weight_map[f'{prefix}{i}.w'] = 'shard.safetensors'
    weight_map['model.embed_tokens.weight'] = 'outer.safetensors'
    weight_map['lm_head.weight'] = 'outer.safetensors'
    rename, groups = reference.slice_weight_map(weight_map, LAYERS)
    assert rename['model.layers.22.w'] == 'model.layers.0.w'
    assert rename['model.layers.25.w'] == 'model.layers.3.w'
    assert 'model.layers.21.w' not in rename and 'model.layers.26.w' not in rename
    assert rename['model.embed_tokens.weight'] == 'model.embed_tokens.weight'
    assert sorted(k for k, _, _ in groups[0]) == ['language_model.model.layers.0.w',
                                                  'model.language_model.layers.0.w', 'model.layers.0.w']
    assert sorted(k for k, _, _ in groups[None]) == ['lm_head.weight', 'model.embed_tokens.weight']
    # Every prefix keeps its own namespace, so both the release and the library ABI load.
    assert 'model.language_model.layers.0.w' in rename.values()
    assert 'language_model.model.layers.3.w' in rename.values()


def test_missing_window_layer_is_refused():
    with pytest.raises(ValueError, match='no tensors for slice position'):
        reference.slice_weight_map({'model.layers.22.w': 's.safetensors'}, LAYERS)


def test_write_slice_snapshot_reindexes_keys_and_truncates_config(tmp_path):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    cfg = metadata()[0]
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    tensors_a = {f'model.layers.{i}.w': torch.full((2,), float(i)) for i in (21, 22, 23)}
    tensors_b = {f'model.layers.{i}.w': torch.full((2,), float(i)) for i in (24, 25, 26)}
    tensors_outer = {'model.embed_tokens.weight': torch.zeros(2), 'lm_head.weight': torch.zeros(2)}
    save_file(tensors_a, src / 'model-00001.safetensors')
    save_file(tensors_b, src / 'model-00002.safetensors')
    save_file(tensors_outer, src / 'outer.safetensors')
    weight_map = {k: 'model-00001.safetensors' for k in tensors_a}
    weight_map.update({k: 'model-00002.safetensors' for k in tensors_b})
    weight_map.update({k: 'outer.safetensors' for k in tensors_outer})
    (src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    (src / 'config.json').write_text(json.dumps(cfg))
    (src / 'REVISION').write_text('deadbeef\n')
    reference.write_slice_snapshot(src, dst, LAYERS)
    index = json.loads((dst / 'model.safetensors.index.json').read_text())['weight_map']
    assert set(index) == {'model.layers.0.w', 'model.layers.1.w', 'model.layers.2.w', 'model.layers.3.w',
                          'model.embed_tokens.weight', 'lm_head.weight'}
    assert (dst / 'REVISION').read_text() == 'deadbeef\n'
    assert json.loads((dst / 'config.json').read_text())['text_config']['num_hidden_layers'] == 4
    # Every derived shard is written fresh; the index is self-consistent.
    for key, name in index.items():
        assert (dst / name).is_file(), key
    from safetensors import safe_open
    with safe_open(dst / index['model.layers.0.w'], framework='pt') as f:
        assert f.get_tensor('model.layers.0.w').tolist() == [22.0, 22.0]
    with safe_open(dst / index['model.layers.3.w'], framework='pt') as f:
        assert f.get_tensor('model.layers.3.w').tolist() == [25.0, 25.0]
    assert index['lm_head.weight'].startswith('outer')
    with pytest.raises(ValueError):
        reference.write_slice_snapshot(src, dst, LAYERS)  # never overwrites


def test_oracle_requires_matching_slice_identity():
    rec = dict(reference.identity('slice', LAYERS), backend='hf', batch=1,
               requests=[dict(prompt_ids=[0], generated_ids=[0] * 32)])
    same = dict(rec, backend='etx-pp2')
    assert oracle.compare(rec, same, 'slice')['status'] == 'TOKENS-EQUAL'
    t = metadata()[0]['text_config']
    other = dict(same, slice=check_slice(t, (21, 22, 23, 24)))
    with pytest.raises(ValueError, match='slices differ'):
        oracle.compare(rec, other, 'slice')
    full = dict(same)
    full.pop('slice')
    with pytest.raises(ValueError, match='no slice provenance'):
        oracle.compare(rec, full, 'slice')
    with pytest.raises(ValueError, match='sliced but the gate is production'):
        oracle.compare(rec, same, 'production')


def test_slice_plan_generates_locally_without_weights(tmp_path):
    """The derived four-layer spec compiles to a plan: layers 0-3, two ranks, no GPU."""
    cfg, api = metadata()
    t = slice_config(cfg['text_config'], LAYERS)
    n = t['num_hidden_layers']
    weights = {f'l{i}': dict(blob_bytes=1, meta_bytes=32 * len(stage.slot_names(t)),
                             blob=f'l{i}.bin', meta=f'l{i}.meta') for i in range(n)}
    for name in ('embedding', 'norm', 'head'):
        weights[name] = dict(blob_bytes=1, meta_bytes=32, blob=name + '.bin', meta=name + '.meta')
    spec = dict(schema=1, model='zai-org/GLM-5.3-Flash', revision=api['sha'],
                config_sha256=sha(EVIDENCE / 'glm-config.json'), fixture=True, compile_only=True,
                config=t, dimensions=stage.dimensions(t, 1072), ranks=partition(t), weights=weights,
                slice=check_slice(cfg['text_config'], LAYERS))
    out = tmp_path / 'build'
    plan = model_build.generate(spec, out, 1)
    assert plan.n_devices == 2
    emitted = json.loads((out / 'plan.json').read_text())
    assert emitted['devices'] == 2 and emitted['shape'] == [1]
    for name in ('plan.json', 'plan_data.h', 'megakernel_d0.hip', 'megakernel_d1.hip', 'load.tsv'):
        assert (out / name).stat().st_size > 0
    layers = [line.split() for line in (out / 'load.tsv').read_text().splitlines() if line.startswith('l')]
    assert len(layers) == 2 * n  # blob + meta per layer
    # The derived window keeps the PP2 ownership: derived layers 0-1 on rank 0, 2-3 on rank 1.
    owner = {name: int(rank) for name, _, rank, _ in layers}
    assert owner['l0_weights'] == owner['l1_weights'] == 0
    assert owner['l2_weights'] == owner['l3_weights'] == 1


def _log(batch=1, ms=lambda pos: pos / 1000):
    rows = [f'GLM-MODEL-STEP pos={pos} batch={batch} ms={ms(pos):.6f}'
            for pos in range(bench.CONTEXT + bench.GEN - 1)]
    return '\n'.join(rows) + '\nGLM-MODEL-EXECUTED batch=1 layers=4 generated_each=32\n'


def test_bench_samples_are_positions_1024_to_1054():
    xs = bench.step_samples(_log(), 1)
    assert len(xs) == 31
    assert xs[0] == pytest.approx(1.024) and xs[-1] == pytest.approx(1.054)


def test_bench_refuses_a_log_without_step_times():
    text = '\n'.join(f'GLM-MODEL-STEP pos={pos} batch=1' for pos in range(1055))
    with pytest.raises(bench.Refusal, match='carries no ms='):
        bench.step_samples(text, 1)


def test_bench_refuses_incomplete_or_invalid_samples():
    with pytest.raises(bench.Refusal, match='incomplete/duplicate'):
        bench.step_samples(_log().replace('pos=700 ', 'pos=699 '), 1)
    with pytest.raises(bench.Refusal, match='invalid decode timing'):
        bench.step_samples(_log(ms=lambda pos: math.nan if pos == 1030 else 1.0), 1)
    with pytest.raises(bench.Refusal, match='invalid decode timing'):
        bench.step_samples(_log(ms=lambda pos: 0.0 if pos == 1030 else 1.0), 1)


def test_bench_summary_and_certificate_fields():
    xs = bench.step_samples(_log(), 1)
    s = bench.summarise(xs, 1)
    assert s['median_decode_wall_ms'] == pytest.approx(1.039)
    assert s['tokens_per_s'] == pytest.approx(1000 / 1.039)
    assert s['ms_per_token'] == pytest.approx(1.039)
    assert s['spread_ms'] == pytest.approx(0.030)
    staged = dict(config=metadata()[0]['text_config'], dimensions=dict(MAXLEN=1072),
                  slice=check_slice(metadata()[0]['text_config'], LAYERS))
    receipt = dict(status='TOKENS-EQUAL', repetitions=2, matched_each=[32], logit_gate='REPORTED',
                   logit_atol=None, numerical_certification='raw full logits compared with HF')
    cert = bench.certificate(staged, Path('/tmp/build'), dict(status='GPU-LINKED', binary_sha256='0' * 64),
                             1, xs, receipt, dict(schema=1), 'slice')
    assert cert['status'] == 'PASS' and cert['gate'] == 'slice' and cert['batch'] == 1
    assert cert['steps'] == 31 and cert['conditions']['layers'] == 45
    assert cert['tokens_matched'] == 32 and cert['tokens_expected'] == 32
    assert cert['qualification']['status'] == 'TOKENS-EQUAL'
    assert json.dumps(cert)  # serialisable

    with pytest.raises(bench.Refusal, match='batch must be one of'):
        bench.summarise(xs, 3)


def test_step_timestamp_is_taken_before_completion_copies():
    """The certificate's timed region ends when both streams retire, not after the D2H checks."""
    host = (ROOT / 'examples/glm53/model_host.hip').read_text()
    launch = host.index('for(auto& h:hosts)h.launch()')
    streams = host.index('h.wait_stream(120.0,start)')
    stamp = host.index('double step_ms=')
    checks = host.index('h.after_wait()')
    assert launch < streams < stamp < checks
    assert 'h.wait(120.0,start)' not in host
    assert host.count('h.wait_stream(120.0,start)') == 1 and 'for(auto& h:hosts)if(!h.wait_stream' in host
    assert 'ETX_CHECK(hipSetDevice(h.device));if(!h.after_wait())' in host


def test_wait_splits_stream_completion_from_device_copies():
    """wait() = wait_stream() then after_wait(); the D2H copies live in the second half."""
    text = (ROOT / 'etx/runtime/include/etx/host.h').read_text()
    wait_at = text.index('bool wait(')
    wait_body = text[wait_at:text.index('// everything the host checks', wait_at)]
    assert 'return wait_stream(timeout_s, t0) && after_wait();' in wait_body
    stream_at = text.index('bool wait_stream(')
    stream_body = text[stream_at:wait_at]
    assert 'hipMemcpyDeviceToHost' not in stream_body
    after_at = text.index('bool after_wait()')
    assert text.index('hipMemcpyDeviceToHost', after_at) > after_at


def _hashed_snapshot(tmp_path):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file({'a': torch.zeros(1)}, tmp_path / 'model-00000.safetensors')
    (tmp_path / 'config.json').write_text('{}\n')
    (tmp_path / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'a': 'model-00000.safetensors'}}))
    return tmp_path


def test_snapshot_hashes_cover_config_index_and_every_shard(tmp_path):
    snap = _hashed_snapshot(tmp_path)
    hashes = reference.snapshot_hashes(snap)
    assert set(hashes) == {'config.json', 'model.safetensors.index.json', 'model-00000.safetensors'}
    # Changing any file HF loads is detectable.
    (snap / 'model-00000.safetensors').write_bytes(b'tampered')
    assert reference.snapshot_hashes(snap)['model-00000.safetensors'] != hashes['model-00000.safetensors']
    snap = _hashed_snapshot(tmp_path / 'second')
    hashes = reference.snapshot_hashes(snap)
    (snap / 'config.json').write_text('{"changed": true}\n')
    assert reference.snapshot_hashes(snap)['config.json'] != hashes['config.json']
    snap = _hashed_snapshot(tmp_path / 'third')
    hashes = reference.snapshot_hashes(snap)
    (snap / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
    assert reference.snapshot_hashes(snap)['model.safetensors.index.json'] != hashes['model.safetensors.index.json']


def test_snapshot_hashes_refuse_a_missing_shard(tmp_path):
    snap = _hashed_snapshot(tmp_path)
    (snap / 'model-00000.safetensors').unlink()
    with pytest.raises(ValueError, match='missing derived snapshot file'):
        reference.snapshot_hashes(snap)


def test_reference_hashes_the_derived_snapshot_before_hf_loads_it():
    text = (ROOT / 'examples/glm53/reference.py').read_text()
    hashed = text.index('derived_hashes=snapshot_hashes(load_src)')
    load = text.index('from_pretrained(load_src')
    recheck = text.index('if snapshot_hashes(load_src)!=derived_hashes')
    provenance = text.index("provenance['slice_snapshot']")
    assert hashed < load < recheck < provenance
