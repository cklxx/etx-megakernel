"""GLM-5.3-Flash module gate: synthetic checkpoint, pinned identity, the production
stager/validator/reference/qualification with gate='module', and the median-logit rule.

Tests that need the production HF reference (FineGrainedFP8 dequantize) need
`accelerate`; they skip without it (run them in the module-gate venv, see
docs/glm/QWEN-BATCH-GLM53-RUNBOOK.md#module-gate).
"""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

from examples.glm53 import module_gate as mg, stage
from examples.glm53.prepare import EVIDENCE, MODULE_EVIDENCE, MODULE_MODEL, metadata
from tools import two_gpu_preflight as pf
from tools import two_gpu_verify as vf

ROOT = Path(__file__).resolve().parent.parent


def test_committed_evidence_is_what_the_generator_derives():
    pytest.importorskip('torch')
    files, ts = mg.evidence()
    for name, text in files.items():
        assert (MODULE_EVIDENCE / name).read_text() == text, name
    cfg, api = metadata(MODULE_EVIDENCE)
    assert api['id'] == MODULE_MODEL and len(api['sha']) == 40
    assert mg.weights_digest(ts) == api['weights_sha256']


def test_module_config_changes_sizes_only():
    release, _ = metadata()
    module, _ = metadata(MODULE_EVIDENCE)
    t, r = module['text_config'], release['text_config']
    changed = {k for k in r if t.get(k) != r[k]}
    assert changed <= set(mg.TEXT) | {'layer_types', 'mlp_layer_types', 'indexer_types', 'linear_attn_config'}
    for key in ('rms_norm_eps', 'hc_eps', 'hc_mult', 'hc_sinkhorn_iters', 'routed_scaling_factor', 'swiglu_limit',
                'index_kpool', 'index_head_dim', 'qk_nope_head_dim', 'v_head_dim', 'scoring_func', 'topk_method'):
        assert t[key] == r[key], key
    assert t['linear_attn_config']['head_dim'] == r['linear_attn_config']['head_dim']
    assert module['quantization_config']['weight_block_size'] == [128, 128]
    # Each rank owns one KDA + one sparse-MLA layer and one dense + one MoE MLP.
    from examples.glm53.prepare import partition
    assert [r['layers'] for r in partition(t)] == [[0, 1], [2, 3]]
    kinds = {(r['layer_types'][i], r['mlp_layer_types'][i]) for i in range(r['num_hidden_layers'])}
    assert all((t['layer_types'][i], t['mlp_layer_types'][i]) in kinds for i in range(4))
    # More pools than the index budget, so sparse selection really drops pools.
    assert mg.CONTEXT // t['index_kpool'] > t['index_topk'] // t['index_kpool']


def test_dtype_rule_reproduces_the_release_inventory():
    """The FP8/BF16/F32 storage rule, applied to the 45 release layers and the MTP layer
    (a sparse-MLA MoE layer without hyper-connection tensors), gives the pinned counts."""
    release, api = metadata()
    t = release['text_config']
    inventory = api['safetensors']['parameters']

    def count(tt, layers, skip=()):
        out = dict(BF16=0, F8_E4M3=0, F32=0)
        for i in layers:
            for key, shape in stage.shapes(tt, i).items():
                if key not in skip:
                    out[mg.storage(tt['layer_types'][i], key, shape)] += math.prod(shape)
        return out
    main = count(t, range(t['num_hidden_layers']))
    mtp_t = copy.deepcopy(t)
    for key, value in (('layer_types', 'deepseek_sparse_attention'), ('mlp_layer_types', 'sparse'), ('indexer_types', 'full')):
        mtp_t[key] = t[key] + [value]
    mtp_t['num_hidden_layers'] += 1
    mtp = count(mtp_t, [t['num_hidden_layers']], skip={k for k in stage.SLOTS if '_hc.' in k})
    assert main['F8_E4M3'] + mtp['F8_E4M3'] == inventory['F8_E4M3']
    assert main['F32'] + mtp['F32'] == inventory['F32']


def test_checkpoint_is_native_fp8_with_block_scales(tmp_path):
    pytest.importorskip('torch')
    from safetensors import safe_open
    api = mg.checkpoint(tmp_path / 'ck')
    assert (tmp_path / 'ck/REVISION').read_text().strip() == api['sha']
    index = json.loads((tmp_path / 'ck/model.safetensors.index.json').read_text())['weight_map']
    with safe_open(str(tmp_path / 'ck/model.safetensors'), 'pt') as f:
        q = f.get_tensor('model.language_model.layers.1.self_attn.q_b_proj.weight')
        s = f.get_tensor('model.language_model.layers.1.self_attn.q_b_proj.weight_scale_inv')
        kda = f.get_tensor('model.language_model.layers.0.self_attn.q_proj.weight')
        bias = f.get_tensor('model.language_model.layers.1.mlp.gate.e_score_correction_bias')
    import torch
    assert q.dtype == torch.float8_e4m3fn and tuple(s.shape) == (8, 3) and s.dtype == torch.float32
    assert len(set(s.flatten().tolist())) == s.numel()          # one distinct scale per 128x128 block
    assert kda.dtype == torch.bfloat16 and bias.dtype == torch.float32
    for e in range(8):
        assert f'model.language_model.layers.2.mlp.experts.{e}.down_proj.weight_scale_inv' in index
    assert {f'model.language_model.layers.0.self_attn.{x}_conv1d.weight' for x in 'qkv'} <= set(index)
    # Deterministic: a second generation is byte-identical.
    mg.checkpoint(tmp_path / 'again')
    assert stage.file_sha(tmp_path / 'again/model.safetensors') == stage.file_sha(tmp_path / 'ck/model.safetensors')


@pytest.fixture(scope='module')
def staged(tmp_path_factory):
    pytest.importorskip('torch')
    d = tmp_path_factory.mktemp('module-gate')
    mg.checkpoint(d / 'checkpoint')
    stage.stage(d / 'checkpoint', d / 'stage', gate='module')
    return d


def test_production_stager_and_validator_keep_the_gates_apart(staged, capsys):
    r = stage.validate(staged / 'stage', gate='module')
    assert r['model'] == MODULE_MODEL and r['fixture'] is False
    assert [x['layers'] for x in r['ranks']] == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match='production gate'):
        stage.validate(staged / 'stage')
    with pytest.raises(ValueError, match='production gate'):
        stage.stage(staged / 'checkpoint', staged / 'as-release')
    assert pf.main(['--glm-stage', str(staged / 'stage'), '--glm-gate', 'module', '--hash']) == 0
    assert pf.main(['--glm-stage', str(staged / 'stage')]) == 2
    assert 'pinned GLM-5.3-Flash checkpoint' in capsys.readouterr().out


def test_unknown_gate_is_refused(staged):
    with pytest.raises(ValueError, match='unknown GLM gate'):
        stage.validate(staged / 'stage', gate='fixture')


def test_module_session_runs_the_production_jobs_with_the_module_gate(monkeypatch):
    from etx.tools.gpu_jobs import load_session
    monkeypatch.setenv('ETX_GLM_GATE', 'module')
    monkeypatch.setenv('ETX_GLM_BATCH', str(mg.BATCH))
    jobs, cap = load_session(ROOT / 'examples/sessions/glm53_model.py')
    job = {j.name: j for j in jobs}
    assert [j.name for j in jobs] == ['apt', 'etx_env', 'model_build', 'model_qualify', 'verify']
    for name in ('model_build', 'model_qualify'):
        assert 'examples.glm53.run' in job[name].cmd and '--gate module' in job[name].cmd
    assert '--glm-only --gate module' in job['verify'].cmd
    assert job['model_qualify'].env['HIP_VISIBLE_DEVICES'] == '0,1'
    assert cap == 2700 and job['model_qualify'].timeout == 1800
    monkeypatch.setenv('ETX_GLM_GATE', 'release')
    with pytest.raises(ValueError, match='invalid GLM gate'):
        load_session(ROOT / 'examples/sessions/glm53_model.py')


def module_receipt(median):
    from tests.test_two_gpu_verify import glm_receipt
    r = glm_receipt()
    _, api = metadata(MODULE_EVIDENCE)
    r.update(gate='module', model=MODULE_MODEL, revision=api['sha'])
    for v in r['verdicts']:
        v['logit_max_abs_per_step'] = [median] * 32
    return r


def write(tmp_path, r):
    (tmp_path / 'qualification').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'qualification/qualification.json').write_text(json.dumps(r))
    return tmp_path


def test_verify_applies_the_median_rule_and_keeps_gates_apart(tmp_path):
    ok = write(tmp_path / 'ok', module_receipt(mg.MEDIAN_LOGIT_BOUND))
    assert vf.glm(ok, 'module')['median_logit_max_abs'] == [mg.MEDIAN_LOGIT_BOUND] * 2
    with pytest.raises(vf.Fail, match='production gate'):
        vf.glm(ok, 'production')
    high = write(tmp_path / 'high', module_receipt(mg.MEDIAN_LOGIT_BOUND * 1.01))
    with pytest.raises(vf.Fail, match='pre-registered module-gate bound'):
        vf.glm(high, 'module')
    from tests.test_two_gpu_verify import glm_receipt
    release = write(tmp_path / 'release', glm_receipt())
    with pytest.raises(vf.Fail, match='module gate'):
        vf.glm(release, 'module')
    assert vf.glm(release, 'production')['gate'] == 'production'


def test_rule_is_registered_between_the_measured_correct_and_buggy_bodies():
    # CPU runs of the native body against this reference (docs/glm/QWEN-BATCH-GLM53-RUNBOOK.md#module-gate)
    correct, smallest_bug = 0.453, 5.02
    assert correct * 3 < mg.MEDIAN_LOGIT_BOUND < smallest_bug / 3


# ----------------------------------------------------------------- production HF reference (accelerate)
@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('accelerate')
    d = tmp_path_factory.mktemp('module-gate-prepared') / 'out'
    verdict = mg.prepare(d)
    return d, verdict


def test_hf_loads_the_checkpoint_exactly_and_the_cpu_body_passes(prepared):
    d, verdict = prepared
    ref = json.loads((d / 'reference.json').read_text())
    assert ref['model'] == MODULE_MODEL and ref['batch'] == mg.BATCH
    assert ref['provenance']['quantization'] == 'FP8 block dequantize -> BF16'
    assert verdict['status'] == 'TOKENS-EQUAL' and verdict['matched_each'] == [32] * mg.BATCH
    assert verdict['module_rule']['ok'] and verdict['logit_argmax_equal']
    assert all(len(set(r['generated_ids'])) > 16 for r in ref['requests'])   # not a degenerate repeat
    assert not any(p.name.startswith('cpu') for p in (d / 'stage').iterdir())  # the pushed stage stays clean


MUTATIONS = {
    'shared expert dropped': ('y[j]=glm_bf(y[j]+expert[j]);GLM_SYNC();\n}', 'y[j]=glm_bf(y[j]+0*expert[j]);GLM_SYNC();\n}'),
    'KDA beta unsquashed': ('beta[j]=glm_bf(glm_sig(beta[j]));', 'beta[j]=glm_bf(beta[j]);'),
    'routed scale dropped': ('probs[k]=probs[k]/den*GLM_ROUTE_SCALE;', 'probs[k]=probs[k]/den;'),
    'kpool tail off by one': ('for(int j=np*GLM_POOL;j<=pos;++j)chosen[j]=1;', 'for(int j=np*GLM_POOL;j<pos;++j)chosen[j]=1;'),
}


@pytest.mark.parametrize('mutation', list(MUTATIONS))
def test_injected_bugs_fail_the_module_gate(prepared, tmp_path, mutation):
    d, _ = prepared
    old, new = MUTATIONS[mutation]
    source = (ROOT / 'examples/glm53/native.h').read_text()
    assert source.count(old) == 1
    (tmp_path / 'examples/glm53').mkdir(parents=True)
    (tmp_path / 'examples/glm53/native.h').write_text(source.replace(old, new))
    verdict = mg.predict(d / 'stage', d / 'reference.json', tmp_path / 'verdict.json', root=tmp_path)
    assert verdict['status'] == 'FAILED', verdict


LAUNCHER = '''#!{python}
"""Test stand-in for build/run: model_host.hip's CLI and output contract, CPU native body."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r})
import numpy as np
from examples.glm53 import cpu_check
stage, load, prompts, prefix = map(Path, sys.argv[1:5])
head, *rows = prompts.read_text().split('\\n')
batch, context = map(int, head.split())
p = [list(map(int, rows[r].split())) for r in range(batch)]
m = cpu_check.CPUModel(stage, gate='module', work=Path(tempfile.mkdtemp()))
m.reset(batch)
returned, generated, logits = [0] * batch, [[] for _ in p], []
for pos in range(context + 31):
    got, _ = m([row[pos] if pos < context else returned[r] for r, row in enumerate(p)])
    returned = [int(x) for x in got.argmax(-1)]
    if pos >= context - 1:
        logits.append(got.astype('<f4'))
        for r, t in enumerate(returned):
            generated[r].append(t)
    print(f'GLM-MODEL-STEP pos={{pos}} batch={{batch}}', flush=True)
np.stack(logits).tofile(str(prefix) + '.logits')
Path(str(prefix) + '.tokens').write_text(''.join(' '.join(map(str, g)) + ' \\n' for g in generated))
print(f'GLM-MODEL-EXECUTED batch={{batch}} layers={{len(m.spec["config"]["layer_types"])}} generated_each=32; reference comparison required')
'''


def test_production_qualification_and_verify_end_to_end_on_a_cpu_launcher(prepared, tmp_path):
    """run.qualify and two_gpu_verify --gate module, unchanged, with build/run replaced by the CPU body."""
    from examples.glm53 import model_build, run
    d, _ = prepared
    spec = stage.validate(d / 'stage', gate='module')
    build = tmp_path / 'record/model'
    model_build.generate(spec, build, mg.BATCH)
    launcher = build / 'run'
    launcher.write_text(LAUNCHER.format(python=sys.executable, root=str(ROOT)))
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    model_build.finalize(build)                       # identity of this stand-in binary
    receipt = run.qualify(d / 'stage', build, d / 'reference.json', tmp_path / 'record/qualification',
                          seconds=600, gate='module')
    assert receipt['status'] == 'TOKENS-EQUAL' and receipt['gate'] == 'module'
    summary = vf.glm(tmp_path / 'record', 'module')
    assert max(summary['median_logit_max_abs']) <= mg.MEDIAN_LOGIT_BOUND
    with pytest.raises(vf.Fail):
        vf.glm(tmp_path / 'record', 'production')
