"""Offline configuration contract, identity and explicit qualification forwarding."""
import json
from pathlib import Path
import sys
import pytest

from etx.models.glm53 import model_build
from examples.glm53 import model, options, run


def test_original_defaults_and_planner_not_compiler_register_claim():
    config = options.Options()
    assert config.to_dict() == dict(threads=256, vgpr_budget=64, waves_per_simd_max=0,
                                    cache_ab=0, routed_slc=0, shared_slc=0, row_tile=1, k_tile=0, lds_bytes=0, phased=1,
                                    devices=2, up_pairs=36, parallel=0, attention_tp=0, up_tail_balance=0)
    assert config.effective_up_pairs == 36
    spec = model_build.production_compile_spec()
    g = model.build_from(spec, options=options.Options(phased=0))
    assert all(r.resource.threads == 256 and r.resource.vgpr == 64 and r.resource.lds_bytes == 0 for r in g.grids)
    changed = model.build_from(spec, options=options.Options(threads=64, vgpr_budget=96, lds_bytes=256, phased=0))
    assert all(r.resource.threads == 64 and r.resource.vgpr == 96 and r.resource.lds_bytes == 0 for r in changed.grids)


@pytest.mark.parametrize('values', [dict(threads=32), dict(threads=192), dict(vgpr_budget=0),
    dict(vgpr_budget=513), dict(cache_ab=2), dict(waves_per_simd_max=9), dict(routed_slc=2), dict(shared_slc=-1),
    dict(row_tile=0), dict(row_tile=33), dict(k_tile=-1), dict(k_tile=2**31),
    dict(lds_bytes=2), dict(lds_bytes=65540), dict(threads=True), dict(unknown=1), dict(phased=2),
    dict(devices=1), dict(devices=3), dict(devices=4, phased=0), dict(devices=8, phased=0),
    dict(up_pairs=0), dict(up_pairs=257), dict(parallel=-1), dict(parallel=3),
    dict(parallel=1, phased=0), dict(parallel=2, phased=0),
    dict(attention_tp=-1), dict(attention_tp=2), dict(attention_tp=1),
    dict(attention_tp=1, parallel=0), dict(attention_tp=1, parallel=1, phased=0),
    dict(up_tail_balance=-1), dict(up_tail_balance=2), dict(up_tail_balance=1, phased=0),
    dict(up_tail_balance=1, up_pairs=32), dict(up_tail_balance=1, up_pairs=37)])
def test_options_fail_closed(values):
    with pytest.raises(ValueError):options.from_dict(values)


def test_configuration_hash_is_canonical_and_changes_each_knob():
    config = options.Options()
    assert options.from_dict(dict(reversed(list(config.to_dict().items())))).digest() == config.digest()
    for name, value in dict(threads=128, vgpr_budget=32, waves_per_simd_max=4, cache_ab=1, routed_slc=1,
                            shared_slc=1, row_tile=2, k_tile=64, lds_bytes=256, phased=0, devices=4,
                            up_pairs=32, parallel=1, up_tail_balance=1).items():
        assert options.from_dict(dict(config.to_dict(), **{name:value})).digest() != config.digest()
    for parallel in (1, 2):
        serial = options.Options(parallel=parallel)
        sharded = options.Options(parallel=parallel, attention_tp=1)
        assert serial.digest() != sharded.digest()
        assert options.from_dict(sharded.to_dict()).digest() == sharded.digest()


@pytest.mark.parametrize('name', ['attention_tp', 'up_tail_balance'])
@pytest.mark.parametrize('value', [True, False, 1.0, '1', None])
def test_parallel_switches_require_actual_integers(name, value):
    with pytest.raises(ValueError, match='integer'):
        options.from_dict(dict(parallel=1, **{name:value}))
    with pytest.raises(ValueError, match='integer'):
        options.Options(parallel=1, **{name:value})


@pytest.mark.parametrize('parallel', [1, 2])
@pytest.mark.parametrize('world', [2, 4, 8])
@pytest.mark.parametrize('balance', [0, 1])
def test_attention_tp_and_balance_are_independent_explicit_options(parallel, world, balance):
    config = options.Options(parallel=parallel, devices=world, attention_tp=1, up_tail_balance=balance)
    assert config.attention_tp == 1 and config.up_tail_balance == balance
    assert config.up_pairs == 36 and config.effective_up_pairs == (32 if balance else 36)
    assert options.from_dict(config.to_dict()) == config
    assert '#define ETX_GLM_ATTENTION_TP 1\n' in config.header()
    assert f'#define ETX_GLM_UP_TAIL_BALANCE {balance}\n' in config.header()
    assert f'#define ETX_GLM_UP_PAIRS {config.effective_up_pairs}\n' in config.header()


@pytest.mark.parametrize('pairs', [1, 32, 36, 256])
def test_effective_up_pairs_preserves_explicit_unbalanced_override(pairs):
    assert options.Options(up_pairs=pairs).effective_up_pairs == pairs


def test_balance_only_preserves_pipeline_and_header_defaults():
    baseline = options.Options()
    balanced = options.Options(up_tail_balance=1)
    assert balanced.to_dict() == dict(baseline.to_dict(), up_tail_balance=1)
    assert balanced.parallel == balanced.attention_tp == 0
    assert balanced.effective_up_pairs == 32 and baseline.effective_up_pairs == 36
    assert '#define ETX_GLM_ATTENTION_TP 0\n' in baseline.header()
    assert '#define ETX_GLM_UP_TAIL_BALANCE 0\n' in baseline.header()
    assert '#define ETX_GLM_UP_PAIRS 36\n' in baseline.header()
    # Equal effective tiling is not equal build identity: the opt-in stays explicit.
    assert balanced.digest() != options.Options(up_pairs=32).digest()


@pytest.mark.parametrize('name', ['attention_parallel', 'tail_balance'])
def test_unsupported_parallel_option_names_fail_closed(name):
    with pytest.raises(ValueError, match='unknown build options'):
        options.from_dict({name:1})


def test_generated_configuration_header_waves_and_build_identity(tmp_path):
    out = tmp_path/'build'
    config = options.Options(threads=64, row_tile=3, k_tile=8, lds_bytes=256, waves_per_simd_max=4)
    model_build.generate(model_build.production_compile_spec(), out, 2, config)
    identity = json.loads((out/'source-identity.json').read_text())
    assert identity['options'] == config.to_dict() and identity['options_sha256'] == config.digest()
    assert 'examples/glm53/native_options.h' in identity['sources']
    assert 'examples/glm53/options.py' in identity['sources']
    assert 'examples/glm53/phases.h' in identity['sources']
    assert all(f'etx/codegen/{name}' in identity['sources'] for name in ('kernel.py','lowering.py','plan_header.py'))
    assert identity['generated']['glm53_options.h'] and identity['generated']['build-options.json']
    source = (out/'megakernel_d0.hip').read_text()
    assert '#define ETX_THREADS 64' in source
    assert 'amdgpu_waves_per_eu(1, 4)' in source
    assert (out/'glm53_options.h').read_text() == config.header()
    (out/'run').write_text('not executed')
    model_build.finalize(out)
    assert model_build.verified_build(out)['options_sha256'] == config.digest()
    (out/'build-options.json').write_text(json.dumps(options.Options().to_dict()))
    with pytest.raises(ValueError):model_build.verified_build(out)


@pytest.mark.parametrize('source', ['native_options.h', 'phases.h', 'kernel.py'])
def test_identity_detects_generated_and_native_options_drift(tmp_path, monkeypatch, source):
    out = tmp_path/'build';model_build.generate(model_build.production_compile_spec(),out)
    (out/'run').write_text('not executed');model_build.finalize(out)
    old_sha = model_build.stage.file_sha
    def changed(path):
        return 'drift' if Path(path).name == source else old_sha(path)
    monkeypatch.setattr(model_build.stage,'file_sha',changed)
    with pytest.raises(ValueError,match='identity'):model_build.verified_build(out)


def test_run_build_passes_explicit_options_after_cleaning_environment(tmp_path,monkeypatch):
    stage = tmp_path/'stage';build = tmp_path/'build';out = tmp_path/'result';cfg = tmp_path/'options.json'
    cfg.write_text(json.dumps(options.Options(row_tile=2).to_dict()))
    monkeypatch.setenv('ETX_GLM_OPTIONS','/bad/ambient.json')
    monkeypatch.setenv('GLM_SHARED_EVENTS','1')
    monkeypatch.setenv('GLMDBG_DUMP_LAYERS','/bad/dump')
    monkeypatch.setattr(run.stage,'validate',lambda *a,**k: {})
    calls=[]
    monkeypatch.setattr(run,'execute',lambda *a: calls.append(a))
    monkeypatch.setattr(sys,'argv',['run','build','--stage',str(stage),'--build',str(build),'--out',str(out),
                                    '--options',str(cfg),'--batch','4'])
    run.main()
    command,log,env,timeout=calls[0]
    assert command[-2:] == ['--options',str(cfg.resolve())]
    assert 'ETX_GLM_OPTIONS' not in env and env['ETX_GLM_BATCH']=='4'
    assert 'GLM_SHARED_EVENTS' not in env and 'GLMDBG_DUMP_LAYERS' not in env


def test_qualify_cannot_supply_options_to_override_existing_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','qualify','--stage',str(tmp_path),'--build',str(tmp_path),
                                   '--out',str(tmp_path),'--options',str(tmp_path/'options.json')])
    with pytest.raises(SystemExit):run.main()


def test_inject_refuses_missing_entrypoint():
    with pytest.raises(ValueError,match='entrypoint'):
        options.inject('// wrong source', options.Options(waves_per_simd_max=4))
