"""Offline configuration contract, identity and explicit qualification forwarding."""
import json
from pathlib import Path
import sys
import pytest

from examples.glm53 import model, model_build, options, run


def test_original_defaults_and_planner_not_compiler_register_claim():
    config = options.Options()
    assert config.to_dict() == dict(threads=256, vgpr_budget=64, waves_per_simd_max=0,
                                    cache_ab=0, routed_slc=0, shared_slc=0, row_tile=1, k_tile=0, lds_bytes=0)
    spec = model_build.production_compile_spec()
    g = model.build_from(spec)
    assert all(r.resource.threads == 256 and r.resource.vgpr == 64 and r.resource.lds_bytes == 0 for r in g.grids)
    changed = model.build_from(spec, options=options.Options(threads=64, vgpr_budget=96, lds_bytes=256))
    assert all(r.resource.threads == 64 and r.resource.vgpr == 96 and r.resource.lds_bytes == 0 for r in changed.grids)


@pytest.mark.parametrize('values', [dict(threads=32), dict(threads=192), dict(vgpr_budget=0),
    dict(vgpr_budget=513), dict(cache_ab=2), dict(waves_per_simd_max=9), dict(routed_slc=2), dict(shared_slc=-1),
    dict(row_tile=0), dict(row_tile=33), dict(k_tile=-1), dict(k_tile=2**31),
    dict(lds_bytes=2), dict(lds_bytes=65540), dict(threads=True), dict(unknown=1)])
def test_options_fail_closed(values):
    with pytest.raises(ValueError):options.from_dict(values)


def test_configuration_hash_is_canonical_and_changes_each_knob():
    config = options.Options()
    assert options.from_dict(dict(reversed(list(config.to_dict().items())))).digest() == config.digest()
    for name, value in dict(threads=128, vgpr_budget=32, waves_per_simd_max=4, cache_ab=1, routed_slc=1,
                            shared_slc=1, row_tile=2, k_tile=64, lds_bytes=256).items():
        assert options.from_dict(dict(config.to_dict(), **{name:value})).digest() != config.digest()


def test_generated_configuration_header_waves_and_build_identity(tmp_path):
    out = tmp_path/'build'
    config = options.Options(threads=64, row_tile=3, k_tile=8, lds_bytes=256, waves_per_simd_max=4)
    model_build.generate(model_build.production_compile_spec(), out, 2, config)
    identity = json.loads((out/'source-identity.json').read_text())
    assert identity['options'] == config.to_dict() and identity['options_sha256'] == config.digest()
    assert 'examples/glm53/native_options.h' in identity['sources']
    assert 'examples/glm53/options.py' in identity['sources']
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


def test_identity_detects_generated_and_native_options_drift(tmp_path, monkeypatch):
    out = tmp_path/'build';model_build.generate(model_build.production_compile_spec(),out)
    (out/'run').write_text('not executed');model_build.finalize(out)
    old_sha = model_build.stage.file_sha
    def changed(path):
        return 'drift' if Path(path).name == 'native_options.h' else old_sha(path)
    monkeypatch.setattr(model_build.stage,'file_sha',changed)
    with pytest.raises(ValueError,match='identity'):model_build.verified_build(out)


def test_run_build_passes_explicit_options_after_cleaning_environment(tmp_path,monkeypatch):
    stage = tmp_path/'stage';build = tmp_path/'build';out = tmp_path/'result';cfg = tmp_path/'options.json'
    cfg.write_text(json.dumps(options.Options(row_tile=2).to_dict()))
    monkeypatch.setenv('ETX_GLM_OPTIONS','/bad/ambient.json')
    monkeypatch.setattr(run.stage,'validate',lambda *a,**k: {})
    calls=[]
    monkeypatch.setattr(run,'execute',lambda *a: calls.append(a))
    monkeypatch.setattr(sys,'argv',['run','build','--stage',str(stage),'--build',str(build),'--out',str(out),
                                    '--options',str(cfg),'--batch','4'])
    run.main()
    command,log,env,timeout=calls[0]
    assert command[-2:] == ['--options',str(cfg.resolve())]
    assert 'ETX_GLM_OPTIONS' not in env and env['ETX_GLM_BATCH']=='4'


def test_qualify_cannot_supply_options_to_override_existing_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','qualify','--stage',str(tmp_path),'--build',str(tmp_path),
                                   '--out',str(tmp_path),'--options',str(tmp_path/'options.json')])
    with pytest.raises(SystemExit):run.main()


def test_inject_refuses_missing_entrypoint():
    with pytest.raises(ValueError,match='entrypoint'):
        options.inject('// wrong source', options.Options(waves_per_simd_max=4))
