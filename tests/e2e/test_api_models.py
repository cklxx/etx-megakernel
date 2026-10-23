"""Model workloads through the external API: declaration files by explicit path and configuration."""
import hashlib
from pathlib import Path

import pytest
from etx.external import CompilerRequest, PlanRequest, compiler_command, emit, model, plan
from etx.testing import model_isolation_probe


ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / 'examples/llm/model.py'
CONFIGS = ROOT / 'examples/llm/configs'


# Synthetic configurations plan in well under a second; the parity case uses the measured model.
def llm(config='phi-synth-a.json', **overrides):
    fields = dict(declaration=str(LLM), declaration_sha256=hashlib.sha256(LLM.read_bytes()).hexdigest(),
                  environment={'ETX_LLM_CONFIG': str(CONFIGS / config)})
    fields.update(overrides)
    return model('llm', config=fields)


def test_llm_declaration_emits_the_legacy_compile_files_byte_for_byte(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('ETX_COHERENCE', raising=False)
    monkeypatch.setenv('ETX_LLM_CONFIG', str(CONFIGS / 'qwen3-30b-a3b.json'))
    legacy = tmp_path / 'legacy'
    assert compiler_command(CompilerRequest('compile', graph=str(LLM), output=str(legacy),
                                            options=dict(inline_tiles=True))) == 0
    capsys.readouterr()
    monkeypatch.delenv('ETX_LLM_CONFIG')
    artifact = emit(plan(PlanRequest(llm('qwen3-30b-a3b.json'), options=dict(inline_tiles=True))))
    names = sorted(path.name for path in legacy.iterdir())
    assert names == ['decisions.log', 'etx_lowering.h', 'etx_tiles.h', 'megakernel_d0.hip', 'plan.json', 'plan_data.h']
    assert {name: artifact.files[name] for name in names} == {name: (legacy / name).read_text() for name in names}
    assert artifact.status == 'SOURCES' and len(artifact.rank_sources) == 1


def test_ambient_configuration_never_reaches_the_declaration(monkeypatch):
    explicit = plan(PlanRequest(llm()))
    monkeypatch.setenv('ETX_LLM_CONFIG', str(CONFIGS / 'phi-synth-c.json'))
    monkeypatch.setenv('ETX_COHERENCE', 'l1')
    assert plan(PlanRequest(llm())).identity == explicit.identity


def test_requests_are_isolated_and_leave_no_model_modules_loaded():
    first = plan(PlanRequest(llm()))
    other = plan(PlanRequest(llm('phi-synth-c.json')))
    again = plan(PlanRequest(llm()))
    assert first.identity == again.identity != other.identity
    probe = model_isolation_probe(llm())
    assert probe['identities'] == (first.identity, first.identity) and probe['leaked'] == []


@pytest.mark.parametrize('overrides, message', [
    (dict(declaration_sha256='0' * 64), 'has sha256'),
    (dict(declaration='examples/llm/model.py'), 'absolute path'),
    (dict(declaration=str(CONFIGS / 'phi-synth-c.json')), 'not a Python file'),
    (dict(environment={'PATH': '/bin'}), 'ETX_'),
    (dict(environment={'ETX_LLM_CONFIG': 3}), 'ETX_'),
    (dict(weights='/models'), 'unknown model configuration'),
])
def test_invalid_model_requests_are_refused_before_planning(overrides, message):
    with pytest.raises(ValueError, match=message):
        plan(PlanRequest(llm(**overrides)))


def test_runtime_state_comes_from_the_declaration():
    with pytest.raises(ValueError, match='runtime state'):
        plan(PlanRequest(llm(), runtime={'topk': [[0]]}))
