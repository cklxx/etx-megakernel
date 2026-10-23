"""Model builds through the external API and the S3 translator; fake compilers, never a device run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
from etx.external import ModelBuildRequest, PlanRequest, build_model, model, execute, ProcessRequest


ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / 'examples/llm/model.py'
SYNTH = ROOT / 'examples/llm/configs/phi-synth-a.json'
FAKE = '''#!/bin/bash
{ printf 'cwd=%s\\n' "$PWD"; printf 'arg=%s\\n' "$@"; } > "$HIPCC_RECORD"
echo "x.hip: remark: Function Name: etx_megakernel_d0 [-Rpass-analysis=kernel-resource-usage]" >&2
echo "x.hip: remark: VGPRs: 128 [-Rpass-analysis=kernel-resource-usage]" >&2
echo "x.hip: remark: ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]" >&2
[ -z "${HIPCC_FAIL:-}" ] || { echo "x.hip:1:1: error: injected" >&2; exit 1; }
out=; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
printf '#!/bin/sh\\n' > "$out"; chmod +x "$out"
'''


def workload():
    return model('llm', config=dict(declaration=str(LLM), declaration_sha256=hashlib.sha256(LLM.read_bytes()).hexdigest(),
                                    environment={'ETX_LLM_CONFIG': str(SYNTH)}))


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin/hipcc').write_text(FAKE)
    (tmp_path / 'bin/hipcc').chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    monkeypatch.setenv('HIPCC_RECORD', str(tmp_path / 'hipcc.args'))
    monkeypatch.delenv('HIPCC_FAIL', raising=False)
    return tmp_path


def request(out, **fields):
    return ModelBuildRequest(PlanRequest(workload(), options=dict(inline_tiles=True)), str(ROOT), str(out),
                             str(ROOT / 'examples/llm/host.hip'), ('-O3', '-DX=1'),
                             (str(ROOT / 'etx/runtime/include'), str(out), str(ROOT)), True, **fields)


def test_build_writes_the_legacy_files_and_links_ranks_with_the_host(compiler):
    out = compiler / 'out'
    result = build_model(request(out))
    assert result['status'] == 'EXECUTABLE' and result['device_execution'] is False
    assert sorted(p.name for p in out.iterdir() if p.is_file()) == [
        'decisions.log', 'etx_lowering.h', 'etx_tiles.h', 'megakernel_d0.hip', 'plan.json', 'plan_data.h', 'run']
    assert (compiler / 'hipcc.args').read_text().splitlines() == [
        f'cwd={ROOT}', 'arg=-O3', 'arg=-DX=1', f'arg=-I', f'arg={ROOT}/etx/runtime/include', 'arg=-I', f'arg={out}',
        'arg=-I', f'arg={ROOT}', 'arg=-Rpass-analysis=kernel-resource-usage', f'arg={out}/megakernel_d0.hip',
        f'arg={ROOT}/examples/llm/host.hip', 'arg=-o', f'arg={out}/run']
    assert result['executable'] == f'{out}/run'
    assert result['executable_sha256'] == hashlib.sha256((out / 'run').read_bytes()).hexdigest()
    assert 'tasks=' in result['summary'] and 'Function Name' in Path(result['log']).read_text()


def test_compiler_failure_is_failed_even_with_a_stale_executable(compiler, monkeypatch):
    out = compiler / 'out'
    assert build_model(request(out))['status'] == 'EXECUTABLE'
    monkeypatch.setenv('HIPCC_FAIL', '1')
    result = build_model(request(out))
    # The legacy shell reported `built` here because the old binary remained; the API does not.
    assert result['status'] == 'FAILED' and result['returncode'] == 1 and result['executable'] is None


def test_translator_prints_the_historical_console_lines(compiler, tmp_path):
    packed = tmp_path / 'packed/phi-synth-a'
    packed.mkdir(parents=True)
    shutil.copy(SYNTH, packed / 'config.json')
    out = tmp_path / 'built'
    env = {key: value for key, value in os.environ.items() if not key.startswith('ETX_')}
    env.update(ETX_OUT=str(out), ETX_PYTHON=sys.executable)
    process = ProcessRequest(('/bin/bash', str(ROOT / 'examples/llm/build.sh'), str(packed), '-DEXTRA=1'),
                             str(tmp_path), 300.0, environment=env, silence_s=300.0)
    result = execute(process, output=tmp_path / 'translator')
    console = (tmp_path / 'translator/output.log').read_text().splitlines()
    assert result.returncode == 0, console
    assert console[0].startswith('  devices=1 domains/device=8 workers/domain=') and console[1].startswith('  tasks=')
    assert console[2] == 'Function Name: etx_megakernel_d0\tVGPRs: 128\tScratchSize [bytes/lane]: 0\t'
    assert console[3:] == [f'built {out}/run']
    assert 'arg=-DEXTRA=1' in (compiler / 'hipcc.args').read_text()
    env['HIPCC_FAIL'] = '1'
    shutil.rmtree(out)
    failed = execute(ProcessRequest(process.argv, process.cwd, 300.0, environment=env, silence_s=300.0),
                     output=tmp_path / 'translator-failed')
    console = (tmp_path / 'translator-failed/output.log').read_text().splitlines()
    assert failed.returncode == 1 and console[-1] == f'build failed: no {out}/run'
    # `paste - - - -` groups every four filtered lines, so the error joins the resource row.
    assert console[-2] == 'Function Name: etx_megakernel_d0\tVGPRs: 128\tScratchSize [bytes/lane]: 0\tx.hip:1:1: error: injected'


@pytest.mark.parametrize('field, value, message', [
    ('output', 'relative/out', 'absolute'), ('compiler', '--evil', 'compiler'), ('deadline_s', 0, 'deadline')])
def test_invalid_build_requests_are_refused(compiler, field, value, message):
    fields = dict(output=str(compiler / 'out'))
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        ModelBuildRequest(PlanRequest(workload()), str(ROOT), fields.pop('output'), str(ROOT / 'examples/llm/host.hip'),
                          **fields)


def test_graph_workloads_are_not_model_builds():
    from etx.external import KernelSpec, kernel
    with pytest.raises(ValueError, match='kind="model"'):
        ModelBuildRequest(PlanRequest(kernel(KernelSpec('splitk_sum', 'graph', dict(provider='splitk_sum')))),
                          str(ROOT), '/tmp/x', str(ROOT / 'examples/llm/host.hip'))


@pytest.mark.skipif(not (os.environ.get('ETX_TEST_ROCM_DOCKER') and os.environ.get('ETX_MODEL_BUILD_BASELINE')),
                    reason='opt-in: ETX_TEST_ROCM_DOCKER image and ETX_MODEL_BUILD_BASELINE legacy capture')
def test_docker_build_is_byte_identical_to_the_retained_legacy_shell(tmp_path):
    # Baseline: packed/qwen3-30b-a3b/config.json and legacy/ (examples/llm/build.sh output, same path rebuilt here).
    baseline = Path(os.environ['ETX_MODEL_BUILD_BASELINE'])
    legacy = baseline / 'legacy'
    record = json.loads((legacy.parent / 'legacy.json').read_text())
    out = Path(record['output'])
    assert not out.exists(), 'the legacy output path must be free so the rebuild uses identical paths'
    config = baseline / 'packed/qwen3-30b-a3b/config.json'
    work = model('llm', config=dict(declaration=str(LLM), declaration_sha256=hashlib.sha256(LLM.read_bytes()).hexdigest(),
                                    environment={'ETX_LLM_CONFIG': str(config)}))
    try:
        result = build_model(ModelBuildRequest(
            PlanRequest(work, options=dict(inline_tiles=True)), str(ROOT), str(out), str(ROOT / 'examples/llm/host.hip'),
            ('-O3', '-std=c++17', '--offload-arch=gfx942'), (str(ROOT / 'etx/runtime/include'), str(out), str(ROOT)),
            True, 'docker:' + os.environ['ETX_TEST_ROCM_DOCKER']))
        assert result['status'] == 'EXECUTABLE'
        for name in ('decisions.log', 'etx_lowering.h', 'etx_tiles.h', 'megakernel_d0.hip', 'plan.json', 'plan_data.h', 'run'):
            assert (out / name).read_bytes() == (legacy / name).read_bytes(), name
    finally:
        shutil.rmtree(out, ignore_errors=True)
