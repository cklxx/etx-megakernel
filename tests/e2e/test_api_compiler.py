"""Generic compiler compatibility through the external surface only."""
from pathlib import Path
import json
import sys

import pytest
from etx.external import (CompilerRequest, ProcessRequest, compiler_command, execute,
                          KernelSpec, PlanRequest, kernel, plan, emit, load_sources)
from etx.testing import forbid_compilation

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('example', ['splitk_sum', 'gemm_rs', 'moe_layer'])
def test_compiler_cli_and_api_write_identical_sources(tmp_path, example):
    graph = str(ROOT / 'examples' / f'{example}.py')
    api_out, cli_out = tmp_path / 'api', tmp_path / 'cli'
    assert compiler_command(CompilerRequest('compile', graph=graph, output=str(api_out))) == 0
    request = ProcessRequest((sys.executable, '-m', 'etx', 'compile', graph, '--out', str(cli_out)),
                             str(ROOT), 30, environment={'PATH': '/usr/bin:/bin'})
    assert execute(request, output=tmp_path / 'command').status == 'complete'
    assert {p.name for p in api_out.iterdir()} == {p.name for p in cli_out.iterdir()}
    for path in api_out.iterdir():
        assert path.read_bytes() == (cli_out / path.name).read_bytes()
    assert (api_out / 'plan_data.h').is_file()
    assert not (api_out / 'manifest.json').exists()  # Historical source-only file inventory.


def test_explain_and_archs_remain_callable_without_creating_a_build(tmp_path, capsys):
    assert compiler_command(CompilerRequest('archs')) == 0
    assert 'gfx942' in capsys.readouterr().out
    assert compiler_command(CompilerRequest('explain', graph=str(ROOT / 'examples' / 'splitk_sum.py'),
                                            output=str(tmp_path / 'unused'), bindings={'n': 2},
                                            simulate=True, seed=7)) == 0
    text = capsys.readouterr().out
    assert 'sim[static ' in text and 'sim[dynamic' in text and 'sim[hybrid ' in text
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('binding', ['requests', 'requestsz'])
def test_deepseek_unsupported_binding_never_reaches_compiler(tmp_path, binding):
    request = CompilerRequest('compile', graph=str(ROOT / 'examples/dsv2lite/graph.py'),
                              output=str(tmp_path / 'bad'), bindings={binding: 8})
    with forbid_compilation(), pytest.raises(ValueError, match='does not accept --bind.*' + binding):
        compiler_command(request)
    assert not (tmp_path / 'bad').exists()
    result = execute(ProcessRequest((sys.executable, '-m', 'etx', 'compile', request.graph,
                                     '--out', request.output, '--bind', f'{binding}=8'),
                                    str(ROOT), 20, environment={'PATH': '/usr/bin:/bin'}),
                     output=tmp_path / 'command')
    assert result.status == 'failed'
    assert 'does not accept --bind' in (tmp_path / 'command/output.log').read_text()
    assert not (tmp_path / 'bad').exists()


def test_declarative_source_cli_matches_api(tmp_path):
    spec = PlanRequest(kernel(KernelSpec('splitk', 'graph', {'provider': 'splitk_sum'})), bindings={'n': 2})
    request_file = tmp_path / 'request.json'
    request_file.write_text(json.dumps(spec.to_dict()))
    result = execute(ProcessRequest((sys.executable, '-m', 'etx', 'sources', str(request_file),
                                     '--out', str(tmp_path / 'sources')), str(ROOT), 30,
                                    environment={'PATH': '/usr/bin:/bin'}), output=tmp_path / 'command')
    assert result.status == 'complete'
    assert load_sources(tmp_path / 'sources') == emit(plan(spec))


def test_deepseek_isolated_cli_compile_through_external_process(tmp_path):
    result = execute(ProcessRequest((sys.executable, '-m', 'etx', 'compile',
                                     str(ROOT / 'examples/dsv2lite/graph.py'), '--arch', 'gfx942',
                                     '--out', str(tmp_path / 'sources'), '--inline-tiles'), str(ROOT), 30,
                                    environment={'PATH': '/usr/bin:/bin',
                                                 'ETX_DSV2_ISOLATE': '0:qkv_5;0:gate_up_5+down_5',
                                                 'ETX_DSV2_ISOLATE_REPS': '2'}), output=tmp_path / 'command')
    assert result.status == 'complete'
    assert (tmp_path / 'sources/megakernel_d0.hip').exists()
    assert 'iso1x1_r0_down_5' in (tmp_path / 'sources/plan.json').read_text()


def test_compiler_request_is_not_implicitly_executed(tmp_path):
    request = CompilerRequest('compile', graph='/does/not/exist.py', output=str(tmp_path / 'unused'))
    assert not (tmp_path / 'unused').exists()
    with pytest.raises(FileNotFoundError):
        compiler_command(request)
    assert not (tmp_path / 'unused').exists()
