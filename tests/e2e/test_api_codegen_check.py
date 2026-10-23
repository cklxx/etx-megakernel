"""External codegen compile/link check; fake toolchains and opt-in Docker, never a device run."""
import json
import os
from pathlib import Path
import shutil

import pytest
from etx.external import check_codegen


GRAPHS = ('splitk_sum', 'moe_layer', 'gemm_rs')
IMPORTS = ('fused_moe_down', 'fused_moe_down_e8', 'fused_moe_up', 'fused_moe_up_e8')


def tool(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(0o755)


def toolchain(root, opt_fail=None):
    """A ROCm-shaped directory whose tools record their arguments instead of compiling."""
    tool(root / 'bin/hipcc', 'if [ "$1" = --version ]; then echo "HIP version: fake"; exit 0; fi\n'
         'while [ $# -gt 1 ]; do [ "$1" = -o ] && out=$2; shift; done\n'
         'echo "hipcc ROCM_PATH=$ROCM_PATH HCC_AMDGPU_TARGET=$HCC_AMDGPU_TARGET" > "$out"\n')
    tool(root / 'llvm/bin/clang', '')
    tool(root / 'llvm/bin/llvm-link', 'cp "$1" "$3"\n')
    fail = f'case "$3" in *{opt_fail}.bc) echo "opt: invalid IR" >&2; exit 3;; esac\n' if opt_fail else ''
    tool(root / 'llvm/bin/opt', fail + 'exit 0\n')
    return str(root)


def codegen(root):
    for name in IMPORTS:
        (root / 'imports').mkdir(parents=True, exist_ok=True)
        (root / 'imports' / (name + '.ll')).write_text(f'; {name}\n')
    for graph in GRAPHS:
        (root / graph).mkdir()
        (root / graph / 'megakernel_d0.hip').write_text('// fixture\n')
    return root


def test_fake_toolchain_links_every_import_and_graph_in_legacy_order(tmp_path, capsys):
    out = codegen(tmp_path / 'codegen')
    result = check_codegen(out, toolchain=toolchain(tmp_path / 'rocm'))
    assert result['status'] == 'LINKED' and result['returncode'] == 0 and result['device_execution'] is False
    assert [step['step'] for step in result['steps']] == (
        ['compiler'] + ['import-' + name for name in IMPORTS] + ['link-' + graph for graph in GRAPHS])
    for graph in GRAPHS:
        assert (out / graph / 'run').read_text() == f'hipcc ROCM_PATH={tmp_path / "rocm"} HCC_AMDGPU_TARGET=gfx942\n'
    assert all((out / 'imports' / (name + '.bc')).read_text() == f'; {name}\n' for name in IMPORTS)
    assert (out / 'compiler-version.txt').read_text() == 'HIP version: fake\n'
    assert (out / 'compiler-sha256.txt').read_text().endswith(f'{tmp_path / "rocm"}/llvm/bin/clang\n')
    assert set(result['outputs']) == ({'compiler-version.txt', 'compiler-sha256.txt'}
                                     | {f'imports/{name}.bc' for name in IMPORTS} | {f'{g}/run' for g in GRAPHS})
    console = [line for line in capsys.readouterr().out.splitlines()]
    assert console == ([f'Verify imported IR {out}/imports/{name}.ll' for name in IMPORTS]
                       + [f'Compile/link {graph} (no execution)' for graph in GRAPHS])
    receipt = json.loads((tmp_path / 'codegen-device-check/device-check.json').read_text())
    assert receipt == result.to_dict()


def test_failing_import_stops_before_linking_and_keeps_its_exit_code(tmp_path, capsys):
    out = codegen(tmp_path / 'codegen')
    result = check_codegen(out, toolchain=toolchain(tmp_path / 'rocm', opt_fail='fused_moe_down_e8'),
                           logs=tmp_path / 'logs')
    assert result['status'] == 'FAILED' and result['returncode'] == 3
    assert [step['step'] for step in result['steps']][-1] == 'import-fused_moe_down_e8'
    assert not any((out / graph / 'run').exists() for graph in GRAPHS)
    assert 'opt: invalid IR' in capsys.readouterr().out
    assert (tmp_path / 'logs/import-fused_moe_down_e8/output.log').read_text() == 'opt: invalid IR\n'


def test_missing_codegen_or_implicit_toolchain_is_refused_before_any_output(tmp_path):
    with pytest.raises(ValueError, match='codegen directory'):
        check_codegen(tmp_path / 'absent', toolchain=str(tmp_path))
    out = codegen(tmp_path / 'codegen')
    for value in ('', '--privileged', None):
        with pytest.raises(ValueError, match='toolchain'):
            check_codegen(out, toolchain=value)
    assert not (tmp_path / 'codegen-device-check').exists()


@pytest.mark.skipif(not (os.environ.get('ETX_TEST_ROCM_DOCKER') and os.environ.get('ETX_DEVICE_CHECK_BASELINE')),
                    reason='opt-in: ETX_TEST_ROCM_DOCKER image and ETX_DEVICE_CHECK_BASELINE legacy capture')
def test_docker_outputs_are_byte_identical_to_the_retained_legacy_shell(tmp_path):
    # ETX_DEVICE_CHECK_BASELINE holds input/ (pristine `etx compile` output) and legacy/ (after the old shell).
    baseline = Path(os.environ['ETX_DEVICE_CHECK_BASELINE'])
    out = tmp_path / 'codegen'
    shutil.copytree(baseline / 'input', out, symlinks=True)
    result = check_codegen(out, toolchain=os.environ['ETX_TEST_ROCM_DOCKER'])
    assert result['status'] == 'LINKED' and result['returncode'] == 0
    assert len(result['outputs']) == 10
    for name, digest in result['outputs'].items():
        assert (out / name).read_bytes() == (baseline / 'legacy' / name).read_bytes(), name
