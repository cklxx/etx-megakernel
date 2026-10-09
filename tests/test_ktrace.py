"""Trace regressions for the committed kernel-comparison pre-registration."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from etx.tools import ktrace
from examples.vllm_llm.kcmp_report import decide
from etx.tools.gpu_jobs import load_jobs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def historical(tmp_path_factory):
    fixtures = ROOT / 'tests/fixtures/ktrace'
    output = tmp_path_factory.mktemp('ktrace')
    for name, info in json.loads((fixtures / 'provenance.json').read_text()).items():
        data = gzip.decompress((fixtures / (name + '.gz')).read_bytes())
        assert len(data) == info['bytes']
        assert hashlib.sha256(data).hexdigest() == info['sha256']
        (output / name).write_bytes(data)
    return output


@pytest.mark.parametrize('label,expected', [
    ('A', {'qkv': 7.34, 'wvSplitK_all': 7.10, 'moe_up': 17.80, 'moe_down': 9.86,
           'topk': 7.98, 'attn': 7.54, 'reduce': 4.01, 'router': 3.97}),
    ('C', {'wvSplitK_all': 6.98, 'moe_up': 17.84, 'moe_down': 9.82,
           'topk': 8.06, 'attn': 7.50, 'reduce': 4.05, 'router': 4.01}),
])
def test_historical_vllm(historical, label, expected):
    got = ktrace.vllm(historical / f'vllm-{label}.csv')
    for phase, us in expected.items():
        assert got[phase]['median_us'] == pytest.approx(us, abs=0.005)


def test_historical_imported(historical):
    trace = historical / 'imported.csv'
    plan = historical / 'imported-plan.txt'
    got = ktrace.etx(trace, plan)
    for phase, us in {'qkv': 8.38, 'o_proj': 7.66, 'router': 4.01, 'topk': 9.98,
                      'moe_up': 24.42, 'moe_down': 12.71, 'lm_head': 135.83}.items():
        assert got[phase]['median_us'] == pytest.approx(us, abs=0.005)
        assert got[phase]['samples'] == (7 if phase == 'lm_head' else 315)
    assert 'attn' not in got and 'reduce' not in got


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_etx_rejects_incomplete(tmp_path):
    plan = tmp_path / 'plan.txt'
    plan.write_text('0 dsl_gemv 1 qkv 1 1 1 0 1\n')
    trace = tmp_path / 'trace.csv'
    write_csv(trace, [dict(Start_Timestamp=0, End_Timestamp=1000, Kernel_Name='etx_unfused_d0',
                           Dispatch_Id=1, Agent_Id=1)])
    with pytest.raises(ValueError, match='incomplete'):
        ktrace.etx(trace, plan)


def test_vendor_ranges_and_guards(tmp_path):
    write_csv(tmp_path / 'x_kernel_trace.csv', [
        dict(Start_Timestamp=1100, End_Timestamp=2100, Kernel_Name='actual_blas'),
        dict(Start_Timestamp=2200, End_Timestamp=3200, Kernel_Name='actual_reduce'),
        dict(Start_Timestamp=4100, End_Timestamp=9100, Kernel_Name='warmup')])
    write_csv(tmp_path / 'x_marker_api_trace.csv', [
        dict(Start_Timestamp=1000, End_Timestamp=4000, Function='kcmp:o_proj:0:0')])
    m = {'o_proj': dict(check_ok=True, bytes=1, iters=1, copies=1)}
    got = ktrace.vendor(tmp_path, m)['o_proj']
    assert got['median_us'] == 2
    assert got['kernel_names'] == ['actual_blas', 'actual_reduce']
    m['o_proj']['bytes'] = 100_000_000
    assert 'cache-contaminated' in ktrace.vendor(tmp_path, m)['o_proj']['refused']
    m['o_proj'].update(bytes=1, copies=2)
    assert 'incomplete' in ktrace.vendor(tmp_path, m)['o_proj']['refused']
    m['o_proj']['check_ok'] = False
    assert 'median_us' not in ktrace.vendor(tmp_path, m)['o_proj']


def test_decisions():
    assert decide([1, 1.01, 1.02], 'qkv').startswith('VENDOR WINS')
    assert decide([-1, 1, 1], 'qkv').startswith('CLOSE: mixed')
    assert decide([0.1, 0.2, 0.3], 'qkv').startswith('CLOSE')
    assert decide([1, 2, 4], 'qkv').startswith('INCONCLUSIVE')
    assert decide([1], 'qkv').startswith('NOT MEASURED')
    assert decide([1, 1.01, 1.02], 'lm_head').startswith('AMBIGUOUS')


def test_session_order_and_shell():
    jobs = load_jobs('examples/sessions/kcmp.py')
    assert [j.name for j in jobs] == ['kc_etx_1', 'kc_vnd_1', 'kc_vnd_2', 'kc_etx_2', 'kc_etx_3', 'kc_vnd_3', 'kc_report']
    for i, job in enumerate(jobs):
        assert job.after_any == [j.name for j in jobs[:i]]
        subprocess.run(['bash', '-n', '-c', job.cmd], check=True)
    assert jobs[-1].lane == 'cpu'
    assert all(j.gpu == 0 for j in jobs[:-1])


def test_vendor_cpu_dry_run():
    result = subprocess.run([sys.executable, 'examples/vllm_llm/vendor_gemv.py', '--dry-run'],
                            check=True, text=True, capture_output=True)
    assert 'stopped before torch import' in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])['measurements'] is None


def test_thresholds_are_verbatim_and_reported(tmp_path, capsys):
    from examples.sessions import kcmp
    from examples.vllm_llm import kcmp_report
    text = (ROOT / 'docs/KCMP-THRESHOLDS.md').read_text()
    thresholds = text.split('## 4. Thresholds, written before the run\n\n')[1].split('\n## 5.')[0]
    assert thresholds in kcmp.__doc__
    kcmp_report.main([str(tmp_path), '--historical-root', str(tmp_path)])
    output = capsys.readouterr().out
    assert thresholds in output
    assert 'PAIR ' not in output
    assert 'NOT MEASURED: incomplete pairs' in output


def test_nvref_command_echo_is_not_evidence(tmp_path):
    from examples.vllm_llm.kcmp_report import nvref_evidence
    from examples.vllm_llm.pairs import Refusal
    (tmp_path / 'gate_all.log').write_text(
        '# job gate_all gpu=0 cwd=/tmp\n'
        '# echo RESULT matched=32 gen=32 requests=1 CHECK-DSL OK GATE-OK\n')
    with pytest.raises((Refusal, ValueError)):
        nvref_evidence(tmp_path)


@pytest.mark.parametrize('name,arch,accepted', [
    ('', 'gfx942:sramecc+:xnack-', True),
    ('', 'gfx1100', False),
    ('AMD Instinct MI300X', '', True),
])
def test_vendor_device_architecture(name, arch, accepted, capsys):
    from types import SimpleNamespace
    from examples.vllm_llm.vendor_gemv import check_device

    properties = SimpleNamespace(name=name, gcnArchName=arch)
    torch = SimpleNamespace(
        __version__='test',
        cuda=SimpleNamespace(get_device_name=lambda index: properties.name,
                             get_device_properties=lambda index: properties))
    if accepted:
        check_device(torch)
        assert capsys.readouterr().out == f"KC_DEVICE {name or '-'} {arch.split(':')[0]} torch test\n"
    else:
        with pytest.raises(RuntimeError) as error:
            check_device(torch)
        assert f'name={name!r}' in str(error.value)
        assert f'arch={arch!r}' in str(error.value)
        assert capsys.readouterr().out == ''


def test_vendor_clean_boundary_summary(tmp_path, capsys):
    write_csv(tmp_path / 'x_kernel_trace.csv', [
        dict(Start_Timestamp=1100, End_Timestamp=4000, Kernel_Name='ends_at_boundary')])
    write_csv(tmp_path / 'x_marker_api_trace.csv', [
        dict(Start_Timestamp=1000, End_Timestamp=4000, Function='kcmp:o_proj:0:0')])
    got = ktrace.vendor(tmp_path, {'o_proj': dict(check_ok=True, bytes=1, iters=1, copies=1)})
    assert got['o_proj']['median_us'] == 2.9
    assert capsys.readouterr().out == (
        'BOUNDARY-SUMMARY lib= ranges=1 crossings=0 max_overhang_us=0.0\n')


@pytest.fixture
def boundary_trace(tmp_path):
    nested = tmp_path / 'nested'
    nested.mkdir()
    write_csv(nested / 'x_kernel_trace.csv', [
        dict(Start_Timestamp=1100, End_Timestamp=4250, Kernel_Name='first_crossing'),
        dict(Start_Timestamp=1200, End_Timestamp=2000, Kernel_Name='contained'),
        dict(Start_Timestamp=2200, End_Timestamp=5670, Kernel_Name='worst_kernel'),
        dict(Start_Timestamp=7100, End_Timestamp=11160, Kernel_Name='later_worst')])
    write_csv(nested / 'x_marker_api_trace.csv', [
        dict(Start_Timestamp=1000, End_Timestamp=4000, Function='kcmp:o_proj:2:21'),
        dict(Start_Timestamp=7000, End_Timestamp=8000, Function='kcmp:o_proj:2:22')])
    return tmp_path


def test_vendor_worst_crossing(boundary_trace, capsys):
    with pytest.raises(ValueError) as error:
        ktrace.vendor(boundary_trace, {}, 'hipblaslt')
    assert str(error.value) == 'kernel crosses vendor range boundary: kcmp:o_proj:2:21'
    assert capsys.readouterr().out.splitlines() == [
        'BOUNDARY-CROSS label=kcmp:o_proj:2:21 in_range=3 overhang_us=1.7 '
        'kernel=worst_kernel kernel_end_us=5.7 marker_end_us=4.0',
        'BOUNDARY-SUMMARY lib=hipblaslt ranges=2 crossings=2 max_overhang_us=3.2']


def test_boundary_report_cli(boundary_trace):
    result = subprocess.run([sys.executable, '-m', 'etx.tools.ktrace', '--trace-dir',
                             str(boundary_trace), '--boundaries'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload, summary = result.stdout.split('BOUNDARY-SUMMARY ')
    assert json.loads(payload) == [
        dict(label='kcmp:o_proj:2:22', in_range=1, overhang_us=3.2,
             kernel='later_worst', kernel_end_us=11.2, marker_end_us=8.0),
        dict(label='kcmp:o_proj:2:21', in_range=3, overhang_us=1.7,
             kernel='worst_kernel', kernel_end_us=5.7, marker_end_us=4.0)]
    assert summary == 'lib= ranges=2 crossings=2 max_overhang_us=3.2\n'
    assert 'duration' not in result.stdout
    assert 'median_us' not in result.stdout
    assert result.stderr == ''


def test_etx_cli_preserved(historical):
    result = subprocess.run([sys.executable, '-m', 'etx.tools.ktrace',
                             str(historical / 'imported.csv'), '--plan',
                             str(historical / 'imported-plan.txt')],
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == ktrace.etx(
        historical / 'imported.csv', historical / 'imported-plan.txt')
