"""Local acceptance for the session graph and fail-closed evidence consumers."""
import json
import subprocess
from pathlib import Path

import pytest

from etx.tools.gpu_jobs import load_jobs, load_session
from examples.sessions import nvchk
from tests.test_import_certificate import envelope, fixture_session
from tests.test_nvref_report import etx



SESSION = Path('examples/sessions/nvchk.py')


def jobs():
    return {j.name: j for j in load_jobs(SESSION)}


def test_record_version_fallbacks():
    graph = jobs()
    cmd = graph['record'].cmd
    sources = ('/opt/rocm/.info/version', '/opt/rocm/core-*/.info/version',
               'hipconfig --version', 'printf unknown')
    assert all(source in cmd for source in sources)
    assert [cmd.index(source) for source in sources] == sorted(cmd.index(source) for source in sources)
    assert 'ROCM-VERSION %s' in cmd
    assert 'sort -V' in cmd
    assert cmd.startswith('set -e; rocm-smi --showproductname; ')
    assert '; uname -r; nproc; ' in cmd
    assert graph['gate_chkx'].deps() == ['build_chkx', 'record']
    for j in graph.values():
        if 'cat /opt/rocm/.info/version' in j.cmd:
            assert 'cat /opt/rocm/.info/version 2>/dev/null || ' in j.cmd, j.name
            assert all(source in j.cmd for source in sources), j.name


def ancestors(name, graph, success_only=True):
    deps = graph[name].deps() + ([] if success_only else graph[name].deps(True))
    return set(deps).union(*(ancestors(d, graph, success_only) for d in deps))


def test_producers_and_certification_precede_every_consumer():
    graph = jobs()
    assert 'ETX_VL_CHECK=1' in graph['build_chkx'].cmd
    assert 'ETX_VL_XCD=0' not in graph['build_chkx'].cmd
    assert {'prep', 'golden_sha'} <= ancestors('build_chkx', graph)
    for name, j in graph.items():
        if '$HOME/sess/b_all' in j.cmd and name != 'build_chkx':
            assert 'build_chkx' in ancestors(name, graph)
        if '$HOME/sess/dump_all' in j.cmd:
            assert name == 'gate_chkx' or 'gate_chkx' in ancestors(name, graph)
    for rnd in range(1, 6):
        for arm in ('etx', 'etxu'):
            assert 'check_certificate' in ancestors(f'{arm}_r{rnd}', graph)
    assert '--check-pos 0' in graph['check_timed'].cmd
    assert '--check-pos 1040' in graph['check_timed'].cmd
    assert '--checks-only' in graph['check_certificate'].cmd
    assert 'check_timed' in graph['check_certificate'].deps()
    assert '--export-build $HOME/sess/b_all' in graph['build_chkx'].cmd
    assert '--session ${ETX_SESS_LOGDIR:-$HOME/sess}' in graph['build_chkx'].cmd


def test_order_and_budget():
    sequence, limit = load_session(SESSION)
    graph = {j.name: j for j in sequence}
    assert limit == 2400
    timing = [j for j in sequence if j.name.startswith(('etx_r', 'etxu_r'))]
    assert len(timing) == 10
    for i in range(0, 10, 2):
        assert {j.name.split('_')[0] for j in timing[i:i+2]} == {'etx', 'etxu'}
    for prev, following in zip(timing, timing[1:]):
        assert prev.name in ancestors(following.name, graph)
    for rnd in range(1, 4):
        assert 'ETX_SYNC=block' in graph[f'block_r{rnd}'].cmd
        assert '--unfused' in graph[f'graph_r{rnd}'].cmd
        assert 'ETX_SYNC=block' not in graph[f'poll_r{rnd}'].cmd
    assert sequence[-1].name == 'vllm_last'
    assert 'ETX_NVCHK_VLLM' in sequence[-1].cmd
    assert '--entrypoint timeout' in sequence[-1].cmd
    assert '300 python3' in sequence[-1].cmd
    assert 'ETX_SESS_REMAINING_SECONDS' in sequence[-1].cmd
    assert all(j.gpu == 0 for j in sequence if j.lane == 'gpu')
    for j in sequence:
        assert subprocess.run(['bash', '-n', '-c', j.cmd], capture_output=True).returncode == 0, j.name


@pytest.mark.parametrize('position', [0, 1040])
def test_check_failure_propagates_without_swallowing(tmp_path, position):
    cmd = jobs()['check_timed'].cmd.replace(nvchk.CERT, 'fake_cert')
    stub = f'''fake_cert() {{
case "$*" in *"--check-pos {position}") echo 'unfiltered failure'; return 7;; esac
printf 'CHECK OK\\n'
}}
'''
    run = subprocess.run(['bash', '-c', stub + cmd], env={'HOME': str(tmp_path),
                         'ETX_SESS_LOGDIR': str(tmp_path)}, capture_output=True, text=True)
    assert run.returncode == 7
    assert 'unfiltered failure' in (tmp_path / f'check_p{position}.txt').read_text()
    if position == 0:
        assert not (tmp_path / 'check_p1040.txt').exists()


@pytest.mark.parametrize('failure', [None, 'missing', 'status', 'identity', 'wrong_arm', 'gpu'])
def test_triplet_report_requires_complete_same_build(tmp_path, capsys, failure):
    fixture_session(tmp_path)
    identity = json.loads((tmp_path / 'build_identity.json').read_text())
    statuses = []
    for rnd in range(1, 4):
        for arm in ('poll', 'block', 'graph'):
            name = f'{arm}_r{rnd}'
            mode = 'graph' if arm == 'graph' else 'fused'
            wall = {'poll': 4.5, 'block': 4.4, 'graph': 4.7}[arm]
            text = f'# job {name} gpu=0\n' + envelope(identity, etx(wall, arm=mode))
            (tmp_path / f'{name}.log').write_text(text)
            statuses.append(dict(name=name, status='ok'))
    p = tmp_path / 'block_r2.log'
    if failure == 'missing':
        p.unlink()
    elif failure == 'status':
        statuses[4]['status'] = 'timeout'
    elif failure == 'identity':
        p.write_text(p.read_text().replace('phase=after', 'phase=wrong'))
    elif failure == 'wrong_arm':
        p.write_text(p.read_text().replace('mode: ETX megakernel', 'mode: unfused'))
    elif failure == 'gpu':
        p.write_text(p.read_text().replace('gpu=0', 'gpu=1'))
    (tmp_path / 'status.json').write_text(json.dumps(statuses))
    assert nvchk.report_sync(tmp_path) == (2 if failure else 0)
    assert ('SYNC NOT-COMPLETE' if failure else 'SYNC supported=True') in capsys.readouterr().out
