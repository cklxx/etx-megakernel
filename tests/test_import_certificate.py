"""Synthetic evidence mutations exercise the certificate's failure boundaries."""
import json
from pathlib import Path

import pytest

from examples.vllm_llm import import_certificate as cert
from examples.vllm_llm.pairs import Refusal
from tests.test_nvref_report import etx, write


def envelope(identity, body):
    tag = cert.digest(identity)
    return f'BUILD-IDENTITY phase=before sha256={tag}\n{body}\nBUILD-IDENTITY phase=after sha256={tag}\n'


def fixture_session(path, count=97):
    path.mkdir(exist_ok=True)
    plan = '# fixture\ntotal 100 parts 5 maxlen 1072 requests 1\n'
    plan += ''.join(f'{n} rms_norm_2d {n // 2} in_norm 1 1 1 0 1 0 0 0 {8 if n == 0 else 1} 112 -1 0 1\n' for n in range(count))
    (path / 'vl_plan.txt').write_text(plan)
    (path / 'imports.json').write_text(json.dumps({'rms_norm_2d': {'block': [256, 1, 1]}}))
    keys = ['executable', 'plan', 'imports', 'import_metadata', 'plan_header', 'plan_json',
            'weights_manifest', 'golden', 'compiler_output', 'device_source/megakernel_d0.hip',
            'device_source/dsl_tiles.hip', 'device_source/imp_tiles.hip']
    identity = dict(schema=1, check_enabled=True, compiler='AMD clang version fixture',
                    files={k: 'a' * 64 for k in keys})
    identity['files']['plan'] = cert.sha(path / 'vl_plan.txt')
    identity['files']['import_metadata'] = cert.sha(path / 'imports.json')
    (path / 'build_identity.json').write_text(json.dumps(identity))
    expected = cert.expected_instances(plan, json.loads((path / 'imports.json').read_text())) if count else []
    for pos in (0, 1040):
        body = ''.join('CHECK-INSTANCE ' + json.dumps(dict(r, position=pos, bytes=128, result='identical', first_byte=-1)) + '\n' for r in expected)
        body += f'CHECK-IMPORTED checked={count} planned={count} replicas={sum(r["replicas"] for r in expected)} position={pos}\nCHECK OK: 0 launch(es) differ'
        (path / f'check_p{pos}.txt').write_text(envelope(identity, body))
    for name, arm in [('gate_chkx', 'fused'), ('gate_unf', 'graph')]:
        write(path, name, envelope(identity, etx(5, arm) + '\nCHECK-DSL OK: fixture\nGATE-OK'))
    for rnd in range(1, 6):
        for arm, mode, ms in [('etx', 'fused', 4.8), ('etxu', 'graph', 5.2)]:
            write(path, f'{arm}_r{rnd}', envelope(identity, etx(ms, mode)))
    return path


@pytest.fixture
def evidence(tmp_path):
    return fixture_session(tmp_path)


def test_complete_97(evidence, capsys):
    assert cert.main([str(evidence), '--require-positions', '0,1040']) == 0
    out = capsys.readouterr().out
    assert '"checked": 97' in out and '"eligible": true' in out


def test_build_identity_records_emission_time_isolation_flag(tmp_path, monkeypatch):
    (tmp_path/'compiler_version.txt').write_text('test compiler')
    monkeypatch.setattr(cert, 'fingerprint', lambda *args: {})
    monkeypatch.setenv('ETX_PREFETCH_ISOLATE_POLLER', '1')
    assert cert.write_build(tmp_path, tmp_path)['overrides']['ETX_PREFETCH_ISOLATE_POLLER'] == '1'


def test_count_comes_from_plan(tmp_path):
    assert cert.validate(fixture_session(tmp_path, 3))['checked'] == 3


MUTATIONS = ['5_of_97', 'duplicate', 'order', 'zero', 'executable', 'plan', 'golden', 'imports',
             'failed', 'missing_position', 'wrong_position', 'metadata', 'bytes', 'first_byte',
             'summary', 'missing_ok', 'stamp', 'envelope', 'compiler', 'check_disabled',
             'missing_fingerprint', 'wrong_hash_shape', 'job_failed', 'gate', 'tokens', 'mode',
             'different_gpu', 'missing_timing', 'timing_fingerprint']


def mutate(path, mutation):
    p = path / 'check_p1040.txt'
    text = p.read_text()
    if mutation == '5_of_97':
        lines = text.splitlines(); seen = 0; out = []
        for line in lines:
            if line.startswith('CHECK-INSTANCE '):
                seen += 1
                if seen > 5:
                    continue
            out.append(line)
        text = '\n'.join(out)
    elif mutation in ('duplicate', 'order', 'metadata', 'wrong_position', 'bytes', 'first_byte'):
        records = [json.loads(s.removeprefix('CHECK-INSTANCE ')) for s in text.splitlines() if s.startswith('CHECK-INSTANCE ')]
        if mutation == 'duplicate': records[-1] = records[0]
        if mutation == 'order': records[0], records[1] = records[1], records[0]
        if mutation == 'metadata': records[-1]['replicas'] = 8
        if mutation == 'wrong_position': records[-1]['position'] = 0
        if mutation == 'bytes': records[-1]['bytes'] = 0
        if mutation == 'first_byte': records[-1]['first_byte'] = 127
        it = iter(records)
        text = '\n'.join('CHECK-INSTANCE ' + json.dumps(next(it)) if s.startswith('CHECK-INSTANCE ') else s for s in text.splitlines())
    elif mutation == 'failed': text = text.replace('CHECK OK: 0', 'CHECK FAILED: 1')
    elif mutation == 'summary': text = text.replace('checked=97', 'checked=5')
    elif mutation == 'missing_ok': text = text.replace('CHECK OK: 0 launch(es) differ', '')
    elif mutation == 'stamp': text = text.replace('phase=after', 'phase=before')
    elif mutation == 'envelope': text += '\nCHECK OK: 0 launch(es) differ'
    elif mutation == 'missing_position': p.unlink(); return
    elif mutation in ('executable', 'plan', 'golden', 'imports', 'compiler', 'check_disabled', 'missing_fingerprint', 'wrong_hash_shape'):
        p = path / 'build_identity.json'; obj = json.loads(p.read_text())
        if mutation == 'compiler': obj['compiler'] = ''
        elif mutation == 'check_disabled': obj['check_enabled'] = False
        elif mutation == 'missing_fingerprint': del obj['files']['golden']
        elif mutation == 'wrong_hash_shape': obj['files']['golden'] = 'not-a-sha'
        else: obj['files'][mutation] = 'b' * 64
        text = json.dumps(obj)
    elif mutation == 'zero':
        p = path / 'vl_plan.txt'; p.write_text('# zero imports\n')
        p = path / 'build_identity.json'; obj = json.loads(p.read_text())
        obj['files']['plan'] = cert.sha(path / 'vl_plan.txt'); text = json.dumps(obj)
    elif mutation == 'job_failed':
        p = path / 'status.json'; text = json.dumps([dict(name='check_timed', status='timeout')])
    else:
        p = path / ('gate_chkx.log' if mutation == 'gate' else 'etxu_r5.log'); text = p.read_text()
        if mutation == 'gate': text = text.replace('CHECK-DSL OK', 'CHECK-DSL missing')
        if mutation == 'tokens': text = text.replace('matched=32', 'matched=31')
        if mutation == 'mode': text = text.replace('unfused (same imported kernels, one launch per grid, HIP graph)', 'ETX megakernel (vLLM kernels imported)')
        if mutation == 'different_gpu': text = text.replace('gpu=0', 'gpu=1')
        if mutation == 'missing_timing': p.unlink(); return
        if mutation == 'timing_fingerprint': text = text.replace('sha256=', 'sha256=b')
    p.write_text(text)


@pytest.mark.parametrize('mutation', MUTATIONS)
def test_evidence_mutations_refuse(evidence, capsys, mutation):
    mutate(evidence, mutation)
    assert cert.main([str(evidence), '--require-positions', '0,1040']) == 2
    assert 'REFUSED' in capsys.readouterr().out


def test_timing_rule_not_waived(evidence):
    for rnd in range(1, 6):
        p = evidence / f'etx_r{rnd}.log'
        p.write_text(p.read_text().replace('wall_ms=4.800', 'wall_ms=5.200'))
    assert cert.main([str(evidence)]) == 1
    assert cert.validate(evidence)['eligible'] is False


def test_checks_only_is_not_claim_eligibility(evidence):
    (evidence / 'etx_r5.log').unlink()
    result = cert.validate(evidence, timing=False)
    assert result['checked'] == 97 and not result['eligible']


@pytest.mark.parametrize('positions', [(), (0, 0), (-1, 0), (0, 1024)])
def test_required_positions_fail_closed(evidence, positions):
    with pytest.raises((Refusal, OSError)):
        cert.validate(evidence, positions)


def test_runtime_rehash_and_compiler_capture(tmp_path, monkeypatch):
    build = tmp_path / 'b'; model = tmp_path / 'm'
    build.mkdir(); model.mkdir(); (build / 'imp').mkdir()
    for name in ('run', 'vl_plan.txt', 'imports.bc', 'plan_data.h', 'plan.json', 'hipcc.log',
                 'megakernel_d0.hip', 'dsl_tiles.hip', 'imp_tiles.hip', 'compiler_version.txt', 'imp/imports.json'):
        (build / name).write_text(name)
    for name in ('vl.manifest', 'golden_c1024.txt'): (model / name).write_text(name)
    monkeypatch.setenv('ETX_VL_CHECK', '1')
    identity = cert.write_build(build, model)
    assert identity['compiler'] == 'compiler_version.txt'
    assert cert.verified_build(build, model) == identity
    (build / 'run').write_text('different executable')
    with pytest.raises(Refusal, match='fingerprint changed'):
        cert.verified_build(build, model)


@pytest.mark.parametrize('mutation,reason', [
    ('stamp', 'missing or mismatched build fingerprints'),
    ('envelope', 'evidence outside fingerprint envelope'),
    ('compiler', 'incomplete CHECK build identity'),
    ('plan', 'archived plan/import metadata hash mismatch'),
    ('mode', 'wrong certified timing arm'),
])
def test_guard_diagnostics(evidence, mutation, reason):
    mutate(evidence, mutation)
    with pytest.raises(Refusal, match=reason):
        cert.validate(evidence)


def test_zero_imports_guard():
    with pytest.raises(Refusal, match='zero planned imports'):
        cert.expected_instances('# no imports\n', {})


def test_duplicate_plan_guard(evidence):
    plan = (evidence / 'vl_plan.txt').read_text()
    plan += plan.splitlines()[2] + '\n'
    with pytest.raises(Refusal, match='duplicate plan instance ID'):
        cert.expected_instances(plan, json.loads((evidence / 'imports.json').read_text()))


def test_error_after_success_guard(evidence):
    p = evidence / 'check_p1040.txt'
    p.write_text(p.read_text().replace('BUILD-IDENTITY phase=after', 'ERROR: late failure\nBUILD-IDENTITY phase=after'))
    with pytest.raises(Refusal, match='failed checker output'):
        cert.validate(evidence)


def test_failed_timing_status_guard(evidence):
    (evidence / 'status.json').write_text(json.dumps([dict(name='etx_r1', status='fail')]))
    with pytest.raises(Refusal, match='etx_r1: job fail'):
        cert.validate(evidence)


def test_missing_source_guard(tmp_path):
    with pytest.raises(Refusal, match='missing generated source'):
        cert.fingerprint(tmp_path, tmp_path)


def test_capture_option_guard(tmp_path):
    with pytest.raises(Refusal, match='unsupported certified host option'):
        cert.capture(tmp_path, tmp_path, ['--golden', 'other.txt'])


def test_capture_identity_changes_during_run(tmp_path, monkeypatch):
    versions = iter([{'files': {'executable': 'a'}}, {'files': {'executable': 'b'}}])
    monkeypatch.setattr(cert, 'verified_build', lambda *_: next(versions))
    monkeypatch.setattr(cert.subprocess, 'call', lambda *_: 0)
    with pytest.raises(Refusal, match='build identity changed during run'):
        cert.capture(tmp_path, tmp_path, ['--repeat', '1'])


def test_failed_capture_has_no_closing_stamp(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cert, 'verified_build', lambda *_: {'files': {'executable': 'a'}})
    monkeypatch.setattr(cert.subprocess, 'call', lambda *_: 4)
    assert cert.capture(tmp_path, tmp_path, []) == 4
    assert 'phase=after' not in capsys.readouterr().out


def test_compiler_failure_cannot_reuse_stale_executable(tmp_path):
    import subprocess
    script = Path('examples/vllm_llm/build.sh').read_text()
    tail = script[script.index('HIPCC=(hipcc -O3'):script.index('# CHECK builds require')]
    stale = tmp_path / 'run'; stale.write_text('old executable'); stale.chmod(0o755)
    prefix = r'''set -euo pipefail
hipcc() { echo 'unfiltered diagnostic'; echo 'compiler failed' >&2; return 7; }
# Nonempty placeholders isolate compiler status propagation from bash 3.2's
# empty-array nounset rule, even when mutating the extracted compiler snippet.
OUT="$1"; ROOT="$1"; PY="$2"; ROCMINC=(-I "$1"); CHK=(-DCHECK); NTF=(-DNT)
VNF=(-DVN); ABL=(-DABL); PREPF=(-DPREP)
'''
    run = subprocess.run(['bash', '-c', prefix + tail, 'bash', str(tmp_path), __import__('sys').executable], capture_output=True, text=True)
    assert run.returncode == 7
    assert (tmp_path / 'hipcc.log').read_text() == 'unfiltered diagnostic\ncompiler failed\n'
