"""The import-certificate command (examples/vllm_llm/import_certificate.py) through the external API."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
from etx.testing import certified_import_session, run_command

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate(*argv):
    return run_command([sys.executable, '-m', 'examples.vllm_llm.import_certificate', *map(str, argv)], cwd=ROOT)


@pytest.fixture
def evidence(tmp_path):
    return certified_import_session(tmp_path)


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
        obj['files']['plan'] = _sha(path / 'vl_plan.txt'); text = json.dumps(obj)
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


def test_complete_97(evidence):
    run = certificate(evidence, '--require-positions', '0,1040')
    assert run.returncode == 0
    assert '"checked": 97' in run.stdout and '"eligible": true' in run.stdout


@pytest.mark.parametrize('mutation', MUTATIONS)
def test_evidence_mutations_refuse(evidence, mutation):
    mutate(evidence, mutation)
    run = certificate(evidence, '--require-positions', '0,1040')
    assert run.returncode == 2 and 'REFUSED' in run.stdout


def test_timing_rule_not_waived(evidence):
    for rnd in range(1, 6):
        p = evidence / f'etx_r{rnd}.log'
        p.write_text(p.read_text().replace('wall_ms=4.800', 'wall_ms=5.200'))
    run = certificate(evidence)
    assert run.returncode == 1 and '"eligible": false' in run.stdout
