"""CPU-only benchmark failure boundaries; synthetic output is never GPU evidence."""
import json
from pathlib import Path
import subprocess

import pytest

from examples.vllm_llm import bench as b
from examples.vllm_llm import import_certificate as cert
from examples.vllm_llm.pairs import Refusal
from tests.test_import_certificate import envelope, fixture_session
from tests.test_qwen300_hour import host_text


@pytest.mark.parametrize('mutation', ['none', 'tokens', 'teacher', 'missing', 'duplicate', 'extra_run', 'nan', 'dsl', 'graph'])
def test_decode_population_and_correctness(mutation):
    text = host_text()+'CHECK-DSL OK: fixture\n'
    if mutation == 'tokens': text = text.replace('32/32', '31/32')
    if mutation == 'teacher': text = text.replace('match HF greedy', 'match HF greedy (teacher-forced argmax)')
    if mutation == 'missing': text = '\n'.join(l for l in text.splitlines() if '"step": 1024' not in l)
    if mutation == 'duplicate': text += 'HOUR-WALL {"run":0,"step":1024,"ms":4.18}\n'
    if mutation == 'extra_run': text += 'run 1: 32/32 generated tokens match HF greedy\n'
    if mutation == 'nan': text = text.replace('"ms": 4.18', '"ms": NaN')
    if mutation == 'dsl': text = text.replace('CHECK-DSL OK', 'CHECK-DSL FAILED')
    if mutation == 'graph': text = text.replace('ETX megakernel', 'unfused HIP graph')
    if mutation == 'none': assert b.decode_samples(text) == [4.18]*31
    else:
        with pytest.raises((Refusal, ValueError)): b.decode_samples(text)


@pytest.mark.parametrize('failure', [None, 'missing_input', 'build', 'imports', 'tokens', 'source', 'identity', 'timeout'])
def test_standalone_pipeline_fail_closed(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b, 'prerequisites', lambda model, batch=1: None)
    if failure == 'missing_input':
        def missing(model, batch=1): raise Refusal('missing input: golden_c1024.txt')
        monkeypatch.setattr(b, 'prerequisites', missing)
    monkeypatch.setattr(b.subprocess, 'check_output', lambda *args, **kwargs: 'a'*40)
    seen = []
    reads = [0]
    def hashes():
        reads[0] += 1
        return {'source.py': ('b' if failure == 'source' and reads[0] > 1 else 'a')*64}
    monkeypatch.setattr(b, 'source_hashes', hashes)
    fixture = fixture_session(tmp_path/'fixture', count=3)
    identity = json.loads((fixture/'build_identity.json').read_text())
    identity['hipcc_flags'] = b.BASE_FLAGS
    identity['overrides'] = dict(b.KEPT_ENV)
    (fixture/'build_identity.json').write_text(json.dumps(identity))
    # Restamp synthetic evidence after adding the flags; real checker/validator
    # still owns coverage and byte-identity decisions in this integration test.
    for pos in (0, 1040):
        p = fixture/f'check_p{pos}.txt'
        body = '\n'.join(p.read_text().splitlines()[1:-1])
        p.write_text(envelope(identity, body))
    def verified(build, model):
        if failure == 'identity' and len(seen) > 3:
            raise Refusal('build/model fingerprint changed')
        return identity
    monkeypatch.setattr(cert, 'verified_build', verified)
    def export(build, model, archive):
        archive.mkdir()
        for name in ('build_identity.json', 'vl_plan.txt', 'imports.json'):
            (archive/name).write_bytes((fixture/name).read_bytes())
    monkeypatch.setattr(cert, 'export_build', export)
    def command(argv, log, env, timeout):
        seen.append(log.stem)
        if log.stem == 'hardware': return 'AMD Instinct MI300X VF'
        if log.stem == 'build':
            if failure == 'build': raise Refusal('build: exit 1')
            (log.parent/'hipcc_command.json').write_text('["hipcc", "-O3"]')
            (log.parent/'vl_plan.txt').write_bytes((fixture/'vl_plan.txt').read_bytes())
            (log.parent/'imp').mkdir()
            (log.parent/'imp/imports.json').write_bytes((fixture/'imports.json').read_bytes())
            return ''
        if log.stem.startswith('check_'):
            text = (fixture/(log.stem+'.txt')).read_text()
            if failure == 'imports' and log.stem == 'check_p1040':
                text = '\n'.join(l for l in text.splitlines() if not l.startswith('CHECK-INSTANCE '))
            return text
        if failure == 'timeout': raise Refusal('decode: timeout')
        text = host_text()+'CHECK-DSL OK: fixture\n'
        if failure == 'tokens': text = text.replace('32/32', '31/32')
        return envelope(identity, text)
    monkeypatch.setattr(b, 'command', command)
    assert b.main([str(tmp_path/'model')]) == (0 if failure is None else 1)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith('BUILD-IDENTITY ')
    assert out[1].startswith('CORRECTNESS ')
    if failure:
        assert out[1].startswith('CORRECTNESS FAIL ')
        assert out[-1].startswith('BENCH FAIL ')
        assert not any(l.startswith(('TIMING ', 'BENCH PASS ')) for l in out)
        if failure in ('missing_input', 'build', 'imports'): assert 'decode' not in seen
    else:
        assert 'tokens=32/32 imports=6/6' in out[1]
        assert out[2].startswith('TIMING median_decode_wall_ms=4.180000 tokens_per_s=239.23 ')
        assert 'steps=31' in out[2] and 'spread_ms=0.000000' in out[2]
        assert out[-1] == 'BENCH PASS 4.180000 239.23'
        assert seen == ['hardware', 'build', 'check_p0', 'check_p1040', 'decode']


def test_kept_recipe_clears_inherited_experiments(monkeypatch, tmp_path):
    for key in ('ETX_VL_UC', 'ETX_VL_ABLATE', 'ETX_INBODY_BUILD', 'ETX_SESS_MAX_SECONDS', 'DSL_ABLATE_WSUM'):
        monkeypatch.setenv(key, '1')
    env = b.environment(tmp_path)
    assert not any(k in env for k in ('ETX_VL_UC', 'ETX_VL_ABLATE', 'ETX_INBODY_BUILD', 'ETX_SESS_MAX_SECONDS', 'DSL_ABLATE_WSUM'))
    for key, value in b.KEPT_ENV.items(): assert env[key] == value
    assert env['ETX_HIPCC_FLAGS'] == b.BASE_FLAGS


def test_cpu_box_is_a_clear_failure(monkeypatch, capsys):
    exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda p: False if str(p) == '/dev/kfd' else exists(p))
    assert b.main([]) == 1
    out = capsys.readouterr().out
    assert 'BENCH FAIL no AMD GPU available' in out
    assert 'Traceback' not in out and 'TIMING ' not in out


def test_command_retains_failed_output_and_bounds_wait(tmp_path):
    import sys
    for name, script, timeout in [('failed', 'print("full evidence", flush=True); raise SystemExit(3)', 10),
                                   ('hung', 'import time; print("stuck", flush=True); time.sleep(30)', .1)]:
        log = tmp_path/(name+'.log')
        with pytest.raises(Refusal, match='exit 3' if name == 'failed' else 'timeout'):
            b.command([sys.executable, '-c', script], log, b.environment(tmp_path), timeout)
        assert ('full evidence' if name == 'failed' else 'stuck') in log.read_text()
