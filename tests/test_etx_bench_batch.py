"""CPU-only B2/B4/B8 benchmark boundaries: per-batch recipe, certificate and gates. Synthetic output is never GPU evidence."""
import hashlib
import json

import pytest

from examples.vllm_llm import batch_correctness as bc
from examples.vllm_llm import bench as b
from examples.vllm_llm import import_certificate as cert
from examples.vllm_llm.pairs import Refusal
from tests.test_import_certificate import envelope

PER_REQUEST = 3


def plan(batch):
    text = f'# fixture\ntotal 100 parts 5 maxlen 1072 requests {batch}\n'
    rows = [(n, n % batch) for n in range(PER_REQUEST * batch)]
    return text + ''.join(f'{n} rms_norm_2d {n // batch} in_norm 1 1 1 0 1 0 0 0 1 112 -1 {req} 1\n' for n, req in rows)


def check_text(identity, expected, pos, drop=None):
    rows = [dict(r, position=pos, bytes=128, result='identical', first_byte=-1) for r in expected if r['id'] != drop]
    body = ''.join('CHECK-INSTANCE ' + json.dumps(r) + '\n' for r in rows)
    body += (f'CHECK-IMPORTED checked={len(rows)} planned={len(rows)} replicas={len(rows)} position={pos}\n'
             'CHECK OK: 0 launch(es) differ')
    return envelope(identity, body)


def decode_text(batch, cost=9.5, matched=None):
    matched = matched or [32] * batch
    text = 'ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=\nmode: ETX megakernel (vLLM kernels imported)\n'
    for t in range(1024, 1055):
        text += 'HOUR-WALL ' + json.dumps(dict(run=0, step=t, ms=cost)) + '\n'
    text += ''.join(f'run 0 request {r}: {m}/32 generated tokens match HF greedy\n' for r, m in enumerate(matched))
    text += (f'RESULT model=qwen requests={batch} matched={min(matched)} gen=32 median_ms=9.000 wall_ms={cost:.3f} '
             f'tokens_per_s=1 matched_each={",".join(map(str, matched))}\n')
    return text


def coverage_text(batch, positions=1055):
    return ''.join(f'CHECK-COVERAGE OK request {r}: {positions} position(s), 0 wrong beyond ties, {r % 2} tie-tolerated '
                   '(HF margin <= 0.5)\n' for r in range(batch))


def test_request_goldens_are_the_qualified_names():
    assert cert.request_goldens(1) == ['golden_c1024.txt']
    for batch in bc.BATCHES:
        assert cert.request_goldens(batch) == bc.golden_names(batch)


def test_recipes():
    env, flags = b.recipe(1)
    assert env == b.KEPT_ENV and flags == b.BASE_FLAGS                     # the B1 kept build is unchanged
    for batch in (2, 4, 8):
        env, flags = b.recipe(batch)
        assert env['ETX_VL_REQUESTS'] == str(batch) and env['ETX_VL_BATCH'] == '0' and env['ETX_VL_NB'] == '1'
        assert env['ETX_VL_CHECK'] == '1' and flags == b.BASE_FLAGS
        env, flags = b.recipe(batch, 'plain')
        assert env == bc.recipe(batch) and flags == b.REPORT_FLAGS
    with pytest.raises(Refusal):
        b.recipe(1, 'plain')
    with pytest.raises(Refusal):
        b.recipe(3)


def test_multi_request_fingerprint_binds_every_request_reference(tmp_path, monkeypatch):
    build = tmp_path / 'b'; model = tmp_path / 'm'
    build.mkdir(); model.mkdir(); (build / 'imp').mkdir()
    for name in ('run', 'imports.bc', 'plan_data.h', 'plan.json', 'hipcc.log', 'megakernel_d0.hip', 'dsl_tiles.hip',
                 'imp_tiles.hip', 'compiler_version.txt', 'imp/imports.json'):
        (build / name).write_text(name)
    (model / 'vl.manifest').write_text('m')
    (build / 'vl_plan.txt').write_text(plan(1))
    (model / 'golden_c1024.txt').write_text('g')
    one = cert.fingerprint(build, model)
    assert not any(k.startswith('request_') for k in one)                   # one-request identities are unchanged
    (build / 'vl_plan.txt').write_text(plan(4))
    with pytest.raises(FileNotFoundError):                                    # a missing request reference refuses
        cert.fingerprint(build, model)
    for name in bc.golden_names(4):
        (model / name).write_text(name)
        (model / name.replace('golden', 'prefill', 1)).write_text(name)
    monkeypatch.setenv('ETX_VL_CHECK', '1')
    identity = cert.write_build(build, model)
    assert {k for k in identity['files'] if k.startswith('request_')} == (
        {f'request_golden/{n}' for n in bc.golden_names(4)}
        | {f'request_prefill/{n.replace("golden", "prefill", 1)}' for n in bc.golden_names(4)})
    (model / 'prefill_c1024_r291.txt').write_text('changed')
    with pytest.raises(Refusal, match='fingerprint changed'):
        cert.verified_build(build, model)


def test_capture_names_each_request_golden(tmp_path, monkeypatch):
    build = tmp_path / 'b'; build.mkdir()
    (build / 'vl_plan.txt').write_text(plan(4))
    identity = dict(files={})
    monkeypatch.setattr(cert, 'verified_build', lambda build, model: identity)
    seen = []
    monkeypatch.setattr(cert.subprocess, 'call', lambda argv: seen.append(argv) or 0)
    assert cert.capture(build, tmp_path / 'm', ['--repeat', '1']) == 0
    assert seen[0][-8:] == ['--golden', 'golden_c1024.txt', '--golden1', 'golden_c1024_r97.txt',
                            '--golden2', 'golden_c1024_r194.txt', '--golden3', 'golden_c1024_r291.txt']
    with pytest.raises(Refusal, match='unsupported'):                        # the caller cannot pick goldens
        cert.capture(build, tmp_path / 'm', ['--golden1', 'other.txt'])


@pytest.mark.parametrize('mutation', ['none', 'tokens', 'missing_request', 'result', 'teacher', 'steps'])
def test_batch_decode_gate(mutation):
    batch = 4
    text = decode_text(batch)
    if mutation == 'tokens': text = decode_text(batch, matched=[32, 32, 31, 32])
    if mutation == 'missing_request': text = '\n'.join(l for l in text.splitlines() if 'request 2:' not in l)
    if mutation == 'result': text = text.replace('requests=4', 'requests=2')
    if mutation == 'teacher': text = text.replace('match HF greedy\n', 'match HF greedy (teacher-forced argmax)\n', 1)
    if mutation == 'steps': text = '\n'.join(l for l in text.splitlines() if '"step": 1054' not in l)
    if mutation == 'none':
        assert b.decode_samples(text, batch) == [9.5] * 31
    else:
        with pytest.raises((Refusal, ValueError)):
            b.decode_samples(text, batch)


def test_batch_coverage_gate():
    assert b.coverage(coverage_text(4), 4) == [0, 1, 0, 1]
    with pytest.raises(Refusal):
        b.coverage(coverage_text(4, positions=1054), 4)
    with pytest.raises(Refusal):
        b.coverage('\n'.join(coverage_text(4).splitlines()[:3]), 4)


def pipeline(tmp_path, monkeypatch, batch, failure=None, recipe='kept'):
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b, 'prerequisites', lambda model, batch=1: None)
    monkeypatch.setattr(b.subprocess, 'check_output', lambda *args, **kwargs: 'a' * 40)
    monkeypatch.setattr(b, 'source_hashes', lambda: {'source.py': 'a' * 64})
    overrides, flags = b.recipe(batch, recipe)
    plan_text = plan(batch)
    identity = dict(schema=1, check_enabled=True, compiler='AMD clang version fixture', hipcc_flags=flags,
                    overrides=dict(overrides) if failure != 'recipe' else dict(overrides, ETX_VL_REQUESTS='1'),
                    files={k: 'a' * 64 for k in ('executable', 'imports', 'plan_header', 'plan_json', 'weights_manifest',
                                                   'golden', 'compiler_output', 'device_source/megakernel_d0.hip',
                                                   'device_source/dsl_tiles.hip', 'device_source/imp_tiles.hip')})
    imports = json.dumps({'rms_norm_2d': {'block': [256, 1, 1]}})
    identity['files']['plan'] = hashlib.sha256(plan_text.encode()).hexdigest()
    identity['files']['import_metadata'] = hashlib.sha256(imports.encode()).hexdigest()
    expected = cert.expected_instances(plan_text, json.loads(imports))
    monkeypatch.setattr(cert, 'verified_build', lambda build, model: identity)
    def export(build, model, archive):
        archive.mkdir()
        (archive / 'build_identity.json').write_text(json.dumps(identity))
        (archive / 'vl_plan.txt').write_text(plan_text)
        (archive / 'imports.json').write_text(imports)
    monkeypatch.setattr(cert, 'export_build', export)
    seen = []
    def command(argv, log, env, timeout):
        seen.append((log.stem, timeout))
        assert env['ETX_HIPCC_FLAGS'] == flags and all(env[k] == v for k, v in overrides.items())
        if log.stem == 'hardware': return 'AMD Instinct MI300X VF'
        if log.stem == 'build':
            (log.parent / 'hipcc_command.json').write_text('["hipcc", "-O3"]')
            (log.parent / 'vl_plan.txt').write_text(plan_text if failure != 'plan' else plan(1))
            (log.parent / 'imp').mkdir()
            (log.parent / 'imp/imports.json').write_text(imports)
            return ''
        if log.stem.startswith('check_'):
            pos = int(log.stem[len('check_p'):])
            drop = expected[-1]['id'] if failure == 'imports' and pos == 1040 else None
            return check_text(identity, expected, pos, drop)
        if log.stem == 'coverage':
            return envelope(identity, coverage_text(batch, 1054 if failure == 'coverage' else 1055))
        return envelope(identity, decode_text(batch, matched=[32] * (batch - 1) + [31 if failure == 'tokens' else 32]))
    monkeypatch.setattr(b, 'command', command)
    out_json = tmp_path / 'result.json'
    rc = b.main(['--batch', str(batch), '--recipe', recipe, '--json', str(out_json), str(tmp_path / 'model')])
    return rc, seen, json.loads(out_json.read_text())


@pytest.mark.parametrize('batch', [2, 4, 8])
@pytest.mark.parametrize('recipe', ['kept', 'plain'])
def test_batch_pipeline_times_only_after_every_gate(tmp_path, monkeypatch, capsys, batch, recipe):
    rc, seen, result = pipeline(tmp_path, monkeypatch, batch, recipe=recipe)
    out = capsys.readouterr().out.splitlines()
    assert rc == 0, out
    assert [s for s, _ in seen] == ['hardware', 'build', 'check_p0', 'check_p1040', 'coverage', 'decode']
    assert dict(seen)['check_p0'] == b.CHECK_TIMEOUT[batch]
    assert out[0].startswith('BUILD-IDENTITY ')
    record = json.loads(out[0].split(' ', 1)[1])
    assert record['conditions'] == dict(gpu='MI300X', batch=batch, context=1024, greedy_tokens=32, recipe=recipe)
    assert out[1].startswith(f'CORRECTNESS PASS batch={batch} recipe={recipe} tokens={32 * batch}/{32 * batch} ')
    assert f'imports={2 * PER_REQUEST * batch}/{2 * PER_REQUEST * batch} ({PER_REQUEST}/request' in out[1]
    assert f'coverage={batch}x1055 positions' in out[1]
    tps = batch * 1000 / 9.5
    assert out[2] == (f'TIMING batch={batch} recipe={recipe} median_decode_wall_ms=9.500000 tokens_per_s={tps:.2f} '
                      f'ms_per_token={9.5 / batch:.6f} spread_ms=0.000000 spread=max-min steps=31')
    assert out[-1] == f'BENCH PASS 9.500000 {tps:.2f} batch={batch} recipe={recipe}'
    assert result['status'] == 'PASS' and result['batch'] == batch and result['median_decode_wall_ms'] == 9.5
    assert result['tokens_per_s'] == pytest.approx(tps) and len(result['samples_ms']) == 31
    assert set(result['byte_identity_per_request']) == {'0', '1040'}
    assert result['byte_identity_per_request']['1040']['instances_per_request'] == PER_REQUEST


@pytest.mark.parametrize('failure', ['tokens', 'imports', 'coverage', 'recipe', 'plan'])
def test_batch_failure_reports_no_timing(tmp_path, monkeypatch, capsys, failure):
    rc, seen, result = pipeline(tmp_path, monkeypatch, 4, failure)
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert result['status'] == 'FAIL' and result['timing'] is None and 'median_decode_wall_ms' not in result
    assert not any(l.startswith(('TIMING ', 'BENCH PASS ')) for l in out)
    assert out[-1].startswith('BENCH FAIL ')
    if failure in ('imports', 'recipe', 'plan'):
        assert 'decode' not in [s for s, _ in seen]
