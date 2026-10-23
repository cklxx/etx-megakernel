"""External sealed-session acceptance; synthetic CPU children are not GPU evidence."""
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path
import json
import sys
import tarfile

import pytest
from etx.external import (SealedPreparationRequest, SealedSessionRequest,
                          prepare_sealed_session, run_sealed_session)
from etx.testing import (synthetic_sealed_stages, interrupted_sealed_export,
                         sealed_cli_list_probe, sealed_historical_replay_probe, sealed_resume_probe)

EXPECTED = {f'G{i:02}' for i in range(1, 14)}
MODULES = {'G01': 'router', 'G02': 'columns', 'G03': 'columns', 'G04': 'routing',
           'G05': 'columns', 'G06': 'issue_order', 'G07': 'issue_order',
           'G08': 'issue_order', 'G09': 'row_ops', 'G10': 'row_ops',
           'G11': 'row_ops', 'G12': 'validation', 'G13': 'columns'}


def config(tmp_path, records=('G09',)):
    anchor = tmp_path / 'explicitly-synthetic-cpu-input.txt'
    anchor.write_text('Synthetic CPU operational fixture, not native input or performance evidence.\n')
    seal = {str(anchor): hashlib.sha256(anchor.read_bytes()).hexdigest()}
    stages = {}
    for record in records:
        module = MODULES[record]
        cmd = [sys.executable, '-m', 'examples.experiment_program.' + module]
        if module == 'columns':
            cmd += ['--record', record, '--dry-run']
        elif module == 'issue_order':
            cmd += ['--record', record, 'preflight']
        elif module == 'row_ops':
            cmd += ['execute', '--dry-run']
        else:
            cmd += ['preflight']
        stages[record] = dict(preflight=cmd, command=cmd, input_files=seal,
                              timeout_seconds=2, decision_file=record + '.json')
    path = tmp_path / 'synthetic-session.json'
    path.write_text(json.dumps(dict(base='b573664faa4e7988dff353bc03aca4ce8561a729',
                                    source_snapshot=seal, budget_seconds=10, records=stages)))
    return path


def test_real_cli_lists_exact_thirteen_disabled_records():
    rows = run_sealed_session(SealedSessionRequest('list'))['records']
    assert sealed_cli_list_probe()['records'] == rows
    assert len(rows) == 13 and {r['id'] for r in rows} == EXPECTED
    assert all(r['enabled'] is False and r['module'] == MODULES[r['id']] for r in rows)


def test_all_thirteen_configuration_dryrun_has_no_output_side_effect(tmp_path):
    path = config(tmp_path, tuple(sorted(EXPECTED)))
    out = tmp_path / 'never-created'
    value = run_sealed_session(SealedSessionRequest('dry-run', config=str(path), output=str(out)))
    assert set(value['selected']) == EXPECTED and value['enabled'] is False
    assert not out.exists()


@pytest.mark.parametrize('dangerous', [
    ['execute'], ['execute', '--device-execution-authorized'], ['execute', '--execute'], ['build'],
])
def test_device_or_mutating_preflight_actions_refused(tmp_path, dangerous):
    path = config(tmp_path)
    value = json.loads(path.read_text())
    value['records']['G09']['preflight'] = [sys.executable, '-m', 'examples.experiment_program.row_ops'] + dangerous
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run_sealed_session(SealedSessionRequest('dry-run', config=str(path), output=str(tmp_path / 'unused')))


def test_actual_columns_missing_capture_is_not_preflight_pass(tmp_path):
    path = config(tmp_path, ('G02',))
    out = tmp_path / 'actual-safe-columns-preflight'
    result = run_sealed_session(SealedSessionRequest('preflight', config=str(path), output=str(out)))
    state = json.loads((out / 'session.json').read_text())
    assert result.to_dict() == state
    assert state['stages']['G02']['status'] in ('MISSING_PREREQUISITE', 'BLOCKED_NATIVE_CAPTURE')
    report = json.loads((out / 'G02-preflight.log').read_text())
    assert report['status'] == 'BLOCKED_NATIVE_CAPTURE' and not report['enabled']


def test_cpu_preflight_resume_preserves_completed_log(tmp_path):
    path = config(tmp_path)
    out = tmp_path / 'synthetic-resume'
    with synthetic_sealed_stages() as calls:
        run_sealed_session(SealedSessionRequest('preflight', config=str(path), output=str(out)))
        log = out / 'G09-preflight.log'
        before = log.read_bytes()
        run_sealed_session(SealedSessionRequest('preflight', config=str(path), output=str(out), resume=True))
    assert len(calls) == 1 and log.read_bytes() == before
    assert json.loads((out / 'session.json').read_text())['stages']['G09']['status'] == 'PREFLIGHT_PASS'
    assert json.loads(before)['synthetic_cpu_only'] is True


def test_run_requires_explicit_activation_before_any_child(tmp_path):
    path = config(tmp_path)
    out = tmp_path / 'disabled'
    with synthetic_sealed_stages(forbid=True):
        with pytest.raises(ValueError, match='disabled'):
            run_sealed_session(SealedSessionRequest('run', config=str(path), output=str(out)))
    assert not out.exists()


def test_interrupted_local_export_recovers_without_overwriting_record(tmp_path):
    record = tmp_path / 'record'
    record.mkdir()
    (record / 'raw.log').write_text('SYNTHETIC retained raw bytes\n')
    destination = record.with_suffix('.tar')
    request = SealedSessionRequest('export', output=str(record))
    with interrupted_sealed_export():
        with pytest.raises(OSError, match='interrupted'):
            run_sealed_session(request)
    assert destination.with_name(destination.name + '.owner').is_file() and not destination.exists()
    digest = run_sealed_session(request)['sha256']
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    with tarfile.open(destination) as archive:
        assert archive.extractfile('raw.log').read() == (record / 'raw.log').read_bytes()


def test_export_refuses_symlink_and_unowned_archive(tmp_path):
    record = tmp_path / 'record'
    record.mkdir()
    personal = tmp_path / 'personal-sentinel'
    personal.write_text('untouched')
    (record / 'escape').symlink_to(personal)
    with pytest.raises(ValueError, match='symlink'):
        run_sealed_session(SealedSessionRequest('export', output=str(record)))
    # A distinct archive has no owner left behind by the refused symlink export.
    unowned_record = tmp_path / 'unowned'
    unowned_record.mkdir()
    unowned = unowned_record.with_suffix('.tar')
    unowned.write_text('owned by nobody here')
    with pytest.raises(ValueError, match='unowned'):
        run_sealed_session(SealedSessionRequest('export', output=str(unowned_record)))
    assert personal.read_text() == 'untouched' and unowned.read_text() == 'owned by nobody here'


@pytest.mark.parametrize('exit_code,write_decision', [(7, True), (0, False)])
def test_cpu_failure_or_missing_decision_stops_later_stages(tmp_path, exit_code, write_decision):
    path = config(tmp_path, ('G09', 'G06'))
    out = tmp_path / 'synthetic-stop'
    with synthetic_sealed_stages(write_decision=write_decision, exit_code=exit_code) as calls:
        with pytest.raises(ValueError):
            run_sealed_session(SealedSessionRequest('run', config=str(path), output=str(out), execute=True))
    state = json.loads((out / 'session.json').read_text())
    assert len(calls) == 1 and state['status'] == 'NOT_COMPLETE' and 'G09' not in state['stages']
    assert (out / 'G06-run.log').is_file() and out.with_suffix('.tar').is_file()


def test_real_historical_retained_record_replay_is_not_fresh_performance():
    result = sealed_historical_replay_probe()
    assert result['historical_only'] is True and result['fresh_performance_claim'] is False
    assert result['tokens_per_invocation'] == 32 and len(result['recomputed']['gains_ns']) == 5
    assert result['recomputed']['keep'] is False and result['retained_verdict'] == 'NO-GO'


def test_cpu_local_miss_never_launches_model_stage(tmp_path):
    path = config(tmp_path, ('G09', 'G12'))
    out = tmp_path / 'synthetic-local-miss'
    with synthetic_sealed_stages() as calls:
        run_sealed_session(SealedSessionRequest('run', config=str(path), output=str(out), execute=True))
    state = json.loads((out / 'session.json').read_text())
    assert len(calls) == 1 and state['stages']['G12']['status'] == 'NOT_ADMITTED'
    assert not (out / 'G12-run.log').exists()
    decision = json.loads((out / 'G09.json').read_text())
    assert decision['synthetic_cpu_only'] is True and decision['admitted'] is False


def test_synthetic_dispatch_restores_nested_and_exceptional_contexts(tmp_path):
    path = config(tmp_path)
    with synthetic_sealed_stages() as outer:
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            with synthetic_sealed_stages() as inner:
                run_sealed_session(SealedSessionRequest('preflight', config=str(path),
                                                        output=str(tmp_path / 'inner')))
                raise RuntimeError('synthetic interruption')
        run_sealed_session(SealedSessionRequest('preflight', config=str(path),
                                                output=str(tmp_path / 'outer')))
    assert len(inner) == len(outer) == 1
    # After both tokens reset, the actual safe columns child must run, not the
    # synthetic PASS child. This is a missing-input gate, not device execution.
    actual = config(tmp_path, ('G02',))
    result = run_sealed_session(SealedSessionRequest('preflight', config=str(actual),
                                                    output=str(tmp_path / 'restored')))
    assert result['stages']['G02']['status'] == 'MISSING_PREREQUISITE'
    evidence = json.loads((tmp_path / 'restored' / 'G02-preflight.log').read_text())
    assert evidence['status'] == 'BLOCKED_NATIVE_CAPTURE' and not evidence['enabled']
    assert 'synthetic_cpu_only' not in evidence


def test_preparation_seals_thirteen_disabled_commands_without_running_them(tmp_path):
    destination = tmp_path / 'prepared.json'
    out = tmp_path / 'future-run'
    request = SealedPreparationRequest(inventory={'staging_root': str(tmp_path / 'staged')},
                                       destination=str(destination), output=str(out))
    with synthetic_sealed_stages(forbid=True):
        prepared = prepare_sealed_session(request)
        dry_run = run_sealed_session(SealedSessionRequest('dry-run', config=str(destination), output=str(out)))
    assert set(prepared['records']) == EXPECTED
    assert json.loads(destination.read_text()) == prepared.to_dict()
    sealed_paths = prepared['source_snapshot']
    assert any(Path(path).as_posix().endswith('/etx/build/toolchain.py') for path in sealed_paths)
    assert any(Path(path).as_posix().endswith('/examples/experiment_program/model_build.py') for path in sealed_paths)
    assert any(Path(path).as_posix().endswith('/examples/vllm_llm/build.sh') for path in sealed_paths)
    assert set(dry_run['selected']) == EXPECTED and dry_run['enabled'] is False
    for record, stage in prepared['records'].items():
        assert stage['input_files'] and any(digest == '0' * 64 for digest in stage['input_files'].values())
        assert stage['decision_file'].startswith(record + '/')
        assert '--execute' not in stage['preflight'] and '--device-execution-authorized' not in stage['preflight']
    assert not out.exists()
    before = destination.read_bytes()
    with pytest.raises(ValueError):
        prepare_sealed_session(request)
    assert destination.read_bytes() == before


def test_sealed_requests_and_results_are_detached(tmp_path):
    inventory = {'staging_root': str(tmp_path / 'staged'), 'options': {'fold_decision_active': False}}
    request = SealedPreparationRequest(inventory, str(tmp_path / 'config.json'), str(tmp_path / 'out'))
    inventory['options']['fold_decision_active'] = True
    assert request.inventory['options']['fold_decision_active'] is False
    with pytest.raises(FrozenInstanceError):
        request.output = str(tmp_path / 'elsewhere')
    records = ['G09']
    session = SealedSessionRequest('list', records=records)
    records.append('G12')
    assert session.records == ('G09',)
    result = run_sealed_session(session)
    detached = result['records']
    detached[0]['enabled'] = True
    assert result['records'][0]['enabled'] is False


@pytest.mark.parametrize('values', [
    {'action': 'unknown'}, {'action': 'run'}, {'action': 'list', 'schema_version': True},
    {'action': 'list', 'records': 'G01'}, {'action': 'list', 'records': ['G14']},
    {'action': 'list', 'records': ['G01', 'G01']}, {'action': 'list', 'resume': 1},
    {'action': 'list', 'execute': 1}, {'action': 'list', 'output': '\0'},
])
def test_sealed_request_validation_has_no_side_effect(values):
    with pytest.raises(ValueError):
        SealedSessionRequest(**values)


def test_source_mutation_refused_before_output_creation(tmp_path):
    path = config(tmp_path)
    value = json.loads(path.read_text())
    source = Path(next(iter(value['source_snapshot'])))
    source.write_text('changed synthetic source')
    out = tmp_path / 'must-not-exist'
    with synthetic_sealed_stages(forbid=True):
        with pytest.raises(ValueError, match='sealed file'):
            run_sealed_session(SealedSessionRequest('preflight', str(path), str(out)))
    assert not out.exists()



def prepare_config(inventory, destination, output):
    return prepare_sealed_session(SealedPreparationRequest(
        inventory, str(destination), str(output))).to_dict()


def test_export_recovery_and_tamper(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'raw.log').write_text('unfiltered\n')
    run_sealed_session(SealedSessionRequest('export',output=str(root)))
    run_sealed_session(SealedSessionRequest('export',output=str(root)))
    import hashlib
    seal={str(root/'raw.log'):hashlib.sha256((root/'raw.log').read_bytes()).hexdigest()}
    sealed_resume_probe({'config_sha256':'a','source_revision':'b','sealed':seal},'a','b')
    for c,s in [('other','b'),('a','other')]:
        with pytest.raises(ValueError):sealed_resume_probe({'config_sha256':'a','source_revision':'b','sealed':seal},c,s)
    (root/'raw.log').write_text('changed')
    with pytest.raises(ValueError):sealed_resume_probe({'config_sha256':'a','source_revision':'b','sealed':seal},'a','b')


def test_staging_emits_all_thirteen_real_commands_and_missing_prerequisites(tmp_path):
    result=prepare_config({'staging_root':str(tmp_path/'staged')},tmp_path/'session.json',tmp_path/'future-run')
    assert set(result['records'])=={f'G{i:02}' for i in range(1,14)}
    for record,stage in result['records'].items():
        assert stage['input_files'] and any(d=='0'*64 for d in stage['input_files'].values())
        assert stage['decision_file'].startswith(record+'/')
        assert '--execute' not in stage['preflight'] and '--device-execution-authorized' not in stage['preflight']
    assert not (tmp_path/'future-run').exists()
    with pytest.raises(ValueError):prepare_config({'staging_root':str(tmp_path)},tmp_path/'session.json',tmp_path/'future-run')


def test_staging_registered_column_program_and_conditional_validation_root(tmp_path):
    program=tmp_path/'columns.json';build=tmp_path/'actual-columns-build';build.mkdir();(build/'identity.json').write_text('{}')
    program.write_text(json.dumps(dict(schema='etx.columns.program.v1',record='G02',build=str(build),capture_root=str(tmp_path/'captures'))))
    out=tmp_path/'future'
    value=prepare_config({'staging_root':str(tmp_path/'staged'),'records':{'G02':{'program':str(program)}}},tmp_path/'session.json',out)
    stage=value['records']['G02']
    assert '--program' in stage['command'] and str(program) in stage['input_files']
    assert stage['decision_file']=='G02/program-report.json'
    assert str(build/'identity.json') in stage['input_files']
    assert stage['admission']['build']==str(build)
    with pytest.raises(ValueError,match='program build'):
        prepare_config({'staging_root':str(tmp_path/'staged'),'records':{'G02':{'program':str(program),'build':str(tmp_path/'wrong')}}},tmp_path/'wrong-session.json',out)
    command=value['records']['G12']['command']
    assert command[command.index('--micro-session-directory')+1]==str(out)


def test_model_validation_aggregate_budget_is_distinct_from_native_call_deadline(tmp_path):
    value=prepare_config({'staging_root':str(tmp_path/'staged'),'records':{'G12':{'timeout_seconds':12000}}},tmp_path/'session.json',tmp_path/'future')
    assert run_sealed_session(SealedSessionRequest('dry-run',str(tmp_path/'session.json'),str(tmp_path/'future')))['stages']['G12']['timeout_seconds']==12000
    value['records']['G01']['timeout_seconds']=12000
    (tmp_path/'invalid.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='timeout'):run_sealed_session(SealedSessionRequest('dry-run',str(tmp_path/'invalid.json'),str(tmp_path/'future')))
