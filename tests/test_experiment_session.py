"""Synthetic process/session faults only. Never native or performance evidence."""
import json
from pathlib import Path
import sys
import pytest
from examples.experiment_program.session import run_logged, export_record, verify_resume, eligible


def test_full_log_and_failure(tmp_path):
    log=tmp_path/'full.log'
    assert run_logged([sys.executable,'-c','import sys;print("raw stdout");print("raw stderr",file=sys.stderr)'],log,seconds=3)==0
    assert 'raw stdout' in log.read_text() and 'raw stderr' in log.read_text()
    assert run_logged([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'fail.log',seconds=3)==7


def test_deadline_is_not_shortened_success(tmp_path):
    with pytest.raises(TimeoutError):run_logged([sys.executable,'-c','import time;time.sleep(2)'],tmp_path/'timeout.log',seconds=.02)
    assert (tmp_path/'timeout.log').is_file()


def test_dependency_and_validation_gates():
    assert eligible('G05',{'G04':dict(complete=True)}, {})
    assert not eligible('G05',{'G04':dict(complete=False)}, {})
    assert not eligible('G12',{'G01':dict(admitted=False)}, {})
    assert eligible('G12',{'G01':dict(admitted=True)}, {})
    assert not eligible('G13',{'G02':dict(admitted=True)}, {})
    assert eligible('G13',{'G02':dict(admitted=True)}, {'coalescing_decision_active':True})


def test_export_recovery_and_tamper(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'raw.log').write_text('unfiltered\n')
    export_record(root,tmp_path/'record.tar')
    export_record(root,tmp_path/'record.tar')
    import hashlib
    seal={str(root/'raw.log'):hashlib.sha256((root/'raw.log').read_bytes()).hexdigest()}
    verify_resume({'config_sha256':'a','source_revision':'b','sealed':seal},'a','b')
    for c,s in [('other','b'),('a','other')]:
        with pytest.raises(ValueError):verify_resume({'config_sha256':'a','source_revision':'b','sealed':seal},c,s)
    (root/'raw.log').write_text('changed')
    with pytest.raises(ValueError):verify_resume({'config_sha256':'a','source_revision':'b','sealed':seal},'a','b')


def test_self_authored_winner_cannot_unlock_model(tmp_path):
    from examples.experiment_program.session import decision_for_dependencies
    p=tmp_path/'decision.json';p.write_text(json.dumps(dict(record='G01',admitted=True)))
    with pytest.raises(ValueError,match='admission'):decision_for_dependencies('G01',{},p)
    p.write_text(json.dumps(dict(record='G06',status='MICRO_MISS')))
    assert decision_for_dependencies('G06',{},p)['admitted'] is False


def test_capture_observation_never_uses_optimization_admission(tmp_path):
    from examples.experiment_program.session import decision_for_dependencies
    p=tmp_path/'capture.json';p.write_text(json.dumps(dict(record='G04',complete=True)))
    with pytest.raises((ValueError,KeyError)):decision_for_dependencies('G04',{},p)


def test_staging_emits_all_thirteen_real_commands_and_missing_prerequisites(tmp_path):
    from examples.experiment_program.stage import prepare
    result=prepare({'staging_root':str(tmp_path/'staged')},tmp_path/'session.json',tmp_path/'future-run')
    assert set(result['records'])=={f'G{i:02}' for i in range(1,14)}
    for record,stage in result['records'].items():
        assert stage['input_files'] and any(d=='0'*64 for d in stage['input_files'].values())
        assert stage['decision_file'].startswith(record+'/')
        assert '--execute' not in stage['preflight'] and '--device-execution-authorized' not in stage['preflight']
    assert not (tmp_path/'future-run').exists()
    with pytest.raises(ValueError):prepare({'staging_root':str(tmp_path)},tmp_path/'session.json',tmp_path/'future-run')


def test_runner_binds_only_owned_output_prefixes(tmp_path):
    from examples.experiment_program.session import bind_output
    old=tmp_path/'old';new=tmp_path/'new'
    c=dict(output_directory=str(old),records={'G01':{'command':['replay',str(old/'G01')],'input_files':{str(old)+'-personal':'x'}}})
    result=bind_output(c,new)
    assert result['records']['G01']['command'][1]==str(new/'G01')
    assert str(old)+'-personal' in result['records']['G01']['input_files']
    assert c['output_directory']==str(old)


def test_source_snapshot_covers_machine_spec_and_native_hosts():
    from examples.experiment_program.source_seal import snapshot,ROOT
    value=snapshot()
    for relative in ('etx/machine/arch/gfx942.yaml','examples/experiment_program/specs.json',
                     'examples/experiment_program/router/replay.cpp','examples/llm/configs/qwen3-30b-a3b.json'):
        assert str((ROOT/relative).resolve()) in value


def test_dependency_order_places_validation_before_fold_exposure():
    from examples.experiment_program.session import execution_order
    assert execution_order(['G11','G12','G01'])==['G01','G12','G11']
    all_records=execution_order([f'G{i:02}' for i in range(1,14)])
    assert len(all_records)==len(set(all_records))==13
    assert all_records.index('G02')<all_records.index('G03')
    assert all_records.index('G04')<all_records.index('G05')
    assert all_records.index('G10')<all_records.index('G12')<all_records.index('G11')
    with pytest.raises(ValueError):execution_order(['G01','G01'])


def test_staging_registered_column_program_and_conditional_validation_root(tmp_path):
    from examples.experiment_program.stage import prepare
    program=tmp_path/'columns.json';build=tmp_path/'actual-columns-build';build.mkdir();(build/'identity.json').write_text('{}')
    program.write_text(json.dumps(dict(schema='etx.columns.program.v1',record='G02',build=str(build),capture_root=str(tmp_path/'captures'))))
    out=tmp_path/'future'
    value=prepare({'staging_root':str(tmp_path/'staged'),'records':{'G02':{'program':str(program)}}},tmp_path/'session.json',out)
    stage=value['records']['G02']
    assert '--program' in stage['command'] and str(program) in stage['input_files']
    assert stage['decision_file']=='G02/program-report.json'
    assert str(build/'identity.json') in stage['input_files']
    assert stage['admission']['build']==str(build)
    with pytest.raises(ValueError,match='program build'):
        prepare({'staging_root':str(tmp_path/'staged'),'records':{'G02':{'program':str(program),'build':str(tmp_path/'wrong')}}},tmp_path/'wrong-session.json',out)
    command=value['records']['G12']['command']
    assert command[command.index('--micro-session-directory')+1]==str(out)


def test_model_validation_aggregate_budget_is_distinct_from_native_call_deadline(tmp_path):
    from examples.experiment_program.stage import prepare
    from examples.experiment_program.session import configuration
    value=prepare({'staging_root':str(tmp_path/'staged'),'records':{'G12':{'timeout_seconds':12000}}},tmp_path/'session.json',tmp_path/'future')
    assert configuration(tmp_path/'session.json')['records']['G12']['timeout_seconds']==12000
    value['records']['G01']['timeout_seconds']=12000
    (tmp_path/'invalid.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='timeout'):configuration(tmp_path/'invalid.json')
