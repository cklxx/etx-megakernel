"""Sealed process/evidence conformance; operational cases live in external E2E."""
import json
from pathlib import Path
import pytest
from etx.testing import sealed_child_probe, sealed_contract_probe, sealed_dependency_probe


def test_full_log_and_failure(tmp_path):
    log=tmp_path/'full.log'
    assert sealed_child_probe('raw-stdout-stderr',log=log)==0
    assert 'raw stdout' in log.read_text() and 'raw stderr' in log.read_text()
    assert sealed_child_probe('exit7',log=tmp_path/'fail.log')==7


def test_deadline_is_not_shortened_success(tmp_path):
    with pytest.raises(TimeoutError):sealed_child_probe('short-deadline',log=tmp_path/'timeout.log')
    assert (tmp_path/'timeout.log').is_file()


def test_dependency_and_validation_gates():
    assert sealed_dependency_probe('G05',{'G04':dict(complete=True)}, {})
    assert not sealed_dependency_probe('G05',{'G04':dict(complete=False)}, {})
    assert not sealed_dependency_probe('G12',{'G01':dict(admitted=False)}, {})
    assert sealed_dependency_probe('G12',{'G01':dict(admitted=True)}, {})
    assert not sealed_dependency_probe('G13',{'G02':dict(admitted=True)}, {})
    assert sealed_dependency_probe('G13',{'G02':dict(admitted=True)}, {'coalescing_decision_active':True})


def test_self_authored_winner_cannot_unlock_model(tmp_path):
    p=tmp_path/'decision.json';p.write_text(json.dumps(dict(record='G01',admitted=True)))
    with pytest.raises(ValueError,match='admission'):sealed_contract_probe('decision',record='G01',value={},path=p)
    p.write_text(json.dumps(dict(record='G06',status='MICRO_MISS')))
    assert sealed_contract_probe('decision',record='G06',value={},path=p)['admitted'] is False


def test_capture_observation_never_uses_optimization_admission(tmp_path):
    p=tmp_path/'capture.json';p.write_text(json.dumps(dict(record='G04',complete=True)))
    with pytest.raises((ValueError,KeyError)):sealed_contract_probe('decision',record='G04',value={},path=p)


def test_runner_binds_only_owned_output_prefixes(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    c=dict(output_directory=str(old),records={'G01':{'command':['replay',str(old/'G01')],'input_files':{str(old)+'-personal':'x'}}})
    result=sealed_contract_probe('binding',value=c,output=new)
    assert result['records']['G01']['command'][1]==str(new/'G01')
    assert str(old)+'-personal' in result['records']['G01']['input_files']
    assert c['output_directory']==str(old)


def test_source_snapshot_covers_machine_spec_and_native_hosts():
    probe=sealed_contract_probe('source-scope');ROOT=Path(probe['root']);value=probe['sources']
    for relative in ('etx/machine/arch/gfx942.yaml','examples/experiment_program/specs.json',
                     'examples/experiment_program/router/replay.cpp','examples/llm/configs/qwen3-30b-a3b.json'):
        assert str((ROOT/relative).resolve()) in value


def test_dependency_order_places_validation_before_fold_exposure():
    def execution_order(value):return sealed_contract_probe('order',value=value)['order']
    assert execution_order(['G11','G12','G01'])==['G01','G12','G11']
    all_records=execution_order([f'G{i:02}' for i in range(1,14)])
    assert len(all_records)==len(set(all_records))==13
    assert all_records.index('G02')<all_records.index('G03')
    assert all_records.index('G04')<all_records.index('G05')
    assert all_records.index('G10')<all_records.index('G12')<all_records.index('G11')
    with pytest.raises(ValueError):execution_order(['G01','G01'])
