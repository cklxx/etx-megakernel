"""Operational acceptance. Synthetic CPU children are never GPU evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest
from examples.experiment_program import session
from examples.experiment_program.protocol import sha

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={f'G{i:02}' for i in range(1,14)}


def config(tmp_path,records=('G09',)):
    anchor=tmp_path/'explicitly-synthetic-cpu-input.txt';anchor.write_text('Synthetic CPU operational fixture, not native input or performance evidence.\n')
    stages={}
    for record in records:
        module=session.MODULES[record];cmd=[sys.executable,'-m','examples.experiment_program.'+module]
        if module=='columns':cmd+=['--record',record,'--dry-run']
        elif module=='issue_order':cmd+=['--record',record,'preflight']
        elif module=='row_ops':cmd+=['execute','--dry-run']
        else:cmd+=['preflight']
        stages[record]=dict(preflight=cmd,command=cmd,input_files={str(anchor):sha(anchor)},timeout_seconds=2,decision_file=record+'.json')
    p=tmp_path/'synthetic-session.json';p.write_text(json.dumps(dict(base='b573664faa4e7988dff353bc03aca4ce8561a729',source_snapshot={str(Path(session.__file__).resolve()):sha(session.__file__)},budget_seconds=10,records=stages)))
    return p


def cpu_dispatch(monkeypatch,decision=True,exit_code=0):
    original=session.run_logged;calls=[]
    def run(argv,log,*,seconds):
        calls.append(argv);log=Path(log)
        script=("import sys,json; print(json.dumps({'status':'PASS','synthetic_cpu_only':True}),flush=True); " if '-preflight.log' in log.name else "import sys; print('SYNTHETIC_CPU_ONLY stdout',flush=True); print('SYNTHETIC_CPU_ONLY stderr',file=sys.stderr,flush=True); ")
        if decision:
            # Decision exists only for operational state-machine testing. It
            # cannot satisfy G12's real raw-evidence admission interfaces.
            record=log.name.split('-')[0]
            script+='from pathlib import Path; import json; Path('+repr(str(log.parent/(record+'.json')))+").write_text(json.dumps({'synthetic_cpu_only':True,'admitted':False,'complete':True})); "
        script+='sys.exit('+str(exit_code)+')'
        return original([sys.executable,'-u','-c',script],log,seconds=seconds)
    monkeypatch.setattr(session,'run_logged',run)
    return calls


def test_real_cli_lists_exact_thirteen_disabled_records():
    r=subprocess.run([sys.executable,'-m','examples.experiment_program','list'],cwd=ROOT,text=True,capture_output=True,timeout=10,check=True)
    rows=json.loads(r.stdout)
    assert len(rows)==13 and {r['id'] for r in rows}==EXPECTED
    assert all(r['enabled'] is False and r['module']==session.MODULES[r['id']] for r in rows)


def test_all_thirteen_configuration_dryrun_has_no_output_side_effect(tmp_path,capsys):
    p=config(tmp_path,tuple(sorted(EXPECTED)));out=tmp_path/'never-created'
    assert session.main(['dry-run','--config',str(p),'--out',str(out)])==0
    value=json.loads(capsys.readouterr().out)
    assert set(value['selected'])==EXPECTED and value['enabled'] is False
    assert not out.exists()


@pytest.mark.parametrize('dangerous',[
    ['execute'],['execute','--device-execution-authorized'],['execute','--execute'],['build'],
])
def test_device_or_mutating_preflight_actions_refused(tmp_path,dangerous):
    p=config(tmp_path);c=json.loads(p.read_text());c['records']['G09']['preflight']=[sys.executable,'-m','examples.experiment_program.row_ops']+dangerous;p.write_text(json.dumps(c))
    with pytest.raises(ValueError):session.configuration(p)


def test_actual_columns_missing_capture_is_not_preflight_pass(tmp_path):
    p=config(tmp_path,('G02',));out=tmp_path/'actual-safe-columns-preflight'
    assert session.main(['preflight','--config',str(p),'--out',str(out)])==0
    state=json.loads((out/'session.json').read_text())
    assert state['stages']['G02']['status'] in ('MISSING_PREREQUISITE','BLOCKED_NATIVE_CAPTURE')
    report=json.loads((out/'G02-preflight.log').read_text())
    assert report['status']=='BLOCKED_NATIVE_CAPTURE' and not report['enabled']


def test_cpu_preflight_resume_preserves_completed_log(tmp_path,monkeypatch):
    p=config(tmp_path);out=tmp_path/'synthetic-resume';calls=cpu_dispatch(monkeypatch)
    session.main(['preflight','--config',str(p),'--out',str(out)])
    log=out/'G09-preflight.log';before=log.read_bytes()
    session.main(['preflight','--config',str(p),'--out',str(out),'--resume'])
    assert len(calls)==1 and log.read_bytes()==before
    assert json.loads((out/'session.json').read_text())['stages']['G09']['status']=='PREFLIGHT_PASS'


def test_run_requires_explicit_activation_before_any_child(tmp_path,monkeypatch):
    p=config(tmp_path);out=tmp_path/'disabled'
    monkeypatch.setattr(session,'run_logged',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(ValueError,match='disabled'):session.main(['run','--config',str(p),'--out',str(out)])
    assert not out.exists()


def test_cpu_subprocess_keeps_unfiltered_stdout_stderr_and_refuses_overwrite(tmp_path):
    log=tmp_path/'synthetic-child.log'
    argv=[sys.executable,'-u','-c',"import sys;print('SYNTHETIC stdout');print('SYNTHETIC stderr',file=sys.stderr)"]
    assert session.run_logged(argv,log,seconds=2)==0
    assert set(log.read_text().splitlines())=={'SYNTHETIC stdout','SYNTHETIC stderr'}
    with pytest.raises(FileExistsError):session.run_logged(argv,log,seconds=2)


def test_deadline_kills_own_cpu_process_group(tmp_path):
    marker=tmp_path/'grandchild-must-not-survive'
    grandchild="import time;from pathlib import Path;time.sleep(.3);Path("+repr(str(marker))+").write_text('SYNTHETIC orphan')"
    parent='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+']);print("SYNTHETIC CPU deadline child",flush=True);time.sleep(10)'
    with pytest.raises(TimeoutError,match='no performance verdict'):session.run_logged([sys.executable,'-u','-c',parent],tmp_path/'deadline.log',seconds=.08)
    time.sleep(.35)
    assert not marker.exists()
    assert 'SYNTHETIC CPU' in (tmp_path/'deadline.log').read_text()


def test_interrupted_local_export_recovers_without_overwriting_record(tmp_path,monkeypatch):
    record=tmp_path/'record';record.mkdir();(record/'raw.log').write_text('SYNTHETIC retained raw bytes\n')
    destination=tmp_path/'retained.tar';opened=tarfile.open
    def fail(*a,**k):raise OSError('SYNTHETIC interrupted export')
    monkeypatch.setattr(session.tarfile,'open',fail)
    with pytest.raises(OSError,match='interrupted'):session.export_record(record,destination)
    assert destination.with_name(destination.name+'.owner').is_file() and not destination.exists()
    monkeypatch.setattr(session.tarfile,'open',opened)
    digest=session.export_record(record,destination)
    assert digest==sha(destination)
    with tarfile.open(destination) as archive:assert archive.extractfile('raw.log').read()==(record/'raw.log').read_bytes()


def test_export_refuses_symlink_and_unowned_archive(tmp_path):
    record=tmp_path/'record';record.mkdir();personal=tmp_path/'personal-sentinel';personal.write_text('untouched')
    (record/'escape').symlink_to(personal)
    with pytest.raises(ValueError,match='symlink'):session.export_record(record,tmp_path/'symlink.tar')
    unowned=tmp_path/'unowned.tar';unowned.write_text('owned by nobody here')
    with pytest.raises(ValueError,match='unowned'):session.export_record(record,unowned)
    assert personal.read_text()=='untouched' and unowned.read_text()=='owned by nobody here'


def test_resume_rejects_changed_sealed_bytes_and_incomplete_state(tmp_path):
    log=tmp_path/'raw.log';log.write_text('SYNTHETIC initial')
    state=dict(config_sha256='c',source_revision='r',status='COMPLETE',sealed={str(log):sha(log)})
    session.verify_resume(state,'c','r')
    with pytest.raises(ValueError,match='identity'):session.verify_resume(state,'changed','r')
    log.write_text('SYNTHETIC changed')
    with pytest.raises(ValueError,match='sealed'):session.verify_resume(state,'c','r')
    for status in ('RUNNING','FAILED','NOT_COMPLETE'):
        with pytest.raises(ValueError,match='ambiguous'):session.verify_resume({**state,'sealed':{},'status':status},'c','r')


def test_dependency_and_optional_decision_gates_are_independent():
    assert not session.eligible('G05',{},{});assert session.eligible('G05',{'G04':{'complete':True}}, {})
    assert not session.eligible('G03',{'G02':{'admitted':False}}, {})
    assert session.eligible('G03',{'G02':{'admitted':True}}, {})
    assert not session.eligible('G13',{'G02':{'admitted':True}}, {})
    assert session.eligible('G13',{'G02':{'admitted':True}}, {'coalescing_decision_active':True})
    assert not session.eligible('G11',{'G12':{'complete':True}}, {})
    assert session.eligible('G11',{'G12':{'complete':True}}, {'fold_decision_active':True})
    assert not session.eligible('G12',{'G09':{'admitted':False}}, {})
    assert session.eligible('G12',{'G09':{'admitted':True}}, {})


@pytest.mark.parametrize('exit_code,write_decision',[(7,True),(0,False)])
def test_cpu_failure_or_missing_decision_stops_later_stages(tmp_path,monkeypatch,exit_code,write_decision):
    p=config(tmp_path,('G09','G06'));out=tmp_path/'synthetic-stop';calls=cpu_dispatch(monkeypatch,decision=write_decision,exit_code=exit_code)
    with pytest.raises(ValueError):session.main(['run','--execute','--config',str(p),'--out',str(out)])
    state=json.loads((out/'session.json').read_text())
    assert len(calls)==1 and state['status']=='NOT_COMPLETE' and 'G09' not in state['stages']
    assert (out/'G06-run.log').is_file() and out.with_suffix('.tar').is_file()


def test_real_historical_retained_record_replay_is_not_fresh_performance():
    from examples.experiment_program.local_acceptance import historical_replay
    r=historical_replay()
    assert r['historical_only'] is True and r['fresh_performance_claim'] is False
    assert r['tokens_per_invocation']==32 and len(r['recomputed']['gains_ns'])==5
    assert r['recomputed']['keep'] is False and r['retained_verdict']=='NO-GO'


def test_cpu_local_miss_never_launches_model_stage(tmp_path,monkeypatch):
    p=config(tmp_path,('G09','G12'));out=tmp_path/'synthetic-local-miss';calls=cpu_dispatch(monkeypatch)
    session.main(['run','--execute','--config',str(p),'--out',str(out)])
    state=json.loads((out/'session.json').read_text())
    assert len(calls)==1 and state['stages']['G12']['status']=='NOT_ADMITTED'
    assert not (out/'G12-run.log').exists()
