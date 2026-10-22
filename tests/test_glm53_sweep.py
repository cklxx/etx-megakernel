"""Compiled survivor enumeration is fail closed and never a performance result."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from examples.glm53 import model_build, options, sweep


def assembly(kernel='etx_megakernel_d0',lds=272,scratch=0,vgpr=80,sgpr=42,spills=0,occupancy=4):
    return f'''\t.size {kernel}, .Lfunc_end0-{kernel}
; Kernel info:
; NumVgprs: {vgpr}
; Occupancy: {occupancy}
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: {kernel}
    .vgpr_count: {vgpr}
    .sgpr_count: {sgpr}
    .vgpr_spill_count: {spills}
    .sgpr_spill_count: 0
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: {scratch}
    .wavefront_size: 64
...
\t.end_amdgpu_metadata
'''


def axes(**overrides):
    result={k:[v] for k,v in options.Options().to_dict().items()};result.update(overrides)
    return result


def run_sweep(tmp_path,monkeypatch,axes_value=None,**kw):
    def compiled(build,*args,**kwargs):
        model_build.finalize(build,local=True)
        return [dict(rank=0,vgpr=80),dict(rank=1,vgpr=80)]
    monkeypatch.setattr(sweep,'compile_candidate',compiled)
    return sweep.sweep(model_build.production_compile_spec(),tmp_path/'out',axes_value or axes(),[1],
                       llvm=Path('/compiler'),headers=Path('/headers'),devlibs=Path('/devlibs'),**kw)


def test_parse_actual_metadata_scratch_spills_waves_and_total_lds():
    report=sweep.parse_report(assembly(scratch=64,spills=3),'etx_megakernel_d0')
    assert report['vgpr']==80 and report['sgpr']==42
    assert report['vgpr_spills']==3 and report['sgpr_spills']==0
    assert report['scratch_bytes_per_workitem']==64 and report['lds_bytes_per_workgroup']==272
    assert report['compiler_waves_per_simd']==4 and 'NOT measured' in report['basis']


@pytest.mark.parametrize('text',[assembly().replace('    .vgpr_count: 80\n',''),
    assembly().replace('; Occupancy: 4',''),assembly().replace('.name: etx_megakernel_d0','.name: other'),
    'compiler crashed before reporting'])
def test_missing_reports_never_mean_zero_or_success(text):
    with pytest.raises(ValueError):sweep.parse_report(text,'etx_megakernel_d0')


def test_residency_uses_total_lds_and_plan_requirement():
    config=options.Options()
    report=sweep.parse_report(assembly(lds=33000),'etx_megakernel_d0')
    assert sweep.feasibility(deepcopy(report),config)['estimated_workgroups_per_cu_upper_bound']==1
    with pytest.raises(ValueError,match='residency'):
        sweep.feasibility(deepcopy(report),config,2)
    report['lds_bytes_per_workgroup']=65537
    with pytest.raises(ValueError,match='LDS'):sweep.feasibility(report,config)
    report=sweep.parse_report(assembly(occupancy=1),'etx_megakernel_d0')
    with pytest.raises(ValueError,match='residency'):sweep.feasibility(report,config,2)


def test_compiled_survivors_have_both_rank_resources_and_own_identity(tmp_path,monkeypatch):
    result=run_sweep(tmp_path,monkeypatch,axes(row_tile=[1,2]))
    assert result['status']=='COMPILED' and len(result['survivors'])==2
    assert result['selected_winner'] is None and not result['gpu_used'] and not result['performance_measured']
    hashes=[]
    for record,candidate in zip(result['candidates'],result['survivors']):
        assert record['status']=='COMPILED' and len(record['resources'])==2
        assert record['options_sha256'] and record['source_identity_sha256'] and record['build_identity_sha256']
        assert '--options' in record['later_commands'][0] and 'build-options.json' in record['later_commands'][0]
        assert 'gpu-b1' in record['later_commands'][1]
        assert (Path(candidate)/'status.json').is_file()
        hashes.append(record['build_identity_sha256'])
    assert hashes[0]!=hashes[1]


def test_errors_propagate_partial_and_keep_full_candidate_status(tmp_path,monkeypatch):
    calls=[]
    def compile_or_fail(build,config,*args,**kw):
        calls.append(config)
        if config.row_tile==2:raise RuntimeError('compiler exit 17; full log rank0-clang.log')
        model_build.finalize(build,local=True);return [dict(rank=0),dict(rank=1)]
    monkeypatch.setattr(sweep,'compile_candidate',compile_or_fail)
    result=sweep.sweep(model_build.production_compile_spec(),tmp_path/'out',axes(row_tile=[1,2]),[1],
                       llvm=Path('/llvm'),headers=Path('/headers'),devlibs=Path('/devlibs'))
    assert len(calls)==2 and result['status']=='PARTIAL' and len(result['survivors'])==1
    assert result['candidates'][1]['status']=='REJECTED' and 'exit 17' in result['candidates'][1]['error']
    assert json.loads((tmp_path/'out/summary.json').read_text())['status']=='PARTIAL'


def test_enumerate_only_never_claims_compiled(tmp_path,monkeypatch):
    result=run_sweep(tmp_path,monkeypatch,enumerate_only=True)
    assert result['status']=='ENUMERATED' and not result['survivors']
    assert result['candidates'][0]['status']=='ENUMERATED'


def test_zero_candidates_is_not_complete(tmp_path,monkeypatch):
    result=run_sweep(tmp_path,monkeypatch,axes(row_tile=[]))
    assert result['status']=='NOT-COMPLETE' and result['error']=='zero candidates' and not result['survivors']


def test_refuse_overwrite_and_excessive_matrix(tmp_path,monkeypatch):
    run_sweep(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError):run_sweep(tmp_path,monkeypatch)
    fresh=tmp_path/'fresh'
    with pytest.raises(ValueError,match='refusing truncation'):
        run_sweep(fresh,monkeypatch,axes(row_tile=[1,2]),max_candidates=1)
    assert json.loads((fresh/'out/summary.json').read_text())['status']=='NOT-COMPLETE'


def test_real_subprocess_nonzero_and_timeout_save_unfiltered_log(tmp_path):
    log=tmp_path/'failure.log'
    with pytest.raises(RuntimeError,match='exit 17'):
        sweep.run_command([sys.executable,'-c','import sys; print("full stdout"); print("full stderr",file=sys.stderr); sys.exit(17)'],log,3,{})
    text=log.read_text();assert 'full stdout' in text and 'full stderr' in text
    with pytest.raises(RuntimeError,match='timeout'):
        sweep.run_command([sys.executable,'-c','import time; time.sleep(10)'],tmp_path/'timeout.log',0.05,{})
    assert 'TIMEOUT' in (tmp_path/'timeout.log').read_text()


def test_cli_exit_code_honest_for_partial_and_zero_candidates(tmp_path,monkeypatch):
    monkeypatch.setattr(sweep,'sweep',lambda *a,**kw:dict(status='PARTIAL',candidates=[{}],survivors=[]))
    monkeypatch.setattr(sys,'argv',['sweep','--out',str(tmp_path),'--devlibs','/unused'])
    assert sweep.main()==1
    monkeypatch.setattr(sweep,'sweep',lambda *a,**kw:dict(status='NOT-COMPLETE',candidates=[],survivors=[]))
    assert sweep.main()==1


def test_build_rejects_unrecorded_compiler_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('HIPCC_COMPILE_FLAGS_APPEND','-DETX_GLM_ROW_TILE=3')
    with pytest.raises(ValueError,match='unrecorded compiler'):
        sweep.compile_candidate(tmp_path,options.Options(),Path('/llvm'),Path('/headers'),Path('/devlibs'))


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),0,-1])
def test_nonfinite_timeout_refused_before_process_start(tmp_path,monkeypatch,value):
    def forbidden(*a,**kw):raise AssertionError('must not spawn')
    monkeypatch.setattr(sweep.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='finite and positive'):
        sweep.run_command(['not-run'],tmp_path/'no.log',value,{})


def test_unexpected_wait_exception_kills_owned_process_group(tmp_path,monkeypatch):
    class Process:
        pid=987654
        calls=0
        def wait(self,**kw):
            self.calls+=1
            if self.calls==1:raise RuntimeError('unexpected wait failure')
            return -9
    proc=Process();kills=[]
    monkeypatch.setattr(sweep.subprocess,'Popen',lambda *a,**kw:proc)
    monkeypatch.setattr(sweep.os,'killpg',lambda *a:kills.append(a))
    with pytest.raises(RuntimeError,match='unexpected wait'):
        sweep.run_command(['not-run'],tmp_path/'log',1,{})
    assert kills and kills[0][0]==proc.pid and proc.calls==2


def test_duplicate_axes_refused_not_silently_recompiled(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='duplicate'):
        run_sweep(tmp_path,monkeypatch,axes(row_tile=[1,1]))
    receipt=json.loads((tmp_path/'out/summary.json').read_text())
    assert receipt['status']=='NOT-COMPLETE' and 'duplicate' in receipt['error']


def test_cartesian_batches_and_knobs():
    values=list(sweep.enumerate_candidates(axes(threads=[64,128],row_tile=[1,2]),[1,4]))
    assert len(values)==8
    assert {(b,c['threads'],c['row_tile']) for b,c in values} == {(b,t,r) for b in (1,4) for t in (64,128) for r in (1,2)}


def test_compile_rejects_actual_residency_below_exact_plan_requirement(tmp_path,monkeypatch):
    for name in sweep.UNRECORDED_ENV:monkeypatch.delenv(name,raising=False)
    build=tmp_path/'build';build.mkdir();libs=tmp_path/'libs';libs.mkdir()
    for name in ('ocml.bc','ockl.bc'):(libs/name).write_bytes(b'not used')
    (build/'build-resources.json').write_text(json.dumps(dict(required_workgroups_per_cu=2,devices=2)))
    for rank in (0,1):(build/f'megakernel_d{rank}.hip').write_text('// not compiled\n')
    def fake(command,*a,**kw):
        if str(command[0]).endswith('llc'):
            (build/'mk0.s').write_text(assembly(occupancy=1))
    monkeypatch.setattr(sweep,'run_command',fake)
    with pytest.raises(ValueError,match='residency'):
        sweep.compile_candidate(build,options.Options(),Path('/llvm'),Path('/headers'),libs,min_workgroups_per_cu=1)
    saved=json.loads((build/'compiler-resources.json').read_text())
    assert saved[0]['required_workgroups_per_cu']==2 and saved[0]['estimated_workgroups_per_cu_upper_bound']==1


def eight_rank_build(tmp_path,monkeypatch,failing_rank=None):
    for name in sweep.UNRECORDED_ENV:monkeypatch.delenv(name,raising=False)
    build=tmp_path/'build';build.mkdir();libs=tmp_path/'libs';libs.mkdir()
    for name in ('ocml.bc','ockl.bc'):(libs/name).write_bytes(b'not used')
    (build/'build-resources.json').write_text(json.dumps(dict(required_workgroups_per_cu=1,devices=8)))
    for rank in range(8):(build/f'megakernel_d{rank}.hip').write_text('// not compiled\n')
    compiled=[]
    def fake(command,*a,**kw):
        if str(command[0]).endswith('llc'):
            rank=int(Path(command[-1]).stem.removeprefix('mk'));compiled.append(rank)
            if rank==failing_rank:raise RuntimeError(f'compiler exit 1 rank {rank}')
            Path(command[-1]).write_text(assembly(kernel=f'etx_megakernel_d{rank}'))
    monkeypatch.setattr(sweep,'run_command',fake)
    monkeypatch.setattr(sweep.model_build,'finalize',lambda *a,**kw:None)
    return build,libs,compiled


def test_compile_visits_and_reports_every_generated_rank(tmp_path,monkeypatch):
    build,libs,compiled=eight_rank_build(tmp_path,monkeypatch)
    reports=sweep.compile_candidate(build,options.Options(devices=8,parallel=1),Path('/llvm'),Path('/headers'),libs)
    assert compiled==list(range(8)) and [r['rank'] for r in reports]==list(range(8))
    assert [r['rank'] for r in json.loads((build/'compiler-resources.json').read_text())]==list(range(8))


def test_compile_failure_on_last_rank_rejects_the_build(tmp_path,monkeypatch):
    build,libs,compiled=eight_rank_build(tmp_path,monkeypatch,failing_rank=7)
    with pytest.raises(RuntimeError,match='rank 7'):
        sweep.compile_candidate(build,options.Options(devices=8,parallel=1),Path('/llvm'),Path('/headers'),libs)
    assert compiled==list(range(8))


@pytest.mark.parametrize('devices,kernels',[(8,2),(2,8),(3,3),(None,2)])
def test_compile_refuses_rank_count_that_disagrees_with_generated_kernels(tmp_path,monkeypatch,devices,kernels):
    build,libs,compiled=eight_rank_build(tmp_path,monkeypatch)
    (build/'build-resources.json').write_text(json.dumps(dict(required_workgroups_per_cu=1,devices=devices)))
    for rank in range(kernels,8):(build/f'megakernel_d{rank}.hip').unlink()
    with pytest.raises(ValueError,match='does not match the generated kernels'):
        sweep.compile_candidate(build,options.Options(),Path('/llvm'),Path('/headers'),libs)
    assert compiled==[]
