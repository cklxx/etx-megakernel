"""Local source/protocol tests; no device or performance evidence."""
import hashlib
from pathlib import Path
import pytest
from examples.experiment_program.issue_order.source import transform, prep_source, ATTENTION_SHA, statement_inventory
from examples.experiment_program.issue_order.evidence import native_admission, validate_frame, decide
from examples.experiment_program.protocol import resolve_path

SNAP = Path('/Users/ckl/.hermes/cache/scratch/etx-hardware-complete-research/source/retained-attention.hip')

def source():
    snapshot=resolve_path(SNAP)
    if not snapshot.is_file(): pytest.skip('retained native source unavailable')
    return snapshot.read_text()

@pytest.mark.parametrize('record', ['G06','G07','G08'])
def test_real_reorder_preserves_statements(record):
    old = prep_source() if record == 'G06' else source()
    new = transform(record, old)
    assert old != new
    assert statement_inventory(old) == statement_inventory(new)
    assert old.count('__syncthreads()') == new.count('__syncthreads()')
    if record == 'G06':
        assert new.index('const long long pos = dsl_ap_scalar(positions)') < new.index('ulo = dsl_a_ld64')
        assert new.index('const long long slot = isq ? 0') < new.index('ulo = dsl_a_ld64')
    if record == 'G07':
        assert new.index('rescaled_exp_sum[i] = exp_sums_ptr') < new.index('float max_logit = reg_max_logit[0]')
        assert new.index('tmps[idx] = tmp_out_ptr') < new.index('if (warpid == 0)',new.index('void paged_attention_ll4mi_reduce_kernel'))
    if record == 'G08':
        assert new.index('Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B') < new.index('// fetch Q in shared')

@pytest.mark.parametrize('record', ['G06','G07','G08'])
def test_unknown_or_truncated_source_fails(record):
    with pytest.raises(ValueError): transform(record, 'not the original source')
    with pytest.raises(ValueError): transform('G99', source())

def test_frozen_attention(): assert hashlib.sha256(source().encode()).hexdigest() == ATTENTION_SHA

def test_no_native_evidence_fails_closed():
    with pytest.raises(ValueError): native_admission({}, {}, [], [])

def test_native_same_order_or_scratch_rejected():
    a={'target': [10], 'anchor': [20]}; b={'target':[12], 'anchor':[22]}
    r=[dict(vgpr=40,agpr=0,lds_bytes=100,scratch_bytes_lane=0,vgpr_spills=0,sgpr_spills=0)]
    assert native_admission(a,b,r,r)['status']=='CLOSED_NO_NATIVE_REORDER'
    c={'target':[10], 'anchor':[5]}
    assert native_admission(c,a,r,r)['status']=='ADMITTED_OPERATOR_ONLY'
    bad=[dict(r[0],scratch_bytes_lane=4)]
    assert native_admission(c,a,r,bad)['status']=='CLOSED_RESOURCE'

@pytest.mark.parametrize('pairs', [[],[(100,99)]*4,[(100,99)]*5,[(100,101)]*5,[(100,80),(100,99),(100,80),(100,80),(100,80)]])
def test_no_false_winner(pairs): assert decide(pairs)['status'] != 'MICRO_WINNER'

def test_micro_thresholds():
    assert decide([(100000,99000)]*5)['status']=='MICRO_MISS' # <2%
    assert decide([(100000,97000)]*5)['status']=='MICRO_WINNER'
    assert decide([(10000,9900)]*5)['status']=='MICRO_MISS' # <625ns

def test_missing_frame_fails(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)): validate_frame(tmp_path,{},None)

def test_source_mapping_requires_actual_native_load():
    from examples.experiment_program.issue_order.evidence import native_events
    source='target_load();\nanchor_load();\n'
    asm='.loc 3 1 2 ; wrong.hip:1:2\n global_load_dword v1,v2,off\n.loc 3 2 2 ; control/body.hip:2:2\n global_load_dword v2,v3,off\n'
    assert native_events(asm,source,('target_load','anchor_load'),'control/body.hip')=={'target':[],'anchor':[3]}

def test_replay_parse_requires_all_ordinals():
    from examples.experiment_program.issue_order.__main__ import parse_pair_means as parse_pairs
    row='IO_PAIR ordinal=0 control_ns=100000 candidate_ns=97000 control_host_ns=200000 candidate_host_ns=197000 samples=50 layer=0 position=1040\n'
    with pytest.raises(ValueError): parse_pairs(row*5+'IO_CHECK OK\n')
    with pytest.raises(ValueError): parse_pairs('')
    full=''.join(row.replace('ordinal=0','ordinal='+str(x)) for x in range(5))+'IO_CHECK OK record=G07 layer=0 position=1040 endpoints=1 epochs=516\n'
    assert parse_pairs(full)['status']=='MICRO_WINNER'

def test_captures_invalid_layers_fail_without_reading_payload(tmp_path):
    from examples.experiment_program.issue_order.evidence import validate_captures
    import json
    (tmp_path/'capture.json').write_text(json.dumps({'schema':'etx.c6.capture.v1','synthetic':False,'imports':{},'frames':{}}))
    with pytest.raises(ValueError): validate_captures(tmp_path,{},(0,0))
    with pytest.raises(ValueError): validate_captures(tmp_path,{},(48,))

def test_address_guard_rejects_negative_slot_and_position(tmp_path):
    from examples.experiment_program.issue_order.evidence import address_contract
    import struct
    args=bytearray(112);struct.pack_into('3i',args,64,32,4,128);struct.pack_into('2i',args,80,16,8)
    (tmp_path/'dsl_attnprep.args').write_bytes(args)
    (tmp_path/'positions.before').write_bytes(struct.pack('q',0))
    (tmp_path/'slot_mapping.before').write_bytes(struct.pack('q',-1))
    sizes={'cos_sin':128*2*1072,'kc':67*4*128*16*2,'vc':67*4*128*16*2}
    with pytest.raises(ValueError): address_contract(tmp_path,sizes,0)
    (tmp_path/'slot_mapping.before').write_bytes(struct.pack('q',0))
    assert address_contract(tmp_path,sizes,0)['cache_blocks']==67
    (tmp_path/'positions.before').write_bytes(struct.pack('q',1072))
    with pytest.raises(ValueError): address_contract(tmp_path,sizes,1072)

def test_native_fmax_machine_suffix_is_an_anchor():
    from examples.experiment_program.issue_order.evidence import native_events
    s='tmp_read();\nmax_logit = fmaxf();\n'
    asm='.loc 1 1 1 ; control/attention.hip:1:1\n global_load_ushort v1,v2,off\n.loc 1 2 1 ; control/attention.hip:2:1\n v_max_f32_e32 v1,v2,v3\n'
    assert native_events(asm,s,('tmp_read','max_logit'),'control/attention.hip')=={'target':[1],'anchor':[3]}

def test_replay_rejects_unbound_or_duplicate_correctness_marker():
    from examples.experiment_program.issue_order.__main__ import parse_pair_means as parse_pairs
    row='IO_PAIR ordinal=0 control_ns=100000 candidate_ns=97000 control_host_ns=200000 candidate_host_ns=197000 samples=50 layer=0 position=1040\n'
    pairs=''.join(row.replace('ordinal=0','ordinal='+str(x)) for x in range(5))
    with pytest.raises(ValueError): parse_pairs(pairs+'IO_CHECK OK record=G07 layer=1 position=1040 endpoints=1 epochs=516\n')
    with pytest.raises(ValueError): parse_pairs(pairs+'IO_CHECK OK\n')

@pytest.mark.parametrize('record',['G06','G07','G08'])
def test_sparse_probe_only_adds_landmarks(record):
    import re
    from examples.experiment_program.issue_order.diagnostic import instrument
    old=prep_source() if record=='G06' else source()
    for arm in (old,transform(record,old)):
        new=instrument(record,arm)
        removed=re.sub(r'io_trace_stamp\([0-3]\);','',new)
        assert statement_inventory(removed)==statement_inventory(arm)
        assert new.count('__syncthreads()')==arm.count('__syncthreads()')


def test_probe_qualification_fails_closed():
    from examples.experiment_program.issue_order.diagnostic import qualify
    r={'vgpr':40,'sgpr':20,'agpr':0,'lds_bytes':80,'scratch_bytes_lane':0,'vgpr_spills':0,'sgpr_spills':0}
    assert qualify(r,r,r,1000,1000,1501,True)['status']=='UNQUALIFIED'
    assert qualify(r,r,dict(r,vgpr=44),1000,1000,1001,True)['status']=='UNQUALIFIED'
    assert qualify(r,r,r,1000,1000,1001,False)['status']=='UNQUALIFIED'
    assert qualify(r,r,r,1000,1000,1001,True)['status']=='QUALIFIED_EXPOSURE_ONLY'

@pytest.mark.parametrize('record',['G06','G07','G08'])
def test_probe_requires_all_final_waves_and_active_partitions(record):
    from examples.experiment_program.issue_order.diagnostic import expected_stamps,check_stamps,TRACE_NATIVE,TRACE_PLACEMENT
    expected=expected_stamps(record,1040)
    rows=[{'index':i,'tick':i+1} for i in expected]
    assert check_stamps(record,1040,rows)['coverage']==len(expected)
    with pytest.raises(ValueError):check_stamps(record,1040,rows[:-1])
    with pytest.raises(ValueError):check_stamps(record,1040,rows+[rows[0]])
    final=[i for i in expected if TRACE_NATIVE<=i<TRACE_PLACEMENT]
    assert len(final)=={'G06':64,'G07':64,'G08':320}[record]
    if record=='G08':assert len(expected_stamps(record,0))<len(expected)

@pytest.mark.parametrize('record',['G06','G07','G08'])
def test_model_overlay_is_real_and_original_reference_preserved(record,tmp_path):
    from examples.experiment_program.issue_order.integration import emit_sources
    from examples.vllm_llm.kept_build import BASE_FLAGS
    control=tmp_path/'control';control.mkdir();(control/'imp').mkdir()
    (control/'dsl_tiles.hip').write_text('BEFORE\n'+prep_source()+'\nAFTER')
    (control/'imp/attention.hip').write_text(source())
    output=tmp_path/'overlay'
    got=emit_sources(record,control,output,BASE_FLAGS.split(),local_validation=True)
    assert got['timing_eligible'] is False
    assert got['certificate_reference']['attention_sha256']==ATTENTION_SHA
    selected=output/('dsl_tiles.hip' if record=='G06' else 'attention.hip')
    original=(control/('dsl_tiles.hip' if record=='G06' else 'imp/attention.hip')).read_text()
    assert selected.read_text()!=original
    assert statement_inventory(selected.read_text())==statement_inventory(original)
    with pytest.raises(ValueError):emit_sources(record,control,tmp_path/'bad',[],local_validation=True)


def test_model_overlay_refuses_unadmitted_winner(tmp_path):
    from examples.experiment_program.issue_order.integration import admit_winner
    with pytest.raises(ValueError):admit_winner({'status':'MICRO_MISS'},'G07',{})

def test_resource_reserve_never_changes_instruction_text():
    from examples.experiment_program.issue_order.diagnostic_match import verify_reserve_only
    raw='v_add_u32 v1,v2,v3\n .amdhsa_next_free_vgpr 42\n .amdhsa_next_free_sgpr 14\n .vgpr_count: 42\n .sgpr_count: 20\n'
    padded=raw.replace('42','44').replace('14','28').replace('20','34')
    assert verify_reserve_only(raw,padded)
    with pytest.raises(ValueError):verify_reserve_only(raw,padded.replace('v_add','v_sub'))

def synthetic_io_log():
    """Protocol fixture only; never device/numerical/performance evidence."""
    rows=['STAGE warmup=0'];epoch=2
    for ordinal in (2,0,4,1,3):
        rows.append(f'STAGE pair={ordinal} first={ordinal%2}')
        for arm in (('control','candidate') if ordinal%2==0 else ('candidate','control')):
            epoch+=1;dev=100000 if arm=='control' else 97000
            rows.append(f'IO_SAMPLE arm={arm} device_ns={dev} host_ns={dev+1000} reset_ns=100 epoch={epoch}')
        rows.append(f'IO_PAIR ordinal={ordinal} control_ns=100000 candidate_ns=97000 control_host_ns=101000 candidate_host_ns=98000 samples=1 layer=0 position=1040')
    rows.append(f'IO_CHECK OK record=G07 layer=0 position=1040 endpoints=1 epochs={epoch}')
    return '\n'.join(rows)+'\n'

def test_strict_raw_sample_population_and_means():
    from examples.experiment_program.issue_order.receipts import validate_native_log
    good=synthetic_io_log()
    assert validate_native_log(good,record='G07')['decision']['status']=='MICRO_WINNER'
    with pytest.raises(ValueError):validate_native_log('\n'.join(x for x in good.splitlines() if not x.startswith('IO_SAMPLE'))+'\n')
    with pytest.raises(ValueError):validate_native_log(good.replace('epoch=4','epoch=3'))
    with pytest.raises(ValueError):validate_native_log(good.replace('device_ns=97000','device_ns=96000'))
    with pytest.raises(ValueError):validate_native_log(good.replace('device_ns=97000','device_ns=nan'))

def test_prep_probe_preserves_one_dimensional_launch_abi():
    from examples.experiment_program.issue_order.diagnostic import trace_header
    assert 'gridDim' not in trace_header('G06')
    assert 'blockIdx.y' not in trace_header('G06')
    assert 'blockIdx.z' not in trace_header('G06')
    assert 'gridDim' in trace_header('G08')

def test_winner_cannot_omit_adjacent_boundary_calls(tmp_path):
    import json
    from examples.experiment_program.issue_order.integration import admit_winner
    from examples.experiment_program.issue_order.receipts import validate_native_log
    log=synthetic_io_log();decision=validate_native_log(log,record='G07')['decision']
    (tmp_path/'p1040_L0.log').write_text(log)
    (tmp_path/'preflight.json').write_text(json.dumps({'build_sha256':'unit-build','capture_seal_sha256':'unit-input','frames':{'p1040/L0':{}}}))
    winner={'status':'MICRO_WINNER','record':'G07','build_sha256':'unit-build','capture_seal_sha256':'unit-input','results':{'p1040/L0':decision},'_evidence_directory':str(tmp_path)}
    micro={'record':'G07','_identity_sha256':'unit-build','admission':{'timing_eligible':True}}
    with pytest.raises(ValueError):admit_winner(winner,'G07',micro)

def test_complete_synthetic_winner_reaches_integration_hook(tmp_path):
    """Synthetic protocol acceptance only; no device data are generated."""
    import json
    from examples.experiment_program.issue_order.integration import admit_winner
    from examples.experiment_program.issue_order.receipts import validate_native_log
    results={}
    for pos in (0,1023,1024,1040):
        raw=synthetic_io_log().replace('position=1040','position='+str(pos))
        (tmp_path/f'p{pos}_L0.log').write_text(raw)
        results[f'p{pos}/L0']=validate_native_log(raw,record='G07')['decision']
    (tmp_path/'preflight.json').write_text(json.dumps({'build_sha256':'unit-build','capture_seal_sha256':'unit-input','frames':{name:{} for name in results},'population':{'pairs':5,'warmup_rounds':1,'samples_per_arm_per_pair':1,'seed':617}}))
    winner={'status':'MICRO_WINNER','record':'G07','build_sha256':'unit-build','capture_seal_sha256':'unit-input','results':results,'_evidence_directory':str(tmp_path)}
    micro={'record':'G07','_identity_sha256':'unit-build','admission':{'timing_eligible':True}}
    assert admit_winner(winner,'G07',micro)

def test_device_dependency_receipts_cannot_be_omitted(tmp_path):
    from examples.experiment_program.issue_order.build import dependency_inputs
    with pytest.raises(ValueError):dependency_inputs(tmp_path,['kernel.d'])
    (tmp_path/'kernel.d').write_text('target: /code/definitely-missing-etx-test-header.h\n')
    with pytest.raises(ValueError):dependency_inputs(tmp_path,['kernel.d'])

def test_probe_carries_actual_placement_for_every_final_wave():
    from examples.experiment_program.issue_order.diagnostic import expected_stamps,TRACE_PLACEMENT
    ids=expected_stamps('G08',1040)
    assert len([i for i in ids if i>=TRACE_PLACEMENT])==320

def test_device_dependency_receipt_includes_system_math_headers(tmp_path):
    from examples.experiment_program.issue_order.build import dependency_flags
    assert '-sys-header-deps' in dependency_flags(tmp_path/'device.d')

def test_staged_build_resolves_dependencies_without_weakening_hashes(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.issue_order.__main__ import verify_build
    staged=tmp_path/'stage';staged.mkdir();dep=staged/'header.h';dep.write_text('frozen native input')
    build=staged/'build';build.mkdir()
    h=hashlib.sha256(dep.read_bytes()).hexdigest()
    (build/'identity.json').write_text(json.dumps({'hashes':{},'source_hashes':{},'dependency_files':{'/origin/dependencies/header.h':h}}))
    mapping=tmp_path/'paths.json';mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/origin/dependencies':str(staged),'/origin/build':str(build)}}))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    assert verify_build(Path('/origin/build'))['dependency_files']
    dep.write_text('changed')
    with pytest.raises(ValueError):verify_build(Path('/origin/build'))

def test_staged_capture_root_is_resolved_before_failclosed_checks(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.issue_order.evidence import validate_captures
    stage=tmp_path/'captures';stage.mkdir()
    (stage/'capture.json').write_text(json.dumps({'schema':'etx.c6.capture.v1','synthetic':False,'imports':{},'frames':{}}))
    mapping=tmp_path/'paths.json';mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/origin/captures':str(stage)}}))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    with pytest.raises(ValueError,match='duplicate/invalid layer'):validate_captures('/origin/captures',{},(48,))

@pytest.mark.parametrize('module',['__main__','diagnostic'])
def test_explicit_missing_dependency_mapping_cannot_fall_back_to_host(tmp_path,monkeypatch,module):
    import json,importlib
    verify_build=importlib.import_module('examples.experiment_program.issue_order.'+module).verify_build
    host=tmp_path/'host';header=host/'code/native/header.h';header.parent.mkdir(parents=True);header.write_text('same bytes remain at original host')
    build=tmp_path/'build';build.mkdir();h=hashlib.sha256(header.read_bytes()).hexdigest()
    (build/'identity.json').write_text(json.dumps({'hashes':{},'source_hashes':{},'dependency_files':{'/code/native/header.h':h}}))
    mapping=tmp_path/'paths.json';mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/code':str(tmp_path/'missing-stage')}}))
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:host));monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    with pytest.raises((ValueError,FileNotFoundError)):verify_build(build)

@pytest.mark.parametrize('module',['__main__','diagnostic'])
def test_explicit_identity_dependency_mapping_pins_original_path(tmp_path,monkeypatch,module):
    import json,importlib
    verify_build=importlib.import_module('examples.experiment_program.issue_order.'+module).verify_build
    host=tmp_path/'host';header=host/'code/native/header.h';header.parent.mkdir(parents=True);header.write_text('host fallback is not staged')
    build=tmp_path/'build';build.mkdir();h=hashlib.sha256(header.read_bytes()).hexdigest()
    (build/'identity.json').write_text(json.dumps({'hashes':{},'source_hashes':{},'dependency_files':{'/code/native/header.h':h}}))
    mapping=tmp_path/'paths.json';mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/code':'/code'}}))
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:host));monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    with pytest.raises((ValueError,FileNotFoundError)):verify_build(build)

def test_nonmatching_stage_prefix_retains_default_mac_mount_lookup(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.issue_order.paths import dependency_path
    host=tmp_path/'host';header=host/'code/native/header.h';header.parent.mkdir(parents=True);header.write_text('actual native input')
    mapping=tmp_path/'paths.json';mapping.write_text(json.dumps({'schema':'etx.path-map.v1','paths':{'/unrelated-capture':str(tmp_path/'capture')}}))
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:host));monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    assert dependency_path('/code/native/header.h')==header

@pytest.mark.parametrize('record',['G06','G07','G08'])
def test_model_overlay_accepts_current_dsl_without_overwriting_reference(record,tmp_path):
    from examples.experiment_program.issue_order.integration import emit_sources
    from examples.vllm_llm.kept_build import BASE_FLAGS
    control=tmp_path/'reference';(control/'imp').mkdir(parents=True)
    old='HISTORICAL\n'+prep_source();(control/'dsl_tiles.hip').write_text(old);(control/'imp/attention.hip').write_text(source())
    current=tmp_path/'current-kept-dsl.hip';current.write_text('void dsl_gemv_router_fold() {}\n'+prep_source())
    result=emit_sources(record,control,tmp_path/'candidate',BASE_FLAGS.split(),local_validation=True,dsl_source=current)
    assert (control/'dsl_tiles.hip').read_text()==old
    assert result['certificate_reference']['dsl_tiles_sha256']==hashlib.sha256(old.encode()).hexdigest()
    assert result['dsl_source_reference']['sha256']==hashlib.sha256(current.read_bytes()).hexdigest()
    if record=='G06':
        candidate=Path(result['candidate_source']).read_text()
        assert 'void dsl_gemv_router_fold() {}' in candidate
        assert statement_inventory(candidate)==statement_inventory(current.read_text())
    else:assert result['required_compile_inputs']['ETX_VL_DSL_TILES']==str(current)
    current.write_text('missing the exact frozen prep body')
    with pytest.raises(ValueError):emit_sources(record,control,tmp_path/'bad-current',BASE_FLAGS.split(),local_validation=True,dsl_source=current)


def owned_matrix_root(tmp_path):
    import json
    root=tmp_path/'new-matrix';root.mkdir()
    token='test-owned-new-root'
    (root/'.matrix-owner.json').write_text(json.dumps({'schema':'etx.local-matrix.owner.v1','root':str(root),'token':token}))
    cell=root/'qwen3-8b_base';cell.mkdir()
    return root,cell,token


def test_new_matrix_ir_archive_exact_roundtrip(tmp_path):
    import gzip
    from examples.experiment_program.issue_order.local_matrix_prepare import archive_new_ir,verify_ir_archive
    root,cell,token=owned_matrix_root(tmp_path);payload=b'; actual generated native IR\n'*1000
    original=cell/'mk.ll';original.write_bytes(payload)
    row=archive_new_ir(cell,root,token,completed_exit=0)
    assert not original.exists()
    assert gzip.decompress((cell/'mk.ll.gz').read_bytes())==payload
    assert row['uncompressed_sha256']==hashlib.sha256(payload).hexdigest()
    assert row['uncompressed_bytes']==len(payload) and verify_ir_archive(row)


def test_new_matrix_ir_archive_tamper_rejected(tmp_path):
    from examples.experiment_program.issue_order.local_matrix_prepare import archive_new_ir,verify_ir_archive
    root,cell,token=owned_matrix_root(tmp_path);(cell/'mk.ll').write_bytes(b'IR'*1000)
    row=archive_new_ir(cell,root,token,completed_exit=0)
    with (cell/'mk.ll.gz').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises((ValueError,OSError)):verify_ir_archive(row)


@pytest.mark.parametrize('occupied',['mk.ll.gz','ir-archive.json'])
def test_new_matrix_ir_archive_refuses_occupied_destination(tmp_path,occupied):
    from examples.experiment_program.issue_order.local_matrix_prepare import archive_new_ir
    root,cell,token=owned_matrix_root(tmp_path);original=cell/'mk.ll';original.write_bytes(b'new IR');archive=cell/occupied;archive.write_bytes(b'preserve old evidence')
    with pytest.raises((ValueError,FileExistsError)):archive_new_ir(cell,root,token,completed_exit=0)
    assert original.read_bytes()==b'new IR' and archive.read_bytes()==b'preserve old evidence'


@pytest.mark.parametrize('failure',['unowned','failed-cell','symlink'])
def test_new_matrix_ir_archive_restricts_owned_successful_new_files(tmp_path,failure):
    from examples.experiment_program.issue_order.local_matrix_prepare import archive_new_ir
    root,cell,token=owned_matrix_root(tmp_path);original=cell/'mk.ll'
    if failure=='symlink':
        external=tmp_path/'preserved.ll';external.write_bytes(b'protected');original.symlink_to(external)
    else:original.write_bytes(b'new IR')
    with pytest.raises((ValueError,OSError)):
        archive_new_ir(cell,root,'wrong-token' if failure=='unowned' else token,completed_exit=1 if failure=='failed-cell' else 0)
    assert original.exists() and not (cell/'mk.ll.gz').exists()


def matrix_archive_fixture(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.issue_order import matrix_archive
    monkeypatch.setattr(matrix_archive,'ROOT',tmp_path)
    root=tmp_path/'build/new-matrix';root.mkdir(parents=True);cell=root/'qwen3-8b_base';cell.mkdir()
    owner=root/'.matrix-owner.json';owner.write_text(json.dumps({'schema':'etx.local-matrix.owner.v1','root':str(root),'token':'new-owned-root'}))
    (cell/'plan.json').write_text('generated-plan '*100)
    (cell/'mk.s').write_text('actual-native-ISA '*100)
    (cell/'matrix.log').write_text('unfiltered native command log\n'*100)
    (cell/'configuration.json').write_text(json.dumps({'model':'qwen3-8b','variant':'base'}))
    manifest=tmp_path/'original-manifest.json';manifest.write_text(json.dumps({'source_root':str(tmp_path),'owner_receipt_sha256':hashlib.sha256(owner.read_bytes()).hexdigest(),'results':[{'selection':'qwen3-8b:base','exit_code':0,'status':'PASS','command':['bash','examples/vllm_llm/local_matrix.sh',str(root)],'output_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cell.iterdir()}}]}))
    return root,cell,manifest,hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_new_matrix_overlay_preserves_original_manifest_and_every_byte(tmp_path,monkeypatch):
    import gzip,json
    from examples.experiment_program.issue_order.matrix_archive import archive_matrix,verify_overlay
    root,cell,manifest,h=matrix_archive_fixture(tmp_path,monkeypatch);before=manifest.read_bytes();log=(cell/'matrix.log').read_bytes();config=(cell/'configuration.json').read_bytes();native=(cell/'mk.s').read_bytes()
    overlay=archive_matrix(root,manifest,tmp_path/'overlay',expected_manifest_sha256=h,min_bytes=128)
    assert manifest.read_bytes()==before and (cell/'matrix.log').read_bytes()==log and (cell/'configuration.json').read_bytes()==config
    assert not (cell/'mk.s').exists()
    entry=next(v for k,v in overlay['archives'].items() if k.endswith('/mk.s'))
    assert gzip.decompress(Path(entry['gzip_path']).read_bytes())==native
    assert overlay['original_manifest_sha256']==h and verify_overlay(tmp_path/'overlay/manifest.json')


@pytest.mark.parametrize('defect',['manifest','owner','source','symlink','hardlink','occupied'])
def test_new_matrix_overlay_rejects_unbound_or_occupied_files(tmp_path,monkeypatch,defect):
    import json,os
    from examples.experiment_program.issue_order.matrix_archive import archive_matrix
    root,cell,manifest,h=matrix_archive_fixture(tmp_path,monkeypatch);out=tmp_path/'overlay'
    if defect=='manifest':manifest.write_text(manifest.read_text()+' ')
    elif defect=='owner':(root/'.matrix-owner.json').write_text('{}')
    elif defect=='source':(cell/'mk.s').write_text('changed')
    elif defect=='symlink':(cell/'mk.s').unlink();(cell/'mk.s').symlink_to(cell/'matrix.log')
    elif defect=='hardlink':os.link(cell/'mk.s',tmp_path/'outside-ISA')
    elif defect=='occupied':(cell/'mk.s.program-archive.gz').write_bytes(b'preserved')
    with pytest.raises((ValueError,OSError)):archive_matrix(root,manifest,out,expected_manifest_sha256=h,min_bytes=128)
    assert (cell/'matrix.log').is_file() and (cell/'mk.s').exists()


def test_new_matrix_overlay_rejects_native_failure(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.issue_order.matrix_archive import archive_matrix
    root,cell,manifest,h=matrix_archive_fixture(tmp_path,monkeypatch);v=json.loads(manifest.read_text());v['results'][0]['exit_code']=1;manifest.write_text(json.dumps(v));h=hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):archive_matrix(root,manifest,tmp_path/'overlay',expected_manifest_sha256=h,min_bytes=128)
    assert (cell/'mk.s').is_file()


def test_matrix_continuation_selects_only_eleven_unbuilt_cells():
    from examples.experiment_program.issue_order.matrix_continue import remaining_cells
    from examples.experiment_program.issue_order.local_matrix_prepare import CELLS
    previous={'results':[{'selection':m+':'+v,'exit_code':0,'status':'PASS' if i<4 else 'ARCHIVE_FAILED_SOURCE_PRESERVED'} for i,(m,v) in enumerate(CELLS[:5])]}
    assert remaining_cells(previous)==CELLS[5:]
    assert len(remaining_cells(previous))==11


@pytest.mark.parametrize('defect',['duplicate','native-failure','unknown','unfinished'])
def test_matrix_continuation_refuses_ambiguous_pass_population(defect):
    from examples.experiment_program.issue_order.matrix_continue import remaining_cells
    row={'selection':'qwen3-8b:base','exit_code':0,'status':'PASS'};previous={'results':[row]}
    if defect=='duplicate':previous['results'].append(dict(row))
    elif defect=='native-failure':row['exit_code']=1
    elif defect=='unknown':row['selection']='unknown:base'
    else:row['status']='RUNNING'
    with pytest.raises(ValueError):remaining_cells(previous)


def test_final_matrix_successor_requires_exact_original5_plus_partial8():
    from examples.experiment_program.issue_order.matrix_finish import completed_prefix
    from examples.experiment_program.issue_order.local_matrix_prepare import CELLS
    rows=[{'selection':m+':'+v,'exit_code':0,'status':'PASS'} for m,v in CELLS]
    original={'results':rows[:5]};partial={'original_sha256':'bound-original','status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[5:13]}
    assert completed_prefix(original,partial,'bound-original')==rows[:13]


@pytest.mark.parametrize('defect',['original-sha','duplicate','missing','native-failure','controller-status'])
def test_final_matrix_successor_rejects_unbound_or_duplicate_completed_prefix(defect):
    from examples.experiment_program.issue_order.matrix_finish import completed_prefix
    from examples.experiment_program.issue_order.local_matrix_prepare import CELLS
    rows=[{'selection':m+':'+v,'exit_code':0,'status':'PASS'} for m,v in CELLS];original={'results':rows[:5]};partial={'original_sha256':'bound-original','status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[5:13]}
    if defect=='original-sha':partial['original_sha256']='other'
    elif defect=='duplicate':partial['results'][0]=dict(rows[0])
    elif defect=='missing':partial['results'].pop(0)
    elif defect=='native-failure':partial['results'][0]['exit_code']=1
    else:partial['status']='RUNNING'
    with pytest.raises(ValueError):completed_prefix(original,partial,'bound-original')


def test_last_matrix_successor_preserves_exact_fourteen_native_prefix():
    from examples.experiment_program.issue_order.matrix_last import last_prefix
    from examples.experiment_program.issue_order.local_matrix_prepare import CELLS
    rows=[{'selection':m+':'+v,'exit_code':0,'status':'PASS'} for m,v in CELLS];a={'results':rows[:5]};b={'original_sha256':'original','status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[5:13]};c={'original_sha256':'original','already_native_pass':13,'status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[13:14]}
    assert last_prefix(a,b,c,'original')==rows[:14]


@pytest.mark.parametrize('defect',['wrong-sha','duplicate','failed-native','wrong-count','unfinished'])
def test_last_matrix_successor_refuses_unproven_fourteen_prefix(defect):
    from examples.experiment_program.issue_order.matrix_last import last_prefix
    from examples.experiment_program.issue_order.local_matrix_prepare import CELLS
    rows=[{'selection':m+':'+v,'exit_code':0,'status':'PASS'} for m,v in CELLS];a={'results':rows[:5]};b={'original_sha256':'original','status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[5:13]};c={'original_sha256':'original','already_native_pass':13,'status':'BLOCKED_STORAGE_BEFORE_CELL','results':rows[13:14]}
    if defect=='wrong-sha':c['original_sha256']='wrong'
    elif defect=='duplicate':c['results'][0]=rows[0]
    elif defect=='failed-native':c['results'][0]['exit_code']=1
    elif defect=='wrong-count':c['already_native_pass']=12
    else:c['status']='RUNNING'
    with pytest.raises(ValueError):last_prefix(a,b,c,'original')


def test_matrix_last_fresh_prefix_preserves_previous_partial(tmp_path,monkeypatch):
    from examples.experiment_program.issue_order import matrix_last
    monkeypatch.setattr(matrix_last,'ROOT',tmp_path)
    old=tmp_path/'build/local-matrix-last-01-mixtral-8x7b_base';old.mkdir(parents=True);(old/'receipt').write_text('preserve')
    rows=matrix_last.build_roots('local-matrix-storage-relief',(('mixtral-8x7b','base'),('mixtral-8x7b','chain')))
    assert rows==[tmp_path/'build/local-matrix-storage-relief-01-mixtral-8x7b_base',tmp_path/'build/local-matrix-storage-relief-02-mixtral-8x7b_chain']
    assert (old/'receipt').read_text()=='preserve' and not any(p.exists() for p in rows)


@pytest.mark.parametrize('prefix',['../escape','/outside','other-root','local-matrix-','local-matrix-x/y'])
def test_matrix_last_build_prefix_refuses_escaping_or_unregistered_root(tmp_path,monkeypatch,prefix):
    from examples.experiment_program.issue_order import matrix_last
    monkeypatch.setattr(matrix_last,'ROOT',tmp_path)
    with pytest.raises(ValueError):matrix_last.build_roots(prefix,(('mixtral-8x7b','base'),))


def test_matrix_last_build_prefix_refuses_occupied_or_dangling_root(tmp_path,monkeypatch):
    from examples.experiment_program.issue_order import matrix_last
    monkeypatch.setattr(matrix_last,'ROOT',tmp_path);root=tmp_path/'build/local-matrix-storage-relief-01-mixtral-8x7b_base';root.parent.mkdir()
    root.symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):matrix_last.build_roots('local-matrix-storage-relief',(('mixtral-8x7b','base'),))
    assert root.is_symlink()
