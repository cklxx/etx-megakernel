import math
import pytest
from examples.experiment_program.row_ops.source import source, mapping, retained_addresses, translation_unit
from examples.experiment_program.row_ops.protocol import qualify, admit_down


def test_exact_row_ownership_and_retained_operand_addresses():
    for k,epl in [(2048,8),(768,4)]:
        visits=mapping(k,epl)
        assert sorted({r for wave in visits for r,seg in wave}) == list(range(64))
        assert all(len({r for r,s in wave})==4 for wave in visits)
        assert all(len({w for w,v in enumerate(visits) if any(r==row for r,s in v)})==1 for row in range(64))
    old,new=retained_addresses()
    assert len(old)==128 and len(new)==32
    assert [new[s*8+i] for row,s,i in old] == [s*512+3*8+i for row,s,i in old]


def test_g09_retains_fp32_and_keeps_scratch_and_fallback():
    s=source('g09')
    assert 'float retained[4][8]' in s
    assert 'dsl_dot_retained' in s
    assert 'pl[(wv * nrt + cur) * NB + b] += v;' in s
    assert 'd.K == 2048 && gn == 32' in s
    assert 'else if (d.K %' in s
    assert 'dsl_bfr(retained' not in s


def test_g10_whole_row_up_down_preserves_virtual_zeros_and_join():
    s=source('g10_up'); d=source('g10_down')
    assert 'w2 == wv ? (0.f + v) : 0.f' in s
    assert 'all gate/up results visible before SiLU' in s
    assert 'd.K == 768 && r1 - r0 == 64' in d
    assert 'row_single_down<4, DSL_UN, 1>' in d
    assert 'v[0] * ws' in d
    assert not admit_down([1,1,1,1,-1],[20]*5)
    assert admit_down([1]*5,[20]*5)


def test_g11_no_norm_scratch_reuse_complete_landmarks():
    s=translation_unit('g11_probe','')
    for landmark in ('fold_begin','fold_end','norm_begin','norm_ready','output_ready'):
        assert landmark in s
    assert '__shared__ unsigned long long row_probe' in s
    assert 'rows + (size_t)(s0 + u) * K + 8 * i' in s
    assert 'red[etx_wave()]' not in s
    assert 'row_record + blockIdx.x * 16' in s


def test_probe_qualification_fails_closed():
    records=[{'task':t,'wave':w,'end':20+t+w,'begin':1} for t in range(2) for w in range(16)]
    resources={'vgpr':64,'sgpr':64,'lds':21120,'scratch':0}
    x=qualify(records,2,resources,resources,resources,[10,10],[10.2,10.3],[10.1,10.2],True)
    assert x['qualified'] and x['last_task_wave']==[1,15]
    assert not qualify(records[:-1],2,resources,resources,resources,[10],[10],[10],True)['qualified']
    assert not qualify(records,2,resources,{**resources,'vgpr':65},resources,[10],[10],[10],True)['qualified']
    assert not qualify(records,2,resources,resources,resources,[10],[10.6],[10],True)['qualified']


def test_virtual_zero_fp32_bit_pattern_and_cancellation():
    import struct
    def fp(v):return struct.unpack('<f',struct.pack('<f',v))[0]
    def bits(v):return struct.pack('<f',v)
    for owner in range(16):
        for v in (-0.,0.,1e-30,-1e-30,1e10,-1e10):
            partial=[0.]*16;partial[owner]=fp(partial[owner]+v)
            control=0.
            for x in partial:control=fp(control+x)
            candidate=0.
            for w2 in range(16):candidate=fp(candidate+(fp(0.+v) if w2==owner else 0.))
            assert bits(control)==bits(candidate)


def test_checking_host_capture_is_real_original_boundary(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    from etx.testing import sealed_original_host
    from etx.testing.row_ops import row_capture_probe

    def no_process(*args, **kwargs):
        raise AssertionError('historical capture conformance must not spawn Git or other processes')

    monkeypatch.setattr(subprocess, 'Popen', no_process)
    retained=sealed_original_host()
    result=row_capture_probe(tmp_path)
    assert result['historical_only'] and not result['device_qualification']
    assert result['provenance']==retained['provenance']
    assert hashlib.sha256((tmp_path/'original.hip').read_bytes()).hexdigest()==retained['provenance']['sha256']
    s=result['source']
    assert s.index('row_capture.before(')<s.index('launch_imported(x.n + 1);')<s.index('row_capture.after();')
    assert '".wqkv"' in s
    assert s.index('row_ops/capture.h')<s.index('int main(')


def test_plan_queue_and_event_predecessors_are_exact(tmp_path):
    import json
    from examples.experiment_program.row_ops.capture import plan_edges
    p=tmp_path/'plan.json'
    p.write_text(json.dumps(dict(types={'moe_down_0':{'id':0},'qkv_1':{'id':1}},descs=[dict(id=0,type=0,coord=[0],domain=0,worker=0),dict(id=1,type=1,coord=[0],domain=0,worker=0)],static_queues={'0:0':[0,1]},task_in=[[],[0]],ev_inst=[['after_down',[0],[0]]])))
    edges=plan_edges(p,'dsl_gemv_fold',1)
    assert edges['tasks'][0]['queue']['previous']==0
    assert edges['events']['0']['producers']==[0]
    p.write_text(p.read_text().replace('[0, 1]','[0]'))
    with pytest.raises(ValueError,match='queue predecessor'):plan_edges(p,'dsl_gemv_fold',1)


def test_declared_pair_orders_and_full_window_shape(monkeypatch):
    import struct
    from examples.experiment_program.row_ops import replay
    args=bytearray(160);struct.pack_into('<4i',args,48,768,2048,32,24)
    monkeypatch.setattr(replay,'validate_pack',lambda p:bytes(args))
    p=dict(role='dsl_moe_up',first=0,count=192,position=0,arguments={'file':'/native/args'},buffers=[],pointers=[],references=[])
    text,declared=replay.config(p,'/build','g09','/output')
    assert len(declared['orders'])==5 and all(sorted(x)==[0,1] for x in declared['orders'])
    assert 'row_dsl_moe_up' in text
    with pytest.raises(ValueError,match='complete fixed native'):replay.config({**p,'count':24},'/build','g09','/output')


def test_g11_full_export_population_reserves_storage_before_execution(tmp_path,monkeypatch):
    from examples.experiment_program.row_ops import replay
    from types import SimpleNamespace
    import struct
    args=tmp_path/'args';raw=bytearray(160);struct.pack_into('<2i',raw,52,2048,20);args.write_bytes(raw)
    pack=dict(count=256,role='dsl_gemv_fold',arguments=dict(file=str(args)),references=[dict(bytes=10240),dict(bytes=4096)])
    monkeypatch.setattr(replay.shutil,'disk_usage',lambda p:SimpleNamespace(free=3<<30))
    with pytest.raises(ValueError,match='protected2GiB'):replay.output_reservation(pack,3,4096,tmp_path/'probe')
    monkeypatch.setattr(replay.shutil,'disk_usage',lambda p:SimpleNamespace(free=8<<30))
    result=replay.output_reservation(pack,3,4096,tmp_path/'probe')
    assert result['estimated_export_bytes']>3<<30 and result['protected_free_floor_bytes']==2<<30


def test_native_row_configuration_relocates_all_pack_paths_without_editing_pack(tmp_path,monkeypatch):
    import json,struct
    from examples.experiment_program.row_ops import replay
    mapping=tmp_path/'map.json';mapping.write_text(json.dumps(dict(schema='etx.path-map.v1',paths={'/ORIGIN':str(tmp_path/'staged')})))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    args=bytearray(160);struct.pack_into('<4i',args,48,768,2048,32,24)
    monkeypatch.setattr(replay,'validate_pack',lambda p:bytes(args))
    pack=dict(role='dsl_moe_up',first=0,count=192,position=0,arguments={'file':'/ORIGIN/args'},buffers=[dict(file='/ORIGIN/buffer',offset=0,bytes=4)],pointers=[],references=[])
    old=json.dumps(pack,sort_keys=True)
    text,_=replay.config(pack,'/ORIGIN/build','g09',tmp_path/'run')
    assert str(tmp_path/'staged/args') in text and str(tmp_path/'staged/buffer') in text and str(tmp_path/'staged/build/control.hsaco') in text
    assert json.dumps(pack,sort_keys=True)==old


def test_row_readmission_rereads_every_raw_export_and_intermediate(tmp_path):
    """Synthetic CPU bytes exercise evidence validation, not GPU correctness."""
    import json
    from examples.experiment_program.row_ops.results import summarize
    prefix=tmp_path/'CPU_SYNTHETIC';golden=tmp_path/'original.bin';golden.write_bytes(b'gold')
    import struct
    args=tmp_path/'args';raw=bytearray(160);struct.pack_into('<2i',raw,52,2048,32);args.write_bytes(raw)
    pack=dict(count=1,role='dsl_moe_up',arguments=dict(file=str(args)),references=[dict(file=str(golden),bytes=4)])
    rows=[]
    for sample in range(5):
        from examples.experiment_program.row_ops.protocol import invocation_orders
        for arm in invocation_orders(2,5)[sample]:
            rows.append(dict(sample=sample,arm=arm,device_us=20-arm,host_wall_us=25-arm,exact=True))
            (tmp_path/f'CPU_SYNTHETIC.s{sample}.a{arm}.r0.bin').write_bytes(b'gold')
    for arm in range(2):(tmp_path/f'CPU_SYNTHETIC.a{arm}.intermediates').write_bytes(bytes(256))
    (tmp_path/'CPU_SYNTHETIC.timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert summarize(prefix,pack,{},'g09')['exact']
    changed=tmp_path/'CPU_SYNTHETIC.s4.a1.r0.bin';changed.write_bytes(b'bad!')
    with pytest.raises(ValueError,match='raw|export'):summarize(prefix,pack,{},'g09')
    changed.write_bytes(b'gold');(tmp_path/'CPU_SYNTHETIC.a1.intermediates').write_bytes(b'x'+bytes(255))
    with pytest.raises(ValueError,match='intermediate'):summarize(prefix,pack,{},'g09')


def test_g11_cannot_execute_without_current_matching_g12_trace(monkeypatch,tmp_path):
    import struct
    from examples.experiment_program.row_ops import replay
    args=bytearray(160);struct.pack_into('<4i',args,48,5120,2048,20,0)
    monkeypatch.setattr(replay,'validate_pack',lambda p:bytes(args))
    with pytest.raises(ValueError,match='matching G12 trace'):
        replay.config(dict(role='dsl_gemv_fold',first=0,count=256,position=1040),'/native','g11_probe',tmp_path/'probe')


def test_nonfinite_admission_never_passes():
    assert not admit_down([1,float('nan'),1,1,1],[20]*5)
    assert not admit_down([1]*5,[float('inf')]*5)


def test_winner_source_has_only_mechanism_no_probe_or_check(tmp_path):
    from examples.experiment_program.row_ops.integration import emit_winner
    from etx.models.vllm_llm.bf16_300.source import current_source
    original=tmp_path/'original.hip';original.write_text(current_source())
    for variant,marker in [('G09','float retained[4][8]'),('G10_UP','row_single_up'),('G10_DOWN','row_single_down')]:
        output=tmp_path/(variant+'.hip');m=emit_winner(variant,original,output);s=output.read_text()
        assert marker in s and 'RowRecord' not in s and 'row_probe' not in s
        assert m['candidate_sha256']!=m['original_sha256'] and not m['prior_certificates_valid']


def test_g11_padding_preserves_all_instructions_and_raw_source(tmp_path):
    from examples.experiment_program.row_ops.padding import match,NAMES
    base='.amdhsa_kernel row_dsl_gemv_fold\n.amdhsa_next_free_sgpr {n}\n.end_amdhsa_kernel\n; real instructions\ns_mov_b32 s0, 1\ns_endpgm\n.name: row_dsl_gemv_fold\n.sgpr_count: {total}\n'
    for name,n in zip(NAMES,(71,75,71)):(tmp_path/(name+'.s')).write_text(base.format(n=n,total=n+6))
    report=match(tmp_path)
    for name in NAMES:
        source=(tmp_path/(name+'.s')).read_text()
        assert 's_mov_b32 s0, 1\ns_endpgm' in source
        assert '.amdhsa_next_free_sgpr 75' in source and '.sgpr_count: 81' in source
        assert __import__('pathlib').Path(report[name]['raw_assembly']).is_file()


def test_output_poison_classification_preserves_live_input_aliases():
    from examples.experiment_program.row_ops.replay import reference_output_only
    fold=dict(role='dsl_gemv_fold',pointers=[dict(offset=8,buffer=1),dict(offset=16,buffer=2),dict(offset=64,buffer=3),dict(offset=72,buffer=4),dict(offset=80,buffer=4),dict(offset=88,buffer=5)])
    assert reference_output_only(fold,dict(buffer=1))
    assert reference_output_only(fold,dict(buffer=2))
    assert not reference_output_only(fold,dict(buffer=4))
    assert not reference_output_only(dict(role='dsl_moe_up',pointers=[dict(offset=8,buffer=2),dict(offset=16,buffer=2)]),dict(buffer=2))
    assert reference_output_only(dict(role='dsl_moe_up',pointers=[dict(offset=8,buffer=1),dict(offset=16,buffer=2)]),dict(buffer=2))


def test_native_alias_relocation_cannot_split_input_and_output():
    import struct
    from examples.experiment_program.row_ops.replay import validate_alias_map
    args=bytearray(160);struct.pack_into('<Q',args,8,0x1000);struct.pack_into('<Q',args,16,0x1000)
    p=dict(pointers=[dict(offset=8,buffer=0,add=0),dict(offset=16,buffer=0,add=0)],buffers=[dict(bytes=64)])
    validate_alias_map(p,args)
    with pytest.raises(ValueError,match='canonical allocation'):validate_alias_map(dict(pointers=[dict(offset=8,buffer=0,add=0),dict(offset=16,buffer=1,add=0)],buffers=[dict(bytes=64),dict(bytes=64)]),args)


def test_disjoint_output_slice_of_canonical_buffer_is_still_poisoned():
    import struct
    from examples.experiment_program.row_ops.replay import reference_output_only
    args=bytearray(160);struct.pack_into('<2i',args,48,768,2048);struct.pack_into('<i',args,96,8);struct.pack_into('<i',args,108,128)
    p=dict(role='dsl_moe_up',pointers=[dict(offset=8,buffer=0,add=0),dict(offset=16,buffer=0,add=4096)])
    assert reference_output_only(p,dict(buffer=0,offset=4096,bytes=12288),args)
    assert not reference_output_only(p,dict(buffer=0,offset=0,bytes=12288),args)
