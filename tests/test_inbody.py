"""CPU gates: source isolation, lossless records, completion and provenance."""
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import pytest

from dataclasses import replace

from etx.codegen import emit_kernel
from etx.passes.plan import environment_options
from etx.codegen.inbody import instrument
from etx.tools import inbody
from etx.models.vllm_llm.bf16_300 import dispatcher, launch, launch_source, pack


def test_tick_waits_before_using_async_scalar_result(tmp_path):
    """Compile the real helper for gfx942; no device execution or GPU needed."""
    root = Path(__file__).resolve().parents[1]
    llvm = Path(os.environ.get('ETX_LLVM', '/opt/homebrew/opt/llvm/bin'))
    if not all(p.exists() for p in (llvm/'clang++', llvm/'llc')):
        pytest.skip('local gfx942 compiler unavailable')
    source = tmp_path/'tick.cpp'
    source.write_text('#include <stdint.h>\n#define __device__\n'
                      '#define __forceinline__ inline __attribute__((always_inline))\n'
                      '#include "etx/inbody.h"\n'
                      'extern "C" __attribute__((amdgpu_kernel)) void stamp(uint64_t* out) '
                      '{ out[0] = etx_inbody_tick(); }\n')
    ir, assembly = tmp_path/'tick.ll', tmp_path/'tick.s'
    for command in (
        [str(llvm/'clang++'), '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx942',
         '-nogpulib', '-ffreestanding', '-std=c++17', '-O3', '-S', '-emit-llvm',
         '-I', str(root/'etx/runtime/include'), str(source), '-o', str(ir)],
        [str(llvm/'llc'), '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3',
         str(ir), '-o', str(assembly)],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    code = assembly.read_text()
    # Old code copied stale SGPRs to VGPRs before its unrelated pointer-load wait.
    assert re.search(r's_memrealtime s\[\d+:\d+\]\s+s_waitcnt lgkmcnt\(0\)', code)
    assert 'global_store_dwordx2' in code


def fixture():
    plan = dict(types={'moe_up_0':dict(id=0), 'moe_down_0':dict(id=1)},
        descs=[dict(id=i,type=i,coord=[0],domain=0,mode='static') for i in range(2)],
        domains=1,workers_per_domain=1,
        events=[dict(name='done',offset=0,shape=[1])],
        ev_inst=[['done',[0],[0]]],task_in=[[],[0]])
    raw = dict(version=1,tick_ns=10,waves_per_task=2,
        legacy=[[100,110,200,0],[201,240,400,0]],poll=[[0],[230]],events=[[215,220,0]],
        waves=[[180,190],[185,192],[350,380],[360,390]])
    return plan,raw


def binary(raw):
    words=[inbody.MAGIC,1,len(raw['legacy']),len(raw['events']),raw['waves_per_task'],10]
    for key in ('legacy','poll','events','waves'):
        words.extend(v for r in raw[key] for v in r)
    return struct.pack('<'+'Q'*len(words),*words)


def test_record_roundtrip_and_legacy_semantics(tmp_path):
    plan,raw=fixture()
    file=tmp_path/'raw'; file.write_bytes(binary(raw))
    got=inbody.read_raw(file)
    assert got==raw
    report=inbody.analyze(plan,got,{'moe_down':430000}, {'moe_down_0':.2})
    task=report['tasks'][1]
    assert task['record']==[201,240,400,0,220,230]
    assert task['poll_minus_arrive_bounds_us']==[.1,.15]
    assert task['tile_return_us']==1.2
    assert task['all_wave_drain_tail_us']==.3
    assert task['barrier_tail_us']==.1
    assert task['classes_us']['bytes_estimate']==.1
    assert task['classes_us']['machinery_residual_estimate']==pytest.approx(1.3)
    assert report['critical_us']==pytest.approx(report['span_us'])
    assert report['acceptance']['verdict']=='NOT-ESTABLISHED'


@pytest.mark.parametrize('mutation', ['short','extra','magic','version','waves','tick'])
def test_corrupt_binary_rejected(tmp_path,mutation):
    _,r=fixture(); data=bytearray(binary(r))
    if mutation=='short': data=data[:-8]
    elif mutation=='extra': data+=b'\0'*8
    else:
        off={'magic':0,'version':8,'waves':32,'tick':40}[mutation]
        struct.pack_into('<Q',data,off,99)
    f=tmp_path/'raw'; f.write_bytes(data)
    with pytest.raises(ValueError): inbody.read_raw(f)


@pytest.mark.parametrize('mutation', ['event','poll','wave','legacy'])
def test_missing_completion_never_counts_as_zero(mutation):
    p,r=fixture()
    if mutation=='event': r['events'][0]=[0,0,0]
    elif mutation=='poll': r['poll'][1]=[0]
    elif mutation=='wave': r['waves'][2]=[0,0]
    else: r['legacy'][1][2]=0
    with pytest.raises(ValueError): inbody.analyze(p,r)


@pytest.mark.parametrize('sentinel', [0, 1])
def test_unwritten_poll_buffer_has_specific_refusal(tmp_path, sentinel):
    plan, raw = fixture()
    raw['poll'] = [[sentinel] for _ in raw['poll']]
    path = tmp_path/'unwritten.bin'
    path.write_bytes(binary(raw))
    with pytest.raises(ValueError, match='poll buffer not written.*task 1'):
        inbody.analyze(plan, inbody.read_raw(path))


@pytest.mark.parametrize('poll', [200, 214, 241, (1 << 64)-1])
def test_strict_success_order_still_refuses_non_sentinel_corruption(poll):
    plan, raw = fixture()
    raw['poll'][1] = [poll]
    with pytest.raises(ValueError, match='invalid success order/clock for task 1'):
        inbody.analyze(plan, raw)


def test_success_can_precede_producer_after_stamp():
    p,r=fixture(); r['events'][0]=[215,235,0]
    # Consumer can observe zero before the producer's diagnostic write.
    t=inbody.analyze(p,r)['tasks'][1]
    assert t['poll_minus_arrive_bounds_us']==[0,.15]


def test_arrival_producer_identity_must_match_dependency():
    p,r=fixture(); r['events'][0][2]=1
    with pytest.raises(ValueError,match='invalid producer'): inbody.analyze(p,r)


def test_hop_absent_and_impossible_floor_are_unknown():
    p,r=fixture()
    t=inbody.analyze(p,r,{'moe_down':999999999})['tasks'][1]
    assert t['classes_us']['hop_estimate'] is None
    assert t['classes_us']['machinery_residual_estimate'] is None


def test_default_emission_unchanged(monkeypatch):
    p=dispatcher.graph('/tmp/inbody_tiles.hip')
    monkeypatch.delenv('ETX_INBODY_EMIT',raising=False)
    original=emit_kernel(p)
    # Emission follows the plan's options; the process environment is not read.
    monkeypatch.setenv('ETX_INBODY_EMIT','1')
    assert emit_kernel(p)==original
    p.options=replace(p.options,**environment_options({'ETX_INBODY_EMIT':'1'}))
    diagnostic=emit_kernel(p)
    assert diagnostic==instrument(original)
    assert '#define ETX_TARGET(idx) (void)ETX_INBODY_ARRIVE(p, 0 + (idx), tid,' in diagnostic
    assert diagnostic.count('etx_call_body(sym, &ctx);   // phase 2: the only call site of each body')==1


def test_original_publication_not_replaced():
    s=launch_source.candidate_source('body',2,3,4)
    assert s.count('ETX_RELEASE_DOMAIN(); __syncthreads();')==2
    assert s.count('probe.save(records, blk);')==2
    assert 'records[1 + blockIdx.x * 16 + etx_wave()]' in s


def test_lossless_sparse_records_and_last_wave(tmp_path):
    rows=[(100,200,10,3,0,4)]+[(180,210+w,10,w,0,4) for w in range(16)]
    path=tmp_path/'raw'; path.write_bytes(b''.join(launch.RECORD.pack(*r) for r in rows))
    r=launch.read_records(path,dict(count=1,first=10),1,10,3,4)
    assert len(r['records'])==17
    assert r['completion'][0]['last_completed']==[(10,15)]
    assert not r['completion'][0]['sample_is_last']
    rows[7]=(0,0,10,6,0,4)
    path.write_bytes(b''.join(launch.RECORD.pack(*r) for r in rows))
    with pytest.raises(ValueError): launch.read_records(path,dict(count=1,first=10),1,10,3,4)


def portable_pack(tmp_path):
    block=bytearray(160)
    struct.pack_into('<i',block,120,1)
    struct.pack_into('<i',block,156,1)
    args=tmp_path/'arguments.bin'; args.write_bytes(block)
    ref=tmp_path/'reference.bin'; ref.write_bytes(b'original')
    bufs=[]
    for i in range(5):
        f=tmp_path/f'buf{i}'; f.write_bytes(b'input')
        bufs.append(dict(file=str(f),offset=0,bytes=5,sha256=pack.sha(f)))
    return dict(version=1,precision='bf16',batch=1,layer=0,first=0,count=1,total_tasks=1,
        arguments_sha256=pack.sha(args),reference=dict(sha256=pack.sha(ref),bytes=8),buffers=bufs),args,ref


def test_reference_and_arguments_are_independent_immutable_inputs(tmp_path):
    p,args,ref=portable_pack(tmp_path)
    assert len(launch.validate_pack(p,ref,args,0,0,0,0))==160
    ref.write_bytes(b'candidate')
    with pytest.raises(ValueError,match='reference changed'): launch.validate_pack(p,ref,args,0,0,0,0)


def test_original_argument_mutation_rejected(tmp_path):
    p,args,ref=portable_pack(tmp_path)
    args.write_bytes(b'\xff'*160)
    with pytest.raises(ValueError,match='argument block changed'): launch.validate_pack(p,ref,args,0,0,0,0)


def test_standalone_topk_fold_alias_ignores_logits_but_topk_recompute_is_rejected(tmp_path):
    p,args,ref=portable_pack(tmp_path)
    b=bytearray(args.read_bytes()); struct.pack_into('<Q',b,112,0x1234)
    args.write_bytes(b); p['arguments_sha256']=pack.sha(args)
    # The production _fold MoE exports use the SAME bodies at mode zero.
    assert launch.validate_pack(p,ref,args,0,0,0,0)==b
    struct.pack_into('<i',b,104,2)
    args.write_bytes(b); p['arguments_sha256']=pack.sha(args)
    with pytest.raises(ValueError,match='unfolded'): launch.validate_pack(p,ref,args,0,0,0,0)


def test_snapshot_finalizer_requires_completed_check(tmp_path):
    f=tmp_path/'failed.log'; f.write_text('CHECK FAILED: 1 launch(es) differ')
    with pytest.raises(ValueError,match='completed'): pack.finalize(tmp_path,'host','plan',f)


def test_session_report_requires_all_five_full_runs():
    from examples.sessions.inbody import samples,perturbation,MAX_SESSION_SECONDS,JOBS
    text='\n'.join('INBODY-RUN '+json.dumps(dict(run=i,matched=32,gen=32,wall_ms=4+i*.001)) for i in range(5))
    s=samples(text)
    assert perturbation(s,s,s)['off_within_spread']
    with pytest.raises(ValueError): samples(text.replace('"matched": 32','"matched": 31',1))
    assert MAX_SESSION_SECONDS==900 and len(JOBS)==1 and JOBS[0]['timeout']==900


def test_off_identity_normalizes_only_source_locations():
    from etx.models.vllm_llm.bf16_300.inbody_cpu import normalize_ir
    source='call void asm "", ""(), !srcloc !1\n!1 = !{i32 123, i64 456}\n!2 = !{i64 789}\nadd i64 456, 1\n'
    result=normalize_ir(source)
    assert '!1 = !{i32 0, i64 0}' in result
    assert '!2 = !{i64 789}' in result and 'add i64 456, 1' in result
