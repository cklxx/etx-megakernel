"""Protocol tests; synthetic payloads here are never native/model evidence."""
import hashlib
import json
from pathlib import Path
import pytest
from examples.experiment_program.columns.protocol import (
    ProtocolError, stable_groups, validate_capture, EpochCompletion, queue_reachable,
    mechanism_source, decide_diagnostic,
)

def test_stable_group_identity_and_singletons():
    picks = [[7, 3, 8], [3, 7, 1], [9, 3, 2], [7, 4, 5]]
    g = stable_groups(picks)
    assert [x['expert'] for x in g] == [7, 3, 8, 1, 9, 2, 4, 5]
    assert g[0]['slots'] == [(0, 0), (1, 1), (3, 0)]
    assert g[1]['slots'] == [(0, 1), (1, 0), (2, 1)]
    assert sum(len(x['slots']) for x in g) == 12
    assert sum(x['singleton'] for x in g) == 6
    with pytest.raises(ProtocolError): stable_groups([[1, 1]])

def fixture(tmp_path):
    payload = tmp_path / 'payload.bin'; payload.write_bytes(b'\0' * 128)
    return {'schema': 1, 'record': 'G02', 'family': 'fleet', 'role': 'qkv',
            'source_revision': '554133d744b4591c9cf87c68b6817df781137c9b',
            'source_sha256': 'a' * 64, 'input_provenance': 'native-capture',
            'model': 'DeepSeek-Coder-V2-Lite-Base', 'context': 1024, 'precision': 'bf16',
            'positions': [1024,1040], 'batch': 2, 'N': 4, 'K': 8, 'ld': 8,
            'workers': 37, 'xcds': 8, 'threads': 256,
            'arrivals_ticks': [0,10], 'epoch': 1,
            'payload': {'path': str(payload), 'bytes': 128, 'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()},
            'segments': [{'name':'weights','offset':0,'bytes':64,'dtype':'bf16'},
                         {'name':'inputs','offset':64,'bytes':64,'dtype':'fp32'}],
            'golden_sha256': 'b' * 64, 'capture_complete': True,
            'per_column_dag': 'fleet-fma8-wave-xor32-to1-bf16',
            'source_bound': True, 'placement_bound': True}

def test_capture_fail_closed(tmp_path):
    c = fixture(tmp_path)
    assert validate_capture(c)['record'] == 'G02'
    for key,val in [('source_bound',False), ('capture_complete',False), ('precision','fp16'),
                    ('source_revision','HEAD'),('positions',[0]),('arrivals_ticks',[0,-1]),('placement_bound',False)]:
        bad = dict(c); bad[key] = val
        with pytest.raises(ProtocolError): validate_capture(bad)
    c['payload']['bytes'] -= 1
    with pytest.raises(ProtocolError): validate_capture(c)

def test_capture_overlap_duplicate_and_missing(tmp_path):
    c = fixture(tmp_path); c['segments'][1]['offset'] = 60
    with pytest.raises(ProtocolError): validate_capture(c)
    c = fixture(tmp_path); c['segments'].append(c['segments'][0])
    with pytest.raises(ProtocolError): validate_capture(c)
    c = fixture(tmp_path); Path(c['payload']['path']).unlink()
    with pytest.raises(ProtocolError): validate_capture(c)

def test_all_writing_waves_and_stale_epoch():
    c = EpochCompletion(2, 4, 1)
    for task in range(2):
        for wave in range(4):
            if (task,wave) != (1,3): c.publish(task,wave,1,drained=True)
    assert not c.ready(1)
    with pytest.raises(ProtocolError): c.reset(2)
    with pytest.raises(ProtocolError): c.publish(1,3,0,drained=True)
    with pytest.raises(ProtocolError): c.publish(1,3,1,drained=False)
    c.publish(1,3,1,drained=True)
    assert c.ready(1)
    with pytest.raises(ProtocolError): c.publish(1,3,1,drained=True)
    c.reset(2)
    assert not c.ready(2)
    with pytest.raises(ProtocolError): c.ready(1)

def test_queue_proof_rejects_producer_block():
    assert queue_reachable([['u0','d0'],['u1','d1']], {'d0':['u0'],'d1':['u1']})
    assert not queue_reachable([['d0','u0']], {'d0':['u0']})
    assert not queue_reachable([['u0','u0']], {})

def test_diagnostic_cannot_price_historical_deficit():
    x = decide_diagnostic([10]*5,[8]*5,[6]*5)
    assert x['verdict'] == 'DOMINANT_NEW_ISOLATED'
    assert x['historical_attribution'] == 'UNKNOWN'
    assert decide_diagnostic([10]*5,[8]*5,[9]*5)['verdict'] == 'PARTIAL_OR_UNKNOWN'
    with pytest.raises(ProtocolError): decide_diagnostic([10]*4,[8]*4,[6]*4)

def test_mechanism_not_flag_only():
    s = mechanism_source()
    assert 'load_weight' in s and 'fma8' in s and 'wave_sum' in s
    assert 'for (int b = 0; b < B' in s
    assert 's_waitcnt' in s and 'release' in s

def test_native_admission_requires_source_bound_golden_and_completion(tmp_path):
    from examples.experiment_program.columns.runner import admit_native
    c=fixture(tmp_path)
    with pytest.raises(ProtocolError, match='decode position'):
        admit_native(c,tmp_path)

def test_pair_parser_rejects_truncated_duplicate_or_bad_request_endpoint():
    from examples.experiment_program.columns.runner import decide_pairs
    rows=[]
    for pair in range(5):
        population=[('control',10),('candidate',9)] if pair%2==0 else [('candidate',9),('control',10)]
        for arm,ms in population:
            rows.append({'pair':pair,'ordinal':pair,'arm':arm,'device_ms':ms,'host_ms':ms,
                         'request_ms':[{'request':0,'ready':0,'done':ms,'latency':ms}]})
    assert decide_pairs(rows)['verdict']=='MICRO_PASS'
    for bad in [rows[:-1],rows+[rows[0]], [dict(x,host_ms=-1) for x in rows]]:
        with pytest.raises(ProtocolError): decide_pairs(bad)
    bad=[dict(x) for x in rows];bad[0]['request_ms']=[{'request':0,'ready':3,'done':2,'latency':-1}]
    with pytest.raises(ProtocolError): decide_pairs(bad)

def test_historical_stream_is_exact_source_bound(tmp_path):
    from examples.experiment_program.columns.build import extract_stream
    source=tmp_path/'gemv.py';source.write_text("STREAM = 'pretend math'\n")
    with pytest.raises(ProtocolError,match='source hash'):
        extract_stream(source,'b5eaa67')

def test_cli_all_flags_off_no_native_inputs(capsys):
    from examples.experiment_program.columns.__main__ import main
    for record in ('G02','G03','G05','G13'):
        assert main(['--record',record,'--dry-run'])==0
        report=json.loads(capsys.readouterr().out)
        assert report['enabled'] is False
        assert report['status']=='BLOCKED_NATIVE_CAPTURE'
    assert main(['--record','G02','--execute'])==2
    assert json.loads(capsys.readouterr().out)['enabled'] is False

def test_native_capture_patcher_refuses_changed_anchors():
    from examples.experiment_program.columns.patch_capture import patch_kernel,patch_host,patch_shim
    for fn in (patch_kernel,patch_host,patch_shim):
        with pytest.raises(ProtocolError,match='anchor'):
            fn('unrelated native source')

def test_source_bound_capture_binding_sites():
    from examples.experiment_program.columns.patch_capture import patch_kernel,patch_host,patch_shim
    from examples.dsv2lite.pinned import fleet_export
    root=Path(__file__).resolve().parents[1]
    fleet=fleet_export()/'src/kernels/fleet_kernel.hip'
    out=patch_kernel(fleet.read_text())
    q=out.index('etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,1')
    assert out.rfind('rms_norm_lds',0,q) < q < out.index('gemv_rows(w.layer_qkv[L]',q)
    u=out.index('etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,2')
    assert u < out.index('expert_gate_up(u, s.vec',u)
    d=out.index('etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,3')
    assert out.rfind('stage_vector(h + col0',0,d) < d < out.index('expert_down_chunk(u, s.vec',d)
    shim=patch_shim((root/'examples/dsv2lite/fleet_shim.hip').read_text())
    assert 'tc.column_capture=fp->column_capture' in shim and 'etx_column_finish' in shim
    host=patch_host((root/'examples/dsv2lite/host.hip').read_text())
    assert 'q.layers_ok!=27' in host and 'COLUMN CAPTURE COMPLETE' in host
    assert 'column_dir && last?column_sinks[r]:nullptr' in host

def test_g13_actual_fixed_queue_interventions():
    from examples.experiment_program.columns.g13_graph import build, queue_variants
    g=build()
    assert sum(int(x.grid[0])*int(x.grid[1]) for x in g.grids if 'up' in x.name)==384
    assert sum(int(x.grid[0])*int(x.grid[1]) for x in g.grids if 'down' in x.name)==512
    proof=queue_variants()
    assert proof['combined_both_up_first']['reachable']
    assert proof['independent_request_order']['reachable']
    assert not proof['combined_request_order']['reachable']
    assert proof['combined_request_order']['admitted'] is False
    assert proof['same_task_bodies'] and proof['same_worker_assignment']

def test_g04_dependency_never_treats_capture_as_optimization(tmp_path):
    from examples.experiment_program.columns.runner import verify_dependency
    p=tmp_path/'fake.json';p.write_text(json.dumps({'record':'G04','complete':True,'admitted':True}))
    with pytest.raises(ProtocolError):verify_dependency(p,needed=['G04'])

def test_model_integration_local_preparation_cannot_admit(tmp_path):
    from examples.experiment_program.columns.integration import prepare,prepare_grouped,frozen_graph,transform_qkv_graph,transform_grouped_graph
    root=Path(__file__).resolve().parents[1]
    rev='554133d744b4591c9cf87c68b6817df781137c9b'
    with pytest.raises(ProtocolError):prepare({},tmp_path/'no',batch=2,baseline_revision=rev)
    q=prepare({},tmp_path/'q',batch=2,baseline_revision=rev,local_validation=True)
    assert q['timing_eligible'] is False and q['local_validation'] is True
    grouped=prepare_grouped({},tmp_path/'g',batch=2,baseline_revision=rev,host_source=root/'examples/dsv2lite/host.hip',local_validation=True)
    assert grouped['timing_eligible'] is False and grouped['local_validation'] is True
    qgraph=transform_qkv_graph(frozen_graph(2),2,q['source'])
    assert sum(grid.name.startswith('columns_qkv') for grid in qgraph.grids)==27
    graph=transform_grouped_graph(frozen_graph(2),2,grouped['source'])
    assert sum(grid.name.startswith('column_metadata') for grid in graph.grids)==26
    for layer in range(1,27):
        assert graph.grids.index(graph.grid(f'column_metadata_{layer}')) < graph.grids.index(graph.grid(f'r0_gate_up_{layer}'))
        assert graph.grids.index(graph.grid(f'r1_gate_up_{layer}')) < graph.grids.index(graph.grid(f'r0_down_{layer}'))


def test_group_model_requires_both_native_roles(tmp_path):
    from examples.experiment_program.columns.integration import prepare_grouped
    winner=dict(record='G05',batch=2,status='MICRO_PASS_BOTH_REGIMES',native_exact=True,
                resource_admitted=True,native_pick_coverage=True,singleton_fallback_passed=True,role='gate_up')
    with pytest.raises(ProtocolError,match='both routed'):
        prepare_grouped(winner,tmp_path/'model',batch=2,baseline_revision='554133d744b4591c9cf87c68b6817df781137c9b',host_source=Path('examples/dsv2lite/host.hip'))


def test_native_row_parser_rejects_truncation_and_incomplete():
    from examples.experiment_program.columns.capture_pack import decode_rows
    import struct
    with pytest.raises(ProtocolError,match='row extent'):
        decode_rows(b'broken',family='fleet')
    raw=bytearray(18944)
    struct.pack_into('<Q',raw,0,0x455458434f4c3031)
    with pytest.raises(ProtocolError,match='complete'):
        decode_rows(bytes(raw),family='fleet')


def test_native_outputs_are_bitcopies_with_complete_unique_ownership():
    from examples.experiment_program.columns.capture_pack import original_values,staged
    import struct
    rows=[dict(h=[0]*20+[2],indices=(0,2),outputs=(0x3f80,0xbf80),inputs=struct.pack('<2I',0x3f800000,0xbf800000)),
          dict(h=[0]*20+[2],indices=(1,3),outputs=(0x4000,0xc000),inputs=struct.pack('<2I',0x3f800000,0xbf800000))]
    assert original_values(rows,4)==struct.pack('<4H',0x3f80,0x4000,0xbf80,0xc000)
    assert staged(rows,family='dsl')==struct.pack('<2H',0x3f80,0xbf80)
    with pytest.raises(ProtocolError,match='overlap'):
        original_values(rows+rows[:1],4)
    with pytest.raises(ProtocolError,match='coverage'):
        original_values(rows[:1],4)


def test_sealed_payload_supports_explicit_staging_prefix(tmp_path,monkeypatch):
    from examples.experiment_program.columns.protocol import sha256
    import json,hashlib
    actual=tmp_path/'staged';actual.mkdir();(actual/'bytes').write_bytes(b'unchanged')
    mapping=tmp_path/'mapping.json';mapping.write_text(json.dumps(dict(schema='etx.path-map.v1',paths={'/original/native':str(actual)})))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    assert sha256('/original/native/bytes')==hashlib.sha256(b'unchanged').hexdigest()


def test_native_etx_micro_fixed_request_slots_and_queue_reachability(monkeypatch):
    from examples.experiment_program.columns.micro_graph import build,verify_queues,freeze_native_budget
    from etx.passes import compile_graph,PassOptions
    from etx.machine import load_machine
    from etx.codegen import emit_plan_json
    # Planning emits the real persistent request slots, without any GPU/device work.
    monkeypatch.setenv('ETX_COLUMN_BATCH','2');monkeypatch.setenv('ETX_COLUMN_KIND','qkv')
    g=build()
    assert [grid.grid for grid in g.grids if grid.name.startswith('service')]==[(8,37),(8,37)]
    assert '38+' in g.grid('service_r1').worker_map
    plan=compile_graph(g,load_machine('gfx942'),{}, {},PassOptions(inline_tiles=True,relay='off'))
    with pytest.raises(ProtocolError,match='worker budget'):
        verify_queues(json.loads(emit_plan_json(plan)))
    freeze_native_budget(plan)
    receipt=verify_queues(json.loads(emit_plan_json(plan)))
    assert receipt['native_worker_pins'] and receipt['stable_owner_all_possible_groups_reachable']
    assert receipt['workers_per_domain']==75 and receipt['actual_tasks']==594


def test_dispatch_arena_alias_requires_actual_copy_and_lifetime_barriers(monkeypatch):
    from examples.experiment_program.columns.micro_graph import build,freeze_native_budget
    from examples.experiment_program.columns.dispatch_lds import alias_dispatch
    from etx.passes import compile_graph,PassOptions
    from etx.codegen import emit_kernel
    monkeypatch.setenv('ETX_COLUMN_BATCH','4');monkeypatch.setenv('ETX_COLUMN_KIND','group')
    plan=compile_graph(build(),'gfx942',{}, {},PassOptions(inline_tiles=True,prefetch=False,relay='off'));freeze_native_budget(plan)
    source=emit_kernel(plan,0);new,proof=alias_dispatch(source,4)
    assert proof['arena_bytes']==32768 and proof['numeric_storage_unchanged'] and not proof['alias_live_state']
    assert '__shared__ etx_task s_task;' not in new
    assert 'column_micro_arena[32768]' in new
    assert 'every lane owns its descriptor before any staging write' in new
    with pytest.raises(ProtocolError,match='post-body'):
        alias_dispatch(source.replace('etx_run_task(p, cur, worker, domain, tid);\n    __syncthreads();','etx_run_task(p, cur, worker, domain, tid);'),4)


def test_registered_program_is_complete_and_stops_native_failure(tmp_path):
    from examples.experiment_program.columns.program import registered,run_program
    cases=registered('G02');assert len(cases)==10 and cases[1]['depends']==['qkv-B2']
    assert {c.get('native_xcd') for c in cases[2:]}=={6,7}
    config=tmp_path/'program.json';root=tmp_path/'captures'
    config.write_text(json.dumps(dict(schema='etx.columns.program.v1',record='G02',build=str(tmp_path/'build'),capture_root=str(root),dependencies=[])))
    report=run_program(config,'G02',None,execute=False,case_main=lambda args:pytest.fail('missing inputs must not execute'))
    assert report['status']=='BLOCKED' and len(report['cases'])==10
    for position in (1024,1040):
        p=root/'qkv-B2'/f'p{position}';p.mkdir(parents=True);(p/'capture.json').write_text(json.dumps(dict(record='G02',batch=2,role='qkv')))
    calls=[]
    def refuse(args):calls.append(args);return 2
    report=run_program(config,'G02',None,execute=False,case_main=refuse)
    assert report['status']=='BLOCKED' and report['cases'][0]['status']=='REFUSED' and len(calls)==1
    calls.clear()
    with pytest.raises(ProtocolError,match='stops the column program'):
        run_program(config,'G02',tmp_path/'failed',execute=True,case_main=refuse)
    assert len(calls)==1 and json.loads((tmp_path/'failed/program-report.json').read_text())['status']=='ABORTED_NATIVE_FAILURE'


def test_g05_early_request_cannot_be_hidden_by_final_span_gain():
    from examples.experiment_program.columns.runner import qualify_rows
    rows=[]
    for ordinal in range(5):
        arms=['control','candidate'] if ordinal%2==0 else ['candidate','control']
        for arm in arms:
            latency=10 if arm=='control' else 11
            rows.append(dict(pair=ordinal,ordinal=ordinal,arm=arm,host_ms=20 if arm=='control' else 10,device_ms=10,
                             request_ms=[dict(request=0,ready=0,done=latency,latency=latency),dict(request=1,ready=1,done=9,latency=8)]))
    verdict=qualify_rows(rows,dict(record='G05',arrivals_ticks=[0,100]))
    assert verdict['verdict']=='MICRO_MISS' and len(verdict['early_requests_held'])==5


def test_nonmonotonic_ready_arrivals_have_independent_free_producer_queues(monkeypatch):
    from examples.experiment_program.columns import micro_graph,g03_graph,g13_graph
    monkeypatch.setenv('ETX_COLUMN_BATCH','4');monkeypatch.setenv('ETX_COLUMN_KIND','qkv')
    for graph in (micro_graph.build(),g03_graph.build(),g13_graph.build()):
        ready=[grid for grid in graph.grids if grid.name.startswith('ready_r')]
        assert [grid.domain_map for grid in ready]==[f'x->({r})' if grid.grid==(1,) else f'xy->({r})' for r,grid in enumerate(ready)]
        # Arrival order r2,r0,r3,r1 must not be serialized as r0,r1,r2,r3.
        assert len({grid.domain_map for grid in ready})==len(ready)
    proof=g13_graph.queue_variants()
    assert proof['independent_ready_producers'] is True
    assert proof['nonmonotonic_arrivals_replayable'] is True


def test_only_g05_candidate_constructs_and_transfers_group_metadata():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'examples/experiment_program/columns'
    state=(root/'micro_state.h').read_text();runtime=(root/'micro_runtime.h').read_text()
    assert state.index('ColumnMicroGroup groups[24]')>state.index('deadline;')
    assert 'if(c.record=="G05" && fused)' in runtime
    assert 'offsetof(ColumnMicroState,groups)' in runtime
    assert 'group_metadata_bytes' in runtime


def test_every_poisoned_reference_endpoint_requires_finite_golden():
    from examples.experiment_program.columns.runner import require_finite_golden
    import struct
    require_finite_golden(struct.pack('<2H',0x3f80,0xbf80),2)
    with pytest.raises(ProtocolError,match='nonfinite'):
        require_finite_golden(struct.pack('<H',0xffff),2)
    with pytest.raises(ProtocolError,match='nonfinite'):
        require_finite_golden(struct.pack('<I',0x7fc00000),4)
