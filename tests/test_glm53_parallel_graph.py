"""TP/EP placement and publication contracts; source authored, not GPU qualification."""
from dataclasses import replace

import pytest
from etx.ir import Scope
from etx.passes import compile_graph, verify_plan
from examples.glm53 import model, model_build
from examples.glm53.options import Options


def graph_fixture(mode, world, attention_tp=0, up_tail_balance=0, batch=1):
    # Compile-only shapes: byte-slicing correctness belongs to parallel_layout
    # tests. These sentinel blobs must never be used for numerical execution.
    spec=model_build.production_compile_spec()
    spec['config']=dict(spec['config'],num_hidden_layers=2,
                        layer_types=['linear_attention','deepseek_sparse_attention'],
                        mlp_layer_types=['dense','sparse'])
    spec['weights']={k:v for k,v in spec['weights'].items() if k in ('l0','l1','embedding','norm','head')}
    spec['ranks']=[dict(rank=0,layers=[0]),dict(rank=1,layers=[1])]
    weights={}
    for layer in range(2):
        suffixes=[('common',0)]+[(f'r{r}',r) for r in range(world)]
        if attention_tp:suffixes += [(f'a{r}',r) for r in range(world)]
        for suffix,rank in suffixes:
            weights[f'l{layer}_{suffix}']=dict(rank=rank,blob_bytes=1,meta_bytes=spec['weights'][f'l{layer}']['meta_bytes'])
    layout=dict(mode='tp' if mode==1 else 'ep',devices=world,weights=weights,attention_tp=bool(attention_tp))
    return model.build_from(spec,batch=batch,options=Options(parallel=mode,devices=world,attention_tp=attention_tp,
                                                up_tail_balance=up_tail_balance),layout=layout)


@pytest.mark.parametrize('mode', [1,2])
@pytest.mark.parametrize('world', [2,4,8])
@pytest.mark.parametrize('batch', [1,2,4,8])
def test_parallel_ffn_weights_private_outputs_and_system_edges(mode,world,batch):
    graph=graph_fixture(mode,world,batch=batch)
    plan=compile_graph(graph,'gfx942',{'B':batch},{})
    assert not verify_plan(plan) and plan.n_devices==world
    grids={gr.name:gr for gr in graph.grids}
    assert grids['head'].device==grids['argmax'].device==grids['feedback'].device==0
    assert all(gr.device==0 for name,gr in grids.items() if name.endswith(('_hc','_p1','_p2','_kda','_mla','_o','_rt')))
    assert plan.events['e_argmax'].scope!=Scope.SYSTEM
    for layer in ([0,1] if mode==1 else [1]):
        prefix=f'l{layer}'
        combine=grids[prefix+'_combine']
        assert combine.device==0 and plan.events[f'e_{prefix}_route'].scope==Scope.SYSTEM
        assert plan.tensor_placement[prefix+'_rt']=='fine_grained'
        for rank in range(world):
            local=f'{prefix}_r{rank}'
            for phase in ('dispatch','up','down'):assert grids[local+'_'+phase].device==rank
            for phase in ('up','down'):
                reads=grids[local+'_'+phase].reads
                assert local+'_weights' in reads and local+'_meta' in reads
                assert prefix+'_common_weights' not in reads
                assert not any(f'{prefix}_r{other}_weights' in reads for other in range(world) if other!=rank)
            assert local+'_dots' in combine.reads
            assert plan.events[f'e_{local}_dispatch'].scope!=Scope.SYSTEM
            assert (plan.events[f'e_{local}_down'].scope==Scope.SYSTEM)==(rank!=0)
            if rank:assert plan.tensor_placement[local+'_dots']=='fine_grained'
    if mode==2:
        assert grids['l0_up'].device==grids['l0_down'].device==0
        assert 'l0_combine' not in grids


@pytest.mark.parametrize('mode', [1,2])
@pytest.mark.parametrize('world', [2,4,8])
@pytest.mark.parametrize('batch', [1,2,4,8])
def test_attention_tp_placement_local_scratch_and_publication(mode,world,batch):
    graph=graph_fixture(mode,world,attention_tp=1,batch=batch)
    plan=compile_graph(graph,'gfx942',{'B':batch},{})
    assert not verify_plan(plan) and plan.n_devices==world
    grids={gr.name:gr for gr in graph.grids}
    dimensions=model_build.production_compile_spec()['dimensions']
    assert grids['head'].device==grids['argmax'].device==grids['feedback'].device==0
    assert 'norm_weights' in grids['head'].reads
    for layer,linear in [(0,True),(1,False)]:
        prefix=f'l{layer}'
        assert grids[prefix+'_attention_input'].device==grids[prefix+'_attention_combine'].device==0
        assert grids[prefix+'_hc'].device==grids[prefix+'_hcf'].device==grids[prefix+'_rt'].device==0
        assert plan.events[f'e_{prefix}_attention_input'].scope==Scope.SYSTEM
        assert plan.tensor_placement[prefix+'_attention_input']=='fine_grained'
        assert prefix+'_state' not in graph.tensors
        assert all(prefix+'_'+phase not in grids for phase in ('p1','p2','kda','mla','o'))
        combine=grids[prefix+'_attention_combine']
        for rank in range(world):
            local=f'{prefix}_a{rank}'
            phases=['dispatch','project','heads','output']+([] if linear else ['expand'])
            for phase in phases:assert grids[local+'_'+phase].device==rank
            assert (local+'_expand' in grids)==(not linear)
            for phase in ['project','heads','output']+([] if linear else ['expand']):
                gr=grids[local+'_'+phase]
                assert local+'_weights' in gr.reads and local+'_meta' in gr.reads
                assert prefix+'_common_weights' not in gr.reads
                assert not any(f'{prefix}_a{other}_weights' in gr.reads for other in range(world) if other!=rank)
            dispatch=grids[local+'_dispatch']
            assert prefix+'_attention_input' in dispatch.reads and local+'_input' in dispatch.writes
            assert f'e_{prefix}_attention_input' in dispatch.in_edges
            for phase in ['dispatch','project','heads']+([] if linear else ['expand']):
                assert plan.events[f'e_{local}_{phase}'].scope!=Scope.SYSTEM
            assert (plan.events[f'e_{local}_output'].scope==Scope.SYSTEM)==(rank!=0)
            assert local+'_partial' in combine.reads
            assert f'e_{local}_output' in combine.in_edges
            if rank:assert plan.tensor_placement[local+'_partial']=='fine_grained'
            heads=grids[local+'_heads']
            assert heads.grid[1]==dimensions['LH' if linear else 'NH']//world
            assert heads.consts[1:]==(int(linear),rank,world)
            assert len(plan.inst.tasks[heads.name])==batch*heads.grid[1]
            assert heads.resource.lds_bytes==4*model.lds_floats(dimensions)
            for suffix in ['input','p1','att','state']+([] if linear else ['p2']):
                tensor=local+'_'+suffix
                users=[gr for gr in graph.grids if tensor in gr.reads or tensor in gr.writes]
                assert users and {gr.device for gr in users}=={rank}
                assert plan.tensor_placement[tensor]!='fine_grained'
            # Capacity is still global per rank; locality does not certify reduced state bytes.
            state=graph.tensors[local+'_state']
            assert state.shape==('B',dimensions['STATE_KDA' if linear else 'STATE_MLA'])
            assert state.role=='runtime' and state.bytes_per_elem==4
            assert state.nbytes({'B':batch})==batch*state.shape[1]*4
    # Full TP shards dense and sparse FFNs; head-TP+EP retains the dense FFN at root.
    assert ('l0_r0_up' in grids)==(mode==1)
    assert ('l0_up' in grids)==(mode==2)
    assert 'l1_r0_up' in grids


@pytest.mark.parametrize('batch', [1,2,4,8])
def test_explicit_disabled_variants_preserve_default_pp2_graph(batch):
    spec=model_build.production_compile_spec()
    baseline=model.build_from(spec,batch=batch)
    explicit=model.build_from(spec,batch=batch,options=Options(devices=2,phased=1,parallel=0,
                                                   attention_tp=0,up_tail_balance=0,up_pairs=36))
    assert baseline==explicit
    names={gr.name for gr in baseline.grids}
    assert 'l0_kda' in names and 'l0_attention_input' not in names
    assert not any('_a0_' in name or '_r0_' in name for name in names)
    assert next(gr.device for gr in baseline.grids if gr.name=='head')==1


@pytest.mark.parametrize('world', [2,4,8])
@pytest.mark.parametrize('batch', [1,2,4,8])
@pytest.mark.parametrize('balance', [0,1])
def test_phased_pp_batch_binding_scales_request_buffers_not_workgroup_lds(world,batch,balance):
    spec=model_build.production_compile_spec()
    config=Options(devices=world,up_tail_balance=balance)
    graph=model.build_from(spec,batch=batch,options=config)
    single=model.build_from(spec,batch=1,options=config)
    assert graph==single                 # B is symbolic, not an Options field.
    dimensions=spec['dimensions']
    for name,tensor in graph.tensors.items():
        if tensor.shape and tensor.shape[0]=='B':
            assert tensor.nbytes({'B':batch})==batch*tensor.nbytes({'B':1}),name
    for gr in graph.grids:
        if len(gr.grid)==2:
            assert gr.grid[0]=='B' and gr.resource.lds_bytes==4*model.lds_floats(dimensions)
            assert graph.events['e_'+gr.name].shape==('B',)
            assert all(edge.text=='bt->b' for edge in gr.out_edges.values())
    state=graph.tensors['l0_state']
    assert state.nbytes({'B':batch})==batch*dimensions['STATE_KDA']*4


@pytest.mark.parametrize('world', [2,4,8])
@pytest.mark.parametrize('batch', [1,2,4,8])
def test_pp_tail_balance_changes_only_up_granularity(world,batch):
    spec=model_build.production_compile_spec()
    baseline=model.build_from(spec,batch=batch,options=Options(devices=world))
    balanced=model.build_from(spec,batch=batch,options=Options(devices=world,up_tail_balance=1))
    assert baseline.tensors==balanced.tensors
    old={gr.name:gr for gr in baseline.grids}
    changed=[]
    for gr in balanced.grids:
        previous=old[gr.name]
        if gr!=previous:
            assert gr.name.endswith('_up')
            assert replace(gr,grid=previous.grid,consts=previous.consts)==previous
            assert gr.consts[1:]==previous.consts[1:]
            changed.append(gr.name)
        before=baseline.events['e_'+gr.name] if 'e_'+gr.name in baseline.events else None
        after=balanced.events['e_'+gr.name] if 'e_'+gr.name in balanced.events else None
        if before!=after:
            assert gr.name.endswith('_up') and replace(after,wait_count=before.wait_count)==before
    sparse=[f'l{i}_up' for i,kind in enumerate(spec['config']['mlp_layer_types']) if kind=='sparse']
    assert changed and sparse
    assert all(old[name].grid[1]==512 and balanced.grid(name).grid[1]==576 for name in sparse)
    assert all(gr==old[gr.name] for gr in balanced.grids if not gr.name.endswith('_up'))


@pytest.mark.parametrize('mode', [1,2])
@pytest.mark.parametrize('world', [2,4,8])
@pytest.mark.parametrize('attention_tp', [0,1])
@pytest.mark.parametrize('batch', [1,2,4,8])
def test_parallel_tail_balance_preserves_every_other_phase(mode,world,attention_tp,batch):
    baseline=graph_fixture(mode,world,attention_tp=attention_tp,batch=batch)
    balanced=graph_fixture(mode,world,attention_tp=attention_tp,up_tail_balance=1,batch=batch)
    assert baseline.tensors==balanced.tensors
    old={gr.name:gr for gr in baseline.grids}
    changed=[]
    for gr in balanced.grids:
        previous=old[gr.name]
        if gr!=previous:
            assert gr.name.endswith('_up')
            assert replace(gr,grid=previous.grid,consts=previous.consts)==previous
            assert gr.consts[1:]==previous.consts[1:]
            changed.append(gr.name)
    assert changed
    local=world if mode==1 else 1
    assert old['l1_r0_up'].grid[1]==512//local
    assert balanced.grid('l1_r0_up').grid[1]==576//local
    for name,event in balanced.events.items():
        previous=baseline.events[name]
        if event!=previous:
            assert name.endswith('_up') and replace(event,wait_count=previous.wait_count)==previous
