"""Canonical G11 only: tiny CPU gate/source tests, not GPU observations."""
import copy,json
from pathlib import Path
import pytest
from examples.experiment_program import publication
from examples.experiment_program import publication_subgraph as sg

W=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (W/'build/review-repair-publication-attempt2/identity.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_failed_legacy_whole_model_is_not_an_exact_point_or_scope_waiver_gate():
    result=publication.admission(W/'build/review-repair-publication-attempt2/identity.json')
    assert result['admitted'] is False and result['status']=='CLOSED_NATIVE_DIAGNOSTIC_RESOURCE'
    assert result['owner_protocol_change_required'] is False
    assert result['exact_atomic_point_required'] is False
    assert result['whole_model_instrumented_build_required'] is False
    assert result['canonical_subgraph_alternative_allowed'] is True


def profiles():
    r=dict(vgpr=88,agpr=0,vgpr_spills=0,sgpr_spills=14,scratch_bytes_lane=0,lds_bytes=21152,static_blocks_per_cu=1)
    return {m:dict(r) for m in sg.MODES},{m:dict(rsrc1=1,rsrc2=2,group_segment=21152,private_segment=0) for m in sg.MODES}


@pytest.mark.parametrize('fault',['none','probe_spills','empty_spills','placement_allocation','capacity'])
def test_canonical_resource_gate_is_not_removed(fault):
    resources,physical=profiles()
    if fault=='probe_spills':resources['probe']['sgpr_spills']=46
    if fault=='empty_spills':resources['empty']['vgpr_spills']=1
    if fault=='placement_allocation':physical['probe']['rsrc1']=3
    if fault=='capacity':resources['control']['static_blocks_per_cu']=0
    actual=sg.resource_gate(resources,physical)
    assert actual['resources_equal']==(fault in ('none','probe_spills','empty_spills'))
    assert actual['admitted']==(fault in ('none','probe_spills','empty_spills'))
    assert actual['gpu_qualification']=='UNMEASURED'
    assert actual['exact_atomic_point_required'] is False


@pytest.mark.skipif(not (W/'build/review-repair-models-complete/row-original-checker/plan.json').is_file() or not (W/'build/publication-gap-subgraph4/plan-L1.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_actual_current_subgraph_keeps_all_original_tasks_and_worker_pins():
    root=W/'build/publication-gap-subgraph4';p=json.loads((W/'build/review-repair-models-complete/row-original-checker/plan.json').read_text())
    for layer in sg.LAYERS:
        c=sg.extract(p,layer);g=json.loads((root/f'plan-L{layer}.json').read_text())
        assert len(c['producers'])==len(c['consumers'])==256
        for d in g['descs']:
            actual=next(o for o in c['producers' if d['type']==0 else 'consumers'] if o['coord']==d['coord'])
            assert (actual['worker'],actual['domain'])==(d['worker'],d['domain'])
    broken=copy.deepcopy(p);first=sg.extract(p,1)['consumers'][0]['id'];event=broken['task_in'][first][0];broken['ev_inst'][event][2].pop()
    with pytest.raises(ValueError,match='complete final predecessor'):sg.extract(broken,1)


@pytest.mark.skipif(not (W/'build/review-repair-models-complete/row-original-checker/dsl_tiles.hip').is_file() or not (W/'build/publication-gap-subgraph4/control.hip').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_disjoint_read_fold_norm_hooks_do_not_replace_arithmetic_or_add_consumer_stamp_waits():
    original=(W/'build/review-repair-models-complete/row-original-checker/dsl_tiles.hip').read_text();instrumented=sg.tiles(original)
    # Preserve complete original arithmetic lines verbatim; only diagnostic hooks added.
    for line in ('          z[2 * j] += __uint_as_float(rv[u][j] << 16); z[2 * j + 1] += __uint_as_float(rv[u][j] & 0xffff0000u);','  ss = etx_block_sum(ss, red);','  const float sv = rsqrtf(ss / (float)K + d.eps);'):
        assert original.count(line)==instrumented.count(line)
    assert 'disjoint slot-read end / fold begin' in instrumented
    assert 'LABELED forced-read-drain diagnostic' in instrumented
    assert 'never aliases norm scratch' in instrumented
    assert 'while(g11_phase' not in instrumented and 'while (g11_phase' not in instrumented
    for mode in sg.MODES:
        code=(W/f'build/publication-gap-subgraph4/{mode}.hip').read_text()
        assert '#define ETX_DSL_COH 1' in code and 'ETX_RELEASE_DOMAIN();   // every wave drains its own stores' in code


@pytest.mark.skipif(not (W/'build/publication-gap-subgraph4/identity.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_prior_subgraph_attempt_cannot_inherit_changed_source_admission():
    with pytest.raises(ValueError,match='phase source differs'):
        publication.admission(W/'build/publication-gap-subgraph4/identity.json')


@pytest.mark.skipif(not (W/'build/review-repair-publication-attempt2/identity.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_current_subgraph_terminal_stages_without_launching_a_child(tmp_path,monkeypatch):
    from examples.experiment_program import stage,session
    identity=W/'build/review-repair-publication-attempt2/identity.json';cfg=tmp_path/'session-config.json';out=tmp_path/'preflight'
    stage.prepare(dict(staging_root=str(tmp_path/'staged'),records={'G11':dict(publication_preparation=str(identity))}),cfg,out)
    monkeypatch.setattr(session,'run_logged',lambda *a,**k:pytest.fail('unqualified subgraph cannot launch a child'))
    assert session.main(['preflight','--config',str(cfg),'--out',str(out),'--record','G11'])==0
    result=json.loads((out/'session.json').read_text())['stages']['G11']
    assert result['status']=='CLOSED_LOCAL_DIAGNOSTIC'
    assert result['evidence']['admitted'] is False
