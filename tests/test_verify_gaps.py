from etx.tools import verify_gaps as vg
import pytest


def fixture():
    return '''k:
 s_load_dword s4, s[0:1], 0
 s_waitcnt lgkmcnt(0)
 s_endpgm
.Lfunc_end0:
 .amdhsa_kernel k
; NumVgprs: 8
; TotalNumSgprs: 12
; ScratchSize: 0
; LDSByteSize: 32
; Occupancy: 8
'''


def test_lossless_index_and_resources(tmp_path):
    p=tmp_path/'k.s'; p.write_text(fixture())
    r=vg.report(p)['kernels']['k']
    assert [i['index'] for i in r['instructions']]==[0,1,2]
    assert r['instructions'][0]['assembly_line']==2
    assert r['resources']['LDSByteSize']==32
    assert r['instructions'][1]['text']=='s_waitcnt lgkmcnt(0)'


def test_missing_kernel_fails_closed():
    with pytest.raises(ValueError): vg.kernels('s_endpgm')


def test_independent_load_after_wait_is_counted_but_not_hoist_permission():
    s=fixture().replace(' s_endpgm',' s_load_dword s8, s[0:1], 4\n s_endpgm')
    r=vg.memory_order(vg.kernels(s)['k']['instructions'])
    assert r['register_independent_load_sites']==1
    assert r['candidates'][0]['independent_prior_loads']==[0]
    assert r['proven_overlapable_loads'] is None


def test_transitive_address_dependence_and_barrier_are_not_independent():
    s=fixture().replace(' s_endpgm',' s_add_u32 s8, s4, 4\n s_load_dword s9, s[8:9], 0\n s_barrier\n s_load_dword s12, s[0:1], 8\n s_endpgm')
    r=vg.memory_order(vg.kernels(s)['k']['instructions'])
    assert r['register_independent_load_sites']==0
    assert len(r['sequence'])==5


def test_lds_layout_unknown_type_fails_closed():
    with pytest.raises(ValueError,match='unsupported'): vg.type_layout('opaque',{})
    assert vg.type_layout('{ [8 x i32], i32 }',{})==(36,4)


def test_spill_pair_through_branch_and_lane_not_broadcast():
    from etx.tools import spill_chain as sc
    s='''k:
 s_load_dword s4, s[0:1], 0
 v_writelane_b32 v125, s4, 7
 s_branch .Ljoin
.Ljoin:
 v_readlane_b32 s5, v125, 7
 v_mov_b32 v1, s5
 v_add_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2]
 s_endpgm
.Lfunc_end0:
 .amdhsa_kernel k
'''
    r=sc.analyze(s,'k',['v125'])
    assert r['reloads'][0]['reaching_stores']==[1]
    assert r['may_chain_reload_counts']['sgpr_lane']==1
    assert r['executed_critical_reload_count'] is None
    assert sc.analyze(s,'k',[])['reloads']==[]


def test_spill_reaching_store_merge_and_non_chain_scratch():
    from etx.tools import spill_chain as sc
    s='''k:
 global_load_dword v1, v[2:3], off
 s_cbranch_scc0 .Lother
 scratch_store_dword off, v1, off offset:4
 s_branch .Ljoin
.Lother:
 scratch_store_dword off, v1, off offset:4
.Ljoin:
 scratch_load_dword v8, off, off offset:4
 scratch_load_dword v9, off, off offset:8
 v_add_f32_dpp v8, v8, v8 quad_perm:[1,0,3,2]
 s_endpgm
.Lfunc_end0:
 .amdhsa_kernel k
'''
    r=sc.analyze(s,'k',[])
    assert r['reloads'][0]['reaching_stores']==[2,4]
    assert r['may_chain_reload_counts']['scratch']==1
    assert r['unmatched_reload_indices']==[6]


def test_final_shuffle_add_across_branch_and_debug_labels():
    from etx.tools import spill_chain as sc
    s='''k:
 s_load_dword s4, s[0:1], 0
 v_writelane_b32 v125, s4, 2
 v_readlane_b32 s5, v125, 2
 v_mov_b32 v1, s5
 v_mov_b32_dpp v2, v1 quad_perm:[1,0,3,2]
.Ltmp1:
 s_cbranch_scc0 .Lexit
 v_add_f32 v1, v1, v2
.Lexit:
 s_endpgm
.Lfunc_end0:
 .amdhsa_kernel k
'''
    r=sc.analyze(s,'k',['v125'])
    assert r['final_reduction_indices']==[6]
    assert r['may_chain_reload_counts']['sgpr_lane']==1


def test_swizzle_operand_commas_preserve_dependency():
    s=fixture().replace(' s_load_dword s4, s[0:1], 0',' ds_bpermute_b32 v4, v1, v2').replace(' s_endpgm',' v_add_f32 v5, v5, v4\n ds_swizzle_b32 v6, v5 offset:swizzle(SWAP,16)\n s_endpgm')
    assert vg.memory_order(vg.kernels(s)['k']['instructions'])['register_independent_load_sites']==0
