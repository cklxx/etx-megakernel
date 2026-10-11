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

