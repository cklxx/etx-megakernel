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
