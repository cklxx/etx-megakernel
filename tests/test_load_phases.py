import copy
import struct
import pytest
from etx.tools import inbody
from etx.ops import gemv


def healthy():
    types = {f'{role}_0': dict(id=i) for i, role in enumerate(inbody.LOAD_ROLES)}
    descs, legacy, waves, offsets, data = [], [], [], [0], []
    for ty, (role, (count, size)) in enumerate(inbody.LOAD_ROLES.items()):
        for _ in range(count):
            descs.append(dict(type=ty, coord=[0]))
            legacy.append([90,100,200,0]); waves.extend([[190,195]]*2)
            for _ in range(2):
                data.extend([1,110,150,size//2]+[0]*21)
            offsets.append(len(data))
    return dict(types=types, descs=descs), dict(version=3, tick_ns=10, waves_per_task=2,
        legacy=legacy, waves=waves, poll=[[0]]*len(descs), events=[[1,2,0]],
        loads=dict(capacity=8, offsets=offsets, data=data))


def test_full_grid_union_excludes_dot_and_never_sums_parallel_waves():
    p,r=healthy(); result=inbody.load_phases(p,r,layers=[0])
    for row in result['totals']:
        assert row['load_union_us']==pytest.approx(.4)
        nt,size=inbody.LOAD_ROLES[row['role']]
        assert row['bytes']==nt*size
        assert row['logical_TB_s']==pytest.approx(nt*size/(.4*1e6))
        assert row['actionable'] is False
    assert inbody.union_ticks([(1,3),(2,4),(6,8)])==5


@pytest.mark.parametrize('bad', ['old', 'zero', 'overflow', 'clock', 'bytes', 'coverage'])
def test_missing_or_invalid_load_stamps_fail_closed(bad):
    p,r=healthy()
    if bad=='old': del r['loads']
    elif bad=='zero': r['loads']['data'][1]=0
    elif bad=='overflow': r['loads']['data'][0]=9
    elif bad=='clock': r['loads']['data'][2]=201
    elif bad=='bytes': r['loads']['data'][3]=1
    else: p['descs'].pop()
    with pytest.raises(ValueError, match='load-phase'): inbody.load_phases(p,r,layers=[0])


def test_versioned_record_reader_roundtrip(tmp_path):
    p,r=healthy()
    words=[inbody.MAGIC,3,len(p['descs']),1,2,10]
    for key in ('legacy','poll','events','waves'): words.extend(v for row in r[key] for v in row)
    words.extend([8,len(r['loads']['data']),*r['loads']['offsets'],*r['loads']['data']])
    path=tmp_path/'load.bin';path.write_bytes(struct.pack('<'+'Q'*len(words),*words))
    assert inbody.read_raw(path)==r
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match='load-phase'): inbody.read_raw(path)


def test_load_overlay_ordinary_source_identical(monkeypatch):
    monkeypatch.delenv('ETX_INBODY_EMIT', raising=False); ordinary=gemv.emit_tiles()
    monkeypatch.setenv('ETX_INBODY_EMIT','0');assert ordinary==gemv.emit_tiles()
    monkeypatch.setenv('ETX_INBODY_EMIT','1');source=gemv.emit_tiles()
    assert source.count('lt.start();')==2
    assert source.count('lt.stop(min(U, se - s0)')==2
    assert 'dsl_gemv_rows_nb<8, DSL_UW, 1>(lt, We' in source
    assert 'void dsl_gemv_rows_nb(etx_load_trace& lt,' in source
