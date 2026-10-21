import struct
from pathlib import Path
import pytest
from examples.experiment_program.routing import decode_records


def raw(batch=2):
    rows=[]
    for r in range(batch):
        b=bytearray()
        for t in range(32):
            for l in range(1,27):
                for x in range(8):
                    b+=struct.pack('<16I',33+t,1,*range(6),*[0x3e000000]*6,0,0)
        rows.append(bytes(b))
    return rows


def test_complete_native_record_decode():
    v=decode_records(raw(),batch=2)
    assert v['requests']==2 and len(v['effective_batch_histogram'])>0
    assert not v['performance_claim']


@pytest.mark.parametrize('fault',['truncated','duplicate','missing','wrong_epoch','expert','nan'])
def test_native_capture_faults_fail_closed(fault):
    rows=raw();b=bytearray(rows[0])
    if fault=='truncated':b=b[:-4]
    if fault=='duplicate':struct.pack_into('<I',b,4,2)
    if fault=='missing':struct.pack_into('<I',b,4,0)
    if fault=='wrong_epoch':struct.pack_into('<I',b,0,32)
    if fault=='expert':struct.pack_into('<I',b,8,64)
    if fault=='nan':struct.pack_into('<I',b,32,0x7fc00000)
    rows[0]=bytes(b)
    with pytest.raises(ValueError):decode_records(rows,batch=2)
