"""Read actual AMDHSA kernel descriptor allocation words, without external tools."""
import struct
from pathlib import Path


def descriptors(path):
    data=Path(path).read_bytes()
    if data[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',data,18)[0]!=224:raise ValueError('expected little-endian ELF64 AMDGPU')
    shoff=struct.unpack_from('<Q',data,40)[0];size,count=struct.unpack_from('<HH',data,58)
    if size!=64 or shoff+size*count>len(data):raise ValueError('section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*size) for i in range(count)]
    result={}
    for s in sections:
        if s[1]!=2:continue
        strings=sections[s[6]];st=data[strings[4]:strings[4]+strings[5]]
        if s[9]!=24:raise ValueError('symbol table entry')
        for offset in range(s[4],s[4]+s[5],s[9]):
            name,info,other,section,value,n=struct.unpack_from('<IBBHQQ',data,offset)
            name=st[name:st.find(b'\0',name)].decode()
            if not name.endswith('.kd'):continue
            sec=sections[section];offset=sec[4]+value-sec[3];kd=data[offset:offset+64]
            if len(kd)!=64:raise ValueError('kernel descriptor')
            group,private,kernarg=struct.unpack_from('<III',kd)
            rsrc3,rsrc1,rsrc2=struct.unpack_from('<III',kd,44);props=struct.unpack_from('<H',kd,56)[0]
            result[name[:-3]]=dict(raw=kd.hex(),group_segment=group,private_segment=private,kernarg_segment=kernarg,rsrc1=rsrc1,rsrc2=rsrc2,rsrc3=rsrc3,properties=props)
    if not result:raise ValueError('no actual kernel descriptors')
    return result


def allocation_signature(descriptor,agpr_count):
    # On gfx942 RSRC3[5:0] is ACCUM_OFFSET (in groups of four VGPRs).
    # With zero allocated AGPRs this locates an empty accumulator bank; retain
    # its raw value but exclude that unused offset from actual capacity equality.
    d=descriptor
    return dict(group_segment=d['group_segment'],private_segment=d['private_segment'],kernarg_segment=d['kernarg_segment'],rsrc1=d['rsrc1'],rsrc2=d['rsrc2'],rsrc3=d['rsrc3'] if agpr_count else d['rsrc3']&~63,properties=d['properties'],agpr_count=agpr_count)
