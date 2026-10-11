#!/usr/bin/env python3
"""Read whole-kernel resources and retained-source identities (no execution)."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/tmp/etx-isafixes-whole')
report={'compiler':'ROCm 7.2.4 LLVM 22, x86 offline container',
        'import_bundle_sha256':hashlib.sha256((ROOT/'build/qwen300-rocm-on/imports.bc').read_bytes()).hexdigest(),
        'modes':{}}
for mode in ('baseline','c1','c5','combined'):
    work=OUT/mode; text=(work/'whole.s').read_text(); values={}; current=None
    for line in text.splitlines():
        m=re.search(r'\.amdhsa_kernel\s+(\w+)',line)
        if m: current=m[1]; values[current]={}
        m=re.match(r'; (NumVgprs|TotalNumSgprs|ScratchSize|LDSByteSize|Occupancy): (\d+)',line)
        if m and current: values[current][m[1]]=int(m[2])
        m=re.search(r'\.name:\s+(\w+)',line)
        if m: current=m[1]
        m=re.search(r'\.(sgpr_spill_count|vgpr_spill_count):\s+(\d+)',line)
        if m and current in values: values[current][m[1]]=int(m[2])
    report['modes'][mode]={'resources':values,'sha256':{name:hashlib.sha256((work/name).read_bytes()).hexdigest()
        for name in ('dsl_tiles.hip','megakernel_d0.hip','plan.json','whole.ll','whole.s')}}
print(json.dumps(report,indent=2))
