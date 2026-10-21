"""One role-dependent ABI for compact fp32 own-arena exports, without padding."""
import json,struct
from pathlib import Path
from examples.experiment_program.protocol import resolve_path,require
ARENA_BYTES=16384
RECORD_BYTES=72
WAVES=16
CONTRACT=json.loads(Path(__file__).with_suffix('.json').read_text())

def intermediate_extent(pack,args=None):
    args=resolve_path(pack['arguments']['file']).read_bytes() if args is None else args
    require(len(args)==160,'row arguments ABI')
    require(pack['role'] in CONTRACT,'row export role')
    c=CONTRACT[pack['role']];values=0 if c['argument_offset'] is None else c['multiplier']*struct.unpack_from('<i',args,c['argument_offset'])[0]
    require(0<=values*4<=ARENA_BYTES,'row intermediate arena extent')
    return pack['count']*values*4

def cpp():
    lines=['// Generated from export_contract.json by export_contract.cpp().','#pragma once','static size_t row_intermediate_values(const std::string& role,const std::vector<char>& args) {']
    for role,c in CONTRACT.items():
        body='return 0;' if c['argument_offset'] is None else f'int value; memcpy(&value,args.data()+{c["argument_offset"]},4); if(value<0)exit(1); return size_t(value)*{c["multiplier"]};'
        lines.append('  if(role=="row_'+role+'") { '+body+' }')
    lines+=['  fprintf(stderr,"unknown row export role\\n");exit(1);','}','']
    return '\n'.join(lines)
