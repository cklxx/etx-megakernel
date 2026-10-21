"""Explicit diagnostic-only register reservations; instruction bytes unchanged.

Original optimization code objects remain immutable. An optional matched
uninstrumented module is used only to qualify sparse probes. Raw original,
matched-original, empty and probe timings must separately stay within500ns.
Actual scratch or spills reject this path; reservations never hide them.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
from .build import ROOT,command
from .diagnostic import verify_build
from .evidence import require,sha

RX=re.compile(r'(?s)(\.amdhsa_kernel\s+(\S+)\n)(.*?)(\.end_amdhsa_kernel)')

def counts(source):
    result={}
    for match in RX.finditer(source):
        if match[2] in ('io_control','io_candidate'):
            result[match[2]]={key:int(re.search(r'\.amdhsa_next_free_'+key+r' (\d+)',match[3])[1]) for key in ('vgpr','sgpr')}
    require(set(result)=={'io_control','io_candidate'},'reserve kernel coverage');return result

def reserve(source,targets):
    original=counts(source);changes={}
    def patch(match):
        name=match[2]
        if name not in targets:return match[0]
        body=match[3];changes[name]={}
        for kind in ('vgpr','sgpr'):
            before=original[name][kind];after=targets[name][kind]
            require(after>=before,'reserve cannot reduce registers')
            body=re.sub(r'(\.amdhsa_next_free_'+kind+r' )\d+',lambda m:m[1]+str(after),body)
            changes[name][kind]={'original':before,'reserved':after}
        return match[1]+body+match[4]
    padded=RX.sub(patch,source)
    for name,fields in changes.items():
        for kind,pair in fields.items():
            pattern=r'(?s)(\.name:\s*'+re.escape(name)+r'\s*\n.*?\.'+kind+r'_count:\s*)(\d+)'
            padded,n=re.subn(pattern,lambda m:m[1]+str(int(m[2])+pair['reserved']-pair['original']),padded)
            require(n==1,'reserve metadata coverage')
    return padded,changes

def verify_reserve_only(original,padded):
    def stripped(text):
        text=re.sub(r'(\.amdhsa_next_free_(?:vgpr|sgpr) )\d+',r'\1RESERVED',text)
        return re.sub(r'(\.(?:vgpr|sgpr)_count:\s*)\d+',r'\1RESERVED',text)
    require(stripped(original)==stripped(padded),'reservation changed code or non-register metadata')
    return True

def text_bytes(path):
    data=Path(path).read_bytes();require(data[:6]==b'\x7fELF\x02\x01','native ELF')
    offset=struct.unpack_from('<Q',data,40)[0];size,count,strings=struct.unpack_from('<HHH',data,58)
    require(size==64 and offset+size*count<=len(data),'native section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    names=sections[strings];names=data[names[4]:names[4]+names[5]]
    for section in sections:
        name=names[section[0]:names.find(b'\0',section[0])]
        if name==b'.text':return data[section[4]:section[4]+section[5]]
    raise ValueError('missing native instruction section')

def prepare(base,probe,out,headers):
    from examples.vllm_llm.bf16_300.build import resources
    from examples.experiment_program.row_ops.elf import descriptors,allocation_signature
    base=Path(base);probe=Path(probe);out=Path(out);raw=verify_build(base)
    diag=json.loads((probe/'identity.json').read_text())
    require(diag['base_build_sha256']==sha(base/'identity.json'),'probe/base identity')
    for name,digest in diag['hashes'].items():require(sha(probe/name)==digest,'stale raw diagnostic '+name)
    require(all(not r[k] for r in diag['resources'] for k in ('scratch_bytes_lane','vgpr_spills','sgpr_spills')),'actual probe scratch/spills cannot be matched away')
    require(not out.exists(),'fresh matching output');out.mkdir(parents=True)
    sources={'uninstrumented':(base/'kernel.s').read_text(),'diagnostic':(probe/'kernel.s').read_text()}
    targets={name:{kind:max(counts(s)[name][kind] for s in sources.values()) for kind in ('vgpr','sgpr')} for name in ('io_control','io_candidate')}
    llvm=Path(headers)/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin';record={};commands=[]
    for role,source in sources.items():
        padded,changes=reserve(source,targets);verify_reserve_only(source,padded);(out/(role+'.raw.s')).write_text(source);(out/(role+'.s')).write_text(padded)
        for name,cmd in ((role+'_object',[llvm/'clang++','-target','amdgcn-amd-amdhsa','-mcpu=gfx942','-c',out/(role+'.s'),'-o',out/(role+'.o')]),(role+'_link',[llvm/'ld.lld','-shared',out/(role+'.o'),'-o',out/(role+'.hsaco')]),(role+'_notes',[llvm/'llvm-readobj','--notes',out/(role+'.hsaco')])):
            commands.append(list(map(str,cmd)));command(cmd,out,name)
        reference=(base if role=='uninstrumented' else probe)/'kernel.hsaco'
        require(text_bytes(out/(role+'.hsaco'))==text_bytes(reference),'reservation altered native instructions')
        native=descriptors(out/(role+'.hsaco'));rs=[r for r in resources((out/(role+'.s')).read_text()) if r['kernel'] in targets]
        for r in rs:
            sgpr=re.search(r'(?s)\.name:\s*'+r['kernel']+r'\s.*?\.sgpr_count:\s*(\d+)',padded)
            r['sgpr']=int(sgpr[1]);r['descriptor']=native[r['kernel']];r['physical_signature']=allocation_signature(native[r['kernel']],r['agpr'])
        import hashlib
        record[role]={'changes':changes,'raw_source_sha256':sha(out/(role+'.raw.s')),'resources':rs,'text_sha256':hashlib.sha256(text_bytes(reference)).hexdigest(),'native_text_unchanged':True}
    equal={arm:next(r for r in record['uninstrumented']['resources'] if r['kernel']=='io_'+arm)['physical_signature']==next(r for r in record['diagnostic']['resources'] if r['kernel']=='io_'+arm)['physical_signature'] for arm in ('control','candidate')}
    value={'schema':'etx.issue-order.matched-diagnostic.v1','record':raw['record'],'base_sha256':sha(base/'identity.json'),'probe_sha256':sha(probe/'identity.json'),'roles':record,'physical_equal':equal,'status':'MATCHED_DIAGNOSTIC_ONLY' if all(equal.values()) else 'RESOURCE_REJECTED','commands':commands,'gpu_execution':'NOT_RUN','optimization_modules_modified':False,'required_runtime_qualification':'all raw original/matched original/empty/probe byte gates and <=500ns perturbation vs raw original','source_hashes':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'examples/experiment_program/row_ops/elf.py':sha(ROOT/'examples/experiment_program/row_ops/elf.py')},'hashes':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':value['status'],'physical_equal':equal,'roles':record},indent=2));return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--headers',type=Path,default=Path.home()/'code/rocm-headers');p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.inside:prepare(a.base,a.probe,a.out,a.headers);return
    def mapped(path):return '/code/'+str(Path(path).resolve().relative_to(Path.home()/'code'))
    cmd=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(Path.home()/'code')+':/code','-v',str(Path.home()/'.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root')+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','-e','TMPDIR='+mapped(a.out.parent),'-e','PYTHONPATH='+mapped(ROOT),'-e','PYTHONDONTWRITEBYTECODE=1','-w',mapped(ROOT),'etx-rocm22','python3','-m','examples.experiment_program.issue_order.diagnostic_match','--inside','--base',mapped(a.base),'--probe',mapped(a.probe),'--out',mapped(a.out),'--headers',mapped(a.headers)]
    subprocess.run(cmd,check=True,timeout=180)
if __name__=='__main__':main()
