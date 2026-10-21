"""Reserve equal native descriptor registers; preserve every instruction byte."""
import json,re,os,shutil,subprocess,struct,hashlib
from pathlib import Path
from .protocol import require,sha
from .build_support import HOME_CODE,DEPS,cpath
from .row_ops.elf import descriptors,allocation_signature

def text_bytes(path):
    data=Path(path).read_bytes();off=struct.unpack_from('<Q',data,40)[0];size,n,si=struct.unpack_from('<HHH',data,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,off+i*size) for i in range(n)];ss=sections[si];names=data[ss[4]:ss[4]+ss[5]]
    for s in sections:
        if names[s[0]:names.find(b'\0',s[0])]==b'.text':return data[s[4]:s[4]+s[5]]
    raise ValueError('missing native text')

def match(source,out,groups,*,write_budget_bytes=40<<20):
    source=Path(source).resolve();out=Path(out).resolve();require(not out.exists(),'preserve diagnostic build');required=sum((source/(n+suffix)).stat().st_size for group in groups for n in group for suffix in ('.s','.o','.hsaco'))*2+(2<<20);require(write_budget_bytes>=required,'descriptor reserve budget below bounded outputs');require(shutil.disk_usage(source).free>(2<<30)+write_budget_bytes,'diagnostic descriptor reserve requires2GiB floor+declared write budget')
    out.mkdir();raw=json.loads((source/'identity.json').read_text());report={};resources={}
    native_names={x for group in groups for x in group}
    for p in source.iterdir():
        if p.is_file() and p.name!='identity.json' and not any(p.name==n+suffix for n in native_names for suffix in ('.s','.o','.hsaco')):os.link(p,out/p.name)
    tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','etx-rocm22']
    commands=[]
    for group in groups:
        files={n:(source/(n+'.s')).read_text() for n in group};limits={}
        rx=re.compile(r'(?s)(\.amdhsa_kernel\s+(\S+)\n)(.*?)(\.end_amdhsa_kernel)')
        for text in files.values():
            for m in rx.finditer(text):
                lds=int(re.search(r'\.amdhsa_group_segment_fixed_size (\d+)',m[3])[1]);limits[m[2],'lds']=max(limits.get((m[2],'lds'),0),lds)
                for kind in ('vgpr','sgpr'):
                    count=int(re.search(r'\.amdhsa_next_free_'+kind+r' (\d+)',m[3])[1]);limits[m[2],kind]=max(limits.get((m[2],kind),0),count)
        for n,text in files.items():
            def edit(m):
                body=m[3]
                for kind in ('vgpr','sgpr'):body=re.sub(r'(\.amdhsa_next_free_'+kind+r' )\d+',lambda q:q[1]+str(limits[m[2],kind]),body)
                body=re.sub(r'(\.amdhsa_group_segment_fixed_size )\d+',lambda q:q[1]+str(limits[m[2],'lds']),body)
                return m[1]+body+m[4]
            (out/(n+'.s')).write_text(rx.sub(edit,text))
            for label,args in [('object',[tool+'llvm-mc','-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',cpath(out/(n+'.s')),'-o',cpath(out/(n+'.o'))]),('link',[tool+'ld.lld','-shared',cpath(out/(n+'.o')),'-o',cpath(out/(n+'.hsaco'))])]:
                cmd=docker+args
                with (out/(n+'.matched-'+label+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=60)
                commands.append(dict(argv=cmd,exit=r.returncode));require(r.returncode==0,'diagnostic matched native build failed')
            require(text_bytes(source/(n+'.hsaco'))==text_bytes(out/(n+'.hsaco')),'descriptor reserve changed instructions')
            ds=descriptors(out/(n+'.hsaco'));resources[n]={kernel:allocation_signature(d,next(r['agpr'] for r in raw['resources'][n] if r['kernel']==kernel)) for kernel,d in ds.items()}
            report[n]=dict(original_module_sha256=sha(source/(n+'.hsaco')),matched_module_sha256=sha(out/(n+'.hsaco')),instruction_sha256=hashlib.sha256(text_bytes(out/(n+'.hsaco'))).hexdigest(),instruction_bytes_unchanged=True)
        require(all(resources[n]==resources[group[0]] for n in group),'diagnostic physical allocations still differ')
    (out/'allocation-matching.json').write_text(json.dumps(dict(groups=groups,modules=report,physical_resources=resources,commands=commands),indent=2)+'\n')
    value=dict(raw,physical_resources=resources,raw_build_identity=dict(path=str(source/'identity.json'),sha256=sha(source/'identity.json')),files={str(p):sha(p) for p in out.iterdir() if p.is_file()})
    if 'hashes' in raw:value['hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');return value
