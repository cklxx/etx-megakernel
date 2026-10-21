"""Recompile only the changed export-contract host; reuse sealed device modules."""
import json,os,subprocess,shutil
from pathlib import Path
from examples.experiment_program.protocol import require,sha,verify_files
from examples.experiment_program.build_support import HOME_CODE,ROOT,DEPS,cpath,dependencies

def refresh(source,out,*,diagnostic):
    source=Path(source).resolve();out=Path(out).resolve();identity=json.loads((source/'identity.json').read_text())
    require(not out.exists(),'preserve host refresh output');require(shutil.disk_usage(source).free>(2<<30)+(40<<20),'replay host reserve above2GiB')
    if identity.get('files'):verify_files(identity['files'])
    else:
        for name,digest in identity['hashes'].items():require(sha(source/name)==digest,'changed sealed row build')
    out.mkdir();skip={'identity.json','replay','host.d','host.log','host-help.log','commands.json','image-dependencies.json','host-refresh.log'}
    for p in source.iterdir():
        if p.is_file() and p.name not in skip:os.link(p,out/p.name)
    host=Path(__file__).with_name('diagnostic_replay.cpp' if diagnostic else 'replay.cpp')
    args=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','etx-rocm22','/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/clang++','-std=c++17','-O2','-D__HIP_PLATFORM_AMD__','-isystem','/code/rocm-headers/inc','-I'+cpath(ROOT),'-Xclang','-dependency-file','-Xclang',cpath(out/'host.d'),'-Xclang','-MT','-Xclang','host',cpath(host),'-ldl','-o',cpath(out/'replay')]
    with (out/'host-refresh.log').open('x') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    (out/'host-refresh.exit').write_text(str(r.returncode)+'\n');require(r.returncode==0,'changed replay host link failed')
    (out/'commands.json').write_text(json.dumps([dict(argv=args,exit=r.returncode)],indent=2)+'\n')
    value=dict(identity,host_source_sha256=sha(host),reused_native_identity=dict(path=str(source/'identity.json'),sha256=sha(source/'identity.json')),dependency_files=dependencies(out),files={str(p):sha(p) for p in out.iterdir() if p.is_file()},gpu_executed=False)
    if not diagnostic:value['hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');return value
