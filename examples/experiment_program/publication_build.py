"""One bounded serial offline G11 build; preserve raw and matched artifacts."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path
from .build_support import ROOT,HOME_CODE,DEPS,cpath,dependencies
from .publication_subgraph import prepare,MODES,LAYERS
from .publication_capture import checking_source,BASE
from .publication_run import interface_sources
from .row_ops.elf import descriptors,allocation_signature
from examples.vllm_llm.bf16_300.build import resources
FLOOR=2<<30

def build(native,out,runtime,*,budget=20<<20):
    """20MiB bounded outputs/temporaries, stop at reserve+4MiB, no device run."""
    native=resolve_path(native);out=resolve_path(out);runtime=resolve_path(runtime)
    require(not out.exists(),'preserve owned build attempt');require(budget>=20<<20,'G11 bounded build minimum20MiB');require(shutil.disk_usage(out.parent).free>FLOOR+budget,'G11 native build budget plus protected2GiB reserve unavailable')
    c=prepare(native,out);records=[];measurements=[]
    name='etx-g11-publication-completion-build';tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    runtime_system=runtime.parents[3]/'usr/lib/x86_64-linux-gnu'
    docker=['docker','run','--rm','--name',name,'--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu:'+cpath(runtime.parent)+':'+cpath(runtime_system),'-e','TMPDIR='+cpath(out),'-w',cpath(ROOT),'etx-rocm22']
    def save():
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n');(out/'storage.json').write_text(json.dumps(dict(floor_bytes=FLOOR,budget_bytes=budget,safety_stop_bytes=FLOOR+(4<<20),measurements=measurements),indent=2)+'\n')
    def run(label,args):
        require(shutil.disk_usage(out).free>FLOOR+(4<<20),'stop before storage reserve')
        cmd=docker+args;record=dict(label=label,argv=cmd,log=str(out/(label+'.log')));records.append(record);start=time.monotonic();print('BUILD_START',label,flush=True)
        with (out/(label+'.log')).open('x') as f:
            process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            while process.poll() is None:
                free=shutil.disk_usage(out).free;measurements.append(dict(label=label,elapsed_seconds=time.monotonic()-start,free_bytes=free))
                if free<FLOOR+(4<<20) or time.monotonic()-start>180:
                    record['stop_reason']='STORAGE_MARGIN' if free<FLOOR+(4<<20) else 'STEP_DEADLINE';subprocess.run(['docker','stop','--time','1',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10);process.wait(timeout=10);record['exit']=process.returncode;save();raise RuntimeError(record['stop_reason'])
                time.sleep(.25)
            record['exit']=process.returncode;record['seconds']=time.monotonic()-start;(out/(label+'.exit')).write_text(str(process.returncode)+'\n');save();require(process.returncode==0,'native build failed: '+label)
        print('BUILD_EXIT',label,record['exit'],flush=True)
    common=['-x','hip','--cuda-device-only','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc','--rocm-device-lib-path=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode','-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-I'+cpath(out/'include'),'-I'+cpath(out),'-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(ROOT),*c['flags']]
    profiles={};physical={}
    for mode in MODES:
        extra=['-DETX_INBODY=1','-DG11_CONTROL=1'] if mode=='control' else ['-DETX_INBODY=1','-DG11_EMPTY='+str(int(mode=='empty'))]
        run(mode+'-device',[tool+'clang++',*common,*extra,'-Xclang','-dependency-file','-Xclang',cpath(out/(mode+'.d')),'-Xclang','-MT','-Xclang',mode,'-Rpass-analysis=kernel-resource-usage','-S',cpath(out/(mode+'.hip')),'-o',cpath(out/(mode+'.s'))])
        run(mode+'-object',[tool+'clang++','-target','amdgcn-amd-amdhsa','-mcpu=gfx942','-c',cpath(out/(mode+'.s')),'-o',cpath(out/(mode+'.o'))])
        run(mode+'-module',[tool+'ld.lld','-shared',cpath(out/(mode+'.o')),'-o',cpath(out/(mode+'.hsaco'))])
        profiles[mode]=resources((out/(mode+'.s')).read_text());ds=descriptors(out/(mode+'.hsaco'));physical[mode]={r['kernel']:allocation_signature(ds[r['kernel']],r['agpr']) for r in profiles[mode]}
    link=['-L'+cpath(runtime.parent),'-Wl,-rpath,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime_system),'-lamdhip64']
    for layer in LAYERS:
        label=f'host-L{layer}';run(label,[tool+'clang++','-x','c++','-std=c++17','-O2','-D__HIP_PLATFORM_AMD__','-DETX_INBODY=1','-DG11_LAYER='+str(layer),'-I/code/rocm-headers/inc','-I'+cpath(out),'-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(ROOT),'-Xclang','-dependency-file','-Xclang',cpath(out/(label+'.d')),'-Xclang','-MT','-Xclang',label,'-fuse-ld=lld',cpath(Path(__file__).with_name('publication_host.cpp')),*link,'-o',cpath(out/label)])
        run(label+'-help',[cpath(out/label),'--help'])
    # Original full-model module and host-side registration object are reused
    # byte-for-byte. Only the nine-block checking capture host is compiled.
    checker=out/'checker';checker.mkdir();old=json.loads((native/'build-identity.json').read_text());verify_files(old['hashes'])
    excluded={'build-identity.json','build_identity.json','run','commands.json','hipcc.log','host-link.log','host-link.exit'}
    for p in native.iterdir():
        if p.is_file() and p.name not in excluded:os.link(p,checker/p.name)
    (checker/'imp').mkdir();os.link(native/'imp/imports.json',checker/'imp/imports.json')
    capture=out/'capture-host.hip';capture.write_text(checking_source(subprocess.check_output(['git','show',BASE+':examples/vllm_llm/host.hip'],cwd=ROOT,text=True)))
    common=['-x','hip','--cuda-host-only','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc','-nogpulib','-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(native),'-I'+cpath(ROOT),*old['flags'],*['-I'+cpath(Path(p).parent) for p in old.get('reused_imports',{})]]
    run('capture-host-object',[tool+'clang++',*common,'-Xclang','-dependency-file','-Xclang',cpath(out/'capture.d'),'-Xclang','-MT','-Xclang','capture','-c',cpath(capture),'-o',cpath(out/'capture.o')])
    run('capture-host-link',[tool+'clang++','-fuse-ld=lld',cpath(native/'linked-module-host.o'),cpath(out/'capture.o'),*[cpath(p) for p in old['original_reference_objects']],*link,'-o',cpath(checker/'run')])
    run('capture-loader',['ldd',cpath(checker/'run')]);require('not found' not in (out/'capture-loader.log').read_text(),'genuine runtime dependency closure missing')
    deps=dependencies(out);verify_files(deps)
    (checker/'hipcc.log').write_text('REUSED exact device module '+sha(native/'model.hsaco')+'\n'+json.dumps(records,indent=2)+'\n'+(out/'capture-host-object.log').read_text()+(out/'capture-host-link.log').read_text())
    original=dict(old,host_link_exit=0,host_source_sha256=sha(capture),dependency_files={**old['dependency_files'],**deps},capture_source_guard=interface_sources(),device_reuse=dict(identity=str(native/'build-identity.json'),sha256=sha(native/'build-identity.json'),module_sha256=sha(native/'model.hsaco')),hashes={str(p):sha(p) for p in checker.iterdir() if p.is_file()})
    (checker/'build-identity.json').write_text(json.dumps(original,indent=2)+'\n')
    value=dict(c,resources=profiles,physical_resources=physical,compiler=dict(path=str(HOME_CODE/'rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/clang++'),sha256=sha(HOME_CODE/'rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/clang++')),dependency_files=deps,commands=records,native_build_complete=True,files={str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    value['interface']=dict(directory=str(out),capture_build=str(checker),host_layers=list(LAYERS),sources=interface_sources(),files={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('identity.json',)},commands=[r for r in records if r['label'].startswith(('host-','capture-'))],completed=[r['label'] for r in records if r['label'].startswith(('host-','capture-'))])
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path);p.add_argument('--matched',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args();require(bool(a.native)!=bool(a.matched),'choose --native build or --matched host finalization');v=finish(a.matched,a.out,a.runtime) if a.matched else build(a.native,a.out,a.runtime);print(json.dumps(dict(native_complete=v['native_build_complete'],gpu_executed=False)))


def finish(matched,out,runtime):
    """Seal final Python interfaces and relink only changed local host sources.

    The actual gfx942 instruction streams and their original failed/raw receipts
    are preserved; final descriptors are re-read by admission.
    """
    matched=resolve_path(matched);out=resolve_path(out);runtime=resolve_path(runtime)
    require(not out.exists(),'preserve final interface attempt');require(shutil.disk_usage(out.parent).free>FLOOR+(6<<20),'final host relink requires6MiB above2GiB floor')
    old=json.loads((matched/'identity.json').read_text());verify_files(old['files']);raw=resolve_path(old['raw_build_identity']['path']).parent
    # Only Python admission/reader logic and host-side endpoints can change in
    # this finalization; native phase source remains exactly the compiled source.
    from .publication_subgraph import tiles
    require((raw/'tiles.hip').read_text()==tiles((resolve_path(old['native'])/'dsl_tiles.hip').read_text()),'changed device source requires a real affected device rebuild')
    out.mkdir()
    for p in matched.iterdir():
        if p.is_file() and p.name not in ('identity.json','image-dependencies.json') and not p.name.startswith('host-L'):os.link(p,out/p.name)
    commands=[];measurements=[];name='etx-g11-publication-final-hosts';tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    docker=['docker','run','--rm','--name',name,'--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu:'+cpath(runtime.parent)+':'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-e','TMPDIR='+cpath(out),'-w',cpath(ROOT),'etx-rocm22']
    for layer in LAYERS:
        label=f'host-L{layer}'
        cmd=[tool+'clang++','-x','c++','-std=c++17','-O2','-D__HIP_PLATFORM_AMD__','-DETX_INBODY=1','-DG11_LAYER='+str(layer),'-I/code/rocm-headers/inc','-I'+cpath(raw),'-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(ROOT),'-Xclang','-dependency-file','-Xclang',cpath(out/(label+'.d')),'-Xclang','-MT','-Xclang',label,'-fuse-ld=lld',cpath(Path(__file__).with_name('publication_host.cpp')),'-L'+cpath(runtime.parent),'-Wl,-rpath,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-lamdhip64','-o',cpath(out/label)]
        for step,args in ((label,cmd),(label+'-help',[cpath(out/label),'--help'])):
            argv=docker+args;start=time.monotonic();print('FINAL_BUILD_START',step,flush=True)
            with (out/(step+'.log')).open('x') as f:
                p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    free=shutil.disk_usage(out).free;measurements.append(dict(label=step,free_bytes=free,elapsed_seconds=time.monotonic()-start))
                    if free<FLOOR+(2<<20) or time.monotonic()-start>60:
                        subprocess.run(['docker','stop','--time','1',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10);p.wait(timeout=10);raise RuntimeError('final host storage/deadline stop')
                    time.sleep(.25)
            record=dict(label=step,argv=argv,exit=p.returncode,log=str(out/(step+'.log')));commands.append(record);(out/(step+'.exit')).write_text(str(p.returncode)+'\n');(out/'final-commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(p.returncode==0,'final host build failed '+step)
    (out/'final-storage.json').write_text(json.dumps(dict(floor_bytes=FLOOR,budget_bytes=6<<20,measurements=measurements),indent=2)+'\n')
    checker=out/'checker';checker.mkdir();prior=raw/'checker';b=json.loads((prior/'build-identity.json').read_text())
    for p in prior.iterdir():
        if p.is_file() and p.name!='build-identity.json':os.link(p,checker/p.name)
    (checker/'imp').mkdir();os.link(prior/'imp/imports.json',checker/'imp/imports.json')
    capture_deps=out/'capture-dependencies';capture_deps.mkdir();os.link(raw/'capture.d',capture_deps/'capture.d')
    dep=dependencies(capture_deps);original=json.loads((resolve_path(old['native'])/'build-identity.json').read_text());b.update(dependency_files={**original['dependency_files'],**dep},hashes={str(p):sha(p) for p in checker.iterdir() if p.is_file()})
    (checker/'build-identity.json').write_text(json.dumps(b,indent=2)+'\n')
    native_deps=dict(old['dependency_files']);native_deps.update(dependencies(out));verify_files(native_deps)
    selected=[r for r in old['interface']['commands'] if r['label'].startswith('capture-')]+commands
    files={str(p):sha(p) for p in out.rglob('*') if p.is_file()};interface=dict(directory=str(out),capture_build=str(checker),host_layers=list(LAYERS),sources=interface_sources(),files=files,commands=selected,completed=[r['label'] for r in selected])
    result=dict(old,interface=interface,dependency_files=native_deps,files=files,finalization=dict(prior_identity=str(matched/'identity.json'),prior_sha256=sha(matched/'identity.json'),host_source_sha256=sha(Path(__file__).with_name('publication_host.cpp')),device_instructions_rebuilt=False,reason='correct host timing endpoint/warmup and freeze final CPU interface closure'))
    (out/'identity.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':main()
