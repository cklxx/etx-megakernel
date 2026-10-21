"""Ordinary HIP host link reusing an exact sealed device module, no device calls."""
import json,os,shutil
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path
from .build_support import HOME_CODE,DEPS,ROOT,cpath,dependencies
from .model_build import run

def stage_selected_header(native,out):
    """Derive the host ABI from the same selected manifest as the reused module."""
    from etx.importer.adapter import emit_host_header
    selected=json.loads((native/'imp/imports.json').read_text())
    for line in (native/'vl_plan.txt').read_text().splitlines():
        fields=line.split()
        if not fields or fields[0].startswith('#') or fields[0]=='total':continue
        require(fields[1] in selected,'selected plan export lacks descriptor: '+fields[1])
    # Fresh output only: never overwrite a hard-linked legacy descriptor.
    with (out/'imports.h').open('x') as f:f.write(emit_host_header(selected))


def link_existing(native,out,runtime,*,host_source=None):
    native=resolve_path(native);out=resolve_path(out);runtime=resolve_path(runtime)
    old=json.loads((native/'build-identity.json').read_text());verify_files(old['hashes'])
    require(runtime.is_file(),'genuine runtime missing')
    require(shutil.disk_usage(out.parent).free>(2<<30)+(40<<20),'ordinary host link requires reserve plus2GiB floor')
    out.mkdir(exist_ok=False);(out/'tmp').mkdir()
    # Compiler outputs (including dependency files) must never alias the input.
    # Private copies retain bytes without allowing a host rebuild to mutate seals.
    for p in native.iterdir():
        if p.is_file() and p.name not in ('build-identity.json','run','host-link.log','host-link.exit','commands.json','image-dependencies.json','imports.h','hipcc.log','hipcc_command.json'):
            shutil.copyfile(p,out/p.name)
    if host_source:stage_selected_header(native,out)
    elif (native/'imports.h').exists():os.link(native/'imports.h',out/'imports.h')
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu:'+cpath(runtime.parent)+':'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-e','TMPDIR='+cpath(out/'tmp'),'-w',cpath(ROOT),'etx-rocm22']
    tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/'
    commands=[]
    def action(name,args):
        cmd=docker+args;commands.append(cmd);run(cmd,out,name,seconds=180)
    action('bundle',[tool+'clang-offload-bundler','--type=o','--bundle-align=4096','--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942','--input=/dev/null','--input='+cpath(native/'model.hsaco'),'--output='+cpath(out/'linked.hipfb')])
    common=['-x','hip','--cuda-host-only','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc','-nogpulib','-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(out),'-I'+cpath(native),'-I'+cpath(ROOT),*old['flags'],*['-I'+cpath(Path(p).parent) for p in old.get('reused_imports',{})]]
    action('module-host-object',[tool+'clang++',*common,'-Xclang','-fcuda-include-gpubinary','-Xclang',cpath(out/'linked.hipfb'),'-c',cpath(native/'megakernel_d0.hip'),'-o',cpath(out/'linked-module-host.o')])
    host=out/'host.o'
    if host_source:
        host=out/'linked-capture-host.o'
        action('capture-host-object',[tool+'clang++',*common,'-Xclang','-dependency-file','-Xclang',cpath(out/'capture-host.d'),'-Xclang','-MT','-Xclang','capture-host','-c',cpath(host_source),'-o',cpath(host)])
    original=[resolve_path(p) for p in old.get('original_reference_objects',{})]
    action('host-link',[tool+'clang++','-fuse-ld=lld',cpath(out/'linked-module-host.o'),cpath(host),*[cpath(p) for p in original],'-L'+cpath(runtime.parent),'-Wl,-rpath,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-lamdhip64','-o',cpath(out/'run')])
    # ldd performs loader dependency resolution only; never invoke the model.
    action('loader-dependencies',['ldd',cpath(out/'run')])
    require('not found' not in (out/'loader-dependencies.log').read_text(),'genuine loader dependency closure incomplete')
    dep_files={**old['dependency_files'],**dependencies(out)} if host_source else old['dependency_files']
    runtime_files={str(p):sha(p) for root in (runtime.parent,runtime.parents[3]/'usr/lib/x86_64-linux-gnu') for p in root.glob('*.so*') if p.is_file()}
    metadata_files={}
    if (native/'imp/imports.json').is_file():
        (out/'imp').mkdir();os.link(native/'imp/imports.json',out/'imp/imports.json');metadata_files[str(out/'imp/imports.json')]=sha(out/'imp/imports.json')
    (out/'hipcc_command.json').write_text(json.dumps(commands,indent=2)+'\n')
    (out/'hipcc.log').write_text(''.join((out/(name+'.log')).read_text() for name in ('module-host-object','capture-host-object','host-link') if (out/(name+'.log')).exists()))
    value=dict(old,metadata_files=metadata_files,dependency_files=dep_files,host_link_exit=0,commands=commands,device_reuse=dict(identity=str(native/'build-identity.json'),sha256=sha(native/'build-identity.json'),module_sha256=sha(native/'model.hsaco')),runtime_files=runtime_files,host_source_sha256=sha(host_source) if host_source else old['host_source_sha256'],hashes={str(p):sha(p) for p in out.iterdir() if p.is_file()},gpu_executed=False)
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');value['hashes'][str(out/'commands.json')]=sha(out/'commands.json')
    (out/'build-identity.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--host-source',type=Path)
    a=p.parse_args();value=link_existing(a.native,a.out,a.runtime,host_source=a.host_source);print(json.dumps(dict(host_link_exit=value['host_link_exit'],native_reused=True,gpu_executed=False),indent=2))

if __name__=='__main__':main()


def complete_qwen_metadata(linked,out,reference):
    """Stage the selected binary with its exact import metadata and legacy sources."""
    linked=resolve_path(linked);out=resolve_path(out);reference=resolve_path(reference)
    n=json.loads((linked/'build-identity.json').read_text());verify_files(n['hashes']);require(n['host_link_exit']==0,'ordinary host link required first')
    original=resolve_path(n['device_reuse']['identity']).parent
    require(sha(original/'model.hsaco')==sha(linked/'model.hsaco'),'metadata belongs to another native module')
    require(not out.exists(),'preserve staged native metadata');out.mkdir();(out/'imp').mkdir()
    for p in linked.iterdir():
        if p.is_file() and p.name not in ('build-identity.json','hipcc.log','compiler_version.txt','hipcc_command.json'):os.link(p,out/p.name)
    import_metadata=original/'imp/imports.json';require(import_metadata.is_file(),'original selected import metadata unavailable')
    os.link(import_metadata,out/'imp/imports.json')
    for name,p in [('imports.bc',resolve_path(next(iter(n['reused_imports'])))),('imp_tiles.hip',reference/'imp_tiles.hip')]:
        destination=out/name
        if destination.exists():require(sha(destination)==sha(p),'selected convenience artifact differs')
        else:os.link(p,destination)
    (out/'hipcc.log').write_text(''.join((out/name).read_text() for name in ('compiler.log','device.log','device-object.log','device-link.log','host-object.log','host-link.log')))
    (out/'compiler_version.txt').write_text((out/'compiler.log').read_text());(out/'hipcc_command.json').write_text(json.dumps(n['commands'],indent=2)+'\n')
    value=dict(n,metadata_completion=dict(previous_identity=str(linked/'build-identity.json'),previous_identity_sha256=sha(linked/'build-identity.json'),selected_import_metadata_source=str(import_metadata)),metadata_files={str(out/'imp/imports.json'):sha(out/'imp/imports.json')},hashes={str(p):sha(p) for p in out.iterdir() if p.is_file()})
    (out/'build-identity.json').write_text(json.dumps(value,indent=2)+'\n');return value
