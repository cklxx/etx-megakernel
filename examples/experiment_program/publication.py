"""Full-chain publication brackets from the original atomic, never a point fiction.

The release atomic has no architectural timestamp output. A safe diagnostic
records before/retired-after bounds without making consumers wait for a stamp.
Those bounds can qualify an exposure interval; they cannot provide an exact
atomic-publication point. Canonical G11 permits separately labeled issue/retirement
bounds on an accepted current subgraph; no exact point or full-model probe is required.
"""
import json,shutil
from pathlib import Path
from .protocol import require,resolve_path,sha

def prepare(native,out):
    from etx.codegen.inbody import instrument
    from .validation_prepare import host_reporting
    from .build_support import ROOT
    native=resolve_path(native);out=resolve_path(out);require(not out.exists(),'preserve publication attempt');out.mkdir()
    private=out/'include/etx';private.mkdir(parents=True)
    for p in (ROOT/'etx/runtime/include/etx').iterdir():
        if p.is_file():shutil.copyfile(p,private/p.name)
    abi=private/'abi.h';s=abi.read_text().replace('  uint64_t* inbody_event;', '  int32_t inbody_empty; // diagnostic-only matched empty mode\n  uint64_t* inbody_event;');abi.write_text(s)
    header=private/'inbody.h';s=header.read_text();anchor='  uint64_t e = (p).inbody_event ? etx_inbody_tick() : 0; \\'
    require(s.count(anchor)==1,'publication completion anchor')
    s=s.replace(anchor,'  asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); /* retire original atomic */ \\\n'+anchor+'\n  if ((p).inbody_empty) b = e; \\')
    header.write_text(s)
    src=out/'megakernel_d0.hip';src.write_text(instrument((native/'megakernel_d0.hip').read_text()))
    host=out/'publication-host.hip';host_reporting(ROOT/'examples/vllm_llm/host.hip',host,'qwen');s=host.read_text()
    # Post-setup mode binding; ordinary source/ABI and decisive pairs are untouched.
    setup='  hst.setup();';require(s.count(setup)==1,'publication host setup')
    s=s.replace(setup,setup+'''
  const char* publication_mode=getenv("ETX_PROGRAM_PUBLICATION_MODE");
  if(publication_mode){
    if(strcmp(publication_mode,"control") && strcmp(publication_mode,"probe") && strcmp(publication_mode,"empty"))die("publication mode");
    hst.p.inbody_empty=!strcmp(publication_mode,"empty");
    if(!strcmp(publication_mode,"control")){hst.p.inbody_event=nullptr;hst.p.inbody_poll=nullptr;hst.p.inbody_wave=nullptr;hst.p.arrival_stamps=nullptr;}
  }
''')
    anchor='        hst.report_phases(etx_type_names,ETX_N_TYPES);'
    require(s.count(anchor)==1,'exact publication export guard')
    s=s.replace(anchor,anchor+'\n        if(const char* raw=getenv("ETX_PROGRAM_PUBLICATION_RAW"))hst.write_inbody(raw);')
    host.write_text(s)
    value=dict(schema='etx.publication.preparation.v1',native=str(native),host=str(host),device=str(src),private_headers=str(out/'include'),modes=['control','probe','empty'],same_binary_and_allocation=True,original_counter_called_once=True,consumer_wait_added=False,publication_point='UNAVAILABLE_ARCHITECTURAL_TIMESTAMP',measurement='safe retired atomic bounds plus same-clock poll/body/wave endpoints',files={str(p):sha(p) for p in out.rglob('*') if p.is_file()},gpu_executed=False)
    (out/'preparation.json').write_text(json.dumps(value,indent=2)+'\n');return value

def analyze(plan_file,raw_file,layers=(1,24,47)):
    from etx.tools.inbody import read_raw,analyze as read
    p=json.loads(resolve_path(plan_file).read_text());raw=read_raw(resolve_path(raw_file));report=read(p,raw)
    # read() checks complete task/producer/wave populations and original edges.
    words=__import__('etx.tools.inbody',fromlist=['event_words']).event_words(p)
    names={x['id']:n for n,x in p['types'].items()};result=[]
    for d in p['descs']:
        if names[d['type']] not in [f'qkv_{l}' for l in layers]:continue
        tid=d['id'];dependencies=p['task_in'][tid];require(dependencies,'fold consumer dependencies absent')
        brackets=[raw['events'][words[e]] for e in dependencies]
        lo=max(r[0] for r in brackets);hi=max(r[1] for r in brackets);poll=raw['poll'][tid][0];body=raw['legacy'][tid]
        require(lo<=poll<=body[1]<=body[2],'full-chain publication/consumer order invalid')
        result.append(dict(task=tid,worker=body[3],producer_tasks=[r[2] for r in brackets],publication_ticks=[lo,hi],publication_to_poll_ns=[max(0,poll-hi)*10,(poll-lo)*10],publication_to_body_done_ns=[max(0,body[2]-hi)*10,(body[2]-lo)*10],atomic_point_ns=None))
    require(result and {names[p['descs'][r['task']]['type']] for r in result}=={f'qkv_{l}' for l in layers},'early/middle/late fold consumer coverage')
    return dict(schema='etx.publication.observation.v1',scope='same native full chain and clock; bounds only, no removable-work price',observations=result,exact_publication_timestamp_available=False,qualification='REQUIRES_MATCHED_NATIVE_CONTROL_EMPTY_AND_500NS_GATE',source_analysis=report)

def compile(preparation,runtime):
    """One common diagnostic binary; control/probe/empty share native allocation."""
    from .build_support import HOME_CODE,DEPS,ROOT,cpath,dependencies
    from .model_build import run
    c=json.loads(resolve_path(preparation).read_text());out=resolve_path(preparation).parent;native=resolve_path(c['native']);runtime=resolve_path(runtime)
    require(shutil.disk_usage(out).free>(2<<30)+(64<<20),'publication build needs64MiB write reserve above2GiB floor')
    require(not (out/'commands.json').exists(),'preserve native publication attempt')
    old=json.loads((native/'build-identity.json').read_text());flags=old['flags']+['-DETX_INBODY=1','-DETX_PRODUCER_LEDGER=1']
    docker=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu:'+cpath(runtime.parent)+':'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-w',cpath(ROOT),'etx-rocm22']
    tool='/code/rocm-headers/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/';commands=[]
    def action(name,args):
        cmd=docker+args;commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');run(cmd,out,name,seconds=240)
    common=['-x','hip','--offload-arch=gfx942','-O3','-std=c++17','-nogpuinc','--rocm-device-lib-path=/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode','-isystem','/code/rocm-headers/inc','-include','__clang_hip_runtime_wrapper.h','-D__HIP_PLATFORM_AMD__','-I'+cpath(out/'include'),'-I'+cpath(native),'-I'+cpath(ROOT/'etx/runtime/include'),'-I'+cpath(ROOT),*['-I'+cpath(Path(p).parent) for p in old.get('reused_imports',{})],*flags]
    imports=next(iter(old['reused_imports']));extra=['-Xclang','-mlink-builtin-bitcode','-Xclang',cpath(imports)]
    action('publication-device',[tool+'clang++',*common,'--cuda-device-only',*extra,'-Rpass-analysis=kernel-resource-usage','-S',cpath(c['device']),'-o',cpath(out/'publication.s')])
    action('publication-object',[tool+'llvm-mc','-triple=amdgcn-amd-amdhsa','-mcpu=gfx942','-filetype=obj',cpath(out/'publication.s'),'-o',cpath(out/'publication.o')])
    action('publication-module',[tool+'ld.lld','-shared',cpath(out/'publication.o'),'-o',cpath(out/'publication.hsaco')])
    action('publication-bundle',[tool+'clang-offload-bundler','--type=o','--bundle-align=4096','--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942','--input=/dev/null','--input='+cpath(out/'publication.hsaco'),'--output='+cpath(out/'publication.hipfb')])
    action('publication-module-host',[tool+'clang++',*common,'--cuda-host-only','-Xclang','-fcuda-include-gpubinary','-Xclang',cpath(out/'publication.hipfb'),'-c',cpath(c['device']),'-o',cpath(out/'publication-host-module.o')])
    action('publication-checker-host',[tool+'clang++',*common,'--cuda-host-only','-c',cpath(c['host']),'-o',cpath(out/'publication-checker-host.o')])
    originals=[cpath(p) for p in old['original_reference_objects']]
    action('publication-host-link',[tool+'clang++','-fuse-ld=lld',cpath(out/'publication-host-module.o'),cpath(out/'publication-checker-host.o'),*originals,'-L'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parent),'-Wl,-rpath-link,'+cpath(runtime.parents[3]/'usr/lib/x86_64-linux-gnu'),'-lamdhip64','-o',cpath(out/'run')])
    action('publication-loader',['ldd',cpath(out/'run')]);require('not found' not in (out/'publication-loader.log').read_text(),'publication runtime loader incomplete')
    from examples.vllm_llm.bf16_300.build import resources
    from .row_ops.elf import descriptors,allocation_signature
    res=resources((out/'publication.s').read_text());d=descriptors(out/'publication.hsaco')
    value=dict(c,host_link_exit=0,flags=flags,resources=res,physical_resources={r['kernel']:allocation_signature(d[r['kernel']],r['agpr']) for r in res},commands=commands,original_reference_objects=old['original_reference_objects'],reused_imports=old['reused_imports'],files={str(p):sha(p) for p in out.rglob('*') if p.is_file()},gpu_executed=False,perturbation='UNMEASURED',qualified=False)
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');return value


def admission(identity):
    """Enforce the registered original-resource gate before any G11 device work."""
    from .protocol import verify_files
    from .native_resources import persistent_capacity
    from examples.vllm_llm.bf16_300.build import resources
    p=resolve_path(identity);c=json.loads(p.read_text())
    if c.get('schema')=='etx.g11.subgraph.preparation.v1':
        from .publication_subgraph import admission as subgraph_admission
        return subgraph_admission(p)
    verify_files(c['files'])
    original=persistent_capacity(resolve_path(c['native']))['raw_metadata']
    probe=next(r for r in resources((p.parent/'publication.s').read_text()) if r['kernel']=='etx_megakernel_d0')
    keys=('vgpr','agpr','lds_bytes','scratch_bytes_lane','vgpr_spills','sgpr_spills')
    equal=all(original[k]==probe[k] for k in keys)
    return dict(status='CLOSED_NATIVE_DIAGNOSTIC_RESOURCE' if not equal else 'INCOMPLETE_CANONICAL_SUBGRAPH_COVERAGE',admitted=False,original=original,probe=probe,resources_equal=equal,exact_publication_timestamp_available=False,publication_measurement='retired original-atomic bounds only; no point timestamp fabricated',owner_protocol_change_required=False,exact_atomic_point_required=False,whole_model_instrumented_build_required=False,canonical_subgraph_alternative_allowed=True,gpu_executed=False)
