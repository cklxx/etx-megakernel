"""Winner-only G12 source/import overlay; original certification reference retained.

Local validation emits disabled model sources. A future actual winner is bound
to its exact micro build/capture identity before any model arm can be enabled.
The caller must build/check the final megakernel; operator allocation is not a
model certificate and no graph/fusion availability is fabricated here.
"""
import argparse
import json
from pathlib import Path
import subprocess
from .source import RECORDS,transform,prep_source,ATTENTION_SHA
from .evidence import require,sha,EXPORTS,decide
from .build import ROOT,command
from examples.vllm_llm.kept_build import BASE_FLAGS,KEPT_ENV
from examples.experiment_program.protocol import resolve_path


def admit_winner(winner,record,micro_identity):
    require(winner.get('status')=='MICRO_WINNER' and winner.get('record')==record,'no admitted issue-order winner')
    require(micro_identity.get('record')==record and micro_identity.get('admission',{}).get('timing_eligible') is True,'unadmitted native micro identity')
    require(winner.get('build_sha256')==micro_identity.get('_identity_sha256'),'winner/build mismatch')
    require(winner.get('capture_seal_sha256') and winner.get('results'),'missing winner native input/timing population')
    logs=resolve_path(winner.get('_evidence_directory',''))
    require(logs.is_dir() and (logs/'preflight.json').is_file(),'missing bound resident raw-log population')
    preflight=json.loads((logs/'preflight.json').read_text())
    require(preflight.get('build_sha256')==winner['build_sha256'] and preflight.get('capture_seal_sha256')==winner['capture_seal_sha256'],'winner/preflight identity mismatch')
    require(set(preflight.get('frames',{}))==set(winner['results']),'missing native boundary population')
    population=preflight.get('population',{})
    require(population.get('pairs')==5 and isinstance(population.get('warmup_rounds'),int) and 1<=population['warmup_rounds']<=100 and isinstance(population.get('samples_per_arm_per_pair'),int) and 1<=population['samples_per_arm_per_pair']<=1000 and isinstance(population.get('seed'),int) and 0<=population['seed']<2**32,'missing/invalid predeclared native population')
    from .receipts import validate_native_log
    import re
    parsed=[re.fullmatch(r'p(0|1023|1024|1040)/L(\d+)',name) for name in winner['results']]
    require(all(parsed),'invalid winner frame name')
    layers={int(m[2]) for m in parsed}
    require(all(0<=layer<48 for layer in layers) and set(winner['results'])=={f'p{position}/L{layer}' for position in (0,1023,1024,1040) for layer in layers},'winner omitted adjacent boundary calls')
    for name,result in winner['results'].items():
        raw=(logs/(name.replace('/','_')+'.log')).read_text()
        require(re.search(r'^IO_CHECK OK record='+record+r' ',raw,re.M) is not None,'wrong native record')
        require(validate_native_log(raw,record=record,expected_warmup=preflight.get('population',{}).get('warmup_rounds'),expected_samples=preflight.get('population',{}).get('samples_per_arm_per_pair'))['decision']==result,'winner threshold/raw timing mismatch')
    targets=[r for n,r in winner['results'].items() if n.startswith('p1040/')]
    require(bool(targets),'missing registered context1040 decision')
    for result in targets:
        gains=result.get('gains_ns',[])
        require(len(gains)==5 and all(isinstance(v,(float,int)) and v>0 for v in gains),'incomplete winner pairs')
        # Recheck the non-percentage gates from raw retained decision payload;
        # exact control pair timing remains in the build-bound native logs.
        import statistics
        median=statistics.median(gains);spread=max(gains)-min(gains)
        require(result.get('status')=='MICRO_WINNER' and median>=625 and median>2*spread,'winner threshold failed')
    return True


def emit_sources(record,original_kept,out,complete_flags,winner=None,micro_identity=None,local_validation=False,dsl_source=None):
    require(record in RECORDS,'unknown integration record')
    require(set(BASE_FLAGS.split()).issubset(set(complete_flags)),'incomplete frozen kept flags')
    require(all(isinstance(f,str) and f.startswith('-') for f in complete_flags),'malformed compiler flag set')
    if not local_validation:admit_winner(winner or {},record,micro_identity or {})
    original_kept=resolve_path(original_kept).resolve();out=Path(out).resolve();require(not out.exists(),'refuse existing integration output')
    attention=original_kept/'imp/attention.hip';dsl=original_kept/'dsl_tiles.hip'
    require(sha(attention)==ATTENTION_SHA,'unchanged original attention identity unavailable')
    original_dsl=dsl.read_text();require(original_dsl.count(prep_source())==1,'complete kept prep island identity')
    # The certification reference remains immutable, while G12 can privately
    # emit the current kept DSL required by its freshly generated model plan.
    # Only its exact prep island is transformed; all other current DSL entries
    # (including narrow-router wrappers) must survive byte for byte.
    selected_dsl=resolve_path(dsl_source).resolve() if dsl_source is not None else dsl
    current_dsl=selected_dsl.read_text()
    require(current_dsl.count(prep_source())==1,'current kept prep island identity')
    out.mkdir(parents=True)
    if record=='G06':
        (out/'dsl_tiles.hip').write_text(current_dsl.replace(prep_source(),transform(record,prep_source()),1))
    else:(out/'attention.hip').write_text(transform(record,attention.read_text()))
    selected=out/('dsl_tiles.hip' if record=='G06' else 'attention.hip')
    value={'schema':'etx.issue-order.model-overlay.v1','record':record,'timing_eligible':not local_validation,'scope':'ONLY_THIS_MECHANISM','complete_flags':list(complete_flags),'kept_env':KEPT_ENV,'certificate_reference':{'directory':str(original_kept),'attention_sha256':sha(attention),'dsl_tiles_sha256':sha(dsl),'unchanged_original':True},'candidate_source_sha256':sha(selected),'candidate_source':str(selected),'winner':winner if not local_validation else None,'model_correctness':'PENDING_CHECK_DSL_97_IMPORTS_AND_32_TOKENS','model_native_resources':'PENDING_FINAL_MODEL_BUILD','fusion':'PENDING_SAME_KERNEL_GRAPH_CERTIFICATE','required_compile_inputs':{'ETX_VL_DSL_TILES':str(selected if record=='G06' else dsl),'ETX_VL_IMPORTS':str(original_kept/'imp/imports.json'),'ETX_VL_ADAPTERS':str(original_kept/'imp_tiles.hip'),'candidate_bundle':str(original_kept/'imports.bc') if record=='G06' else str(out/'imports.bc')},'integration_source_sha256':sha(__file__)}
    value['dsl_source_reference']={'path':str(selected_dsl),'sha256':sha(selected_dsl),'fresh_current_source':dsl_source is not None}
    value['required_compile_inputs']['ETX_VL_DSL_TILES']=str(selected if record=='G06' else selected_dsl)
    (out/'overlay.json').write_text(json.dumps(value,indent=2)+'\n');return value


def compile_import_overlay(overlay,headers,vllm):
    """Compile the replacement under its ORIGINAL export/ABI, then real bundle.

    Other immutable imports are read in place; no original object/source is
    overwritten and no build tree is copied. G06 needs only its generated DSL.
    """
    from etx.importer.spec import SOURCES
    from etx.importer.vllm_kernels import _flags
    from etx.importer.ir_import import import_kernel
    from etx.importer.bundle import OCLC
    import os
    path=resolve_path(overlay);value=json.loads(path.read_text());out=path.parent;record=value['record']
    require(value['integration_source_sha256']==sha(__file__),'stale integration generator')
    require(sha(resolve_path(value['candidate_source']))==value['candidate_source_sha256'],'stale integrated candidate source')
    require(sha(resolve_path(value['dsl_source_reference']['path']))==value['dsl_source_reference']['sha256'],'stale current DSL reference')
    if record=='G06':return value
    original=resolve_path(value['certificate_reference']['directory']);manifest=json.loads((original/'imp/imports.json').read_text());export=EXPORTS[record];desc=manifest[export]
    headers=Path(headers);vllm=Path(vllm);llvm=headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin';libs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    os.environ.update(ETX_LLVM=str(llvm),ETX_ROCM_HEADERS=str(headers/'inc'),ETX_TORCH_HEADERS=str(headers/'pytorch'))
    spec=next(k for source in SOURCES for k in source.kernels if k.export==export);srcspec=next(source for source in SOURCES if spec in source.kernels)
    commands=[]
    def run(cmd,name):commands.append(list(map(str,cmd)));command(cmd,out,name)
    run([llvm/'clang++','-x','hip','--cuda-device-only','--offload-arch=gfx942','-nogpuinc','-nogpulib','-isystem',headers/'inc','-include','__clang_hip_runtime_wrapper.h','-O3','-gline-tables-only','-std=c++17','-S','-emit-llvm',*_flags(vllm,srcspec),out/'attention.hip','-o',out/'attention.ll'],'model_native_ir')
    text,info=import_kernel((out/'attention.ll').read_text(),spec.match,'etx_imp_'+export,tuple(desc['block']),1024,spec.lds_cap,spec.inline,desc['slots'])
    actual=info.to_json()
    for field in ('block','threads','slots','grid_offset','arg_bytes','lds_bytes_per_slot','lds_bytes','barriers','slot_threads','slot_barrier'):
        require(list(actual[field])==list(desc[field]) if field=='block' else actual[field]==desc[field],'model original ABI/slot protocol changed '+field)
    require([(p['ty'],p['offset'],p['size']) for p in actual['params']]==[(p['ty'],p['offset'],p['size']) for p in desc['params']],'model original argument ABI changed')
    (out/(export+'.ll')).write_text(text);run([llvm/'llvm-link',out/(export+'.ll'),'-o',out/(export+'.bc')],'model_assemble')
    exports=[name for name,entry in manifest.items() if not entry.get('dsl')]
    inputs=[out/(name+'.bc') if name==export else original/'imp'/(name+'.bc') for name in exports]
    require(all(p.is_file() for p in inputs),'original immutable import bitcode unavailable')
    run([llvm/'llvm-link',*inputs,'-o',out/'imports.all.bc'],'model_import_link')
    run([llvm/'llvm-link',out/'imports.all.bc','--only-needed',libs/'ocml.bc',libs/'ockl.bc',*[libs/f for f in OCLC],'-o',out/'imports.linked.bc'],'model_library_link')
    keep=','.join('etx_imp_'+name for name in exports)+',etx_lds_arena,etx_imp_slot_barrier'
    run([llvm/'opt','-passes=internalize,globaldce','-internalize-public-api-list='+keep,out/'imports.linked.bc','-o',out/'imports.bc'],'model_bundle')
    value['compiled_overlay']={'commands':commands,'export':export,'same_original_abi':True,'bundle_sha256':sha(out/'imports.bc'),'replacement_ir_sha256':sha(out/(export+'.ll')),'original_inputs':{str(p):sha(p) for p in inputs if p.parent!=out}}
    path.write_text(json.dumps(value,indent=2)+'\n');return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',choices=RECORDS,required=True);p.add_argument('--original-kept',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--flags',required=True);p.add_argument('--local-validation',action='store_true');p.add_argument('--winner',type=Path);p.add_argument('--micro-build',type=Path);p.add_argument('--compile-import',action='store_true');p.add_argument('--headers',type=Path,default=Path.home()/'code/rocm-headers');p.add_argument('--vllm',type=Path,default=Path.home()/'code/vllm');a=p.parse_args()
    for key in ('winner','micro_build','original_kept','headers','vllm'):
        if getattr(a,key,None) is not None:setattr(a,key,resolve_path(getattr(a,key)))
    winner=json.loads(a.winner.read_text()) if a.winner else None;micro=None
    if winner is not None:winner['_evidence_directory']=str(a.winner.parent.resolve())
    if a.micro_build:
        from .__main__ import verify_build
        micro=verify_build(a.micro_build);micro['_identity_sha256']=sha(a.micro_build/'identity.json')
    result=emit_sources(a.record,a.original_kept,a.out,a.flags.split(),winner,micro,a.local_validation)
    if a.compile_import:result=compile_import_overlay(a.out/'overlay.json',a.headers,a.vllm)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
