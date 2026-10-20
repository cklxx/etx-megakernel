"""Generate the actual GLM PP2 ETX plan and its exact weight-load contract."""
import argparse
import json
from pathlib import Path
from etx.codegen import emit_kernel,emit_lowering_header,emit_plan_json
from etx.codegen.plan_header import emit_plan_header
from etx.passes import compile_graph,verify_plan
from etx.passes.plan import PassOptions as CompileOptions
from examples.glm53 import model,stage
from examples.glm53 import options as build_options


def generate(spec,out,batch=1,options=None):
    options=build_options.Options() if options is None else options
    out.mkdir(parents=True,exist_ok=False)
    (out/'build-options.json').write_text(json.dumps(options.to_dict(),indent=2)+'\n')
    (out/'glm53_options.h').write_text(options.header())
    plan=compile_graph(model.build_from(spec,batch,options),'gfx942',{'B':batch},{},CompileOptions(inline_tiles=True))
    errors=verify_plan(plan)
    if errors:raise ValueError(errors)
    (out/'glm53_spec.h').write_text(stage.header(spec['dimensions']))
    (out/'plan.json').write_text(emit_plan_json(plan))
    (out/'build-resources.json').write_text(json.dumps(dict(batch=batch,required_workgroups_per_cu=plan.wg_per_cu,
        workers_per_domain=plan.workers_per_domain,planner_vgpr_budget_estimate=options.vgpr_budget,
        native_staging_lds_bytes=options.lds_bytes),indent=2)+'\n')
    (out/'plan_data.h').write_text(emit_plan_header(plan))
    (out/'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
    for d in (0,1):(out/f'megakernel_d{d}.hip').write_text(build_options.inject(emit_kernel(plan,d),options))
    ownership={f'l{i}':r['rank'] for r in spec['ranks'] for i in r['layers']}
    ownership.update(embedding=0,norm=1,head=1)
    shared=[int(name=='step' or plan.tensor_placement.get(name)=='fine_grained') for name in plan.graph.tensors]
    (out/'glm53_contract.h').write_text('#pragma once\nstatic const int glm_arg_shared[] = {'+','.join(map(str,shared))+'};\n')
    loads=[]
    for name,w in spec['weights'].items():
        for kind,suffix in [('blob','weights'),('meta','meta')]:loads.append(f'{name}_{suffix} {w[kind]} {ownership[name]} {w[kind+"_bytes"]}')
    (out/'load.tsv').write_text('\n'.join(loads)+'\n')
    (out/'build-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'source-identity.json').write_text(json.dumps(source_identity(out),indent=2)+'\n')
    return plan


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch',type=int,default=1);ap.add_argument('--allow-fixture',action='store_true')
    ap.add_argument('--gate',choices=['production','module','slice'],default='production')
    build_options.add_arguments(ap)
    a=ap.parse_args();spec=stage.validate(a.stage,allow_fixture=a.allow_fixture,gate=a.gate);p=generate(spec,a.out,a.batch,build_options.from_arguments(a))
    print(f'GLM-MODEL-PLAN layers={spec["config"]["num_hidden_layers"]} devices={p.n_devices} tasks={len(p.tasks)} fixture={spec["fixture"]}',flush=True)




def production_compile_spec(maxlen=1072):
    """Full pinned shape, without weight files. Code generation only, never runnable."""
    from examples.glm53.prepare import metadata,partition,MODEL,EVIDENCE,sha
    cfg,api=metadata();t=cfg['text_config'];weights={}
    for i in range(t['num_hidden_layers']):
        weights[f'l{i}']=dict(blob_bytes=1,meta_bytes=32*len(stage.slot_names(t)),blob=f'l{i}.bin',meta=f'l{i}.meta')
    for name in ('embedding','norm','head'):weights[name]=dict(blob_bytes=1,meta_bytes=32,blob=name+'.bin',meta=name+'.meta')
    return dict(schema=1,model=MODEL,revision=api['sha'],config_sha256=sha(EVIDENCE/'glm-config.json'),
                fixture=True,compile_only=True,config=t,dimensions=stage.dimensions(t,maxlen),ranks=partition(t),weights=weights)


def source_identity(out):
    root=Path(__file__).resolve().parents[2]
    sources=[root/'examples/glm53'/name for name in ('native.h','native_options.h','options.py','tiles.hip','model_host.hip','model.py','model_build.py','build_model.sh','stage.py')]
    sources+=list((root/'etx/runtime/include').rglob('*.h'))
    generated=['plan.json','plan_data.h','etx_lowering.h','glm53_spec.h','glm53_contract.h','load.tsv','build-spec.json','build-options.json','build-resources.json','glm53_options.h','megakernel_d0.hip','megakernel_d1.hip']
    options=build_options.load(out/'build-options.json')
    if (out/'glm53_options.h').read_text()!=options.header():raise ValueError('inconsistent generated options header')
    return dict(schema=2,arch='gfx942',flags=options.compiler_flags(),options=options.to_dict(),options_sha256=options.digest(),
                sources={str(p.relative_to(root)):stage.file_sha(p) for p in sources},
                generated={n:stage.file_sha(out/n) for n in generated})


def finalize(out,local=False):
    expected=json.loads((out/'source-identity.json').read_text())
    if expected!=source_identity(out):raise ValueError('sources changed during model build')
    result=dict(expected,status='LOCAL-COMPILED' if local else 'GPU-LINKED')
    if not local:result['binary_sha256']=stage.file_sha(out/'run')
    (out/'build-identity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verified_build(out):
    r=json.loads((out/'build-identity.json').read_text())
    if r.get('status')!='GPU-LINKED' or r.get('binary_sha256')!=stage.file_sha(out/'run'):raise ValueError('missing/changed linked model binary')
    if {k:v for k,v in r.items() if k not in ('status','binary_sha256')}!=source_identity(out):raise ValueError('changed model source/generated identity')
    return r

if __name__=='__main__':main()
