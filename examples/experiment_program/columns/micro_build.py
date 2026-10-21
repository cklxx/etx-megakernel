"""Prepare/build fixed native Fleet ETX micrographs offline; never run a device."""
from __future__ import annotations
import argparse,json,os,subprocess,shutil
from pathlib import Path
from .protocol import require,sha256
from .micro_graph import verify_queues
ROOT=Path(__file__).resolve().parents[3]
VARIANTS=[(b,k,x) for b in (2,4) for k in ('qkv','group','shared') for x in ((6,7) if k=='shared' else (6,))]

def prepare(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);result=[]
    old=dict(os.environ)
    try:
        for b,kind,xcd in VARIANTS:
            loc=out/(f'native-B{b}-{kind}'+(f'-x{xcd}' if kind=='shared' else ''))
            require(not loc.exists(),'refusing occupied native micro plan destination');loc.mkdir()
            env=dict(os.environ,ETX_COLUMN_BATCH=str(b),ETX_COLUMN_KIND=kind,ETX_COLUMN_NATIVE_XCD=str(xcd))
            from .micro_graph import build
            from etx.machine import load_machine
            from etx.passes import compile_graph,PassOptions,p5_queues,p6_memory,p7_prefetch
            from etx.codegen import emit_kernel,emit_lowering_header,emit_plan_json
            from etx.codegen.plan_header import emit_plan_header
            os.environ.update(ETX_COLUMN_BATCH=str(b),ETX_COLUMN_KIND=kind,ETX_COLUMN_NATIVE_XCD=str(xcd))
            plan=compile_graph(build(),load_machine('gfx942'),{}, {},PassOptions(inline_tiles=True,relay='off',prefetch=False))
            from .micro_graph import freeze_native_budget
            estimated=freeze_native_budget(plan)
            (loc/'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
            from .dispatch_lds import alias_dispatch
            original=emit_kernel(plan,0);(loc/'dispatcher-unaliased.hip').write_text(original)
            source,alias=alias_dispatch(original,b);(loc/'megakernel_d0.hip').write_text(source)
            (loc/'lds-lifetime-proof.json').write_text(json.dumps(alias,indent=2)+'\n')
            (loc/'plan_data.h').write_text(emit_plan_header(plan,0))
            (loc/'plan.json').write_text(emit_plan_json(plan))
            (loc/'plan.log').write_text(plan.summary()+'\n'+'\n'.join(plan.log)+'\n')
            (loc/'native-budget.json').write_text(json.dumps(dict(native_workers_per_domain=75,required_blocks_per_cu=2,declared_capacity=estimated,compiler_only_until_resource_admitted=True),indent=2)+'\n')
            os.environ.update(ETX_COLUMN_BATCH=str(b),ETX_COLUMN_KIND=kind,ETX_COLUMN_NATIVE_XCD=str(xcd))
            proof=verify_queues(json.loads((loc/'plan.json').read_text()));(loc/'queue-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
            result.append(dict(batch=b,kind=kind,native_xcd=xcd,directory=str(loc),queue_proof_sha256=sha256(loc/'queue-proof.json')))
    finally:
        os.environ.clear();os.environ.update(old)
    (out/'native-plans.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--plans-exist',action='store_true')
    a=p.parse_args();require(shutil.disk_usage(a.out.parent).free>=2*(1<<30),'storage free reserve2GiB')
    result=json.loads((a.out/'native-plans.json').read_text()) if a.plans_exist else prepare(a.out)
    if not a.prepare_only:subprocess.run(['bash',str(Path(__file__).with_name('build_micro.sh')),str(ROOT),str(a.out)],check=True)
    print(json.dumps(dict(status='PREPARED' if a.prepare_only else 'COMPILED',variants=result),indent=2))
if __name__=='__main__':main()
