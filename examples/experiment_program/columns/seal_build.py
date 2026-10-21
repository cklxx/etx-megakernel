"""Seal only actual completed native compiler outputs, with no device claim."""
import argparse,json
from pathlib import Path
from etx.machine import load_machine
from examples.vllm_llm.bf16_300.build import resources
from .protocol import require,sha256

def seal(build):
    out=Path(build).resolve();root=Path(__file__).resolve().parents[3]
    paths=set();native={};caps=load_machine('gfx942').resources
    paths.add(out/'native-ready-final/native-plans.json')
    for variant in json.loads((out/'native-ready-final/native-plans.json').read_text()):
        loc=Path(variant['directory']);require((loc/'micro.hsaco').is_file() and (loc/'columns-run').is_file(),'native ETX compile/link incomplete: '+str(loc))
        from examples.experiment_program.native_resources import persistent_capacity
        capacity=persistent_capacity(loc);r=capacity['raw_metadata'];resident=capacity['resident_capacity'];required=capacity['required_persistent_blocks']
        entry=dict(capacity=capacity,resources=r,resident_capacity=resident,required_native_blocks_per_cu=required,admitted=resident>=required,
                  arithmetic='exact source-bound native helpers; same executable/resource allocation in both arms',gpu_residency='UNMEASURED',
                  status='STATIC_RESOURCE_ADMITTED' if resident>=required else 'CLOSED_RESOURCE',
                  plan_sha256=sha256(loc/'plan.json'),assembly_sha256=sha256(loc/'micro.s'))
        native[loc.name]=entry
        paths.update(p for p in loc.iterdir() if p.is_file())
    for name in ('columns','b5eaa67','ff9ef92'):
        require((out/(name+'.hsaco')).is_file(),'base device module missing')
    resource={name:resources((out/(name+'.s')).read_text()) for name in ('columns','b5eaa67','ff9ef92')}
    require((out/'g13-ready-final/g13.hsaco').is_file() and (out/'g13-ready-run').is_file(),'G13 ETX device/host closure incomplete')
    resource['g13']=resources((out/'g13-ready-final/g13.s').read_text())
    require((out/'g03-ready-final/g03.hsaco').is_file() and (out/'g03-ready-run').is_file(),'G03 native ETX device/host closure incomplete')
    resource['b5eaa67']=resources((out/'g03-ready-final/g03.s').read_text())
    historical_capacity={}
    from examples.experiment_program.native_resources import persistent_capacity
    for record,name in (('G03','g03'),('G13','g13')):
        directory=out/(name+'-ready-final')
        historical_capacity[record]=persistent_capacity(directory)
        require(historical_capacity[record]['resident_capacity']>=historical_capacity[record]['required_persistent_blocks'],record+' native persistent residency failed')
    resource={name:[{k:v for k,v in row.items() if k not in ('numeric_gate','static_blocks_per_cu','actual_co_residency')} for row in rows] for name,rows in resource.items()}
    for rev in ('b5eaa67','ff9ef92'):
        loc=out/f'historical-capture-ready-{rev}'
        require((loc/'capture.hsaco').is_file() and (loc/'host.o').is_file(),'historical capture target/host-object closure incomplete')
        paths.update(p for p in loc.iterdir() if p.is_file())
    for folder in (root/'examples/experiment_program/columns',root/'etx/runtime/include',out/'g13-ready-final',out/'g03-ready-final',out/'cooked-fleet',out/'native-capture-04'):
        paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'compiler-tmp' not in str(p))
    paths.update(p for p in out.iterdir() if p.is_file() and p.name!='identity.json')
    paths.add(root/'examples/experiment_program/hip_dynamic.h');paths.add(root/'examples/experiment_program/native_resources.py');paths.add(root/'examples/experiment_program/protocol.py');paths.add(root/'examples/experiment_program/row_ops/elf.py')
    paths.add(Path('/Users/ckl/.hermes/cache/scratch/etx-experiment-preparation/toolchain.json'))
    sources=json.loads((out/'sources.json').read_text())
    for name in ('b5eaa67','ff9ef92'):paths.add(Path(sources[name]['path']))
    identity=dict(schema=1,local_device_compile_complete=True,all_native_micro_variants=len(native),native_resources=native,historical_capacity=historical_capacity,resources=resource,
                  selected_replay_hosts=[str(out/'g03-ready-run'),str(out/'g13-ready-run')]+[str(Path(v['directory'])/'columns-run') for v in json.loads((out/'native-ready-final/native-plans.json').read_text())],
                  retained_base_host='columns-run is historical compiler evidence only; actual sessions select the ready-final hosts listed above',
                  local_gpu_executed=False,correctness='UNMEASURED',performance='UNMEASURED',
                  files={str(p.resolve()):sha256(p) for p in sorted(paths)},
                  source_families=sources,original_model_host_link='UNAVAILABLE_EXISTING_LOCAL_HIP_RUNTIME; actual host objects only')
    with (out/'identity.json').open('x') as f:f.write(json.dumps(identity,indent=2)+'\n')
    return identity

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);a=p.parse_args();r=seal(a.build)
    print(json.dumps(dict(status='SEALED',native_resources=r['native_resources'],files=len(r['files'])),indent=2))
if __name__=='__main__':main()
