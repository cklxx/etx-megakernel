"""Write a concrete thirteen-record local session from staged-path inventory.

This copies no model data and executes no build/device/remote command. Missing
native captures/configuration remain explicit sealed missing prerequisites.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .protocol import require,sha,relocation_seal
from .session import MODULES,ROOT,SPECS,configuration

BASE='b573664faa4e7988dff353bc03aca4ce8561a729'


def prepare(inventory,destination,output):
    destination=Path(destination).resolve();output=Path(output).resolve()
    require(not destination.exists(),'preserve occupied session configuration')
    require(set(inventory.get('records',{}))<=set(SPECS),'unknown inventory record')
    require(Path(inventory.get('python',sys.executable)).resolve()==Path(sys.executable).resolve(),'generate using the actual staged project Python')
    staged=Path(inventory['staging_root']).resolve();records={}
    for record,module in MODULES.items():
        spec=inventory.get('records',{}).get(record,{})
        missing=staged/'prerequisites'/record
        build=Path(spec.get('build',staged/'build'/record)).resolve()
        base=[sys.executable,'-m','examples.experiment_program.'+module]
        required=[ROOT/'examples/experiment_program/specs.json']
        result=output/record
        admission={'record':record,'build':str(build)}
        if module=='router':
            captures=[Path(p).resolve() for p in spec.get('captures',[missing/'pos0',missing/'pos1040'])]
            args=['--build',str(build),'--out',str(result)]
            for capture in captures:args+=['--capture',str(capture)];required.append(capture/'capture.json')
            preflight=base+['preflight']+args;command=base+['program']+args+['--execute'];decision=result/'decision.json'
        elif module=='columns':
            first='pos1024.json' if record in ('G02','G05') else 'pos0.json'
            captures=[Path(p).resolve() for p in spec.get('captures',[missing/first,missing/'pos1040.json'])]
            args=['--record',record,'--build',str(build),'--out',str(result),'--regime','both']
            for capture in captures:args+=['--capture',str(capture)];required.append(capture)
            for dependency in spec.get('dependencies',[]):args+=['--dependency',str(Path(dependency).resolve())]
            preflight=base+args+['--dry-run'];command=base+args+['--execute'];decision=result/'report.json'
            if 'program' in spec:
                require('captures' not in spec,'choose registered program or individual capture selection')
                program=Path(spec['program']).resolve();registered=json.loads(program.read_text())
                require(registered.get('schema')=='etx.columns.program.v1' and registered.get('record')==record,'registered program record/schema required')
                program_build=Path(registered['build']).resolve()
                require('build' not in spec or build==program_build,'program build differs from inventory build')
                build=program_build;admission['build']=str(build)
                required=[ROOT/'examples/experiment_program/specs.json',program]
                args=['--record',record,'--program',str(program),'--out',str(result)]
                preflight=base+args+['--dry-run'];command=base+args+['--execute'];decision=result/'program-report.json'
        elif module=='issue_order':
            capture=Path(spec.get('capture',missing/'capture')).resolve()
            args=['--build',str(build),'--capture',str(capture),'--layers',spec.get('layers','0,24,47')]
            preflight=base+['--record',record,'preflight']+args
            command=base+['--record',record,'execute']+args+['--out',str(result),'--device-execution-authorized']
            decision=result/'result.json';required.append(capture/'capture.json')
        elif module=='row_ops':
            program=Path(spec.get('program',missing/'row-program.json')).resolve()
            args=['program','--record',record,'--config',str(program),'--output',str(result)]
            if record=='G11':args+=['--trace-session',str(output/'G12')]
            preflight=base+args+['--dry-run'];command=base+args+['--execute'];decision=result/'program-report.json'
            if record=='G11' and program.is_file() and json.loads(program.read_text()).get('schema')=='etx.g11.program.v1':
                base=[sys.executable,'-m','examples.experiment_program.publication_run']
                args=['program','--config',str(program),'--output',str(result),'--trace-session',str(output/'G12')]
                preflight=base+args+['--dry-run'];command=base+args+['--execute']
            admission.update(variant=spec.get('variant','g09' if record=='G09' else 'g10_up'))
            required.append(program)
            if program.is_file():
                recipe=json.loads(program.read_text())
                required.extend(Path(p) for p in recipe.get('inputs',{}))
        else:
            cfg=Path(spec.get('config',missing/'config.json')).resolve()
            args=['--config',str(cfg),'--out',str(result)]
            preflight=base+['preflight']+args;command=base+['capture' if module=='routing' else 'run']+args+['--execute']
            decision=result/('capture.json' if module=='routing' else 'decision.json');required.append(cfg)
            if module=='validation':command+=['--micro-session-directory',str(output)]
        if module not in ('routing','validation'):required.append(build/'identity.json')
        required.extend(Path(p).resolve() for p in spec.get('inputs',[]))
        seals={**{str(p):sha(p) if p.is_file() else '0'*64 for p in required},**relocation_seal()}
        records[record]=dict(preflight=preflight,command=command,input_files=seals,
                             timeout_seconds=spec.get('timeout_seconds',12000 if record=='G12' else 900),decision_file=str(decision.relative_to(output)),admission=admission)
        if record=='G11' and spec.get('publication_preparation'):
            proof=Path(spec['publication_preparation']).resolve()
            records[record]['local_terminal']=dict(kind='G11_PUBLICATION',identity=str(proof),sha256=sha(proof))
            records[record]['input_files'][str(proof)]=sha(proof)
    from .source_seal import snapshot
    import subprocess
    revision=inventory.get('source_revision') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    value=dict(base=BASE,source_snapshot=snapshot(),source_revision=revision,budget_seconds=inventory.get('budget_seconds',14400),records=records,
               options=inventory.get('options',{}),gpu_authorization='SEPARATE_FUTURE_OPERATOR_ACTION',
               staging_root=str(staged),output_directory=str(output))
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(value,indent=2)+'\n')
    configuration(destination)
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--session-output',type=Path,required=True);a=p.parse_args(argv)
    r=prepare(json.loads(a.inventory.read_text()),a.out,a.session_output)
    print(json.dumps(dict(configuration=str(a.out.resolve()),records=list(r['records']),device_execution=False,missing_paths=[path for s in r['records'].values() for path,h in s['input_files'].items() if h=='0'*64]),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
