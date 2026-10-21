from examples.experiment_program.protocol import resolve_path
import argparse,json,subprocess
from pathlib import Path
from .build import prepare,compile
from .replay import config,validate_build
from examples.experiment_program.protocol import sha
from .protocol import admit_down
from .results import summarize
from .capture import checking_host,finalize,adapt_moe


def main(argv=None):
    p=argparse.ArgumentParser(description='G09/G10/G11 local preparation and future authorized native replay')
    sp=p.add_subparsers(dest='action',required=True)
    sp.add_parser('records')
    q=sp.add_parser('seal-producer');q.add_argument('--recipe',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sp.add_parser('program');q.add_argument('--record',choices=('G09','G10','G11'),required=True);q.add_argument('--config',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--trace-session',type=Path);q.add_argument('--execute',action='store_true');q.add_argument('--dry-run',action='store_true')
    q=sp.add_parser('capture-host');q.add_argument('--original',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sp.add_parser('seal-capture');q.add_argument('--directory',type=Path,required=True);q.add_argument('--producer',type=Path,required=True);q.add_argument('--plan',type=Path,required=True);q.add_argument('--trace-proof',type=Path);q.add_argument('--capture-build',type=Path)
    q=sp.add_parser('adapt-moe');q.add_argument('--producer',type=Path,required=True);q.add_argument('--pack',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    for a in ('dry-run','build'):
        q=sp.add_parser(a);q.add_argument('--out',type=Path,required=True);q.add_argument('--diagnostics',action='store_true')
    q=sp.add_parser('execute');q.add_argument('--pack',type=Path,required=True);q.add_argument('--build',type=Path,required=True);q.add_argument('--variant',choices=('g09','g10_up','g10_down','g11_probe'),required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--deadline',type=float,default=120);q.add_argument('--dry-run',action='store_true');q.add_argument('--diagnostic',action='store_true');q.add_argument('--up-admission',type=Path);q.add_argument('--up-evidence',type=Path);q.add_argument('--up-build',type=Path,help='admitted normal build, separate from the selected down diagnostic build')
    a=p.parse_args(argv)
    if a.action=='records': result=dict(records=['G09','G10','G11'],gpu_run=False,down='conditional on exact admitted up',g11='exposure only, opportunistic under G12')
    elif a.action=='seal-producer':
        from .producer import seal
        result=seal(a.recipe,a.out)
    elif a.action=='program':
        from .program import run
        result=run(a.config,a.record,a.output,execute=a.execute,case_main=main,trace_session=a.trace_session)
    elif a.action=='capture-host':
        checking_host(a.original,a.out);result=dict(generated=str(a.out),gpu_run=False)
    elif a.action=='seal-capture':result=finalize(a.directory,a.plan,producer=a.producer,trace_proof=a.trace_proof,capture_build=a.capture_build)
    elif a.action=='adapt-moe':result=adapt_moe(a.pack,a.out,producer=a.producer)
    elif a.action=='build':
        if a.diagnostics:
            from .diagnostics import unit
            from examples.vllm_llm.bf16_300.build import preamble
            from etx.ops import gemv
            from .build import FLAGS
            from examples.experiment_program.build_support import build
            result=build(a.out,{variant+'_'+mode:unit(variant,mode,preamble(gemv.LDS_BYTES)) for variant in ('g09','g10_up','g10_down') for mode in ('control','probe','empty')},Path(__file__).with_name('diagnostic_replay.cpp'),flags=FLAGS)
        else:result=compile(a.out)
    elif a.action=='dry-run':result=dict(generated=str(prepare(a.out)),gpu_run=False)
    else:
        a.pack=resolve_path(a.pack);a.build=resolve_path(a.build);a.output=resolve_path(a.output)
        if a.up_admission:a.up_admission=resolve_path(a.up_admission)
        pack=json.loads(a.pack.read_text())
        if a.diagnostic:
            from examples.experiment_program.protocol import selected_artifacts
            identity=json.loads((a.build/'identity.json').read_text());selected_artifacts(a.build,identity['files'],['replay']+[a.variant+'_'+m+'.hsaco' for m in ('control','probe','empty')])
        else:validate_build(a.build,a.variant)
        if a.variant=='g10_down':
            if not a.up_evidence:p.error('G10 down requires complete raw both-position up evidence, not a summary')
            from .dependencies import up_evidence
            if a.diagnostic and not a.up_build:p.error('down diagnostic requires the admitted normal --up-build')
            up_build=resolve_path(a.up_build) if a.up_build else a.build
            if not a.diagnostic and up_build!=a.build:p.error('normal down requires its own admitted up build')
            dependency=up_evidence(a.up_evidence,up_build,pack['layer'],diagnostic_build=a.build if a.diagnostic else None)
        text,result=config(pack,a.build,a.variant,a.output,a.deadline,diagnostic=a.diagnostic)
        if a.variant=='g10_down':
            result['up_evidence']=dependency
            result['up_build']=str(up_build);result['up_build_identity_sha256']=sha(up_build/'identity.json')
        result['build_identity_sha256']=sha(a.build/'identity.json')
        result['input_pack_sha256']=sha(a.pack)
        result['command']=[str(a.build/'replay'),str(a.output)+'.config']
        if not a.dry_run:
            if any(a.output.parent.glob(a.output.name+'.*')):p.error('preserve existing evidence: output prefix already occupied')
            a.output.parent.mkdir(parents=True,exist_ok=True);resolve_path(str(a.output)+'.config').write_text(text)
            from examples.experiment_program.telemetry import capture
            from contextlib import nullcontext
            scope=capture(str(a.output)+'.telemetry',dict(command=result['command'],pack_sha256=sha(a.pack),build_sha256=sha(a.build/'identity.json'))) if a.variant!='g11_probe' else nullcontext()
            try:
                with scope:
                    with resolve_path(str(a.output)+'.log').open('w') as f:
                        run=subprocess.run(result['command'],stdout=f,stderr=subprocess.STDOUT,timeout=a.deadline)
                        resolve_path(str(a.output)+'.exit').write_text(str(run.returncode)+'\n')
                        if run.returncode:raise ValueError('native row replay failed; complete raw attempt retained')
            except subprocess.TimeoutExpired:
                resolve_path(str(a.output)+'.exit').write_text('TIMEOUT\n');raise
            result['gpu_executed']=True
            if a.diagnostic:
                from .diagnostics import summarize as diagnostic_summary
                di=json.loads((a.build/'identity.json').read_text());di['_directory']=str(a.build)
                result['verdict']=diagnostic_summary(a.output,pack,di,a.variant)
            else:
                di=json.loads((a.build/'identity.json').read_text());di['_directory']=str(a.build)
                result['verdict']=summarize(a.output,pack,di,a.variant)
            resolve_path(str(a.output)+'.result.json').write_text(json.dumps(result,indent=2)+'\n')
        else:result['config']=text
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
