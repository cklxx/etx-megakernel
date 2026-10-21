"""Registered decisive run, sparse matched qualification and quiescent rollover."""
import contextlib,io,json
from pathlib import Path
from ..protocol import require,sha

def run(args,main):
    root=args.out;require(root is not None,'retained program output required');root.mkdir(parents=True,exist_ok=False)
    base=['replay','--build',str(args.build),'--warmup',str(args.warmup),'--population',str(args.population),'--seed',str(args.seed),'--deadline',str(args.deadline),'--execute','--stress']
    for capture in args.capture:base+=['--capture',str(capture)]
    for name,extra in [('control',[]),('probe',['--diagnostic']),('empty',['--diagnostic','--diagnostic-mode','empty'])]:
        with contextlib.redirect_stdout(io.StringIO()):code=main(base+['--out',str(root/name)]+extra)
        require(code in (None,0),'router program child failed')
    from .diagnostics import qualify
    q=qualify(args.build,root/'control',root/'probe',root/'empty')
    require(isinstance(q,dict) and type(q.get('qualified')) is bool,'router diagnostic qualification missing/malformed')
    result=json.loads((root/'control/decision.json').read_text())
    if not q['qualified']:result.update(admitted=False,status='CLOSED_DIAGNOSTIC_QUALIFICATION')
    result.update(schema='etx.router.program.v1',diagnostic=q,raw_control=str((root/'control/decision.json').resolve()),raw_control_sha256=sha(root/'control/decision.json'),stress='QUIESCENT_REALLOCATION_AT_UINT32_BOUNDARY')
    (root/'decision.json').write_text(json.dumps(result,indent=2)+'\n');return result
