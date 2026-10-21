"""Exact complete raw up population, required before any optional down command."""
import json
from examples.experiment_program.protocol import require,resolve_path,sha

def diagnostic_pair(build,diagnostic_build):
    """The down probe is a declared diagnostic of the admitted normal variants."""
    from .build import FLAGS
    from .replay import validate_build
    from .source import translation_unit
    from .diagnostics import unit
    from examples.vllm_llm.bf16_300.build import preamble
    from etx.ops import gemv
    from examples.experiment_program.protocol import selected_artifacts
    build=resolve_path(build);diagnostic_build=resolve_path(diagnostic_build)
    normal=validate_build(build,'g10_up');validate_build(build,'g10_down')
    diagnostic=json.loads((diagnostic_build/'identity.json').read_text())
    require(normal.get('math_flags')==diagnostic.get('flags')==FLAGS,'normal/diagnostic math flags differ')
    selected_artifacts(build,normal.get('files',normal['hashes']),['g10_up.hip','g10_down.hip'])
    names=['g10_down_'+m for m in ('control','probe','empty')]
    selected_artifacts(diagnostic_build,diagnostic['files'],['replay']+[n+s for n in names for s in ('.hip','.hsaco')])
    header=preamble(gemv.LDS_BYTES)
    for variant in ('g10_up','g10_down'):
        require((build/(variant+'.hip')).read_text()==translation_unit(variant,header),'normal up/down source variant differs')
    for mode,name in zip(('control','probe','empty'),names):
        require((diagnostic_build/(name+'.hip')).read_text()==unit('g10_down',mode,header),'diagnostic down source variant differs')


def up_evidence(path,build,layer,*,diagnostic_build=None):
    from .program import readmit
    if diagnostic_build is not None:diagnostic_pair(build,diagnostic_build)
    path=resolve_path(path);e=json.loads(path.read_text());require(e.get('schema')=='etx.row.up-evidence.v1','raw up evidence schema')
    rows=e.get('positions',[]);require(len(rows)==2,'both-position raw up population required');positions=[]
    for d in rows:
        require(d.get('record')=='G10' and d.get('variant')=='g10_up' and resolve_path(d['build'])==resolve_path(build),'raw up variant/build differs')
        readmit(d,layer=layer);positions.append(json.loads(resolve_path(d['pack']).read_text())['position'])
    require(set(positions)=={0,1040},'raw up both-position coverage differs')
    return dict(path=str(path),sha256=sha(path))
