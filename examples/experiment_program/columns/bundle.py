"""Create a G05 up+down admission bundle only from both raw qualified reports."""
import argparse,json
from pathlib import Path
from .runner import admit_model_winner
from .protocol import require,sha256

def bundle(up,down,build,out):
    reports={r:Path(p).resolve() for r,p in [('gate_up',up),('down',down)]}
    values={r:admit_model_winner(p,build) for r,p in reports.items()}
    require(all(v['record']=='G05' and v['role']==r for r,v in values.items()),'real G05 role reports required')
    require(values['gate_up']['batch']==values['down']['batch'],'role batch differs')
    from examples.experiment_program.protocol import resolve_path
    for position in (1024,1040):
        upc=json.loads((reports['gate_up'].parent/f'p{position}-simultaneous/capture.json').read_text())
        downc=json.loads((reports['down'].parent/f'p{position}-simultaneous/capture.json').read_text())
        require(upc['picks']==downc['picks'] and (upc['N'],upc['K'],downc['N'],downc['K'])==(1408,2048,2048,1408),'G05 up/down native identities or dimensions differ')
        def segment(c,name):
            s=next(s for s in c['segments'] if s['name']==name)
            with resolve_path(c['payload']['path']).open('rb') as f:f.seek(s['offset']);return f.read(s['bytes'])
        require(segment(upc,'golden')==segment(downc,'inputs') and segment(upc,'scales')==segment(downc,'scales'),'G05 actual down operands/mix differ from exact matched up outputs')
        us=json.loads(resolve_path(upc['native_producer_seal']).read_text());ds=json.loads(resolve_path(downc['native_producer_seal']).read_text())
        require(us['g04']['sha256']==ds['g04']['sha256'],'G05 role captures use different G04 original populations')
    value=dict(record='G05' ,role='gate_up_down',batch=values['down']['batch'],status='MICRO_PASS_BOTH_REGIMES',
               admitted=True,matched_up_down=True,build_identity_sha256=sha256(Path(build)/'identity.json'),
               role_reports={r:dict(path=str(p),sha256=sha256(p)) for r,p in reports.items()})
    with Path(out).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return admit_model_winner(out,build)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('up','down','build','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();print(json.dumps(bundle(a.up,a.down,a.build,a.out),indent=2))
if __name__=='__main__':main()
