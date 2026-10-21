"""Bounded untimed clocks/power provenance around a resident trial group."""
import contextlib,json,subprocess
from pathlib import Path
from .protocol import sha,require,verify_files

@contextlib.contextmanager
def capture(directory,identity,*,collector=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    argv=['rocm-smi','--showclocks','--showpower']; rows=[]
    def collect(phase):
        log=directory/(phase+'.log')
        try:
            with log.open('x') as f:
                rc=(collector or subprocess.run)(argv,stdout=f,stderr=subprocess.STDOUT,timeout=10,check=False).returncode
            row=dict(phase=phase,argv=argv,exit=rc,log=str(log.resolve()),sha256=sha(log))
        except (OSError,subprocess.SubprocessError) as e:
            row=dict(phase=phase,argv=argv,exit=None,error=str(e))
        rows.append(row)
        (directory/'receipt.json').write_text(json.dumps(dict(schema='etx.telemetry.v1',run_identity=identity,brackets='outside native decisive timing',observations=rows,traffic_counters='NOT_MEASURED'),indent=2)+'\n')
        return row.get('exit')==0
    before=collect('before')
    try:
        require(before,'clocks/power collection failed; retained untimed receipt')
        yield
    finally:
        after=collect('after')
        require(after,'clocks/power collection failed; retained untimed receipt')

def verify(directory,identity):
    p=Path(directory)/'receipt.json';r=json.loads(p.read_text())
    require(r.get('schema')=='etx.telemetry.v1' and r.get('run_identity')==identity,'telemetry run identity differs')
    require([x['phase'] for x in r['observations']]==['before','after'] and all(x['exit']==0 for x in r['observations']),'incomplete/failed clocks/power provenance')
    verify_files({x['log']:x['sha256'] for x in r['observations']})
    return {str(p):sha(p)}
