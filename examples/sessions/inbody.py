"""A single 900-second gpu_jobs measurement job; no provisioning or downloads.

python -m etx.tools.gpu_jobs examples/sessions/inbody.py --dry-run
ETX_INBODY_CONFIG=prepared.json tools/session.sh start IP examples/sessions/inbody.py 0 A

Builds, full weight pack, HF golden and portable snapshots MUST already exist.
The cap includes all loading/checking, not only kernel time. Slow loads can make
the attempt NOT-COMPLETE; do not shorten checks/repetitions to manufacture a pass.
Fetch/delete is the operator/session.sh path, never a new provisioning system.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from etx.tools import inbody as reader
from examples.vllm_llm.bf16_300.pack import sha

MAX_SESSION_SECONDS = 900
JOBS = [dict(name='inbody_attribution',gpu=0,machine='A',timeout=900,quiet_s=180,
             cmd='stdbuf -oL -eL python3 -u -m examples.sessions.inbody --run')]


def samples(text, repetitions=5):
    rows=[json.loads(l.split(' ',1)[1]) for l in text.splitlines() if l.startswith('INBODY-RUN ')]
    if len(rows)!=repetitions or any(r['run']!=i or r['matched']!=32 or r['gen']!=32 or r['wall_ms']<=0 for i,r in enumerate(rows)):
        raise ValueError('NOT-COMPLETE: missing full 32/32 repetitions/report-enabled host')
    values=[r['wall_ms'] for r in rows]
    return dict(values_ms=values,median_ms=statistics.median(values),spread_ms=max(values)-min(values))


def perturbation(original,off,on):
    delta=off['median_ms']-original['median_ms']
    spread=max(original['spread_ms'],off['spread_ms'])
    return dict(off_minus_original_ms=delta,off_spread_bar_ms=spread,
                off_within_spread=abs(delta)<=spread,
                on_minus_off_ms=on['median_ms']-off['median_ms'],
                basis='five per-repeat host-wall medians; arm distributions, NOT interleaved pairs')


def template():
    return dict(model='/absolute/prepared/qwen3-30b-a3b',golden='/absolute/independent/golden_c1024.txt',
        builds={arm:dict(path='/absolute/prepared/'+arm,run_sha256='REQUIRED',plan_sha256='REQUIRED')
                for arm in ('original','off','on')},
        cpu_evidence='/absolute/local/fullmodel/evidence.json',
        byte_rules='/absolute/per_task_bytes.json',hop_estimates=None,
        sparse=[dict(pack='/absolute/snapshot/moe_up/pack.json',build='/absolute/prepared/probe_body',
            selector='candidate',pair='body',layer=0,task=0,wave=0,row=0,iterations=5)],
        prerequisites='same KEY32-on math, VM compiler, report-enabled CHECK builds, full model packed before measurement; no downloads',
        cost_estimate='0.25 GPU-hours at historical $1.99/hour = $0.4975; preparation/VM setup extra, no current price quote')


def run(config,logdir):
    c=json.loads(Path(config).read_text()); logdir=Path(logdir); logdir.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+900
    evidence=json.loads(Path(c['cpu_evidence']).read_text())
    if not evidence['off_ir_code_equal'] or not evidence['off_isa_code_equal'] or not evidence['off_device_text_byte_equal']:
        raise ValueError('OFF code-identity gate failed')
    for arm,b in c['builds'].items():
        path=Path(b['path'])
        if sha(path/'run')!=b['run_sha256'] or sha(path/'plan.json')!=b['plan_sha256']:
            raise ValueError('prepared full-model identity changed: '+arm)
    if len({b['plan_sha256'] for b in c['builds'].values()})!=1:
        raise ValueError('stamp builds changed the task plan')
    # Clear all runtime instrumentation inherited from the operator.
    clean={k:v for k,v in os.environ.items() if not k.startswith(('ETX_TRACE','ETX_EXEC_COUNT','ETX_INBODY_RAW','ETX_BF16_CAPTURE','ETX_UNFUSED_'))}
    clean.update(ETX_VL_PROGRESS='64')
    def step(name,cmd,extra=None):
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('NOT-COMPLETE session deadline')
        path=logdir/(name+'.log')
        if path.exists(): raise ValueError('fresh session logdir required: '+str(path))
        print(f'COMMAND {name} deadline={remaining:.1f}s log={path}',flush=True)
        with path.open('w') as out:
            out.write('# command '+json.dumps(list(map(str,cmd)))+'\n'); out.flush()
            p=subprocess.Popen(list(map(str,cmd)),stdout=out,stderr=subprocess.STDOUT,env=dict(clean,**(extra or {})),start_new_session=True)
            start=time.monotonic()
            try:
                while p.poll() is None:
                    remaining=deadline-time.monotonic()
                    if remaining<=0: raise TimeoutError('NOT-COMPLETE '+name)
                    try: p.wait(timeout=min(15,remaining))
                    except subprocess.TimeoutExpired:
                        print(f'PROGRESS {name} elapsed={time.monotonic()-start:.0f}s pid={p.pid}',flush=True)
                        if time.time()-path.stat().st_mtime>180:
                            for probe in (['ps','-eo','pid,ppid,etime,stat,args'],['rocm-smi']):
                                try: subprocess.run(probe,timeout=min(5,max(.01,deadline-time.monotonic())),check=False,stdout=out,stderr=subprocess.STDOUT)
                                except (OSError,subprocess.TimeoutExpired): pass
                if p.returncode: raise RuntimeError(f'NOT-COMPLETE {name} rc={p.returncode}; full log={path}')
            finally:
                if p.poll() is None: os.killpg(p.pid,signal.SIGKILL); p.wait()
        return path.read_text()
    def host(arm,args):
        b=Path(c['builds'][arm]['path'])
        return [b/'run',c['model'],b/'vl_plan.txt','--golden',c['golden'],'--gen','32',*args]
    report=dict(status='NOT-COMPLETE',gpu=os.environ.get('GPU','0'),wall={},raw_reports=[],sparse=[])
    try:
        # Mandatory fresh same-build import coverage at both positions FIRST.
        for arm in ('original','off','on'):
            for position in (0,1040):
                text=step(f'{arm}_check_{position}',host(arm,['--check','48','--check-pos',str(position)]))
                match=re.search(r'CHECK-IMPORTED checked=(\d+) planned=(\d+) replicas=\d+ position=(\d+)',text)
                instances=[json.loads(l.split(' ',1)[1]) for l in text.splitlines() if l.startswith('CHECK-INSTANCE ')]
                if (not match or match[1]!=match[2] or int(match[1])!=len(instances) or not instances
                    or int(match[3])!=position or 'CHECK OK: 0 launch(es) differ' not in text
                    or any(r['result']!='identical' or r['position']!=position for r in instances)):
                    raise ValueError('incomplete imported-launch correctness '+arm)
        for counters in (0,1):
            measured={}
            # Noninterleaved distributions reduce cold loads; do not use them
            # for a fusion or net-removability verdict.
            for arm in ('original','off','on'):
                extra={} if not counters else dict(ETX_EXEC_COUNT='1')
                raw_path=logdir/f'on_c{counters}.bin'
                if arm=='on': extra['ETX_INBODY_RAW']=str(raw_path.resolve())
                text=step(f'{arm}_counter{counters}',host(arm,['--repeat','5']),extra)
                measured[arm]=samples(text)
            report['wall'][str(counters)]=perturbation(**measured)
            if not report['wall'][str(counters)]['off_within_spread']:
                raise ValueError('OFF timed path moved outside run-to-run spread')
            plan=json.loads((Path(c['builds']['on']['path'])/'plan.json').read_text())
            result=reader.analyze(plan,reader.read_raw(raw_path),json.loads(Path(c['byte_rules']).read_text()),
                json.loads(Path(c['hop_estimates']).read_text()) if c.get('hop_estimates') else None)
            file=logdir/f'phases_counter{counters}.json'; file.write_text(json.dumps(result,indent=2)+'\n')
            report['raw_reports'].append(str(file))
        for i,s in enumerate(c['sparse']):
            p=json.loads(Path(s['pack']).read_text())
            cmd=[sys.executable,'-m','examples.vllm_llm.bf16_300.launch','--pack',s['pack'],'--build',s['build'],
                 '--arguments',p['arguments'],'--reference',p['reference']['file'],'--output',str(logdir/f'sparse_{i}'),
                 '--deadline',str(max(.01,deadline-time.monotonic()))]
            for key in ('selector','pair','layer','task','wave','row','iterations'): cmd+=['--'+key,str(s[key])]
            step(f'sparse_{i}_driver',cmd)
            report['sparse'].append(str(logdir/f'sparse_{i}.json'))
        report.update(status='MEASUREMENT-COMPLETE',acceptance=dict(verdict='NOT-ESTABLISHED',
            exclusive_attributed_fraction=0.,largest_removable_item=None,
            reason='timestamp boundaries do not certify exclusive hops or removable compute; need controlled atomic-item evidence'))
    finally:
        (logdir/'inbody_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 1 # no false attribution success from completion alone


def main():
    if float(os.environ.get('ETX_SESS_MAX_SECONDS', 900)) > MAX_SESSION_SECONDS:
        raise ValueError('inbody measurement hard stop cannot be extended')

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--template',type=Path)
    a=p.parse_args()
    if a.template:
        a.template.write_text(json.dumps(template(),indent=2)+'\n'); return
    if a.run:
        raise SystemExit(run(os.environ['ETX_INBODY_CONFIG'],os.environ['ETX_SESS_LOGDIR']))
    p.error('use gpu_jobs.py --dry-run or --template FILE')


if __name__=='__main__': main()
