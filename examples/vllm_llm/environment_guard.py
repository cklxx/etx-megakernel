"""First, four-minute VM guard. --summarize is CPU only; --run requires a VM.

The guard never provisions. A slow/mismatched/unknown result records evidence
and stops before timed tranches; the owner can reprovision or rebase every row.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics as st
import subprocess
import sys
import time
from examples.vllm_llm.pairs import Refusal

SECONDS = 240
TREE_COMMITS = {'old': '7c85efb', 'hour': 'e068e47'}


def original_medians(text, position):
    if f'CHECK position {position} ' not in text or 'CHECK OK: 0 launch(es) differ' not in text:
        raise ValueError('guard incomplete original-kernel check')
    rows = re.findall(r'^\s+(attn|reduce)\s+L(\d+)\s+\S+\s+identical\s+original\s+([\d.]+) us\s+imported\s+([\d.]+) us$',text,re.M)
    result={}
    for role in ('attn','reduce'):
        rs=[(int(layer),float(orig),float(imp)) for kind,layer,orig,imp in rows if kind==role and int(layer)>0]
        if len(rs)!=47 or {r[0] for r in rs}!=set(range(1,48)) or any(not math.isfinite(v) or v<=0 for r in rs for v in r[1:]):
            raise ValueError(f'guard missing/duplicate original {role} population at {position}')
        result[role]=dict(original_us=st.median(r[1] for r in rs), imported_us=st.median(r[2] for r in rs),
                          count=47, layers=list(range(1,48)))
    return result


def decision(pairs, medians, weights_equal):
    if len(pairs)!=2 or any(set(p)!={'old','hour'} for p in pairs):
        raise ValueError('guard requires two complete adjacent pinned tree pairs')
    if any(not math.isfinite(v) or v<=0 for p in pairs for v in p.values()): raise ValueError('guard invalid wall sample')
    deltas=[p['hour']-p['old'] for p in pairs]
    # A pair crossing the source-drift bar is not hidden by median cancellation.
    drift=any(abs(d) >= .030 and not math.isclose(abs(d),.030,abs_tol=1e-12) for d in deltas)
    at_bar=any(math.isclose(abs(d),.030,abs_tol=1e-12) for d in deltas)
    slow=any(medians[k]['0']['attn']['original_us'] > 7.9 for k in TREE_COMMITS)
    wall_anchor=st.median(p['hour'] for p in pairs)
    rebase=wall_anchor>4.210 and not math.isclose(wall_anchor,4.210,abs_tol=1e-12)
    status='DATA-MISMATCH' if not weights_equal else 'TREE-DRIFT' if drift or at_bar else 'SLOW-VM' if slow else 'REBASE-REQUIRED' if rebase else 'PASS'
    return dict(status=status, deltas_ms=deltas, median_delta_ms=st.median(deltas),
        wall_anchor_ms=wall_anchor, original_attention_p0_slow=slow,
        need_ms=wall_anchor-3.3333, historical_need_valid=not rebase,
        action='continue to A' if status=='PASS' else 'STOP: record, fetch and reprovision; alternatively rebase every ledger row before a new timed session',
        cause='[U] clocks/power/data/root cause; paired tree equality only rules out a current >=KEEP tree delta')


def verify_tree(s, name):
    from examples.sessions.qwen300_hour import BASE_FLAGS
    from examples.vllm_llm import import_certificate as cert
    c=s.c['environment_guard']['trees'][name];b=Path(c['build'])
    identity=cert.verified_build(b,Path(s.c['model']))
    if cert.digest(identity)!=c['identity_sha256']: raise ValueError('guard stale tree identity: '+name)
    seal=json.loads((b/'guard_build.json').read_text())
    if cert.sha(b/'guard_build.json')!=c['seal_sha256']: raise ValueError('guard tree seal changed')
    if not re.fullmatch('[a-f0-9]{40}',seal['source_commit']) or not seal['source_commit'].startswith(TREE_COMMITS[name]):
        raise ValueError('guard wrong source commit: '+name)
    if seal['build_sha256']!=cert.digest(identity): raise ValueError('guard source/build seal mismatch')
    if identity['hipcc_flags']!=BASE_FLAGS or identity['overrides'].get('ETX_DEV_RESET','0')!='0' or not identity['check_enabled']:
        raise ValueError('guard must use same kept flags with reset OFF and CHECK enabled')
    if seal.get('flags') != {'ETX_TOPK_KEY32':'1','ETX_VL_CHECK':'1','ETX_DEV_RESET':'0'}:
        raise ValueError('guard kept KEY32/CHECK/reset flags not sealed')
    if not all(seal['cpu_gates'].get(k) is True for k in ('pytest','gfx942','rocm_build_check')):
        raise ValueError('guard tree CPU/native compilation gates missing')
    ir=Path(c['normalized_device_ir'])
    if cert.sha(ir)!=seal['normalized_device_ir_sha256']: raise ValueError('guard IR seal changed')
    return identity,seal


def hardware_readings(text):
    if 'MI300X' not in text.upper(): raise ValueError('guard needs MI300X product identity')
    cards=set(re.findall(r'GPU\[(\d+)\]',text))
    if len(cards)!=1: raise ValueError('guard requires one-GPU VM readings')
    for clock in ('sclk','mclk'):
        if not re.search(rf'GPU\[\d+\].*{clock} clock level:.*[\d.]+\s*Mhz',text,re.I):
            raise ValueError('guard missing '+clock+' clock reading')
    if not re.search(r'GPU\[\d+\].*(?:Average Graphics Package Power|Current Socket Graphics Package Power)\s*\(W\):\s*\d+(?:\.\d+)?\s*$',text,re.M|re.I):
        raise ValueError('guard missing power reading')
    if not re.search(r'GPU\[\d+\].*Performance Level:\s*(auto|low|high|manual|profile_standard|profile_min_sclk|profile_min_mclk|profile_peak|profile_exit)\s*$',text,re.M|re.I):
        raise ValueError('guard missing perflevel reading')


def verify_current(s):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    gates=json.loads(Path(s.c['cpu_gates']).read_text())
    if not all(gates.get(k) is True for k in ('pytest','gfx942','dry_run','shell','local_matrix_16','rocm_build_check')):
        raise ValueError('guard current CPU gates missing/failed')
    if gates.get('source_hashes')!=h.source_hashes(): raise ValueError('guard current CPU gates stale')
    seal=h.verify_seal(s,'kept')
    identity=cert.verified_build(Path(s.c['builds']['kept']),Path(s.c['model']))
    if identity['hipcc_flags']!=h.BASE_FLAGS or identity['overrides'].get('ETX_DEV_RESET','0')!='0' or identity['check_enabled'] is not True:
        raise ValueError('guard current kept flags/CHECK changed')
    if seal['flags'].get('ETX_DEV_RESET','0')!='0': raise ValueError('guard current reset must be OFF')
    if cert.sha(s.c['normalized_device_ir'])!=s.c['normalized_device_ir_sha256']:
        raise ValueError('guard current IR seal changed')
    return identity


def local_decision(anchors, medians):
    if len(anchors)!=2 or any(not math.isfinite(v) or v<=0 for v in anchors):
        raise ValueError('guard requires two complete kept anchor samples')
    anchor=st.median(anchors)
    slow=medians['0']['attn']['original_us']>7.9
    rebase=anchor>4.210 and not math.isclose(anchor,4.210,abs_tol=1e-12)
    status='SLOW-VM' if slow else 'REBASE-REQUIRED' if rebase else 'PASS-LOCAL-ONLY'
    return dict(status=status,wall_anchor_ms=anchor,original_attention_p0_slow=slow,
                need_ms=anchor-3.3333,historical_need_valid=not rebase,
                action='continue to A' if status=='PASS-LOCAL-ONLY' else 'STOP: record, fetch and reprovision; alternatively rebase every ledger row before a new timed session',
                cause='[U] pinned-tree drift check unavailable; clocks/power/data/root cause')


def run(s):
    from examples.sessions import qwen300_hour as h
    evidence=dict(status='NOT-COMPLETE',action='STOP: missing guard evidence',
                  cause='[U] clocks/power/data/root cause')
    try:
        ok=_run(s,evidence)
    except (OSError,ValueError,KeyError,RuntimeError,TimeoutError,subprocess.SubprocessError,Refusal) as exc:
        evidence.update(status='NOT-COMPLETE',reason=str(exc),action='STOP: missing guard evidence')
        ok=False
    evidence['elapsed_s']=time.monotonic()-s.start
    s.row('GUARD',evidence,evidence['status'],h.REGISTERED['GUARD'])
    return ok


def _run(s, evidence):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    s.deadline=min(s.start+SECONDS,s.end)
    # First VM command, before any timed tranche or optional eligibility work.
    hardware=s.step('GUARD_hardware',['rocm-smi','--showproductname','--showclocks','--showpower','--showperflevel'],timeout=15,expected=5)
    evidence['hardware']=hardware
    hardware_readings(hardware)
    cfg=s.c.get('environment_guard', {})
    trees=cfg.get('trees', {})
    unavailable=[]
    for name in TREE_COMMITS:
        if name not in trees:
            unavailable.append(name+': tree config absent')
        elif not Path(trees[name]['build']).is_dir():
            unavailable.append(name+': build directory absent')
    pinned=not unavailable
    evidence['pinned_trees']='available' if pinned else 'unavailable: '+ '; '.join(unavailable)+' [U]'
    kept=verify_current(s)
    sealed={name:verify_tree(s,name) for name in TREE_COMMITS if name in trees and Path(trees[name]['build']).is_dir()}
    if pinned:
        if sealed['old'][0]['compiler']!=sealed['hour'][0]['compiler']:
            raise ValueError('guard tree compilers differ')
        if sealed['old'][1]['normalized_device_ir_sha256']!=sealed['hour'][1]['normalized_device_ir_sha256']:
            raise ValueError('guard device IR differs; investigate before timing')
    else:
        trees={'kept':dict(build=s.c['builds']['kept'])}
        sealed={'kept':(kept,None)}
    weights=s.c.get('weight_sha256', cfg.get('weight_sha256'))
    if weights is None: raise ValueError('guard staged weight hashes missing')
    required={str(Path(s.c['model'])/'vl.bin'),*map(str,Path(s.c['hf']).glob('*.safetensors'))}
    if set(weights)!=required or len(required)<2: raise ValueError('guard must seal packed and all HF weight files')
    text=s.step('GUARD_weights',['sha256sum',*sorted(required)],timeout=240,expected=100)
    actual={}
    for line in text.splitlines():
        match=re.fullmatch(r'([a-f0-9]{64})\s+\*?(.+)',line)
        if match:
            if match[2] in actual: raise ValueError('guard duplicate weight hash')
            actual[match[2]]=match[1]
    if set(actual)!=required: raise ValueError('guard incomplete weight checksums')
    evidence.update(weights=actual, expected_weights=weights)
    if actual!=weights:
        evidence.update(status='DATA-MISMATCH',action='STOP: restage immutable data before provisioning/timing')
        return False
    medians={};pairs=[];anchors=[]
    evidence.update(original_medians=medians,pairs=pairs,anchor_samples_ms=anchors)
    def capture(name,label,args):
        b=Path(trees[name]['build'])
        text=s.step(label,[sys.executable,'-u','-m','examples.vllm_llm.import_certificate','--run-build',b,'--model',s.c['model'],'--',*args],timeout=40,expected=28)
        cert.stamped(text,sealed[name][0]);return text
    for name in sealed:
        b=Path(trees[name]['build']);archive=s.out/('guard_certificate_'+name)
        cert.export_build(b,Path(s.c['model']),archive);medians[name]={}
        for pos in (0,1040):
            text=capture(name,f'GUARD_{name}_p{pos}',['--check','48','--check-pos',str(pos)])
            (archive/f'check_p{pos}.txt').write_text(text)
            medians[name][str(pos)]=original_medians(text,pos)
        cert.validate(archive,timing=False)
    for rnd in range(2):
        row={}
        for name in ((('old','hour') if rnd==0 else ('hour','old')) if pinned else ('kept',)):
            text=h.token_gate(capture(name,f'GUARD_r{rnd}_{name}',['--repeat','1','--check-dsl']))
            if 'CHECK-DSL OK' not in text: raise ValueError('guard CHECK-DSL missing')
            from examples.vllm_llm.pairs import parse
            runs=parse(text.splitlines()).runs
            if len(runs)!=1 or runs[0].arm!='fused': raise ValueError('guard invalid kept timing')
            row[name]=runs[0].wall_ms
        if pinned: pairs.append(row)
        else: anchors.append(row['kept'])
    result=decision(pairs,medians,True) if pinned else local_decision(anchors,medians['kept'])
    result.update(original_medians=medians,pairs=pairs,weights=actual,
                  elapsed_s=time.monotonic()-s.start, sample_basis='RESULT wall_ms: median of decode wall steps')
    evidence.update(result)
    s.deadline=min(time.monotonic()+h.TRANCHE_SECONDS['A']-15,s.end)
    return result['status'] in ('PASS','PASS-LOCAL-ONLY')



def seal_config(config):
    """CPU-only bind already-built tree artifacts to their checked provenance.

    This does not manufacture test results or build either historical tree.
    Native CPU gates and normalized device IR must already exist.
    """
    from examples.vllm_llm import import_certificate as cert
    c=json.loads(config.read_text())
    c['normalized_device_ir_sha256']=cert.sha(c['normalized_device_ir'])
    trees=c.get('environment_guard', {}).get('trees', {})
    for name, prefix in TREE_COMMITS.items():
        if name not in trees or not Path(trees[name]['build']).is_dir(): continue
        tree=trees[name];source=Path(tree['source_tree'])
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if not commit.startswith(prefix): raise ValueError('guard seal wrong source tree')
        subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD'],check=True)
        gates=json.loads(Path(tree['cpu_gates']).read_text())
        if gates.get('source_commit')!=commit or not all(gates.get(k) is True for k in ('pytest','gfx942','rocm_build_check')):
            raise ValueError('guard seal missing same-commit CPU/native gates')
        b=Path(tree['build']);identity=cert.verified_build(b,Path(c['model']))
        if gates.get('hipcc_flags')!=identity['hipcc_flags']:
            raise ValueError('guard seal flags differ from native CPU gates')
        if gates.get('flags') != {'ETX_TOPK_KEY32':'1','ETX_VL_CHECK':'1','ETX_DEV_RESET':'0'}:
            raise ValueError('guard seal lacks exact kept KEY32/CHECK/reset flags')
        record=dict(flags=gates['flags'],source_commit=commit,build_sha256=cert.digest(identity),cpu_gates=gates,
                    normalized_device_ir_sha256=cert.sha(tree['normalized_device_ir']))
        (b/'guard_build.json').write_text(json.dumps(record,indent=2)+'\n')
        tree.update(identity_sha256=cert.digest(identity),seal_sha256=cert.sha(b/'guard_build.json'))
    config.write_text(json.dumps(c,indent=2)+'\n')
    print('GUARD CPU seal complete; GPU correctness remains unverified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--summarize',type=Path);modes.add_argument('--run',type=Path);modes.add_argument('--seal-config',type=Path)
    p.add_argument('--position',type=int,default=0);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.seal_config: seal_config(a.seal_config);return 0
    if not a.output: p.error('--output required for --summarize/--run')
    if a.summarize:
        result=original_medians(a.summarize.read_text(),a.position)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0
    s=h_session(a.run,a.output)
    try: ok=run(s)
    except (OSError,ValueError,KeyError,RuntimeError,TimeoutError,subprocess.SubprocessError,Refusal) as exc:
        s.row('GUARD',dict(reason=str(exc),action='STOP: missing guard evidence'), 'NOT-COMPLETE', 'all guard populations required');ok=False
    finally: s.finish()
    return 0 if ok else 2


def h_session(config,output):
    from examples.sessions.qwen300_hour import Session
    return Session(json.loads(config.read_text()),output,SECONDS+10)


if __name__=='__main__': raise SystemExit(main())
