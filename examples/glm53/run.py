"""Strict real two-GPU GLM model qualification, using pre-staged assets only."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import subprocess
from examples.glm53 import stage,oracle,reference,model_build
from examples.vllm_llm.batch_correctness import execute

ROOT=Path(__file__).resolve().parents[2]


def compare_logits(expected,got,atol=None):
    """Full-logit HF comparison; reports even when the argmax (token) matches.

    With atol set this is an explicit gate: a numerical bug that leaves the
    greedy token unchanged still fails. Without it the error is recorded only.
    """
    import numpy as np
    if expected.shape!=got.shape:raise ValueError(f'logit shape {got.shape} differs from HF {expected.shape}')
    diff=np.abs(got-expected)
    result=dict(logit_max_abs=float(diff.max()),logit_bit_equal=bool(np.array_equal(got,expected)),
                logit_argmax_equal=bool(np.array_equal(got.argmax(-1),expected.argmax(-1))),
                logit_max_abs_per_step=[float(x) for x in diff.max(axis=(1,2))])
    if atol is None:
        result['logit_gate']='REPORTED'
    else:
        if type(atol) not in (int,float) or not math.isfinite(atol) or atol<0:raise ValueError('logit tolerance must be finite and non-negative')
        result.update(logit_gate='ENFORCED',logit_atol=float(atol))
        if result['logit_max_abs']>atol:
            raise ValueError(f'logit max abs {result["logit_max_abs"]:.9g} exceeds {atol:.9g} although the greedy tokens match')
    return result


def qualify(staged,build,ref_path,out,seconds=7200,logit_atol=None):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema=1,status='NOT-COMPLETE',performance='unmeasured',model_certified=False)
    try:
        before=stage.validate(staged)
        spec=json.loads((build/'build-spec.json').read_text())
        if before!=spec:raise ValueError('build/staged model identity differs')
        identity=model_build.verified_build(build)
        required=['run','build-spec.json','plan.json','plan_data.h','glm53_spec.h','glm53_contract.h','load.tsv','megakernel_d0.hip','megakernel_d1.hip']
        build_hash={name:stage.file_sha(build/name) for name in required}
        ref=json.loads(ref_path.read_text());p=[r['prompt_ids'] for r in ref['requests']]
        # Reuse the strict record validator, before starting any GPU work.
        probe=dict(ref,backend='etx-pp2');oracle.compare(ref,probe)
        plan=json.loads((build/'plan.json').read_text());batch=len(p)
        if plan['shape']!=[batch] or len({len(x) for x in p})!=1 or len(p[0])+32>before['dimensions']['MAXLEN']:raise ValueError('reference differs from compiled batch/context')
        if ref.get('provenance',{}).get('index_sha256')!=before['snapshot_inputs']['index_sha256']:raise ValueError('reference checkpoint index differs from staged inputs')
        source=ref.get('provenance',{}).get('source_shards',{})
        if any(source.get(name)!=digest for name,digest in before['snapshot_inputs']['shards'].items()):raise ValueError('reference source shard identity differs')
        if ref.get('provenance',{}).get('quantization')!='FP8 block dequantize -> BF16':raise ValueError('reference uses incompatible FP8 numerical semantics')
        hf_logits=ref_path.with_suffix('.logits');ref_hash={str(ref_path):stage.file_sha(ref_path),str(hf_logits):stage.file_sha(hf_logits)}
        shape=(32,batch,before['dimensions']['VOCAB'])
        if hf_logits.stat().st_size!=int(np.prod(shape))*4:raise ValueError('truncated HF logits')
        prompt=out/'prompts.txt';prompt.write_text(f'{batch} {len(p[0])}\n'+'\n'.join(' '.join(map(str,row)) for row in p)+'\n')
        verdicts=[]
        # Two independent processes reload weights and reset every cache. Any
        # mismatch in either run is fatal; do not let a later repetition mask it.
        for repetition in range(2):
            prefix=out/f'run{repetition}'
            env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','DSL_ABLATE_'))}
            raw=execute([str((build/'run').resolve()),str(staged.resolve()),str((build/'load.tsv').resolve()),str(prompt.resolve()),str(prefix.resolve())],out/f'run{repetition}.log',env,seconds)
            tag=f'GLM-MODEL-EXECUTED batch={batch} layers=45 generated_each=32; reference comparison required'
            steps=re.findall(r'^GLM-MODEL-STEP pos=(\d+) batch=(\d+)$',raw,re.M)
            if raw.count(tag)!=1 or steps!=[(str(i),str(batch)) for i in range(len(p[0])+31)]:raise ValueError('incomplete/duplicate model execution steps')
            rows=[list(map(int,line.split())) for line in prefix.with_suffix('.tokens').read_text().splitlines()]
            if len(rows)!=batch:raise ValueError('missing/duplicate token output rows')
            candidate=dict(reference.identity(),backend='etx-pp2',batch=batch,
                           requests=[dict(prompt_ids=pr,generated_ids=ids) for pr,ids in zip(p,rows)])
            verdict=oracle.compare(ref,candidate)
            if verdict['status']!='TOKENS-EQUAL':raise ValueError(f'exact greedy mismatch: {verdict}')
            lf=prefix.with_suffix('.logits')
            if lf.stat().st_size!=hf_logits.stat().st_size:raise ValueError('truncated model logits')
            expected=np.memmap(hf_logits,dtype='<f4',mode='r',shape=shape)
            got=np.memmap(lf,dtype='<f4',mode='r',shape=shape)
            if not np.isfinite(got).all() or not np.isfinite(expected).all():raise ValueError('nonfinite logits')
            # Tokens are exact. Compare raw full logits against HF even when the
            # argmax matches; --logit-atol turns the reported error into a gate.
            # No universal full-checkpoint numerical certificate is invented.
            verdict.update(compare_logits(expected,got,logit_atol),
                           repetition=repetition,checkpoint_execution='real two-GPU ETX model',model_certified=False)
            print(f'GLM-LOGIT-GATE {verdict["logit_gate"]} repetition={repetition} max_abs={verdict["logit_max_abs"]:.9g} atol={logit_atol}',flush=True)
            verdicts.append(verdict);prefix.with_suffix('.json').write_text(json.dumps(candidate,indent=2)+'\n')
        if model_build.verified_build(build)!=identity or stage.validate(staged)!=before or {name:stage.file_sha(build/name) for name in required}!=build_hash or {p:stage.file_sha(Path(p)) for p in ref_hash}!=ref_hash:raise ValueError('assets/build/reference changed during qualification')
        receipt.update(status='TOKENS-EQUAL',model=before['model'],revision=before['revision'],batch=batch,
                       generated_each=32,repetitions=2,matched_each=[32]*batch,build_hashes=build_hash,
                       reference_hashes=ref_hash,verdicts=verdicts,
                       logit_gate='ENFORCED' if logit_atol is not None else 'REPORTED',logit_atol=logit_atol,
                       numerical_certification='raw full logits retained and compared with HF; token equality does not certify all operator numerics')
    except Exception as exc:
        receipt['error']=str(exc);raise
    finally:(out/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('operation',choices=['build','qualify']);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--reference',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch',type=int,choices=[1,2,4,8],default=1)
    ap.add_argument('--logit-atol',type=float,help='fail unless every full logit is within this absolute error of HF, even when all tokens match; default reports only')
    a=ap.parse_args();stage.validate(a.stage)
    if a.operation=='build':
        a.out.mkdir(parents=True,exist_ok=False)
        env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','DSL_ABLATE_'))}
        execute(['bash','examples/glm53/build_model.sh',str(a.stage.resolve()),str(a.build.resolve())],a.out/'build.log',dict(env,ETX_GLM_BATCH=str(a.batch)),1800)
    else:
        if a.reference is None:ap.error('--reference is required for qualify')
        print(json.dumps(qualify(a.stage,a.build,a.reference,a.out,logit_atol=a.logit_atol)))

if __name__=='__main__':main()
