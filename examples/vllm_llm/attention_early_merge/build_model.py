"""Build an isolated check/capture/hybrid host with the ordinary model plan.

The compiler shim changes only the final link sources. Import capture calls
pass straight to hipcc. All actual arguments are recorded. No shared file is
rewritten. The extra C6 kernel is a separate entry in the model executable.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .build import ROOT, HERE, clean_env, hybrid_host, sha
from examples.vllm_llm.kept_build import KEPT_ENV, BASE_FLAGS, source_hashes


def prepare(model,out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    env=clean_env(); env.update(KEPT_ENV)
    env['VLLM']=env.get('ETX_VLLM',env.get('VLLM',str(Path.home()/'code/vllm')))
    real=shutil.which('hipcc');
    if not real: raise ValueError('hipcc is missing; this command needs the ROCm host compiler and runtime')
    shim=out/'compiler'; shim.mkdir()
    (out/'c6_hybrid_host.hip').write_text(hybrid_host())
    wrapper='''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
out=pathlib.Path(os.environ['C6_MODEL_OUT'])
args=sys.argv[1:]
host=pathlib.Path(os.environ['C6_SOURCE_ROOT'])/'examples/vllm_llm/host.hip'
if str(host) in args:
    args[args.index(str(host))]=str(out/'c6_hybrid_host.hip')
    old=out/'megakernel_d0.hip'
    merged=out/'c6_merged.hip'
    merged.write_text(old.read_text()+'\\n#include "'+os.environ['C6_SOURCE_ROOT']+'/examples/vllm_llm/attention_early_merge/probe.hip"\\n')
    args[args.index(str(old))]=str(merged)
    (out/'c6_actual_link.json').write_text(json.dumps([os.environ['C6_REAL_HIPCC'],*args],indent=2)+'\\n')
sys.exit(subprocess.call([os.environ['C6_REAL_HIPCC'],*args]))
'''
    (shim/'hipcc').write_text(wrapper); (shim/'hipcc').chmod(0o755)
    env.update(PATH=str(shim)+os.pathsep+env['PATH'],C6_MODEL_OUT=str(out),C6_SOURCE_ROOT=str(ROOT),C6_REAL_HIPCC=real,
               ETX_OUT=str(out),ETX_HIPCC_FLAGS=BASE_FLAGS)
    command=['bash',str(HERE.parent/'build.sh'),str(Path(model).resolve())]
    (out/'c6_build_command.json').write_text(json.dumps(dict(command=command,overrides={k:v for k,v in env.items()
        if k in KEPT_ENV or k in ('ETX_HIPCC_FLAGS','ETX_OUT','C6_REAL_HIPCC')}),indent=2)+'\n')
    print('STAGE model-build evidence',out/'c6_build.log',flush=True)
    with (out/'c6_build.log').open('w') as log:
        rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    (out/'c6_build.rc').write_text(str(rc)+'\n')
    if rc: raise RuntimeError(f'build exit={rc}; see {out/"c6_build.log"}')
    identity=dict(schema='etx.c6.model.v1',plan='ordinary kept model; C6 is a separate correctness entry',
        hashes={n:sha(out/n) for n in ('run','c6_actual_link.json','c6_build_command.json','c6_hybrid_host.hip','c6_merged.hip')},
        sources=source_hashes(),
        capture='same executable, sequential original/import checks; before prep and after reduce',
        hybrid='C6_HYBRID=1; serial full-model replay, no full-megakernel claim')
    (out/'c6_model_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.model,a.out),indent=2))


if __name__=='__main__': main()
