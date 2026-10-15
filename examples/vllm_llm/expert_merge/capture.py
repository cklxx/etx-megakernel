"""Emit a checking-only full-model host, or seal an actual completed capture.

This generates a private host/build script. Shared attention/router files are
not changed. The added hook runs only on the existing --check sequential path.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
from .prepare import HERE, ROOT, sha
from .source import once


def emit(out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    source=(ROOT/"examples/vllm_llm/host.hip").read_text()
    anchor='#ifdef ETX_BF16_CAPTURE\n#include "examples/vllm_llm/bf16_300/capture.h"'
    source=once(source,anchor,'#include "examples/vllm_llm/expert_merge/capture.h"\n'+anchor)
    anchor='      if (x.exp.rfind("dsl_", 0) == 0) {\n'
    hook='''        if (const char* base = getenv("EEM_CAPTURE_DIR")) {
          if (x.req == 0) {
            const int target=atoi(getenv("EEM_CAPTURE_LAYER") ? getenv("EEM_CAPTURE_LAYER") : "23");
            if (target<0 || target>47 || (check_pos!=0 && check_pos!=1040)) die("expert merge capture coverage");
            EemCapture::before(base,x.role,x.layer,check_pos,target,ab.data()+x.off,resid2,topk_w);
          }
        }
'''
    source=once(source,anchor,anchor+hook)
    (out/"capture_host.hip").write_text(source)
    build=(ROOT/"examples/vllm_llm/build.sh").read_text()
    # This generated script is private to the caller's output directory.
    rootline=next(line for line in build.splitlines() if line.startswith("ROOT="))
    build=once(build,rootline,"ROOT="+shlex.quote(str(ROOT))+'; cd "$ROOT"')
    build=once(build,'"$ROOT/examples/vllm_llm/host.hip"',shlex.quote(str(out/"capture_host.hip")))
    build += ('\n"$PY" -m examples.vllm_llm.expert_merge.capture attest '
              +shlex.quote(str(out))+' --host-build "$OUT"\n')
    (out/"build.sh").write_text(build)
    (out/"source.json").write_text(json.dumps(dict(source_host=sha(ROOT/"examples/vllm_llm/host.hip"),
             generated_host=sha(out/"capture_host.hip"),capture_header=sha(HERE/"capture.h")),indent=2)+"\n")
    subprocess.run(["bash","-n",str(out/"build.sh")],check=True)
    print("CAPTURE HOST GENERATED",out,"; no full-model build or run done")


def attest(emitted, host_build):
    e=Path(emitted).resolve(); b=Path(host_build).resolve()
    source=json.loads((e/"source.json").read_text())
    if sha(e/"capture_host.hip") != source["generated_host"] or sha(HERE/"capture.h") != source["capture_header"]:
        raise ValueError("capture source changed during build")
    command=json.loads((b/"hipcc_command.json").read_text())
    if str(e/"capture_host.hip") not in command:
        raise ValueError("build did not compile the capture host")
    record=dict(source=source, emitted=str(e), binary=sha(b/"run"),
                plan=sha(b/"vl_plan.txt"), command=command,
                revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    (b/"capture-build.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")


def run(capture, host_build, packed, layer, position):
    from .runner import command, dump
    c=Path(capture).resolve(); b=Path(host_build).resolve()
    if layer not in range(48) or position not in (0,1040):
        raise ValueError("capture requires a layer 0..47 and position 0/1040")
    identity=json.loads((b/"capture-build.json").read_text())
    if identity["binary"]!=sha(b/"run") or identity["plan"]!=sha(b/"vl_plan.txt"):
        raise ValueError("stale capture build")
    c.mkdir(parents=True,exist_ok=False)
    dump(c/"build.json",identity)
    env=os.environ.copy(); env.update(EEM_CAPTURE_DIR=str(c),EEM_CAPTURE_LAYER=str(layer))
    command([b/"run",Path(packed).resolve(),b/"vl_plan.txt","--check","48","--check-pos",str(position)],
            c/"check.log",timeout=300,env=env)
    seal(c,b)


def seal(capture, host_build):
    c=Path(capture); b=Path(host_build)
    layer,position=map(int,(c/"capture.txt").read_text().split())
    # The actual check command must have finished. A partial capture never
    # becomes a valid pack. Keep the full import-check output beside the data.
    log=c/"check.log"
    text=log.read_text()
    status=json.loads((c/"check.log.status.json").read_text())
    identity=json.loads((c/"build.json").read_text())
    if (status["exit"]!=0 or status["timeout"] or not re.search(r"^CHECK OK: 0 launch\(es\) differ$",text,re.M)
            or "CHECK FAILED" in text or identity["binary"]!=sha(b/"run")
            or identity["plan"]!=sha(b/"vl_plan.txt")):
        raise ValueError("capture requires a completed successful kept --check log")
    coverage=re.findall(r"^CHECK-IMPORTED checked=(\d+) planned=(\d+) replicas=\d+ position=(\d+)$",text,re.M)
    if len(coverage)!=1 or int(coverage[0][0])==0 or coverage[0][0]!=coverage[0][1] or int(coverage[0][2])!=position:
        raise ValueError("capture requires complete same-position imported-launch coverage")
    for marker in (f"EEM-CAPTURE up layer={layer} position={position}",
                   f"EEM-CAPTURE consumer={'head' if layer==47 else 'qkv'} layer={-1 if layer==47 else layer+1} position={position}"):
        if text.count(marker)!=1: raise ValueError("missing/duplicate capture marker: "+marker)
    origin=dict(layer=layer,position=position,host_sha256=sha(b/"run"),check_log_sha256=sha(log),
                selected_ids_sha256=sha(c/"selected_ids.bin"),
                revision=identity["revision"],capture_build=identity,
                sources=json.loads((b/"build_identity.json").read_text()) if (b/"build_identity.json").exists() else {},
                note="Real DSL inputs captured on kept host --check. Full-model candidate is not implemented or accepted.")
    (c/"capture.json").write_text(json.dumps(origin,indent=2,sort_keys=True)+"\n")
    # Validate every field before a real-input label can reach the runner.
    # No duplicate large pack is needed. The captured directory is the pack.
    from .inputs import seal as seal_inputs
    seal_inputs(c,"real-capture",layer==47,origin)
    print("CAPTURE SEALED",c)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("emit","attest","run","seal")); p.add_argument("out",type=Path)
    p.add_argument("--host-build",type=Path); p.add_argument("--packed",type=Path)
    p.add_argument("--layer",type=int,default=23); p.add_argument("--position",type=int,default=1040); a=p.parse_args()
    if a.action=="emit": emit(a.out)
    elif a.action=="attest": attest(a.out,a.host_build)
    elif a.action=="run": run(a.out,a.host_build,a.packed,a.layer,a.position)
    else: seal(a.out,a.host_build)
