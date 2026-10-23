"""Fail-closed one-GPU experiment runner. No provisioning or remote execution.

The session deadline is fixed at its first build. Retry gets a new attempt
directory within that deadline. Logs and certificates are never overwritten.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import statistics
import subprocess
import sys
import time
from . import inputs
from . import paths
from .prepare import generate, compile_native, sha, verify, clean_env

ARMS=("control_diag","candidate_diag","control","candidate")
BASE_FIELDS={"act":8*768*2,"contrib":8*2048*2,"residual":4096,"normalized":4096}
DIAG_FIELDS={"merged":4096,"partials":1024,"inverse":4,"staged":256*2048*2}
CAP_SECONDS=1800


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")


@contextlib.contextmanager
def lock(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/"active.lock").open("a") as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("another expert merge runner owns this session") from None
        yield


def command(cmd, log, timeout=120, env=None):
    """Full logs, exit status, bounded silence and process-group termination."""
    if Path(log).exists(): raise ValueError("log exists; choose a new attempt")
    start=time.monotonic(); output=[]
    print("STAGE exec",cmd,"evidence",log,flush=True)
    with Path(log).open("w") as f:
        p=subprocess.Popen(list(map(str,cmd)),cwd=paths.ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
        sel=selectors.DefaultSelector(); sel.register(p.stdout,selectors.EVENT_READ)
        last=start; timed_out=False
        try:
            while sel.get_map():
                for key,_ in sel.select(.2):
                    b=os.read(key.fileobj.fileno(),65536)
                    if not b: sel.unregister(key.fileobj); continue
                    s=b.decode(errors="replace"); f.write(s); f.flush(); output.append(s)
                    print(s,end="",flush=True); last=time.monotonic()
                now=time.monotonic()
                if now-start > timeout:
                    timed_out=True; os.killpg(p.pid,signal.SIGKILL); break
                if now-last > 45:
                    status=f"SILENCE pid={p.pid} elapsed={now-start:.1f}s log={log}\n"
                    print(status,end="",flush=True); f.write(status); f.flush(); last=now
            rc=p.wait(timeout=5)
        finally:
            if p.poll() is None: os.killpg(p.pid,signal.SIGKILL); p.wait()
            sel.close(); p.stdout.close()
        dump(str(log)+".status.json",dict(cmd=list(map(str,cmd)),exit=rc,timeout=timed_out,seconds=time.monotonic()-start))
    if rc or timed_out: raise ValueError("timeout" if timed_out else f"command failed {rc}: {log}")
    return "".join(output)


def session(root):
    root=Path(root).resolve(); m=json.loads((root/"session.json").read_text())
    if time.time() >= m["deadline_unix"]: raise ValueError("fixed session deadline expired; no automatic budget extension")
    inputs.validate(m["input"])
    if sha(Path(m["input"])/"input.json") != m["input_sha256"]:
        raise ValueError("stale input manifest identity")
    return m


def build(root, input_path):
    root=Path(root).resolve(); inp=Path(input_path).resolve(); data=inputs.validate(inp)
    if (root/"session.json").exists():
        m=session(root)
        if m["input"] != str(inp): raise ValueError("retry cannot change input")
    else:
        m=dict(schema=1,input=str(inp),input_sha256=sha(inp/"input.json"),start_unix=time.time(),
               deadline_unix=time.time()+CAP_SECONDS,kind=data["kind"],head=data["head"])
        dump(root/"session.json",m)
    for arm in ARMS:
        out=root/arm
        if (out/"build.json").exists(): check_build(out); continue
        if out.exists(): raise ValueError(f"incomplete build {out}; retain it and start a fresh explicitly budgeted session")
        session(root)
        generate(out,arm.split("_")[0],arm.endswith("_diag"),data["head"])
        # The subprocess provides the compiler's full stdout and exit record.
        command([sys.executable,"-m","examples.vllm_llm.expert_merge.prepare","native",out],
                root/(arm+"_build.log"),timeout=min(300,session(root)["deadline_unix"]-time.time()))
    print("STAGE BUILT; next: python -m examples.vllm_llm.expert_merge.runner check",root,flush=True)


def check_build(out):
    identity=verify(out); native=json.loads((Path(out)/"build.json").read_text())
    if any(native[k]!=identity[k] for k in identity): raise ValueError("stale native build identity")
    if sha(Path(out)/"run") != native["binary"]: raise ValueError("stale executable identity")
    return dict(id=identity["id"],binary=native["binary"],head=identity["head"],diagnostic=identity["diagnostic"])


def attempt(root, stage):
    # Keep every failed attempt. The deadline is stored in session.json.
    for i in range(1000):
        path=Path(root)/f"{stage}-{i:03d}"
        try: path.mkdir(); return path
        except FileExistsError: pass
    raise ValueError("too many attempts")


def snapshot(path, M, diagnostic, candidate):
    import numpy as np
    fields={**BASE_FIELDS,"output":M*2,**(DIAG_FIELDS if diagnostic else {})}
    result={}
    for round in range(12):
        for name,n in fields.items():
            p=Path(path)/f"r{round}_{name}.bin"
            if not p.is_file() or p.stat().st_size != n: raise ValueError("missing/short correctness output: "+str(p))
            if name in ("partials","inverse"):
                values=np.fromfile(p,dtype="<f4")
                if not np.isfinite(values).all() or (name=="inverse" and (values<=0).any()):
                    raise ValueError("nonfinite/invalid correctness output: "+str(p))
            else:
                bits=np.fromfile(p,dtype="<u2")
                if ((bits & 0x7f80)==0x7f80).any():
                    raise ValueError("nonfinite correctness output: "+str(p))
            result[p.name]=sha(p)
        if candidate and diagnostic:
            import struct
            owners=(Path(path)/f"r{round}_owners.bin").read_bytes()
            if len(owners)!=128 or any(v not in range(8) for v in struct.unpack("<32i",owners)):
                raise ValueError("invalid row owner coverage")
    return result


def certify(root):
    root=Path(root).resolve(); m=session(root); data=inputs.validate(m["input"])
    builds={arm:check_build(root/arm) for arm in ARMS}
    if any(b["head"]!=m["head"] for b in builds.values()): raise ValueError("consumer identity mismatch")
    out=attempt(root,"check"); captures={}
    pointer=root/"certificate.json"
    if pointer.exists():
        (out/"previous-certificate.json").write_bytes(pointer.read_bytes())
        pointer.unlink()  # A failed recheck must never leave timing authorized.
    for arm in ARMS:
        for mode in ("fused","graph"):
            session(root); key=arm+"_"+mode; path=out/key; path.mkdir()
            command([root/arm/"run",m["input"],path,"check",mode],out/(key+".log"),
                    min(180,session(root)["deadline_unix"]-time.time()),clean_env())
            captures[key]=snapshot(path,data["M"],arm.endswith("_diag"),arm.startswith("candidate"))
    reference=captures["control_diag_fused"]
    for key,capture in captures.items():
        for file,digest in capture.items():
            if reference[file]!=digest: raise ValueError("BYTE CHECK FAILED: "+key+" "+file)
    # Every diagnostic consumer must see the same normalized vector, not just
    # task zero. Compare each of 256 copies, for every reset and both modes.
    for key in captures:
        if "_diag_" not in key: continue
        for round in range(12):
            path=out/key
            v=(path/f"r{round}_normalized.bin").read_bytes()
            if (path/f"r{round}_staged.bin").read_bytes()!=v*256:
                raise ValueError("not all consumer normalized inputs match")
    # A positive signed input and its negative must not produce stale up output.
    if reference["r0_act.bin"]==reference["r1_act.bin"]:
        raise ValueError("reuse check lacks changed activation payload")
    certificate=dict(schema=1,status="OK",input_sha256=m["input_sha256"],builds=builds,
                     captures=captures,path=str(out),rounds=12,consumers=256,
                     imports="N/A: zero imports",tokens="PENDING: not a model experiment")
    dump(out/"certificate.json",certificate)
    # The latest pointer can change only after a complete successful check.
    dump(root/"certificate.json",dict(path=str(out/"certificate.json"),sha256=sha(out/"certificate.json")))
    print("STAGE BYTE-CHECK OK",out,"next: python -m examples.vllm_llm.expert_merge.runner time",root,flush=True)
    return certificate


def require_certificate(root):
    m=session(root); pointer=json.loads((Path(root)/"certificate.json").read_text())
    p=Path(pointer["path"])
    if sha(p)!=pointer["sha256"]: raise ValueError("stale correctness certificate")
    c=json.loads(p.read_text())
    if c.get("status")!="OK" or c.get("rounds")!=12 or c.get("consumers")!=256:
        raise ValueError("missing/failed mandatory correctness result")
    if c["input_sha256"]!=m["input_sha256"]: raise ValueError("stale correctness input")
    for arm in ARMS:
        if c["builds"][arm]!=check_build(Path(root)/arm): raise ValueError("stale correctness build")
        for mode in ("fused","graph"):
            key=arm+"_"+mode
            now=snapshot(Path(c["path"])/key,151936 if m["head"] else 5120,arm.endswith("_diag"),arm.startswith("candidate"))
            if now!=c["captures"][key]: raise ValueError("correctness artifacts changed")
    return c


def verdict(pairs,kind):
    if len(pairs)!=5 or any(not math.isfinite(x) for x in pairs):
        raise ValueError("need five complete finite paired gains")
    median=statistics.median(pairs); spread=max(pairs)-min(pairs)
    passed=all(x>0 for x in pairs) and median>=.625 and median>2*spread
    return dict(gains_us=pairs,median_gain_us=median,spread_us=spread,
                rule="five gains >0; median >=0.625 us and >2*(max-min)",
                verdict=("SYNTHETIC-PASS-REAL-INPUT-REQUIRED" if kind=="synthetic" else "REAL-INPUT-SCREEN-PASS") if passed else "STOP-NO-BENEFIT",
                full_model="OFF; no KEEP, fusion or strongest-vLLM claim")


def measure(root):
    root=Path(root).resolve(); require_certificate(root); m=session(root); out=attempt(root,"time")
    results={}; env=clean_env(); env["EEM_CHECKED_IDENTITY"]=sha(root/"certificate.json")
    for round in range(5):
        for mode in ("fused","graph"):
            order=("control","candidate") if round%2==0 else ("candidate","control")
            for arm in order:
                require_certificate(root)  # Recheck immediately before every timed arm.
                key=f"r{round}_{mode}_{arm}"; path=out/key; path.mkdir()
                log=command([root/arm/"run",m["input"],path,"time",mode,"200"],out/(key+".log"),
                            min(90,session(root)["deadline_unix"]-time.time()),env)
                lines=re.findall(r"^EEM-TIME (.+)$",log,re.M)
                if len(lines)!=1: raise ValueError("missing complete timing record")
                row=json.loads(lines[0]); identity=check_build(root/arm)
                if (row["identity"]!=identity["id"] or row["n"]!=200 or row["mode"]!=mode or row["arm"]!=arm
                        or not math.isfinite(row["us"]) or row["us"]<=0):
                    raise ValueError("invalid timing identity/coverage")
                results[key]=row; dump(out/"partial.json",results)
    pairs=[results[f"r{r}_fused_control"]["us"]-results[f"r{r}_fused_candidate"]["us"] for r in range(5)]
    report=verdict(pairs,m["kind"])
    report.update(input_sha256=m["input_sha256"],results=results,
                  window="device events around captured reset + complete up/down/consumer + runtime retirement",
                  system_baseline="N/A for isolated subgraph; mandatory strongest legal vLLM for later model trial")
    dump(out/"report.json",report); print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("build","check","time","status"))
    p.add_argument("root",type=Path); p.add_argument("--input",type=Path); a=p.parse_args()
    try:
        with lock(a.root):
            if a.action=="build": build(a.root,a.input)
            elif a.action=="check": certify(a.root)
            elif a.action=="time": measure(a.root)
            else: print(json.dumps(session(a.root),indent=2))
    except (ValueError,OSError,subprocess.SubprocessError) as e:
        print(f"STAGE {a.action} BLOCKED: {e}\nEVIDENCE {a.root}\n"
              f"RESUME python -m examples.vllm_llm.expert_merge.runner {a.action} {a.root}"
              +(f" --input {a.input}" if a.input else "")+"\nDo not extend the saved deadline. Retain old attempts.",file=sys.stderr,flush=True)
        raise SystemExit(2) from None


if __name__=="__main__": main()
