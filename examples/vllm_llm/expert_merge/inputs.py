"""Portable raw input packs with content hashes. Real capture is a separate gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from .prepare import sha

SIZES = dict(x=4096,residual=4096,norm=4096,ids=32,wts=32,
             w13=8*2*768*2048*2,w2=8*2048*768*2,role=1)


def bf16(a):
    a=np.asarray(a,dtype=np.float32)
    u=a.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype("<u2")


def seal(out, kind, head, provenance):
    out=Path(out)
    m=dict(schema=1,kind=kind,head=head,M=151936 if head else 5120,provenance=provenance,
           imports="N/A: zero imported kernels in this isolated DSL subgraph",
           tokens="PENDING: no full-model acceptance from this probe")
    names={**SIZES,"consumer":m["M"]*2048*2}
    m["files"]={name+".bin":dict(bytes=size,sha256=sha(out/(name+".bin"))) for name,size in names.items()}
    if kind=="real-capture":
        m["capture_evidence"]={name:sha(out/name) for name in ("check.log","check.log.status.json","build.json")}
    (out/"input.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    validate(out)
    return m


def validate(out):
    out=Path(out)
    m=json.loads((out/"input.json").read_text())
    if m.get("schema") != 1 or m.get("kind") not in ("synthetic","real-capture"):
        raise ValueError("input schema/kind")
    if type(m.get("head")) is not bool or m["M"] != (151936 if m["head"] else 5120):
        raise ValueError("input consumer shape")
    if m["kind"]=="real-capture":
        for name in ("check.log","check.log.status.json","build.json"):
            if sha(out/name) != m.get("capture_evidence",{}).get(name):
                raise ValueError("capture proof identity: "+name)
        receipt=json.loads((out/"check.log.status.json").read_text())
        if receipt["exit"]!=0 or receipt["timeout"]:
            raise ValueError("failed capture cannot be a real input")
    names={**SIZES,"consumer":m["M"]*2048*2}
    if set(m["files"]) != {x+".bin" for x in names}:
        raise ValueError("incomplete input pack")
    for name,size in names.items():
        path=out/(name+".bin"); item=m["files"][path.name]
        if path.stat().st_size != size or item["bytes"] != size or sha(path) != item["sha256"]:
            raise ValueError("input size/hash mismatch: "+path.name)
    if (out/"role.bin").read_bytes() != bytes([int(m["head"])]):
        raise ValueError("input role mismatch")
    if (out/"ids.bin").read_bytes() != struct.pack("<8i",*range(8)):
        raise ValueError("packed expert IDs must be slot ordered 0..7")
    w=np.fromfile(out/"wts.bin",dtype="<f4")
    if not np.isfinite(w).all() or (w<0).any() or not w.any():
        raise ValueError("nonfinite/invalid routing weights")
    for name in ("x","residual","norm","w13","w2","consumer"):
        bits=np.memmap(out/(name+".bin"),dtype="<u2",mode="r")
        if np.any((bits & 0x7f80)==0x7f80):
            raise ValueError("nonfinite bf16 input: "+name)
    return m


def synthetic(out,head=False,seed=20261015):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed)
    # Generate in row chunks. No large all-model allocation or remote data.
    for name,n,scale in (("x",2048,1),("residual",2048,1),("norm",2048,.125),
                         ("w13",8*2*768*2048,.02),("w2",8*2048*768,.02),
                         ("consumer",(151936 if head else 5120)*2048,.02)):
        with (out/(name+".bin")).open("wb") as f:
            for first in range(0,n,2048*64):
                a=rng.normal(0,scale,min(2048*64,n-first)).astype(np.float32)
                if name=="norm": a+=1
                f.write(bf16(a).tobytes())
    (out/"ids.bin").write_bytes(struct.pack("<8i",*range(8)))
    (out/"wts.bin").write_bytes(struct.pack("<8f",.25,.2,.15,.12,.1,.08,.06,.04))
    (out/"role.bin").write_bytes(bytes([int(head)]))
    return seal(out,"synthetic",head,dict(seed=seed,model_activations=False,
                limitation="Synthetic mechanism probe only. No real-input or model acceptance."))


def from_capture(out, capture):
    out=Path(out); c=Path(capture)
    if out.exists(): raise ValueError("output exists; preserve earlier inputs")
    meta=json.loads((c/"capture.json").read_text())
    if validate(c)["kind"]!="real-capture":
        raise ValueError("capture has no completed real-input proof")
    if meta["layer"] not in range(48) or meta["position"] not in (0,1040):
        raise ValueError("capture layer/position outside registered coverage")
    if meta["revision"] == "" or meta["host_sha256"] == "":
        raise ValueError("capture lacks source/executable identity")
    import shutil
    out.mkdir(parents=True)
    head=meta["layer"]==47
    for name in (*SIZES,"consumer"):
        shutil.copyfile(c/(name+".bin"),out/(name+".bin"))
    for name in ("check.log","check.log.status.json","build.json"):
        shutil.copyfile(c/name,out/name)
    return seal(out,"real-capture",head,meta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("synthetic","capture","validate")); p.add_argument("out",type=Path)
    p.add_argument("--head",action="store_true"); p.add_argument("--capture",type=Path)
    a=p.parse_args()
    m=synthetic(a.out,a.head) if a.action=="synthetic" else from_capture(a.out,a.capture) if a.action=="capture" else validate(a.out)
    print(json.dumps(m,sort_keys=True))


if __name__=="__main__": main()
