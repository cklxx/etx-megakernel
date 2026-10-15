"""CPU replay of production producer arithmetic and actual --check captures.

CPU rsqrt/exp are diagnostic models. They do not certify GPU instruction bits.
Synthetic fixtures must never be labelled as actual-model captures.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from examples.vllm_llm import router_study as s
from etx.ops import producer as p

ROOT = Path(__file__).resolve().parents[2]
KINDS = dict(o=4096, residual=4096, gamma=4096, router=524288, norm=4096,
             resid2=4096, logits=256, ids=256, mix=256, tei=256)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Arithmetic:
    """Compile the real header, then call it from the numerical model."""
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='etx-producer-math-')
        d = Path(self.temp.name)
        src = d/'math.cpp'
        src.write_text('''#include <cmath>
#include "etx/ops/producer_math.h"
extern "C" void partial(const float* w,const float* z,const float* g,float* out) {
 for(int p=0;p<256;++p) {
  for(int e=0;e<128;++e) out[p*129+e]=etx_producer_dot8(w+e*2048+p*8,z+p*8,g+p*8);
  out[p*129+128]=etx_producer_square8(z+p*8);
 }
}
extern "C" void reduce(const float* partial,float scale,float* out) {
 for(int e=0;e<128;++e) {float score=0.f;
  for(int p=0;p<256;++p) score+=partial[p*129+e];
  out[e]=score*scale;
 }
}
''')
        compiler = str(Path('/opt/homebrew/opt/llvm/bin/clang++')) if Path('/opt/homebrew/opt/llvm/bin/clang++').exists() else shutil.which('clang++') or 'c++'
        self.command = [compiler, '-std=c++17', '-O2', '-ffp-contract=off',
                        '-shared', '-fPIC', '-I', str(ROOT), str(src), '-o', str(d/'math.so')]
        result = subprocess.run(self.command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"CPU header compile rc={result.returncode}: {result.stdout}{result.stderr}")
        self.lib = ctypes.CDLL(str(d/'math.so'))
        fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.lib.partial.argtypes = [fp, fp, fp, fp]
        self.lib.reduce.argtypes = [fp, ctypes.c_float, fp]

    def run(self, w, z, g):
        w,z,g = [np.ascontiguousarray(a, dtype=np.float32) for a in (w,z,g)]
        if w.shape != (128,2048) or z.shape != (2048,) or g.shape != (2048,):
            raise ValueError('Qwen shapes required')
        if not all(np.isfinite(a).all() for a in (w,z,g)):
            raise ValueError('finite inputs required')
        partial = np.zeros((256,129), np.float32)
        self.lib.partial(w,z,g,partial)
        ss = s.ordered_sum(np.concatenate([s.wave(partial[:,128].reshape(4,64)),np.zeros(12,np.float32)]))
        scale = np.float32(1/np.sqrt(np.float32(np.float32(ss/np.float32(2048))+np.float32(1e-6))))
        logits = np.zeros(128,np.float32)
        self.lib.reduce(partial,scale,logits)
        return s.bf(logits), scale, partial


class Protocol:
    """Happens-before simulator. Not a GPU cache or latency emulator."""
    def __init__(self):
        self.epoch=0
        self.reset()

    def reset(self):
        self.visible={}
        self.arrivals=set()
        self.finished=False

    def arrive(self, owner, record, *, release=True, epoch=None):
        if owner not in range(256) or owner in self.arrivals or self.finished:
            raise ValueError('duplicate, invalid, or late owner')
        if epoch is not None and epoch != self.epoch:
            raise ValueError('stale epoch')
        self.arrivals.add(owner)
        if release:
            self.visible[owner]=record.copy()
        return len(self.arrivals)==256

    def finish(self, *, acquire=True, normalized=True, fanout=True):
        if not acquire or set(self.visible)!=set(range(256)) or len(self.arrivals)!=256:
            raise ValueError('missing payload visibility or arrival')
        if not normalized or not fanout:
            raise ValueError('expert input or routing fanout missing')
        if self.epoch==2**32-1:
            raise ValueError('epoch wrap: restart process')
        self.finished=True
        result=np.stack([self.visible[i] for i in range(256)])
        self.epoch+=1
        self.reset()
        return result


def schedule(done, *, producer_us, interference_us, publication_us, readback_reduce_us,
             scale_us, topk_us, normalized_input_us, fanout_us):
    costs=[producer_us,interference_us,publication_us,readback_reduce_us,
           scale_us,topk_us,normalized_input_us,fanout_us]
    if not done or not np.isfinite([*done,*costs]).all() or min([*done,*costs])<0:
        raise ValueError('finite nonnegative times from o_proj start required')
    # Matches the implementation: no parallel norm tail claim. Reduction plus
    # scale, norm/logit publication, existing topk, then all replica output stores.
    last=max(d+producer_us+interference_us+publication_us for d in done)
    finish=last+readback_reduce_us+scale_us+normalized_input_us+topk_us+fanout_us
    return dict(last_producer_ready_us=last, all_expert_inputs_and_routes_ready_us=finish,
                exposed_last_producer_scoring_us=producer_us, status='assumed schedule only')


def seal(directory, *, identity, provenance, positions, layers=48):
    directory=Path(directory)
    if provenance not in ('actual-model-check','synthetic-test-fixture'):
        raise ValueError('explicit provenance required')
    if not identity or not positions or layers not in range(1,49):
        raise ValueError('capture identity and coverage required')
    files={}
    expected={f'L{i}_{k}.bin' for i in range(layers) for k in KINDS}
    if {f.name for f in directory.iterdir()} != expected:
        raise ValueError('capture coverage: missing or extra files')
    for i in range(layers):
        for kind,n in KINDS.items():
            path=directory/f'L{i}_{kind}.bin'
            if path.stat().st_size!=n:
                raise ValueError(f'capture size: {path.name}')
            files[path.name]=sha(path)
    data=dict(schema=1, provenance=provenance, identity=identity, positions=positions,
              layers=layers, files=files)
    with (directory/'capture.json').open('x') as f:
        json.dump(data,f,indent=2);f.write('\n')
    return data


def read_capture(directory, *, actual=False):
    directory=Path(directory)
    m=json.loads((directory/'capture.json').read_text())
    if m.get('schema')!=1 or not m.get('identity') or (actual and m.get('provenance')!='actual-model-check'):
        raise ValueError('capture provenance/identity')
    expected={f'L{i}_{k}.bin' for i in range(m['layers']) for k in KINDS}
    if set(m['files'])!=expected or {f.name for f in directory.iterdir()}!=expected|{'capture.json'}:
        raise ValueError('capture coverage')
    for name,h in m['files'].items():
        path=directory/name
        kind=name.split('_',1)[1][:-4]
        if path.stat().st_size!=KINDS[kind] or sha(path)!=h:
            raise ValueError('capture hash/size mismatch: '+name)
    return m


def tensor(directory,layer,kind):
    dtype='<i4' if kind in ('ids','tei') else '<f4' if kind=='mix' else '<u2'
    x=np.fromfile(Path(directory)/f'L{layer}_{kind}.bin',dtype=dtype)
    if dtype=='<u2': x=(x.astype(np.uint32)<<16).view(np.float32)
    if kind=='router': x=x.reshape(128,2048)
    return x


def replay(directory, arithmetic, *, actual=False):
    m=read_capture(directory,actual=actual)
    rows=[]
    for layer in range(m['layers']):
        o,r,g,w=[tensor(directory,layer,k) for k in ('o','residual','gamma','router')]
        z,base_scale,x=s.norm(o,r,g)
        logits,scale,partial=arithmetic.run(w,z,g)
        base=s.bf(s.dot(w,x))
        ids,mix=s.routing(logits)
        rows.append(dict(layer=layer, **s.compare(base,logits),
                         candidate_weights=mix.tolist(), reference_weights=s.routing(base)[1].tolist(),
                         norm_scale_bits_equal=bool(scale.view(np.uint32)==base_scale.view(np.uint32)),
                         captured_reference_ids_equal=bool(np.array_equal(s.routing(base)[0],tensor(directory,layer,'ids')[:8])),
                         cpu_model_matches_capture_logits=bool(np.array_equal(base,tensor(directory,layer,'logits')))))
    return dict(provenance=m['provenance'], identity=m['identity'], cases=rows,
                status='CPU diagnostic; hardware exactness gate is separate',
                actual_model_replay=actual)


def simulate(samples=32):
    rng=np.random.default_rng(20261015)
    a=Arithmetic(); protocol=Protocol(); rows=[]
    for case in range(samples):
        o,r=[s.bf(rng.normal(size=2048)) for _ in range(2)]
        g=s.bf(rng.uniform(.5,1.5,2048)); w=s.bf(rng.normal(0,.03,(128,2048)))
        z,base_scale,x=s.norm(o,r,g)
        logits,scale,partial=a.run(w,z,g)
        np.testing.assert_array_equal(logits,s.deferred(w,z,g,scale))
        assert scale==base_scale
        for _ in range(3):
            order=rng.permutation(256)
            for n,owner in enumerate(order):
                assert protocol.arrive(int(owner),partial[owner])==(n==255)
            np.testing.assert_array_equal(protocol.finish(),partial)
        row=s.compare(s.bf(s.dot(w,x)),logits)
        row.update(case=case, candidate_weights=s.routing(logits)[1].tolist(),reference_weights=s.routing(s.bf(s.dot(w,x)))[1].tolist())
        rows.append(row)
    # Reuse the retained sparse near-tie witness. Do not regenerate old study.
    witness=json.loads((s.DEFAULT/'numerical.json').read_text())['counterexample']
    w=np.zeros((128,2048),np.float32);w[:7,2]=2;w[7,0]=witness['w7_0'];w[8,1]=witness['w8_1']
    o=np.ones(2048,np.float32);o[0]=3
    z,scale,x=s.norm(o,np.zeros(2048,np.float32),np.ones(2048,np.float32))
    candidate,_,_=a.run(w,z,np.ones(2048,np.float32))
    near=s.compare(s.bf(s.dot(w,x)),candidate)
    assert near['candidate_ids']==witness['comparison']['candidate_ids']
    return dict(provenance='synthetic inputs AND synthetic weights; not captured activations',
                samples=samples, header_sha256=sha(ROOT/'etx/ops/producer_math.h'),
                compiler_command=a.command, cases=rows, near_tie=near,
                protocol_epochs=protocol.epoch, actual_model_replay='PENDING: no retained inputs/weights found',
                summary={k:sum(bool(r[k]) for r in rows) for k in ('ordered_ids_equal','expert_set_equal','weights_bits_equal')},
                schedule=schedule([10,12,11],producer_us=1,interference_us=.5,publication_us=.5,
                                  readback_reduce_us=2,scale_us=.1,topk_us=2,normalized_input_us=3,fanout_us=1),
                performance=dict(current_net_gain_us=None, router_bytes=524288,
                                 router_bandwidth_floor_us=524288/4.3e6,
                                 extra_partial_write_read_bytes=2*p.PARTIALS*4,
                                 assumption='4.3 TB/s; no fewer matrix operations',
                                 measured_costs='none for candidate',
                                 unknown=['producer interference','publication','readback reduction','scale',
                                          'topk integration','normalized expert input','fanout','matching baseline trace']))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--capture',type=Path)
    ap.add_argument('--samples',type=int,default=32);ap.add_argument('--require-actual',action='store_true')
    args=ap.parse_args()
    if args.samples<1: ap.error('positive samples required')
    result=replay(args.capture,Arithmetic(),actual=args.require_actual) if args.capture else simulate(args.samples)
    with args.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','compiler_command')},indent=2),flush=True)

if __name__=='__main__': main()
