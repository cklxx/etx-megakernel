"""Offline Qwen router proposal falsification and trace-anchored sensitivity study.

Not a GPU emulator: rsqrt/exp use NumPy, and the candidate's reduction is a
specified CPU design. No actual-model correctness or current wall prediction.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np

F = np.float32
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'examples/vllm_llm/results/2026-10-15-router-study'


def bf(x):
    """Finite binary32 -> bf16 round-to-nearest-even -> binary32."""
    a = np.asarray(x, dtype=F)
    if not np.all(np.isfinite(a)):
        raise ValueError('finite inputs required')
    u = a.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(F)


def wave(a):
    a = np.asarray(a, dtype=F).copy()
    if a.shape[-1] != 64:
        raise ValueError('wave64 required')
    for off in (32, 16, 8, 4, 2, 1):
        a = F(a + a[..., np.arange(64) ^ off])
    return a[..., 0]


def ordered_sum(a):
    s = np.zeros(a.shape[:-1], dtype=F)
    for i in range(a.shape[-1]):
        s = F(s + a[..., i])
    return s


def fma(a, b, c):
    # Binary32 products and sum evaluated in double then rounded to binary32.
    # For this study's bf16 operands this reproduces the scalar dot FMA steps.
    return F(a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64))


def norm(o, residual, gamma):
    if any(np.shape(x) != (2048,) for x in (o, residual, gamma)):
        raise ValueError('Qwen H=2048 required')
    z = bf(F(bf(o) + bf(residual)))
    # VECNORM: 256 active threads, eight consecutive elements each; 16 waves
    # in the 1024-thread tile, twelve with zero norm contribution.
    v = z.reshape(256, 8)
    ss = np.zeros(256, dtype=F)
    for i in range(8):
        ss = fma(v[:, i], v[:, i], ss)
    total = ordered_sum(np.concatenate([wave(ss.reshape(4, 64)), np.zeros(12, dtype=F)]))
    scale = F(1 / np.sqrt(F(F(total / F(2048)) + F(1e-6))))
    x = bf(F(F(z * scale) * bf(gamma)))
    return z, scale, x


def dot(w, x):
    """Narrow router: 4 x 512, 64 lanes x 8 elements, butterfly then wave order."""
    if w.shape != (128, 2048) or x.shape != (2048,):
        raise ValueError('Qwen router shape required')
    a = w.reshape(128, 4, 64, 8)
    b = x.reshape(4, 64, 8)
    s = np.zeros((128, 4, 64), dtype=F)
    for i in range(8):
        s = fma(a[..., i], b[..., i], s)
    return ordered_sum(wave(s))


def routing(logits):
    """Kept KEY32: rank bf16 logits, lower expert ID on finite ties; fp32 weights."""
    l = bf(logits)
    if l.shape != (128,):
        raise ValueError('E=128 required')
    ids = np.lexsort((np.arange(128), -l))[:8]
    p = np.exp(F(l - l[ids[0]]), dtype=F)
    total = wave(F(p[:64] + p[64:]))
    p = F(p * F(1 / total))
    weights = F(p[ids] / ordered_sum(p[ids]))
    return ids, weights


def deferred(w, z, gamma, scale, width=8, reverse=False):
    """Proposal with gamma included, 256 o_proj owners by default.

    Producer dot8 then fixed ascending producer sum. Reverse is a deliberate
    order sensitivity probe, never a permitted nondeterministic implementation.
    """
    if width <= 0 or 2048 % width:
        raise ValueError('slice width must divide 2048')
    a = w.reshape(128, -1, width)
    v = F(z * gamma).reshape(-1, width)
    s = np.zeros(a.shape[:-1], dtype=F)
    for i in range(width):
        s = fma(a[..., i], v[..., i], s)
    return bf(F(ordered_sum(s[:, ::-1] if reverse else s) * scale))


def compare(a, b):
    ai, aw = routing(a); bi, bw = routing(b)
    return dict(logit_bits_different=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
                ordered_ids_equal=bool(np.array_equal(ai, bi)),
                expert_set_equal=bool(set(ai) == set(bi)),
                weights_bits_equal=bool(np.array_equal(aw.view(np.uint32), bw.view(np.uint32))),
                max_logit_abs=float(np.max(np.abs(a-b))),
                max_slot_weight_abs=float(np.max(np.abs(aw-bw))),
                reference_ids=ai.tolist(), candidate_ids=bi.tolist())


def numerical(samples=64, seed=20261015):
    rng = np.random.default_rng(seed)
    rows = []
    for case in range(samples):
        o = bf(rng.normal(size=2048)); r = bf(rng.normal(size=2048))
        g = bf(rng.uniform(.5, 1.5, size=2048))
        w = bf(rng.normal(0, .03, size=(128, 2048)))
        z, s, x = norm(o, r, g)
        base = bf(dot(w, x))
        variants = {'deferred8': deferred(w,z,g,s),
                    'deferred64': deferred(w,z,g,s,64),
                    'deferred8_reverse': deferred(w,z,g,s,reverse=True),
                    'same_dot_scale_after': bf(F(dot(w,F(z*g))*s)),
                    'omit_gamma': deferred(w,z,np.ones(2048,dtype=F),s)}
        rows.append(dict(case=case, scale=float(s), variants={k:compare(base,v) for k,v in variants.items()}))
    # Analytic adversarial input: a huge residual component sets a non-unit
    # scale; two candidate experts straddle the eighth-place tie after bf16.
    # Search a deterministic *synthetic* sparse pair; store the exact bits.
    z = np.ones(2048, dtype=F); z[0] = F(3)
    zero = np.zeros(2048,dtype=F); g=np.ones(2048,dtype=F)
    z,s,x=norm(z,zero,g)
    witness=None
    for u in range(128, 512):
        w=np.zeros((128,2048),dtype=F)
        w[:7,2]=F(2)
        w[7,0]=bf(F(u/256)); w[8,1]=bf(F(3*u/256))
        base=bf(dot(w,x)); candidate=deferred(w,z,g,s)
        c=compare(base,candidate)
        if not c['expert_set_equal']:
            witness=dict(u=u, z0=3, other_z=1, gamma=1, scale=float(s),
                         w7_0=float(w[7,0]),w8_1=float(w[8,1]),
                         reference_logits=base.tolist(),candidate_logits=candidate.tolist(),comparison=c)
            break
    if witness is None:
        raise AssertionError('sparse counterexample not found')
    return dict(provenance='synthetic; no retained router inputs or weights available',
                seed=seed, samples=samples, cases=rows, counterexample=witness,
                summary={k:{'ordered_id_mismatches':sum(not r['variants'][k]['ordered_ids_equal'] for r in rows),
                            'set_mismatches':sum(not r['variants'][k]['expert_set_equal'] for r in rows),
                            'weight_mismatches':sum(not r['variants'][k]['weights_bits_equal'] for r in rows),
                            'logit_mismatches':sum(r['variants'][k]['logit_bits_different'] for r in rows)}
                         for k in rows[0]['variants']})


def trace_windows(raw, log):
    """Read original trace + its own printed type order; no missing-plan substitution."""
    data = np.frombuffer(raw, dtype='<i8')
    if data.size % 9:
        raise ValueError('truncated trace')
    rows = data.reshape(-1,9)
    names = re.findall(r'^  (\w+)\s+(\d+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$', log, re.M)
    if len(names) != len(np.unique(rows[:,0])):
        raise ValueError('type count mismatch')
    groups={}
    origin=rows[:,5].min()
    for ty,(name,count) in enumerate(names):
        rs=rows[rows[:,0]==ty]
        if len(rs) != int(count) or np.any(rs[:,5]>rs[:,6]) or np.any(rs[:,6]>rs[:,7]):
            raise ValueError('trace/log counts or timestamps mismatch')
        groups[name]=rs
    layers=[]
    for layer in range(48):
        o=groups[f'o_proj_{layer}']; r=groups[f'router_{layer}']; t=groups[f'topk_{layer}_x']; m=groups[f'moe_up_{layer}']
        end=o[:,7].max()
        layers.append(dict(layer=layer, o_done_relative_us=((o[:,7]-end)*.01).tolist(),
                           window_to_last_topk_us=float((t[:,7].max()-end)*.01),
                           window_to_first_moe_us=float((m[:,6].min()-end)*.01),
                           o_router_gap_us=float((r[:,6].min()-end)*.01),
                           router_envelope_us=float((r[:,7].max()-r[:,6].min())*.01),
                           router_topk_gap_us=float((t[:,6].min()-r[:,7].max())*.01),
                           topk_envelope_us=float((t[:,7].max()-t[:,6].min())*.01),
                           topk_moe_gap_us=float((m[:,6].min()-t[:,7].max())*.01)))
    return dict(provenance='historical 2026-10-11 tr_off; pre narrowing/hoist/KEY32; NOT current kept',
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                log_sha256=hashlib.sha256(log.encode()).hexdigest(),
                task_count=len(rows),type_count=len(names),span_us=float((rows[:,7].max()-origin)*.01),layers=layers)


def finish_times(done, partial, visibility, reduction, topk, norm_publish, fanout, norm_parallel=True):
    costs=np.asarray([partial,visibility,reduction,topk,norm_publish,fanout])
    if not np.all(np.isfinite(costs)) or np.any(costs<0):
        raise ValueError('nonnegative finite assumed costs required')
    # Scoring and residual/square production extend every o_proj task.
    last=max(done)+partial+visibility
    # Normalized expert input and routing both must arrive before moe_up.
    tail=max(reduction+topk,norm_publish) if norm_parallel else reduction+topk+norm_publish
    return last+tail+fanout


def costs(windows, timing_identity=None):
    if timing_identity is not None:
        transfer_gain(0, windows['raw_sha256'], timing_identity)
    scenarios=[]
    # Assumptions deliberately span optimistic and pessimistic cases. Not fit.
    for partial in (.25,1.,4.,8.):
        for reduction in (.5,2.,8.):
            for norm_publish in (1.,4.,8.):
                for visibility, fanout in ((.5,1.),(.9,1.8),(1.8,3.)):
                    for norm_parallel in (True,False):
                        tail=dict(partial=partial,visibility=visibility,reduction=reduction,
                                  topk=2.,norm_publish=norm_publish,fanout=fanout,norm_parallel=norm_parallel)
                        savings=[r['window_to_first_moe_us']-finish_times(r['o_done_relative_us'],**tail) for r in windows['layers']]
                        scenarios.append(dict(assumed_us_except_boolean=tail,historical_sum_saving_us=sum(savings),
                                              minimum_layer_saving_us=min(savings)))
    return dict(status='conditional historical schedule sensitivity ONLY; current gain UNKNOWN',
                router_matrix_bytes=128*2048*2, matrix_floor_us=128*2048*2/4.3e6,
                slices=256,partial_bytes=256*128*4,partial_write_plus_read_bytes=2*256*128*4,
                scores_fmas=128*2048,serial_reduction_adds=128*255,
                norm_square_partials_bytes=256*4, normalized_expert_input_bytes=2048*2,
                result_record_bytes_with_tei=8*12,
                scenarios=scenarios)


def transfer_gain(gain, trace_identity, timing_identity):
    """A guard against subtracting an older trace from a newer wall median."""
    if not trace_identity or trace_identity != timing_identity:
        raise ValueError('trace/timing identity mismatch: current gain unknown')
    return gain


def publication(seed, release=True, acquire=True, epoch_match=True):
    """Logical happens-before model, not an AMD memory-system simulation.

    Each producer owns a distinct partial record; counter increments alone
    cannot prove payload visibility. Readers fold by producer ID, never arrival.
    """
    order=np.random.default_rng(seed).permutation(256)
    visible=set()
    for producer in order:
        if release: visible.add(int(producer))
    if not epoch_match: raise ValueError('epoch mismatch')
    if not acquire or len(visible)!=256: raise ValueError('missing payload visibility')
    return dict(reads=len(visible),last_arriver=int(order[-1]),fold_order=list(range(256)))


def audit_routing(path):
    """Reuse original real routing outputs; does not test deferred scaling."""
    files=json.loads(gzip.decompress(path.read_bytes()))
    decoded={}
    for name, item in files.items():
        raw=bytes.fromhex(item['hex'])
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('routing original hash mismatch')
        decoded[name]=np.frombuffer(raw,dtype='<i4' if name.endswith('_ids.bin') else '<f4').reshape(8,8)
    replicas_bad=0; different_ids=0; different_weights=0
    for pos in (0,1040):
        for layer in range(48):
            for kind in ('ids','weights'):
                a=decoded[f'routing_off_p{pos}/L{layer}_{kind}.bin']
                b=decoded[f'routing_on_p{pos}/L{layer}_{kind}.bin']
                replicas_bad+=int(not np.array_equal(a,np.tile(a[0],(8,1))))
                replicas_bad+=int(not np.array_equal(b,np.tile(b[0],(8,1))))
                if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
                    if kind=='ids': different_ids+=1
                    else: different_weights+=1
    return dict(status='real retained KEY32 off/on outputs; NOT a candidate correctness check',
                files=len(files),layer_position_cases=96,replica_mismatches=replicas_bad,
                id_cases_different=different_ids,weight_cases_different=different_weights)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--inputs',type=Path,default=DEFAULT)
    p.add_argument('--samples',type=int,default=64)
    a=p.parse_args()
    if a.samples<1: p.error('samples must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    n=numerical(a.samples)
    w=trace_windows(gzip.decompress((a.inputs/'tr_off.bin.gz').read_bytes()),(a.inputs/'tr_off.log').read_text())
    c=costs(w)
    audit=audit_routing(a.inputs/'routing-originals.json.gz')
    for name,value in [('numerical',n),('windows',w),('costs',c),('routing-audit',audit)]:
        (a.out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print('CPU ONLY; synthetic numerical inputs; historical trace sensitivity',flush=True)
    print(json.dumps(n['summary'],indent=2),flush=True)
    print('counterexample:',json.dumps(n['counterexample']),flush=True)
    print('retained routing:', json.dumps(audit),flush=True)
    print('trace:',w['task_count'],w['type_count'],w['span_us'],w['raw_sha256'],flush=True)
    gains=[x['historical_sum_saving_us'] for x in c['scenarios']]
    print('ASSUMED sweep historical sum saving us:',min(gains),max(gains),'current gain UNKNOWN',flush=True)

if __name__=='__main__': main()
