"""CPU vendor inventory and fail-closed isolated attention+reduce pricing.

No GPU work on import or --inventory. Kernels run only via the session's
already-staged, freshly certified builds; unavailable adapters remain [U].
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

BAR_US = 7.69
# Shape eligibility is distinct from a runnable exact import contract.
CANDIDATES = [
    ('ROCM_ATTN_G8', 'legal control', 'PART=256, block=16, HD=128, GQA=8, MFMA16; existing importer', 10.645),
    ('ROCM_ATTN_PART128', 'excluded', 'imported body hard-codes 256; extra early-exit tasks are not a new specialization', None),
    ('ROCM_ATTN_PART512', 'excluded', 'host PART change does not change kernel 256 stride; misses context tokens', None),
    ('ROCM_ATTN_G1_7_G9_16_HD64', 'excluded', 'changes GQA/head-size specialization; actual NH/NKV=8 and HD=128', None),
    ('ROCM_ATTN_BLOCK32', '[U] adapter required', 'shape supported; existing block-16 K/V writer, strides and block table must change together', None),
    ('PAGED_V1', '[U] adapter required', 'bf16 generic paged attention; different export/grid, no existing compatible output/partial contract', None),
    ('PAGED_V2', '[U] adapter required', 'generic partitioned decode; partition/reduce numerics and adapter need independent checks', None),
    ('TRITON_ATTN_2D', '[U] adapter required', 'num_par_softmax_segments=None/2D; flash K/V layout, grid/softmax order differ', None),
    ('TRITON_ATTN_3D', '[U] adapter required', 'num_par_softmax_segments, TILE_SIZE, BLOCK_M, warps/stages; not arbitrary external backend knobs', None),
    ('AITER_UNIFIED', '[U] adapter required', 'retained 2-kernel result slower; segments/tile/warps/stages and shuffled cache require source-pinned export/layout', 11.42030309139785),
    ('AITER_PA_COMMON', '[U] adapter required', 'paged_attention_common wrapper; HIP/ASM internal dispatch implementation not installed here', None),
    ('AITER_PA_ASM', '[U] adapter required', 'pa_fwd_asm wrapper; Q/K/V scales, layout and implementation legality unverified', None),
    ('AITER_FA', '[U] adapter required', 'flash_attn_varlen_func bf16 family; dense/flash cache and metadata differ; no ready import', None),
    ('MLA_SPARSE', 'excluded', 'Qwen uses ordinary GQA, not MLA or sparse attention', None),
    ('CUDA_XPU_RDNA', 'excluded', 'CUDA-only, XPU or Navi/WMMA kernels do not target gfx942 CDNA', None),
    ('QUANTIZED_ATTENTION', 'excluded', 'FP8/INT4/TurboQuant changes bf16 KV/output algorithm contract', None),
    ('SINK_ALIBI_WINDOW_DCP', 'excluded', 'new sinks, ALiBi, sliding window or distributed context changes this workload', None),
]
FILES = ('vllm/platforms/rocm.py', 'vllm/v1/attention/backends/rocm_attn.py',
         'vllm/v1/attention/backends/rocm_aiter_unified_attn.py',
         'vllm/v1/attention/backends/rocm_aiter_fa.py', 'vllm/_aiter_ops.py',
         'vllm/v1/attention/ops/triton_unified_attention.py',
         'csrc/rocm/attention.cu', 'csrc/attention/paged_attention_v1.cu',
         'csrc/attention/paged_attention_v2.cu')


def inventory(vllm, aiter=None, retained=None):
    sources = {}
    for name in FILES:
        path = Path(vllm)/name
        if not path.is_file(): sources[name] = dict(status='[U] missing source'); continue
        data = path.read_bytes(); lines = data.decode().splitlines()
        knobs = [dict(line=i+1, text=line.strip()) for i,line in enumerate(lines)
                 if re.search(r'num_par_softmax_segments|TILE_SIZE.*=|launch_num_(warps|stages).*=|PARTITION_SIZE.*=|T_PAR_SIZE.*=|case (16|32):', line)]
        sources[name] = dict(sha256=hashlib.sha256(data).hexdigest(), knobs=knobs)
    aiter_sources = {}
    if aiter and Path(aiter).is_dir():
        for path in sorted(Path(aiter).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.cu', '.cuh', '.cpp', '.h') and re.search('attn|attention|pa_decode', str(path)):
                aiter_sources[str(path.relative_to(aiter))] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(['git','-C',str(vllm),'rev-parse','HEAD'], text=True).strip()
    except subprocess.SubprocessError: revision = '[U] not a git tree'
    measured = None
    if retained:
        data=json.loads(Path(retained).read_text()); rows=data['layers']
        if len(rows)!=48 or {r['layer'] for r in rows}!=set(range(48)): raise ValueError('retained X5 incomplete layers')
        measured=dict(attention_us=statistics.mean(r['attention'] for r in rows),
                      reduce_us=statistics.mean(r['reduce'] for r in rows),
                      pair_us=statistics.mean(r['attention']+r['reduce'] for r in rows),
                      source=str(retained), samples_per_layer=len(data['samples'][0]['samples_us']))
    return dict(condition=dict(precision='bf16',batch=1,context=1024,NH=32,NKV=4,HD=128),
        vllm_revision=revision, sources=sources, aiter_sources=aiter_sources,
        complete=False, completeness='[U] pinned image AITER dispatcher/compiled exports and exact adapters not certified; no exhaustive closure',
        candidates=[dict(id=i,status=s,reason=r,preprice_us=p,preprice_basis='[E] critical-body or cross-basis retained estimate' if p is not None else '[U]') for i,s,r,p in CANDIDATES],
        retained=measured, survivor_builds=['kept'])


def prices(text):
    rows=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('X5-SAMPLE ')]
    expected={(layer,role,sample) for layer in range(48) for role in ('attn','reduce') for sample in range(3)}
    if len(rows)!=len(expected) or {(r['layer'],r['role'],r['sample']) for r in rows}!=expected:
        raise ValueError('X5 NOT-COMPLETE isolated sample population')
    if any(r['position']!=1024 or r['replays']!=100 or r['bytes_equal'] is not True or
           any(not math.isfinite(r[k]) or r[k]<=0 for k in ('original_us','imported_us')) for r in rows):
        raise ValueError('X5 invalid condition/correctness/timing')
    per_layer=[]
    for layer in range(48):
        pair=sum(statistics.median(r['imported_us'] for r in rows if r['layer']==layer and r['role']==role)
                 for role in ('attn','reduce'))
        per_layer.append(pair)
    value=statistics.mean(per_layer)
    return dict(samples=rows, per_layer_us=per_layer, mean_us_layer=value, pass_bar=value<=BAR_US,
                R2_ms=max(0.,48*(10.645-value)/1000) if value<=BAR_US else 0.,
                basis='[E] isolated imported whole-grid event time vs critical body; candidate must be measured in full model before KEEP')


def run(s):
    from examples.sessions import qwen300_hour as h
    # No speculative adapter is built during paid time. Candidates not present
    # at preflight remain unknown, not failed performance measurements.
    builds=s.c.get('x5_builds', ['kept'])
    if not builds or len(set(builds))!=len(builds): raise ValueError('X5 missing/duplicate survivors')
    results={}
    for arm in builds:
        h.verify_seal(s,arm); h.certify(s,arm)
        end=s.deadline; s.deadline=min(end,h.time.monotonic()+30)
        try:
            text=h.host(s,arm,'X5_'+arm,['--check','48','--check-pos','1024','--price-attention'])
            from examples.vllm_llm import import_certificate as cert
            # Parse the fresh position-1024 check as well as requiring its
            # immutable build stamps. Full 0/1040 + token gates preceded it.
            build = Path(s.c['builds'][arm])
            ident = cert.verified_build(build,Path(s.c['model']))
            expected = cert.expected_instances((build/'vl_plan.txt').read_text(),
                                               json.loads((build/'imp/imports.json').read_text()))
            cert.check_output(text,expected,1024,ident)
            if 'CHECK OK:' not in text: raise ValueError('X5 import check incomplete')
            results[arm]=prices(text)
        finally: s.deadline=end
    best=max((r['R2_ms'] for r in results.values()),default=0.)
    route=s.rows.get('A2.route',{}).get('value')
    if route is not None: route['attainable_cut_ms']=best
    # This tree cannot certify missing AITER implementations or adapters.
    # Do not accept a config boolean as an exhaustive legality certificate.
    complete=False
    status='SURVIVOR' if best else 'CLOSED' if complete else 'UNRESOLVED'
    value=dict(results=results, R2_ms=best, inventory_complete=complete,
               B1='0: no candidate passes' if status=='CLOSED' else '[U]: incomplete legality inventory' if status=='UNRESOLVED' else '[E] cross-basis opportunity')
    s.row('X5',value,status,h.REGISTERED['X5'])
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--aiter',type=Path);p.add_argument('--retained',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=inventory(a.inventory,a.aiter,a.retained)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=result['complete'],vllm_revision=result['vllm_revision'],
                         retained=result['retained'],survivor_builds=result['survivor_builds'])))


if __name__=='__main__': main()
