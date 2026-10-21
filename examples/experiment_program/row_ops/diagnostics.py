"""Sparse phase selectors, separate matched control/empty/probe native paths."""
import json,struct,statistics
from pathlib import Path
from .source import translation_unit
from .results import RECORD
from .export_contract import intermediate_extent
from examples.experiment_program.protocol import require,resolve_path
PHASES=('setup','clear','pipeline','flush','join','gather','silu','publication')

def unit(variant,mode,preamble):
    require(variant in ('g09','g10_up','g10_down') and mode in ('control','probe','empty'),'registered sparse row diagnostic')
    s=translation_unit(variant,preamble)
    a=s.index('template <int EPL, int U, int NB, class Epi>')
    s=s[:a]+'static __device__ __forceinline__ void row_mark(int point);\n'+s[a:]
    # Instrument original/guarded clones at retained instructions. The single-owner
    # deletion has no clear/gather interval; adjacent boundary stamps explicitly
    # encode that absence instead of inventing removed work or adding barriers.
    s=s.replace('  __syncthreads(); row_mark(0);','  __syncthreads(); row_mark(0); row_mark(7); // setup/activation staging begins')
    clear='  for (int i = (int)threadIdx.x; i < nw * nrt * NB; i += blockDim.x) pl[i] = 0.f;'
    s=s.replace(clear,'  row_mark(8); // original partial clear begins\n'+clear)
    s=s.replace('  int cur = sb / spr;','  row_mark(9); // pipeline/retained-operand setup begins\n  int cur = sb / spr;')
    for name in ('row_single_up','row_single_down'):
        if name not in s:continue
        start=s.index('void '+name+'(');end=s.index('#undef DSL_WP',start)
        body=s[start:end].replace('  row_mark(9);','  row_mark(8); // deleted clear: adjacent boundaries\n  row_mark(9);')
        s=s[:start]+body+s[end:]
    s=s.replace('  if (sb < se) flush();','  row_mark(10); // final wave flush begins\n  if (sb < se) flush();\n  row_mark(11); // final flush complete, join begins')
    s=s.replace('  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {','  row_mark(12); // original join complete, gather begins\n  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {')
    s=s.replace('  __syncthreads(); // all gate/up results visible before SiLU; arena reuse safe','  __syncthreads(); // all gate/up results visible before SiLU; arena reuse safe\n  row_mark(12); // no scratch gather')
    s=s.replace('  __syncthreads(); // owner stores visible, arena reuse safe','  __syncthreads(); // owner stores visible, arena reuse safe\n  row_mark(12); // no scratch gather')
    s=s.replace('  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r,','  row_mark(13); // gather complete, SiLU epilogue begins\n  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r,')
    s=s.replace('  ETX_RELEASE_DOMAIN(); __syncthreads(); row_mark(6);','  row_mark(14); // epilogue complete, publication drain begins\n  ETX_RELEASE_DOMAIN(); __syncthreads(); row_mark(15); row_mark(6);')
    old='#if ROW_PROBE_KIND == 1\n'
    extra='''#if ROW_PROBE_KIND == 3 || ROW_PROBE_KIND == 4
  if(etx_lane()==0){
    const int population=gridDim.x*16;
    const bool selected=(int)(blockIdx.x*16+etx_wave())==row_selection%population;
    if(point==0 || point==5 || point==6) row_probe[etx_wave()][point]=__builtin_amdgcn_s_memrealtime();
    const int phase=row_selection/population;
    if(selected && (point==7+phase || point==8+phase)) {
      const unsigned long long t=__builtin_amdgcn_s_memrealtime();
#if ROW_PROBE_KIND == 4
      if(point==7+phase){row_probe[etx_wave()][1]=t;row_probe[etx_wave()][2]=__builtin_amdgcn_s_memrealtime();}
#else
      row_probe[etx_wave()][point==7+phase?1:2]=t;
#endif
    }
  }
#elif ROW_PROBE_KIND == 1
'''
    require(s.count(old)==1,'sparse phase hook drift');s=s.replace(old,extra)
    # Down has no SiLU: expose its output epilogue rather than inventing SiLU work.
    if variant=='g10_down':s=s.replace('  row_mark(5); // output_ready','  row_mark(13); // down has no SiLU; output-ready boundary\n  row_mark(5); // output_ready')
    s=s.replace('#define ROW_PROBE_KIND 0',f'#define ROW_PROBE_KIND '+str({'control':0,'probe':3,'empty':4}[mode]),1)
    s=s.replace('r.pad=0;', 'asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(r.pad));')
    return s

def summarize(prefix,pack,identity,variant):
    prefix=str(prefix);items=[json.loads(l) for l in resolve_path(prefix+'.timing.jsonl').read_text().splitlines()];population=pack['count']*16*len(PHASES)
    import math
    require(all(type(r.get(k)) in (int,float) and math.isfinite(r[k]) and r[k]>0 for r in items for k in ('device_us','host_wall_us')),'invalid diagnostic timing endpoints')
    indexed={(r['sample'],r['arm']):r for r in items}
    require(len(indexed)==len(items)==population*3 and set(indexed)=={(s,a) for s in range(population) for a in range(3)},'complete sparse selector population required')
    from .protocol import invocation_orders
    require([(r['sample'],r['arm']) for r in items]==[(s,a) for s,order in enumerate(invocation_orders(3,population)) for a in order],'diagnostic raw order differs from declared native replay')
    selected=[];reasons=[]
    for sample in range(population):
        rows=list(RECORD.iter_unpack(resolve_path(prefix+f'.s{sample}.a1.records').read_bytes()))
        require(len(rows)==pack['count']*16,'complete last task/wave proof required')
        for i,row in enumerate(rows):require(row[7:9]==(i//16,i%16) and 0<row[0]<=row[5]<=row[6] and row[9]<8,'sparse body task/wave coverage')
        target=sample%(pack['count']*16);row=rows[target];phase=sample//(pack['count']*16)
        require(row[0]<=row[1]<=row[2]<=row[6] and row[1]>0,'selected sparse phase absent')
        last=max(rows,key=lambda r:r[6]);selected.append(dict(task=row[7],wave=row[8],phase=PHASES[phase],begin=row[1],end=row[2],interval_ticks=row[2]-row[1],last_task_wave=[last[7],last[8]],body_ticks=[r[6]-r[0] for r in rows],native_placement=[dict(task=r[7],wave=r[8],xcd=r[9],hwid=r[10]) for r in rows]))
        for arm in range(3):
            require(indexed[sample,arm]['exact'],'diagnostic byte mismatch')
            for r,ref in enumerate(pack['references']):require(resolve_path(prefix+f'.s{sample}.a{arm}.r{r}.bin').read_bytes()==resolve_path(ref['file']).read_bytes(),'diagnostic raw output differs')
    numerical=[resolve_path(prefix+f'.a{arm}.intermediates').read_bytes() for arm in range(3)]
    require(all(len(x)==intermediate_extent(pack) for x in numerical) and numerical[0]==numerical[1]==numerical[2],'diagnostic intermediate mismatch')
    symbol='row_dsl_moe_down' if variant=='g10_down' else 'row_dsl_moe_up'
    resources=[next(r for r in identity['resources'][variant+'_'+mode] if r['kernel']==symbol) for mode in ('control','probe','empty')]
    from .elf import descriptors,allocation_signature
    signatures=[allocation_signature(descriptors(resolve_path(identity.get('_directory','/missing'))/(variant+'_'+mode+'.hsaco'))[symbol],r['agpr']) for mode,r in zip(('control','probe','empty'),resources)] if identity.get('_directory') else None
    keys=('vgpr','agpr','lds_bytes','scratch_bytes_lane','vgpr_spills','sgpr_spills','static_blocks_per_cu')
    if signatures is None or not signatures[0]==signatures[1]==signatures[2]:reasons.append('actual native resource allocations differ')
    delta=max(abs(indexed[s,a]['device_us']-indexed[s,0]['device_us']) for s in range(population) for a in (1,2))
    if delta>.5:reasons.append('perturbation exceeds500ns')
    return dict(status='QUALIFIED_SPARSE_DIAGNOSTIC' if not reasons else 'UNQUALIFIED_DIAGNOSTIC',qualified=not reasons,reasons=reasons,observations=selected,resources=resources,max_perturbation_us=delta,performance_claim=False)
