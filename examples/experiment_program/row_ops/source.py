"""Isolated exact source interventions; production emitter is never mutated."""
from etx.ops import gemv
from examples.vllm_llm.bf16_300.source import current_source, specialised_source, replace_one

VARIANTS=('control','g11_control','g09','g10_up','g10_down','g11_probe','g11_empty')


def mapping(k,epl):
    spr=k//(64*epl); per=(64*spr+15)//16
    return [[(s//spr,s%spr) for s in range(w*per,min(64*spr,(w+1)*per))] for w in range(16)]


def retained_addresses(lane=3):
    old=[(row,s,i) for row in range(4) for s in range(4) for i in range(8)]
    return old,[s*512+lane*8+i for s in range(4) for i in range(8)]


def helper(s,name,per,epl,scratch=True,retain=False):
    a=s.index('template <int EPL, int U, int NB, class Epi>')
    b=s.index('\n}',s.index('#undef DSL_WP',a))+2
    f=s[a:b].replace('dsl_gemv_rows_nb',name)
    f=replace_one(f,'  typedef typename dsl_wvec<EPL>::T V;',f'  static_assert(EPL == {epl} && NB == 1, "fixed whole-row mapping");\n  typedef typename dsl_wvec<EPL>::T V;')
    f=replace_one(f,'  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);',f'  const int per = {per}, sb = wv * per, se = sb + per;')
    if retain:
        f=replace_one(f,'  int cur = sb / spr;','''  float retained[4][8]; // exact fp32 staged operands; never bf16 repacked
#pragma unroll
  for (int segment=0; segment<4; ++segment)
#pragma unroll
    for (int i=0; i<8; ++i) retained[segment][i] = xl[segment*512 + lane*8 + i];
  int cur = sb / spr;''')
        f=f.replace('dsl_dot(w, xl + b * KS + off)','dsl_dot_retained(w, retained[seg])')
        # SCALAR is required by the fixed recipe. Keep the generic branch valid.
        f=f.replace('retained[seg]);   //','retained[sg - row * spr]);   //')
    if not scratch:
        f=replace_one(f,'  for (int i = (int)threadIdx.x; i < nw * nrt * NB; i += blockDim.x) pl[i] = 0.f;\n  __syncthreads();','')
        f=f.replace('if (lane == 0) pl[(wv * nrt + cur) * NB + b] += v;',
          'if (lane == 0) { float z=0.f; for (int w2=0; w2<16; ++w2) z += w2 == wv ? (0.f + v) : 0.f; epi(r0 + cur, &z); }')
        start=f.index('#ifdef DSL_ABLATE_NORED\n#pragma unroll\n  for (int b')
        end=f.index('#undef DSL_WP',start)
        f=f[:start]+('  __syncthreads(); // all gate/up results visible before SiLU; arena reuse safe\n' if epl==8 else '  __syncthreads(); // owner stores visible, arena reuse safe\n')+f[end:]
    return s[:b]+f+s[b:]


DOT=r'''
static __device__ __forceinline__ float dsl_dot_retained(dsl_u32x4 w, const float* x) {
  float s=0.f;
#pragma unroll
  for (int i=0; i<4; ++i) { s += __uint_as_float(w[i] << 16) * x[2*i]; s += __uint_as_float(w[i] & 0xffff0000u) * x[2*i+1]; }
  return s;
}
'''

PROBE=r'''
struct RowRecord { unsigned long long stamp[7]; unsigned task,wave,xcd,pad; };
__shared__ RowRecord* row_record_ptr;
__shared__ int row_selection;
// Separate LDS storage. Never aliases xs, partials, red, or a consumer's norm scratch.
__shared__ unsigned long long row_probe[16][7];
static __device__ __forceinline__ void row_mark(int point) {
#if ROW_PROBE_KIND == 1
  if (etx_lane()==0 && (point==0 || point>=5 || (int)(blockIdx.x*16+etx_wave())==row_selection)) { unsigned long long t; asm volatile("s_memrealtime %0" : "=s"(t) :: "memory"); row_probe[etx_wave()][point]=t; }
#endif
}
'''


def probe(s):
    # Inject only the retained current norm body, with original read/fold/norm DAG.
    a=s.index('template <int FixedK = 0, int FixedR = -1>'); b=s.index('static __device__ __forceinline__ void dsl_stage_norm(',a)
    f=s[a:b]
    f=replace_one(f,'  float ss = 0.f;','  row_mark(1); // fold_begin: issue, not retired\n  float ss = 0.f;')
    f=f.replace('    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);',
                '    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);\n    row_mark(2); // residual_load_issued; active vector waves only',1)
    # This anchor is after all ordered slot additions/rounding, before normalization.
    f=replace_one(f,'#ifdef ETX_DSL_NORM_HOIST','  row_mark(3); // fold_end / norm_begin\n#ifdef ETX_DSL_NORM_HOIST') if f.count('#ifdef ETX_DSL_NORM_HOIST')==1 else f.replace('#ifdef ETX_DSL_NORM_HOIST','  row_mark(3); // fold_end / norm_begin\n#ifdef ETX_DSL_NORM_HOIST',1)
    f=replace_one(f,'  __syncthreads();\n}','  __syncthreads();\n  row_mark(4); // norm_ready, after original norm barrier\n}')
    return s[:a]+f+s[b:]


def source(variant, with_probes=True, base=None):
    if variant not in VARIANTS: raise ValueError(variant)
    s=current_source() if base is None else base
    if variant=='g09':
        p=s.index('template <int EPL, int U, int NB, class Epi>'); s=s[:p]+DOT+s[p:]
        s=helper(s,'row_retained_up',16,8,retain=True)
        old='if (d.K % (ETX_WAVE_SIZE * 8) == 0 && 2 * gn <= DSL_RCH && gp1 % gspr == 0 && gp2 % gspr == 0 && (gn * gspr) % gp2 == 0) {'
        new='''if (d.K == 2048 && gn == 32 && d.rpt == 32 && gnw == 16 && blockDim.x == 1024) {
    row_retained_up<8, DSL_UW, 1>(We,d.K,0,xs,r0,r0+2*gn,part,[&](int r,const float* v) {
      const int q=r-r0; g[q<gn?q:d.rpt+q-gn]=dsl_bfr(v[0]); },gn,(long long)(d.M-gn)*d.K);
  } else '''+old
        if s.count(old)!=2: raise ValueError('up guard source drift')
        s=s.replace(old,new)
    elif variant in ('g10_up','g10_down'):
        s=helper(s,'row_single_up',16,8,scratch=False)
        old='if (d.K % (ETX_WAVE_SIZE * 8) == 0 && 2 * gn <= DSL_RCH && gp1 % gspr == 0 && gp2 % gspr == 0 && (gn * gspr) % gp2 == 0) {'
        new='''if (d.K == 2048 && gn == 32 && d.rpt == 32 && gnw == 16 && blockDim.x == 1024) {
    row_single_up<8, DSL_UW, 1>(We,d.K,0,xs,r0,r0+2*gn,part,[&](int r,const float* v) {
      const int q=r-r0; g[q<gn?q:d.rpt+q-gn]=dsl_bfr(v[0]); },gn,(long long)(d.M-gn)*d.K);
  } else '''+old
        if s.count(old)!=2:raise ValueError('up source drift')
        s=s.replace(old,new)
        if variant=='g10_down':
            s=helper(s,'row_single_down',12,4,scratch=False)
            old='  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v * ws)); });'
            new='''  if (d.K == 768 && r1 - r0 == 64 && d.rpt == 64 && etx_nwaves() == 16 && blockDim.x == 1024) {
    row_single_down<4, DSL_UN, 1>(We,d.K,0,xs,r0,r1,xs+d.K,[&](int r,const float* v) { dsl_a_stbf(out+r,__float2bfloat16(v[0] * ws)); });
  } else '''+old.lstrip()
            if s.count(old)!=2: raise ValueError('down source drift')
            s=s.replace(old,new)
    if not with_probes:return s
    # All variants reserve the SAME named probe allocation; native allocation is
    # still measured and never assumed equal. Only G11 enables issue stamps.
    idx=s.index('template <int FixedK = 0, int FixedR = -1>')
    s=s[:idx]+PROBE+s[idx:]
    s=probe(s)
    pre=f'#define ROW_PROBE_KIND {1 if variant=="g11_probe" else 2 if variant=="g11_empty" else 0}\n'
    return pre+s


def translation_unit(variant,preamble):
    s=preamble+source(variant)
    for export in ('dsl_moe_up','dsl_moe_down','dsl_gemv_fold'):
      for checking in (False,True):
        suffix='_check' if checking else ''
        check_code='''  const DslArgs cd=dsl_args(a);
  float* check_out=(float*)row_record + blockIdx.x*4096;
  const float* arena=(const float*)etx_lds_arena;
'''
        if export=='dsl_moe_up':
            check_code+='  for(int i=threadIdx.x;i<2*cd.rpt;i+=blockDim.x) check_out[i]=arena[cd.K+i]; // separately-rounded gate/up fp32\n'
        elif export=='dsl_gemv_fold':
            check_code+='  for(int i=threadIdx.x;i<cd.K;i+=blockDim.x) check_out[i]=arena[i]; // current consumer own norm staging only\n'
        else:check_code=''
        s+=f'''\nextern "C" __global__ __launch_bounds__(1024,1) void row_{export}{suffix}(const unsigned char* a,int first,RowRecord* row_record,int selection) {{
  etx_import_init();
#if ROW_PROBE_KIND != 0
  if (threadIdx.x==0) {{ row_record_ptr=row_record + blockIdx.x * 16; row_selection=selection; }}
  for (int i=threadIdx.x; i<16*7; i+=blockDim.x) (&row_probe[0][0])[i]=0;
  __syncthreads(); row_mark(0);
#endif
  dsl_run_{export}(a,first+blockIdx.x);
  row_mark(5); // output_ready issue, before original publication drain
  ETX_RELEASE_DOMAIN(); __syncthreads(); row_mark(6);
#if ROW_PROBE_KIND == 2
  // Empty qualification: same allocation and stores, two adjacent issue stamps.
  if (etx_lane()==0) {{ unsigned long long t; asm volatile("s_memrealtime %0" : "=s"(t) :: "memory"); row_probe[etx_wave()][0]=t; row_probe[etx_wave()][5]=t; row_probe[etx_wave()][6]=t; if ((int)(blockIdx.x*16+etx_wave())==row_selection) for(int j=1;j<5;++j) row_probe[etx_wave()][j]=t; }}
#endif
#if ROW_PROBE_KIND != 0
  if ({"false" if checking else "true"} && etx_lane()==0) {{ RowRecord r; for(int j=0;j<7;++j) r.stamp[j]=row_probe[etx_wave()][j]; r.task=first+blockIdx.x; r.wave=etx_wave(); r.xcd=ETX_DOMAIN_ID(); r.pad=0; row_record_ptr[etx_wave()]=r; }}
#endif
  __syncthreads();
  {check_code if checking else ""}
}}\n'''
    return s
