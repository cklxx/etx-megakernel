"""Head-sharded attention contracts; authored without running tests or compilers.

When explicitly run, the CPU checks use real compact descriptors (including
noncontiguous CONV rows and O column slices), poison nonowned head regions, and
compare bounded heads/state against the original phased attention for 7 steps.
This is not a GPU, production-loader, or full-model qualification. In particular,
O's sharded FP32 reduction is checked against the specified rank-order arithmetic,
not claimed bit-identical to the unsharded dot when world > 1.
"""
import ctypes
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'examples/glm53/attention_phases.h'


def test_attention_helper_public_interfaces_and_default_head_bounds():
    source = HEADER.read_text()
    arguments = {
        'input': ('const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,'
                  'float* x,float* lds'),
        'p1': ('const uint8_t* b,const glm_weight* w,const float* x,float* p1,bool linear,'
               'int rank,int world,int t,int T,float* lds'),
        'p2': ('const uint8_t* b,const glm_weight* w,const float* p1,float* p2,float* state,'
               'int pos,int rank,int world,int t,int T,float* lds'),
        'heads': ('const uint8_t* b,const glm_weight* w,const float* p1,const float* p2,'
                  'float* state,float* att,int pos,bool linear,int rank,int world,int t,int T,float* lds'),
        'o': ('const uint8_t* b,const glm_weight* w,const float* att,float* partial,bool linear,'
              'int rank,int world,int t,int T,float* lds'),
        'combine': ('const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,'
                    'const float* const* partials,float* mid,int world,int t,int T,float* lds'),
    }
    assert '#include "examples/glm53/phases.h"' in source
    for name, want in arguments.items():
        match = re.search(r'GLM_DEV void glm_ph_attention_' + name + r'\((.*?)\)\{', source, re.S)
        assert match, name
        assert re.sub(r'\s+', '', match.group(1)) == re.sub(r'\s+', '', want), name
    phases = (ROOT / 'examples/glm53/phases.h').read_text()
    for name, tail in [('mla', 'int head_begin=0,int head_end=GLM_NH'),
                       ('kda', 'int head_begin=0,int head_end=GLM_LH,bool sharded=false')]:
        match = re.search(r'GLM_DEV void glm_ph_' + name + r'\((.*?)\)\{', phases, re.S)
        assert match
        assert re.sub(r'\s+', '', match.group(1)).endswith(re.sub(r'\s+', '', tail))


CPU_SOURCE = r'''
#define GLM53_CPU 1
#include <string.h>
#include <initializer_list>
#include "examples/glm53/attention_phases.h"

struct floats {
  float* p; int n;
  floats(int n,float value=NAN):p(new float[n]),n(n){for(int i=0;i<n;++i)p[i]=value;}
  ~floats(){delete[] p;}
};
bool same(float a,float b){return isfinite(a)&&isfinite(b)&&a==b;}
bool head_owned(int h,int begin,int end){return h>=begin&&h<end;}
bool state_owned(int i,bool linear,int begin,int end){
  if(linear){
    const int conv=3*GLM_QDIM*GLM_CONV;
    int h=i<conv?(i/GLM_CONV%GLM_QDIM)/GLM_LD:(i-conv)/(GLM_LD*GLM_LD);
    return head_owned(h,begin,end);
  }
  const int j=i%(2*GLM_NH*GLM_AD+2*GLM_ID);
  return j>=2*GLM_NH*GLM_AD||head_owned(j/(2*GLM_AD),begin,end);
}

// Physically compact each selected descriptor, retaining kind and raw payload.
// FP8 scales are expanded to block=1 to preserve arbitrary sub-block slices
// exactly; this tests helper indexing, not production block-scale relocation.
struct local_weights {
  uint8_t* b; glm_weight w[EXPERTS+3*GLM_EXPERTS]={}; uint64_t used=0,capacity;
  const uint8_t* original;const glm_weight* global;
  local_weights(const uint8_t* b,const glm_weight* w,int size):
    b(new uint8_t[(uint64_t)size*8+4096]),capacity((uint64_t)size*8+4096),original(b),global(w){}
  ~local_weights(){delete[] b;}
  uint64_t allocate(uint64_t n){uint64_t off=(used+15)/16*16;used=off+n;return off;}
  void copy(int slot,int r0=0,int nr=-1,int c0=0,int nc=-1,bool conv=false){
    const glm_weight& src=global[slot];if(nr<0)nr=src.rows;if(nc<0)nc=src.cols;
    glm_weight dst=src;dst.rows=nr;dst.cols=nc;
    const int es=glm_esize(src);dst.offset=allocate((uint64_t)nr*nc*es);
    if(src.kind==3){dst.block=1;dst.scale=allocate((uint64_t)nr*nc*sizeof(float));}
    if(used>capacity)return; // Caller reports failure before executing a helper.
    for(int r=0;r<nr;++r)for(int c=0;c<nc;++c){
      const int rr=conv?(r/(nr/3))*GLM_QDIM+r0+r%(nr/3):r0+r,cc=c0+c;
      memcpy(b+dst.offset+((uint64_t)r*nc+c)*es,original+src.offset+((uint64_t)rr*src.cols+cc)*es,es);
      if(src.kind==3){
        int scols=(src.cols+src.block-1)/src.block;
        ((float*)(b+dst.scale))[(uint64_t)r*nc+c]=((const float*)(original+src.scale))[(rr/src.block)*scols+cc/src.block];
      }
    }
    w[slot]=dst;
  }
};

extern "C" int attention_check(const void* bp,const void* wp,int blob_size,int linear,int world,int T){
  const uint8_t* b=(const uint8_t*)bp;const glm_weight* w=(const glm_weight*)wp;
  const int heads=linear?GLM_LH:GLM_NH,dim=linear?GLM_LD:GLM_AD;
  if(world<1||world>8||heads%world||T<=0)return 1;
  const int stride=2*GLM_NH*GLM_AD+2*GLM_ID;
  const int nstate=linear?3*GLM_QDIM*GLM_CONV+GLM_LH*GLM_LD*GLM_LD:GLM_MAXLEN*stride;
  floats ref_state(nstate,0),bounded_state(nstate,0),lds(GLM_LDS_FLOATS);
  local_weights* local[8]={};floats* state[8]={};
  // RAII is not needed for the success path only: avoid leaks on assertion codes.
  struct cleanup {
    local_weights** weights;floats** states;int n;
    ~cleanup(){for(int r=0;r<n;++r){delete weights[r];delete states[r];}}
  } clean{local,state,world};
  for(int rank=0;rank<world;++rank){
    const int begin=rank*(heads/world),end=(rank+1)*(heads/world),off=begin*dim,n=(end-begin)*dim;
    local[rank]=new local_weights(b,w,blob_size);auto& m=*local[rank];state[rank]=new floats(nstate);
    for(int i=0;i<nstate;++i)if(state_owned(i,linear,begin,end))state[rank]->p[i]=0;
    if(linear){
      for(int s:{Q,K,V,FB,GB,DT})m.copy(s,off,n);
      for(int s:{BETA,ALOG})m.copy(s,begin,end-begin);
      m.copy(CONV,off,3*n,0,-1,true);
      for(int s:{FA,GA,ONORM})m.copy(s);
    } else {
      for(int s:{QA,KVA,IK,IG,IW,QAN,KVAN,IKN,IKB,IAPE,IQ})m.copy(s);
      m.copy(QB,off,n);m.copy(KVB,2*off,2*n);
    }
    m.copy(O,0,-1,off,n);
    if(m.used>m.capacity)return 2;
  }
  for(int pos=0;pos<7;++pos){
    glm_act in[GLM_HC*GLM_H];for(int j=0;j<GLM_HC*GLM_H;++j)in[j]=glm_put(((j+3*pos)%13-6)*0.0625f);
    floats hca(GLM_S_HC),x(GLM_H),p1(GLM_S_P1),p2(GLM_S_P2),att(GLM_S_ATT),bounded_att(GLM_S_ATT);
    for(int t=0;t<T;++t)glm_ph_hc(b,w,A_FN,glm_src_act{in},hca.p,t,T,lds.p);
    glm_ph_attention_input(b,w,in,hca.p,x.p,lds.p);
    // Independent original collapse/norm, not another call to the new helper.
    float hc[GLM_HCN],wantx[GLM_H];glm_hc_coeffs(b,w,A_FN,hca.p,hc,lds.p+GLM_LDS_HEAD);
    glm_collapse_norm(b,w[N_ATTN],glm_src_act{in},hc+GLM_HC+GLM_HC*GLM_HC,wantx,lds.p);
    for(int j=0;j<GLM_H;++j)if(!same(x.p[j],wantx[j]))return 10;
    for(int t=0;t<T;++t)glm_ph_p1(b,w,in,hca.p,p1.p,linear,t,T,lds.p);
    if(linear){
      for(int t=0;t<T;++t)glm_ph_kda(b,w,p1.p,ref_state.p,att.p,t,T,lds.p);
    } else {
      for(int t=0;t<T;++t)glm_ph_p2(b,w,p1.p,p2.p,ref_state.p,pos,t,T,lds.p);
      memcpy(bounded_state.p,ref_state.p,nstate*sizeof(float));
      for(int t=0;t<T;++t)glm_ph_mla(b,w,p1.p,p2.p,ref_state.p,att.p,pos,t,T,lds.p);
    }
    // Bounded calls with original descriptors also exercise sharded=false.
    for(int rank=0;rank<world;++rank){
      int begin=rank*(heads/world),end=(rank+1)*(heads/world);
      for(int t=0;t<T;++t){
        if(linear)glm_ph_kda(b,w,p1.p,bounded_state.p,bounded_att.p,t,T,lds.p,begin,end);
        else glm_ph_mla(b,w,p1.p,p2.p,bounded_state.p,bounded_att.p,pos,t,T,lds.p,begin,end);
      }
    }
    for(int i=0;i<heads*dim;++i)if(!same(att.p[i],bounded_att.p[i]))return 11;
    for(int i=0;i<nstate;++i)if(!same(ref_state.p[i],bounded_state.p[i]))return 12;
    floats partial(8*GLM_H),combined(GLM_S_MID),original_mid(GLM_S_MID);const float* outputs[8];
    for(int rank=0;rank<world;++rank){
      const int begin=rank*(heads/world),end=(rank+1)*(heads/world),off=begin*dim,n=(end-begin)*dim;
      auto& m=*local[rank];floats lp1(GLM_S_P1),lp2(GLM_S_P2),la(GLM_S_ATT);
      for(int t=0;t<T;++t)glm_ph_attention_p1(m.b,m.w,x.p,lp1.p,linear,rank,world,t,T,lds.p);
      for(int i=0;i<GLM_S_P1;++i){bool owned=false;
        if(linear){
          if(i<3*GLM_QDIM)owned=head_owned((i%GLM_QDIM)/GLM_LD,begin,end);
          else if(i<3*GLM_QDIM+GLM_LD)owned=true;
          else if(i<3*GLM_QDIM+GLM_LD+GLM_LH)owned=head_owned(i-3*GLM_QDIM-GLM_LD,begin,end);
          else if(i<3*GLM_QDIM+2*GLM_LD+GLM_LH)owned=true;
        } else owned=i<GLM_QR+GLM_KVR+2*GLM_ID+GLM_IH;
        if(owned?!same(lp1.p[i],p1.p[i]):!isnan(lp1.p[i]))return 20;
      }
      if(!linear){
        for(int t=0;t<T;++t)glm_ph_attention_p2(m.b,m.w,lp1.p,lp2.p,state[rank]->p,pos,rank,world,t,T,lds.p);
        for(int i=0;i<GLM_S_P2;++i){
          bool owned=i<GLM_NH*GLM_AD?head_owned(i/GLM_AD,begin,end):i<GLM_NH*GLM_AD+GLM_IH*GLM_ID;
          if(owned?!same(lp2.p[i],p2.p[i]):!isnan(lp2.p[i]))return 21;
        }
      }
      for(int t=0;t<T;++t)glm_ph_attention_heads(m.b,m.w,lp1.p,lp2.p,state[rank]->p,la.p,pos,linear,rank,world,t,T,lds.p);
      for(int i=0;i<GLM_S_ATT;++i){bool owned=i<heads*dim&&head_owned(i/dim,begin,end);
        if(owned?!same(la.p[i],att.p[i]):!isnan(la.p[i]))return 22;
      }
      for(int i=0;i<nstate;++i){bool owned=state_owned(i,linear,begin,end);
        if(owned?!same(state[rank]->p[i],ref_state.p[i]):!isnan(state[rank]->p[i]))return 23;
      }
      outputs[rank]=partial.p+rank*GLM_H;
      for(int t=0;t<T;++t)glm_ph_attention_o(m.b,m.w,la.p,partial.p+rank*GLM_H,linear,rank,world,t,T,lds.p);
      for(int k=0;k<GLM_H;++k){float dot=0;
        for(int j=0;j<n;++j)dot+=glm_w(m.b,m.w[O],k,j)*att.p[off+j];
        if(!same(outputs[rank][k],dot))return 24;
      }
    }
    for(int t=0;t<T;++t)glm_ph_attention_combine(b,w,in,hca.p,outputs,combined.p,world,t,T,lds.p);
    for(int k=0;k<GLM_H;++k){float dot=outputs[0][k];for(int r=1;r<world;++r)dot+=outputs[r][k];
      for(int c=0;c<GLM_HC;++c){float want=glm_mix1(hc,glm_src_act{in},glm_bf(dot),c,k);
        if(!same(combined.p[c*GLM_H+k],want))return 25;
      }
    }
    if(world==1){
      for(int t=0;t<T;++t)glm_ph_o(b,w,in,hca.p,att.p,original_mid.p,t,T,lds.p);
      for(int i=0;i<GLM_S_MID;++i)if(!same(combined.p[i],original_mid.p[i]))return 26;
    }
  }
  return 0;
}

extern "C" int combine_check(const void* bp,const void* wp,int world,int T){
  const uint8_t* b=(const uint8_t*)bp;const glm_weight* w=(const glm_weight*)wp;
  if(world!=2&&world!=4)return 1;
  float lds[GLM_LDS_FLOATS],hca[GLM_S_HC]={},hc[GLM_HCN],partial[4][GLM_H],mid[GLM_S_MID];
  glm_act in[GLM_HC*GLM_H]={};const float* outputs[4];bool witnessed=false;
  for(int r=0;r<world;++r){outputs[r]=partial[r];for(int j=0;j<GLM_H;++j){
    partial[r][j]=j%2?(r==0?1.0e8f:r==1?1.0f:r==2?-1.0e8f:1.0f):
                         (r==0?1.00390625f:r==1?0.00390625f:0.0f);
  }}
  for(int t=0;t<T;++t)glm_ph_attention_combine(b,w,in,hca,outputs,mid,world,t,T,lds);
  glm_hc_coeffs(b,w,A_FN,hca,hc,lds+GLM_LDS_HEAD);
  for(int j=0;j<GLM_H;++j){float sum=partial[0][j],early=glm_bf(partial[0][j]);
    for(int r=1;r<world;++r){sum+=partial[r][j];early+=glm_bf(partial[r][j]);}
    for(int c=0;c<GLM_HC;++c){
      float want=glm_mix1(hc,glm_src_act{in},glm_bf(sum),c,j);
      float wrong=glm_mix1(hc,glm_src_act{in},glm_bf(early),c,j);
      witnessed=witnessed||want!=wrong;
      if(!same(mid[c*GLM_H+j],want))return 30;
    }
  }
#if GLM_BF16
  if(!witnessed)return 31; // The fixture must actually expose premature BF rounding.
#endif
  return 0;
}
'''


@pytest.mark.parametrize('bf16,fp8,fp32_conv', [
    (False, False, True), (True, False, True), (True, False, False), (True, True, True),
])
def test_cpu_head_shards_match_original_state_and_rank_order_combine(tmp_path, bf16, fp8, fp32_conv):
    """Both attention types, head ranks 0/1, pool selection, recurrent steps, excess tiles."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from examples.glm53 import cpu_check
    stage = tmp_path / 'fixture'
    spec, _, _ = cpu_check.fixture(stage, bf16=bf16, fp8=fp8, selection=True, fp32_conv=fp32_conv)
    source = tmp_path / 'attention.cpp'
    source.write_text(CPU_SOURCE)
    library = tmp_path / 'attention.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC',
                    '-I', str(ROOT), '-I', str(stage), str(source), '-o', str(library)],
                   check=True, timeout=120)
    lib = ctypes.CDLL(str(library))
    lib.attention_check.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 4
    lib.attention_check.restype = ctypes.c_int
    lib.combine_check.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
    lib.combine_check.restype = ctypes.c_int
    assert spec['dimensions']['LH'] == spec['dimensions']['NH'] == 2
    for linear, layer in [(1, 'l0'), (0, 'l1')]:
        weights = spec['weights'][layer]
        payload = (stage / weights['blob']).read_bytes()
        blob = ctypes.create_string_buffer(payload)
        meta = ctypes.create_string_buffer((stage / weights['meta']).read_bytes())
        for tiles in (1, 3, 7):
            for world in (1, 2):
                result = lib.attention_check(blob, meta, len(payload), linear, world, tiles)
                assert result == 0, (linear, world, tiles, result)
            for world in (2, 4):
                result = lib.combine_check(blob, meta, world, tiles)
                assert result == 0, (linear, world, tiles, result)
