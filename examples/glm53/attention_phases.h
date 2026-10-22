// Head-sharded attention phases. Descriptors keep native.h's global slot ABI,
// but refer to rank-local payloads. Activations and state retain FULL GLOBAL
// strides: only the owned head regions are read/written (capacity is not sharded).
// Preconditions: world>0, 0<=rank<world, LH/NH divisible by world as applicable;
// t is in [0,T), T>0. The graph/loader must validate these before launching.
// Every rank receives the same H-float normalized x. p1/p2/att/state are private
// to the rank; replicas (low ranks, indexer, FA/GA) must not be concurrently
// written into a shared buffer by different ranks. Phase boundaries need events.
#pragma once
#include "examples/glm53/phases.h"

// Root-only, one workgroup: collapse the mHC streams and apply N_ATTN once.
// x is H floats; lds has the same GLM_LDS_FLOATS capacity as the original phases.
GLM_DEV void glm_ph_attention_input(const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,float* x,float* lds){
  float* hc=lds+16;
  glm_hc_coeffs(b,w,A_FN,hca,hc,lds+GLM_LDS_HEAD);
  glm_collapse_norm(b,w[N_ATTN],glm_src_act{in},hc+GLM_HC+GLM_HC*GLM_HC,x,lds);
}

// KDA: Q/K/V and BETA have local rows; FA/GA are full replicas. Write
// global p1 offsets, leaving other ranks' Q/K/V/BETA regions untouched.
// MLA: QA/KVA/IK/IG/IW are full replicas, including original BF16 boundaries.
GLM_DEV void glm_ph_attention_p1(const uint8_t* b,const glm_weight* w,const float* x,float* p1,bool linear,int rank,int world,int t,int T,float* lds){
  (void)lds;
  if(linear){
    const int headbase=rank*(GLM_LH/world),off=headbase*GLM_LD;
    const glm_seg s[6]={{Q,x,p1+off},{K,x,p1+GLM_QDIM+off},{V,x,p1+2*GLM_QDIM+off},
      {FA,x,p1+3*GLM_QDIM},{BETA,x,p1+3*GLM_QDIM+GLM_LD+headbase},
      {GA,x,p1+3*GLM_QDIM+GLM_LD+GLM_LH}};
    glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,6,t,T);
  } else {
    const glm_seg s[5]={{QA,x,p1},{KVA,x,p1+GLM_QR},{IK,x,p1+GLM_QR+GLM_KVR},
      {IG,x,p1+GLM_QR+GLM_KVR+GLM_ID},{IW,x,p1+GLM_QR+GLM_KVR+2*GLM_ID}};
    glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,5,t,T);
  }
}

// MLA only: normalize the replicated low ranks, then project local QB/KVB
// rows into their global head regions. IQ and the position's indexer state
// are replicated. The indexer tail starts after ALL heads, not local heads.
GLM_DEV void glm_ph_attention_p2(const uint8_t* b,const glm_weight* w,const float* p1,float* p2,float* state,int pos,int rank,int world,int t,int T,float* lds){
  float* qr=lds+GLM_LDS_HEAD;float* kv=qr+GLM_R4(GLM_QR);
  glm_pnorm(b,w[QAN],p1,qr,GLM_QR,lds);glm_pnorm(b,w[KVAN],p1+GLM_QR,kv,GLM_KVR,lds);
  const int headbase=rank*(GLM_NH/world),stride=2*GLM_NH*GLM_AD+2*GLM_ID;
  float* st=state+(uint64_t)pos*stride;
  const glm_seg s[3]={{QB,qr,p2+headbase*GLM_AD},{KVB,kv,st+headbase*2*GLM_AD},{IQ,qr,p2+GLM_NH*GLM_AD}};
  glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,3,t,T);
  if(t==0){
    const float* ik=p1+GLM_QR+GLM_KVR;const float* ig=ik+GLM_ID;
    float part=0;for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=ik[i];
    const float mean=glm_bsum(part,lds)/GLM_ID;
    part=0;for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=(ik[i]-mean)*(ik[i]-mean);
    const float inv=1/sqrtf(glm_bsum(part,lds)/GLM_ID+1e-6f);
    for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)st[2*GLM_NH*GLM_AD+i]=glm_bf((ik[i]-mean)*inv*glm_w(b,w[IKN],i)+glm_w(b,w[IKB],i));
    for(int j=GLM_LANE;j<GLM_ID;j+=GLM_LANES)st[2*GLM_NH*GLM_AD+GLM_ID+j]=ig[j];
  }
}

// KDA additionally consumes local FB/GB/DT/ALOG rows and a CONV descriptor
// containing Q-local | K-local | V-local rows. ONORM remains fully replicated.
// Both attention bodies keep state/att addressing in global head coordinates.
GLM_DEV void glm_ph_attention_heads(const uint8_t* b,const glm_weight* w,const float* p1,const float* p2,float* state,float* att,int pos,bool linear,int rank,int world,int t,int T,float* lds){
  if(linear){
    const int begin=rank*(GLM_LH/world),end=(rank+1)*(GLM_LH/world);
    glm_ph_kda(b,w,p1,state,att,t,T,lds,begin,end,true);
  } else {
    const int begin=rank*(GLM_NH/world),end=(rank+1)*(GLM_NH/world);
    glm_ph_mla(b,w,p1,p2,state,att,pos,t,T,lds,begin,end);
  }
}

// O has H rows and only this rank's input columns (QDIM/world for KDA,
// NH*AD/world for MLA). Raw FP32: NEVER round individual rank partials to BF16.
GLM_DEV void glm_ph_attention_o(const uint8_t* b,const glm_weight* w,const float* att,float* partial,bool linear,int rank,int world,int t,int T,float* lds){
  (void)lds;
  const int off=linear?rank*(GLM_LH/world)*GLM_LD:rank*(GLM_NH/world)*GLM_AD;
  const uint64_t bytes=1;const int rows=w[O].rows;int lo,hi;
  glm_share(&bytes,&rows,1,0,t,T,lo,hi);
  glm_rows<GLM_KBF16|GLM_KFP8>(b,w[O],att+off,lo,hi,[&](int k,float v){partial[k]=v;});
}

// Root-only: H-float partials, fixed rank order, BF16 once AFTER the FP32 sum,
// then the original attention A_FN mix into HC*H FP32 streams.
// FP32 reassociation at O's column-shard boundaries is intentional; bitwise
// equivalence to the unsharded O dot is not asserted for world>1.
GLM_DEV void glm_ph_attention_combine(const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,const float* const* partials,float* mid,int world,int t,int T,float* lds){
  float* hc=lds+16;glm_hc_coeffs(b,w,A_FN,hca,hc,lds+GLM_LDS_HEAD);
  const int lo=(int)((int64_t)GLM_H*t/T),hi=(int)((int64_t)GLM_H*(t+1)/T);
  for(int k=lo+GLM_LANE;k<hi;k+=GLM_LANES){
    float sum=partials[0][k];for(int r=1;r<world;++r)sum+=partials[r][k];
    const float y=glm_bf(sum);
    for(int c=0;c<GLM_HC;++c)mid[c*GLM_H+k]=glm_mix1(hc,glm_src_act{in},y,c,k);
  }
}
