// Verify-only tiles. Include as a translation unit: the existing decode tiles
// remain the source of truth for rounding, DSA selection and cache layout.
#pragma once
#include "examples/glm53/tiles.hip"

// One head scans all positions in causal order, including its convolution and
// recurrent state. args: weights, meta, state, (p1, att) for each position.
extern "C" __device__ void glm53_verify_kda_scan(const etx_ctx* c) {
  float* state=GLM_PR(2,float*,(uint64_t)c->cst[3]);
  for(int p=0;p<c->cst[2];++p){
    const float* in=GLM_PR(3+2*p,const float*,GLM_S_P1);
    float* out=GLM_PR(4+2*p,float*,GLM_S_ATT);
    glm_ph_kda(GLM_PB(0),GLM_PW(1),in,state,out,GLM_PT);
  }
}

// One workgroup owns a request's DSA cache throughout the block. Complete
// each position's KV publication and index top-k before advancing the scan.
// args: weights, meta, state, (step, p1, p2, att) for each position.
extern "C" __device__ void glm53_verify_dsa_scan(const etx_ctx* c) {
  float* state=GLM_PR(2,float*,(uint64_t)c->cst[3]);
  float* lds=(float*)c->lds;
  for(int p=0;p<c->cst[2];++p){
    const int pos=glm53_step_load((const int*)c->args[3+4*p]+2*c->coord[0]);
    const float* p1=GLM_PR(4+4*p,const float*,GLM_S_P1);
    float* p2=GLM_PR(5+4*p,float*,GLM_S_P2);
    float* att=GLM_PR(6+4*p,float*,GLM_S_ATT);
    glm_ph_p2(GLM_PB(0),GLM_PW(1),p1,p2,state,pos,0,1,lds);
    GLM_SYNC();
    glm_ph_mla(GLM_PB(0),GLM_PW(1),p1,p2,state,att,pos,0,1,lds);
    GLM_SYNC();
  }
}

// Each expert consumes a weight once for all positions. Separate graph grids
// provide the global up -> down -> mix boundaries required by verify_moe.h.
#include "examples/glm53/verify_moe.h"
template<int P, bool Down> GLM_DEV void glm_verify_expert(const etx_ctx* c) {
  const float *x[P],*sel[P];float* out[P];
  for(int p=0;p<P;++p){
    if constexpr(Down){
      x[p]=GLM_PR(2+3*p,const float*,GLM_S_ACT);
      sel[p]=GLM_PR(3+3*p,const float*,GLM_S_SEL);
      out[p]=GLM_PR(4+3*p,float*,(GLM_TOPK+1)*GLM_H);
    }else{
      x[p]=GLM_PR(2+3*p,const float*,GLM_S_RT)+GLM_R4(GLM_EXPERTS);
      out[p]=GLM_PR(3+3*p,float*,GLM_S_ACT);
      sel[p]=GLM_PR(4+3*p,const float*,GLM_S_SEL);
    }
  }
  const int T=c->cst[1],e=c->coord[1]/T,t=c->coord[1]%T;
  if constexpr(Down)glm_verify_moe_down<P>(GLM_PB(0),GLM_PW(1),x,sel,out,e,t,T);
  else glm_verify_moe_up<P>(GLM_PB(0),GLM_PW(1),x,sel,out,e,t,T);
}
template<bool Down> GLM_DEV void glm_verify_experts(const etx_ctx* c) {
  switch(c->cst[2]){
    case 2:glm_verify_expert<2,Down>(c);break;
    case 3:glm_verify_expert<3,Down>(c);break;
    case 4:glm_verify_expert<4,Down>(c);break;
    case 5:glm_verify_expert<5,Down>(c);break;
    case 6:glm_verify_expert<6,Down>(c);break;
  }
}
extern "C" __device__ void glm53_verify_expert_up(const etx_ctx* c) {
  glm_verify_experts<false>(c);
}
extern "C" __device__ void glm53_verify_expert_down(const etx_ctx* c) {
  glm_verify_experts<true>(c);
}
extern "C" __device__ void glm53_verify_expert_mix(const etx_ctx* c) {
  const float* partial=GLM_PR(2,const float*,(GLM_TOPK+1)*GLM_H);
  const float* sel=GLM_PR(3,const float*,GLM_S_SEL);
  const float* mid=GLM_PR(4,const float*,GLM_S_MID);
  const float* hcf=GLM_PR(5,const float*,GLM_S_HC);
  glm_act* out=GLM_PR(6,glm_act*,GLM_HC*GLM_H);
  float* lds=(float*)c->lds;float* hc=lds+16;
  glm_hc_coeffs(GLM_PB(0),GLM_PW(1),F_FN,hcf,hc,lds+GLM_LDS_HEAD);
  const int T=c->cst[0],t=c->coord[1];
  const int lo=(int)((int64_t)GLM_H*t/T),hi=(int)((int64_t)GLM_H*(t+1)/T);
  for(int j=lo+GLM_LANE;j<hi;j+=GLM_LANES){
    const float y=glm_verify_moe_mix(partial,sel,j);
    for(int k=0;k<GLM_HC;++k)
      out[k*GLM_H+j]=glm_put(glm_mix1(hc,[&](int i){return mid[i];},y,k,j));
  }
}

// One streamed head row serves every verify position. Normalized activations
// live in their existing head scratch, so LDS capacity does not grow with P.
// Each output dot retains the decode scalar column order and BF16 boundaries.
// args: norm blob/meta, head blob/meta, (input, head scratch) for each row.
template<int P> GLM_DEV void glm_verify_head(const etx_ctx* c) {
  const glm_act* in[P]; float* hs[P]; float inv[P];
  float* red=(float*)c->lds;
  for(int p=0;p<P;++p){
    in[p]=GLM_PR(4+2*p,const glm_act*,GLM_HC*GLM_H);
    hs[p]=GLM_PR(5+2*p,float*,2*GLM_H+GLM_VOCAB);
    float part=0;
    for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){
      float sum=0;
      for(int k=0;k<GLM_HC;++k)sum+=glm_get(in[p][k*GLM_H+j]);
      const float x=glm_bf(sum/GLM_HC);hs[p][j]=x;part+=x*x;
    }
    inv[p]=1.0f/sqrtf(glm_bsum(part,red)/GLM_H+GLM_EPS);
  }
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){
    const float w=glm_w(GLM_PB(0),GLM_PW(1)[0],j);
    for(int p=0;p<P;++p)hs[p][GLM_H+j]=glm_bf(glm_bf(hs[p][j]*inv[p])*w);
  }
  GLM_SYNC();
  const glm_weight& w=GLM_PW(3)[0];
  for(int r=GLM_LANE;r<w.rows;r+=GLM_LANES){
    float acc[P]={};
    for(int j=0;j<w.cols;++j){
      const float v=glm_w(GLM_PB(2),w,r,j);
      for(int p=0;p<P;++p)acc[p]+=v*hs[p][GLM_H+j];
    }
    for(int p=0;p<P;++p)hs[p][2*GLM_H+r]=glm_bf(acc[p]);
  }
}
extern "C" __device__ void glm53_verify_head(const etx_ctx* c) {
  switch(c->cst[2]){
    case 2:glm_verify_head<2>(c);break;
    case 3:glm_verify_head<3>(c);break;
    case 4:glm_verify_head<4>(c);break;
    case 5:glm_verify_head<5>(c);break;
    case 6:glm_verify_head<6>(c);break;
  }
}
