// Transactional verification scans. Checkpoint 0 precedes the block;
// checkpoint p+1 follows input position p. Copies include KDA convolution
// history and the complete DSA cache, so rejected KV bytes never persist.
#pragma once
#include "examples/glm53/verify_phases.h"

__device__ inline void glm53_state_copy(float* dst,const float* src,int n){
  for(int i=GLM_LANE;i<n;i+=GLM_LANES)dst[i]=src[i];
  GLM_SYNC();
}

extern "C" __device__ void glm53_checkpoint_kda(const etx_ctx* c){
  const int count=c->cst[2],size=c->cst[3];
  float* state=GLM_PR(2,float*,(uint64_t)size);
  float* saved=GLM_PR(3+2*count,float*,(uint64_t)(count+1)*size);
  // Each head owns disjoint state, but copying the full state requires one
  // owner. This correctness-first scan runs all heads in one workgroup.
  glm53_state_copy(saved,state,size);
  for(int p=0;p<count;++p){
    for(int h=0;h<c->cst[0];++h){
      glm_ph_kda(GLM_PB(0),GLM_PW(1),GLM_PR(3+2*p,const float*,GLM_S_P1),
                 state,GLM_PR(4+2*p,float*,GLM_S_ATT),h,c->cst[0],(float*)c->lds);
      GLM_SYNC();
    }
    glm53_state_copy(saved+(p+1)*size,state,size);
  }
}

extern "C" __device__ void glm53_checkpoint_dsa(const etx_ctx* c){
  const int count=c->cst[2],size=c->cst[3];
  float* state=GLM_PR(2,float*,(uint64_t)size);
  float* saved=GLM_PR(3+4*count,float*,(uint64_t)(count+1)*size);
  float* lds=(float*)c->lds;
  glm53_state_copy(saved,state,size);
  for(int p=0;p<count;++p){
    const int pos=glm53_step_load((const int*)c->args[3+4*p]+2*c->coord[0]);
    const float* in=GLM_PR(4+4*p,const float*,GLM_S_P1);
    float* tmp=GLM_PR(5+4*p,float*,GLM_S_P2);
    glm_ph_p2(GLM_PB(0),GLM_PW(1),in,tmp,state,pos,0,1,lds);
    GLM_SYNC();
    glm_ph_mla(GLM_PB(0),GLM_PW(1),in,tmp,state,
               GLM_PR(6+4*p,float*,GLM_S_ATT),pos,0,1,lds);
    GLM_SYNC();
    glm53_state_copy(saved+(p+1)*size,state,size);
  }
}
