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

// An expert owns all selected position rows, rather than one position owning
// all experts. args: weights, meta, (rt, act, sel) for each position.
// cst: (total tiles, row tiles per expert, positions, 0).
// Consecutive tile ranges own consecutive experts (EXPERTS = shared).
extern "C" __device__ void glm53_verify_expert_up(const etx_ctx* c) {
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  const int T=c->cst[1],expert=c->coord[1]/T;
  const int gs=expert==GLM_EXPERTS?SG:EXPERTS+3*expert;
  const int mi=w[gs].rows,t=c->coord[1]%T;
  const int lo=(int)((int64_t)mi*t/T),hi=(int)((int64_t)mi*(t+1)/T);
  if(lo==hi)return;
  for(int p=0;p<c->cst[2];++p){
    const float* rt=GLM_PR(2+3*p,const float*,GLM_S_RT);
    float* act=GLM_PR(3+3*p,float*,GLM_S_ACT);
    const float* sel=GLM_PR(4+3*p,const float*,GLM_S_SEL);
    int k=expert==GLM_EXPERTS?GLM_TOPK:-1;
    for(int j=0;j<GLM_TOPK;++j)if((int)sel[j]==expert)k=j;
    if(k<0)continue;
    float* x=(float*)c->lds+GLM_LDS_HEAD;
    for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)x[j]=rt[GLM_R4(GLM_EXPERTS)+j];
    GLM_SYNC();
    const int S=glm_group<GLM_KFP8>(w[gs]);
    glm_each(0,(hi-lo+1)/2,S,[&](int u,int sl){
      const glm_weight* ws[4]={&w[gs],&w[gs+1],&w[gs],&w[gs+1]};
      const float* xs[4]={x,x,x,x};
      const int r0=lo+2*u,r1=r0+1<hi?r0+1:r0;
      int rows[4]={r0,r0,r1,r1};float acc[4];
      glm_dots<4,GLM_KFP8>(b,ws,rows,xs,acc,S,sl);
      if(sl==0)for(int i=0;i<2;++i)if(r0+i<hi){
        const float gt=glm_bf(acc[2*i]),up=glm_bf(acc[2*i+1]);
        const float a=fminf(gt,GLM_CLIP),v=fmaxf(-GLM_CLIP,fminf(up,GLM_CLIP));
        act[k*GLM_INTERMAX+r0+i]=glm_bf(glm_bf(a*glm_sig(a))*v);
      }
    });
    GLM_SYNC();
  }
}
