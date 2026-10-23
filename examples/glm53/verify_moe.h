// Multi-position MoE building blocks. No routing or state mutation occurs here.
#pragma once
#include "examples/glm53/native.h"

// Reader injection lets a host byte ledger observe the actual weight accesses.
// Production uses the normal dequantizer (including its BF16 rounding).
struct glm_verify_moe_reader {
  GLM_DEV float operator()(const uint8_t* b,const glm_weight& w,int r,int c) const {
    return glm_w(b,w,r,c);
  }
};

// Inputs are P independent rows, with precomputed sorted top-k IDs/probabilities.
// Expert GLM_EXPERTS denotes the shared expert, stored in activation slot TOPK.
// Call once for each expert in the union, or all experts (unused ones return).
// Each matrix output row has exactly one lane/tile owner; all P accumulators
// consume a weight before the next weight is read. Splitting P into calls loses
// the read-once property. Caller validates unique IDs and 0 <= expert <= EXPERTS.
// Require tiles > 0, 0 <= tile < tiles, and ordinary staged MoE dimensions.
// These primitives do not replace WP-S2 graph symbols by inclusion: its owner
// must schedule up/down/mix boundaries and supply the pointer arrays explicitly.
template<int P> GLM_DEV bool glm_verify_moe_slots(
    const float* const* sel,int expert,int (&slot)[P]) {
  static_assert(P>0,"verify MoE needs at least one row");
  bool used=false;
  for(int p=0;p<P;++p){
    slot[p]=expert==GLM_EXPERTS?GLM_TOPK:-1;
    for(int k=0;k<GLM_TOPK;++k)if((int)sel[p][k]==expert)slot[p]=k;
    used=used||slot[p]>=0;
  }
  return used;
}

// x[p] is the normalized H-vector (not the router-logit prefix of rt).
// act[p] has (TOPK+1)*INTERMAX floats, matching the ordinary decode layout.
template<int P,class Read=glm_verify_moe_reader> GLM_DEV void glm_verify_moe_up(
    const uint8_t* b,const glm_weight* w,const float* const* x,
    const float* const* sel,float* const* act,int expert,int tile=0,int tiles=1,
    Read read=Read{}) {
  int slot[P];if(!glm_verify_moe_slots<P>(sel,expert,slot))return;
  const int gs=expert==GLM_EXPERTS?SG:EXPERTS+3*expert;
  const int lo=(int)((int64_t)w[gs].rows*tile/tiles);
  const int hi=(int)((int64_t)w[gs].rows*(tile+1)/tiles);
  for(int r=lo+GLM_LANE;r<hi;r+=GLM_LANES){
    float gate[P]={},up[P]={};
    for(int c=0;c<w[gs].cols;++c){
      const float g=read(b,w[gs],r,c),u=read(b,w[gs+1],r,c);
      for(int p=0;p<P;++p)if(slot[p]>=0){gate[p]+=g*x[p][c];up[p]+=u*x[p][c];}
    }
    for(int p=0;p<P;++p)if(slot[p]>=0){
      const float a=fminf(glm_bf(gate[p]),GLM_CLIP);
      const float v=fmaxf(-GLM_CLIP,fminf(glm_bf(up[p]),GLM_CLIP));
      act[p][slot[p]*GLM_INTERMAX+r]=glm_bf(glm_bf(a*glm_sig(a))*v);
    }
  }
}

// Independent expert down results: partial[p] has (TOPK+1)*H floats.
// Finish ALL up calls before any down call; finish ALL down calls before mix.
// These are graph/kernel boundaries, not a block-local synchronization promise.
template<int P,class Read=glm_verify_moe_reader> GLM_DEV void glm_verify_moe_down(
    const uint8_t* b,const glm_weight* w,const float* const* act,
    const float* const* sel,float* const* partial,int expert,int tile=0,int tiles=1,
    Read read=Read{}) {
  int slot[P];if(!glm_verify_moe_slots<P>(sel,expert,slot))return;
  const int ds=expert==GLM_EXPERTS?SD:EXPERTS+3*expert+2;
  const int lo=(int)((int64_t)w[ds].rows*tile/tiles);
  const int hi=(int)((int64_t)w[ds].rows*(tile+1)/tiles);
  for(int r=lo+GLM_LANE;r<hi;r+=GLM_LANES){
    float sum[P]={};
    for(int c=0;c<w[ds].cols;++c){
      const float v=read(b,w[ds],r,c);
      for(int p=0;p<P;++p)if(slot[p]>=0)sum[p]+=v*act[p][slot[p]*GLM_INTERMAX+c];
    }
    for(int p=0;p<P;++p)if(slot[p]>=0)
      partial[p][slot[p]*GLM_H+r]=glm_bf(sum[p]);
  }
}

// Match decode's expert-ID order and BF16 rounding. The caller feeds this
// MoE output to its ordinary mHC residual mix; no residual is added twice.
GLM_DEV float glm_verify_moe_mix(const float* partial,const float* sel,int col) {
  float y=0;
  for(int k=0;k<GLM_TOPK;++k)
    y=glm_bf(y+glm_bf(partial[k*GLM_H+col]*sel[GLM_TOPK+k]));
  return glm_bf(y+partial[GLM_TOPK*GLM_H+col]);
}
