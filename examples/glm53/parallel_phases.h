// GLM FFN-only tensor/expert parallel helpers. Attention and final mHC mixing
// remain on the token owner. Include after phases.h (also included below).
#pragma once
#include "examples/glm53/phases.h"

// All pointers are already advanced to one request. Packet layout is unchanged:
// rt = [R4(EXPERTS) router logits | H normalized inputs], sel = [TOPK ids | TOPK probabilities].
// Each rank owns private act[GLM_S_ACT] and expert_out[(TOPK+1)*H]. Output slot k
// denotes selected position k, NOT expert ID. Shared is slot TOPK; dense uses slot 0.
//
// Packed metadata retains global slot numbers. EP retains complete matrices for
// expert_id % world == rank and SG/SU/SD on shared_owner only. TP retains all
// selected expert slots with gate/up rows and down columns sharded consistently.
// Both payload and scale offsets are relative to the rank's LOCAL packed blob.
// TP scale repacking must preserve original 128x128 blocks (block-aligned cuts).
// Dispatch must copy rt/sel locally before up; all down producers must finish
// before owner combine. These helpers do not publish events or transfer packets.

GLM_DEV int glm_parallel_slot(const float* sel,bool moe,int k){
  return !moe?MG:k<GLM_TOPK?EXPERTS+3*(int)sel[k]:SG;
}
GLM_DEV int glm_parallel_owner(const float* sel,bool moe,int k,int world,int shared_owner){
  return moe&&k<GLM_TOPK?(int)sel[k]%world:shared_owner;
}

// Exactly one workgroup per request, on the token owner. Reuse the original
// tie breaking, probability normalization and ascending-expert-ID permutation.
GLM_DEV void glm_ph_parallel_route(const uint8_t* b,const glm_weight* w,const float* rt,float* sel,bool moe,float* lds){
  float* score=lds+GLM_LDS_HEAD;
  float* biased=score+GLM_R4(GLM_EXPERTS);
  float* probs=biased+GLM_R4(GLM_EXPERTS);
  int* ids=(int*)(probs+GLM_R4(GLM_TOPK));
  if(moe){
    glm_topk(b,w,rt,score,biased,ids,probs,lds);
    for(int k=GLM_LANE;k<GLM_TOPK;k+=GLM_LANES){sel[k]=(float)ids[k];sel[GLM_TOPK+k]=probs[k];}
  }else{
    for(int k=GLM_LANE;k<GLM_S_SEL;k+=GLM_LANES)sel[k]=k==GLM_TOPK?1.0f:0.0f;
  }
}

// Flatten the owned selected slots' LOCAL intermediate rows, then partition
// that one range among T tiles. Most tiles touch only one selected slot rather
// than issuing a tiny dot loop for every expert. Selection/metadata are uniform
// across the workgroup, so an empty tile can return before loading x or syncing.
// Nonowned descriptors and activations are never read. Unowned act need not be
// initialized: down zeroes their output slots without reading act.
GLM_DEV void glm_ph_parallel_up(const uint8_t* b,const glm_weight* w,const float* rt,const float* sel,float* act,
                               bool moe,bool tensor_parallel,int rank,int world,int shared_owner,int t,int T,float* lds){
  const int ne=moe?GLM_TOPK+1:1;
  int slots[GLM_TOPK+1],ends[GLM_TOPK+2],count=0;ends[0]=0;
  for(int k=0;k<ne;++k){
    if(!tensor_parallel&&glm_parallel_owner(sel,moe,k,world,shared_owner)!=rank)continue;
    const int gs=glm_parallel_slot(sel,moe,k);
    slots[count]=k;ends[count+1]=ends[count]+w[gs].rows;++count;
  }
  const int total=ends[count];
  const int tile_lo=(int)((int64_t)total*t/T),tile_hi=(int)((int64_t)total*(t+1)/T);
  if(tile_hi<=tile_lo)return;
  float* x=lds+GLM_LDS_HEAD;
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)x[j]=rt[GLM_R4(GLM_EXPERTS)+j];
  GLM_SYNC();
  for(int q=0;q<count;++q){
    if(tile_hi<=ends[q]||tile_lo>=ends[q+1])continue;
    const int k=slots[q],gs=glm_parallel_slot(sel,moe,k);
    const int lo=(tile_lo>ends[q]?tile_lo:ends[q])-ends[q];
    const int hi=(tile_hi<ends[q+1]?tile_hi:ends[q+1])-ends[q];
    const int S=glm_group<GLM_KFP8>(w[gs]);
    glm_each(0,(hi-lo+1)/2,S,[&](int u,int sl){
      const glm_weight* ws[4]={&w[gs],&w[gs+1],&w[gs],&w[gs+1]};
      const float* xs[4]={x,x,x,x};
      const int r0=lo+2*u,r1=r0+1<hi?r0+1:r0;
      int row[4]={r0,r0,r1,r1};float acc[4];
      glm_dots<4,GLM_KFP8>(b,ws,row,xs,acc,S,sl);
      if(sl==0)for(int i=0;i<2;++i)if(r0+i<hi){
        const float gt=glm_bf(acc[2*i]),up=glm_bf(acc[2*i+1]);
        const float a=fminf(gt,GLM_CLIP),v=fmaxf(-GLM_CLIP,fminf(up,GLM_CLIP));
        act[(uint64_t)k*GLM_INTERMAX+r0+i]=glm_bf(glm_bf(a*glm_sig(a))*v);
      }
    });
  }
}

// Full hidden output per selected expert, kept separate until ordered combine.
// TP: raw FP32 local-K partials; rounding must follow the rank sum, not precede it.
// EP: a complete dot, rounded exactly where the original down phase rounds it.
// Every rank writes ALL (TOPK+1)*H output words, including zero for unowned and
// unused dense slots. Thus reused scratch cannot leak stale expert outputs.
GLM_DEV void glm_ph_parallel_down(const uint8_t* b,const glm_weight* w,const float* act,const float* sel,float* expert_out,
                                 bool moe,bool tensor_parallel,int rank,int world,int shared_owner,int t,int T,float* lds){
  (void)lds;
  const int lo=(int)((int64_t)GLM_H*t/T),hi=(int)((int64_t)GLM_H*(t+1)/T);
  const int ne=moe?GLM_TOPK+1:1;
  for(int k=0;k<GLM_TOPK+1;++k){
    const bool owned=k<ne&&(tensor_parallel||glm_parallel_owner(sel,moe,k,world,shared_owner)==rank);
    float* dst=expert_out+(uint64_t)k*GLM_H;
    if(!owned){for(int j=lo+GLM_LANE;j<hi;j+=GLM_LANES)dst[j]=0;continue;}
    const int ds=glm_parallel_slot(sel,moe,k)+2;
    glm_rows<GLM_KFP8>(b,w[ds],act+(uint64_t)k*GLM_INTERMAX,lo,hi,
      [&](int j,float v){dst[j]=tensor_parallel?v:glm_bf(v);});
  }
}

// rank_outputs[r] points at rank r's per-request expert_out, not a host pointer
// table. A wrapper can build this small pointer array from explicit graph args.
// EP reads only the owning rank. TP sums raw partials in increasing rank order,
// then restores the per-expert BF16 boundary before probabilities/accumulation.
// No rank-local sum over experts is permitted: BF16 addition is nonassociative.
GLM_DEV void glm_ph_parallel_combine(const uint8_t* b,const glm_weight* w,const float* sel,const float* const* rank_outputs,
                                    const float* mid,const float* hcf,glm_act* out,bool moe,bool tensor_parallel,
                                    int world,int shared_owner,int t,int T,float* lds){
  float* hc=lds+16;glm_hc_coeffs(b,w,F_FN,hcf,hc,lds+GLM_LDS_HEAD);
  const int lo=(int)((int64_t)GLM_H*t/T),hi=(int)((int64_t)GLM_H*(t+1)/T);
  for(int j=lo+GLM_LANE;j<hi;j+=GLM_LANES){
    auto expert=[&](int k){
      const uint64_t off=(uint64_t)k*GLM_H+j;
      if(!tensor_parallel)return rank_outputs[glm_parallel_owner(sel,moe,k,world,shared_owner)][off];
      float v=rank_outputs[0][off];for(int r=1;r<world;++r)v+=rank_outputs[r][off];
      return glm_bf(v);
    };
    float y;
    if(moe){
      y=0;
      for(int k=0;k<GLM_TOPK;++k)y=glm_bf(y+glm_bf(expert(k)*sel[GLM_TOPK+k]));
      y=glm_bf(y+expert(GLM_TOPK));
    }else y=expert(0);
    for(int c=0;c<GLM_HC;++c)out[c*GLM_H+j]=glm_put(glm_mix1(hc,[&](int i){return mid[i];},y,c,j));
  }
}
