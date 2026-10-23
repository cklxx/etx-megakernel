#include "etx/abi.h"
#include "examples/glm53/native.h"
// Host-updated per-step controls cross both GPUs and may be read by CUs with
// only DOMAIN predecessors. SYSTEM atomic loads must bypass stale XCD caches.
static __device__ __forceinline__ int glm53_step_load(const int* p) {
#if defined(__CUDACC__) && !defined(__HIP__)
  int v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
#else
  return __hip_atomic_load(p,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
extern "C" __device__ __attribute__((always_inline)) void glm53_embed(const etx_ctx* c) {
  int r=c->coord[0];const int* step=(const int*)c->args[0];const uint8_t* b=(const uint8_t*)c->args[1];
  const glm_weight* w=(const glm_weight*)c->args[2];glm_act* out=(glm_act*)c->args[3]+r*GLM_HC*GLM_H;
  for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES)out[j]=glm_put(glm_w(b,w[0],glm53_step_load(step+2*r+1),j%GLM_H));
}
extern "C" __device__ __attribute__((always_inline)) void glm53_layer(const etx_ctx* c) {
  int r=c->coord[0];const int* step=(const int*)c->args[0];
  glm_layer((const uint8_t*)c->args[1],(const glm_weight*)c->args[2],
    (const glm_act*)c->args[3]+r*GLM_HC*GLM_H,(glm_act*)c->args[4]+r*GLM_HC*GLM_H,
    (float*)c->args[5]+(uint64_t)r*c->cst[2],(float*)c->args[6]+(uint64_t)r*GLM_SCRATCH,
    glm53_step_load(step+2*r),c->cst[0],c->cst[1]);
}
extern "C" __device__ __attribute__((always_inline)) void glm53_head(const etx_ctx* c) {
  int r=c->coord[0];const glm_act* input=(const glm_act*)c->args[0]+r*GLM_HC*GLM_H;
  const uint8_t* nb=(const uint8_t*)c->args[1];const glm_weight* nw=(const glm_weight*)c->args[2];
  const uint8_t* hb=(const uint8_t*)c->args[3];const glm_weight* hw=(const glm_weight*)c->args[4];
  float* x=(float*)c->args[5]+(uint64_t)r*(2*GLM_H+GLM_VOCAB);float* y=x+GLM_H;float* logits=y+GLM_H;
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){float sum=0;for(int k=0;k<GLM_HC;++k)sum+=glm_get(input[k*GLM_H+j]);x[j]=glm_bf(sum/GLM_HC);}GLM_SYNC();
  glm_norm(nb,nw[0],x,y,GLM_H);glm_mv(hb,hw[0],y,logits);
  if(GLM_LANE==0){int best=0;for(int j=1;j<GLM_VOCAB;++j)if(logits[j]>logits[best])best=j;((int*)c->args[6])[r]=best;}
}
extern "C" __device__ __attribute__((always_inline)) void glm53_feedback(const etx_ctx* c) {
  if(GLM_LANE==0)((int*)c->args[1])[c->coord[0]]=((const int*)c->args[0])[c->coord[0]];
}

// ---- phased layers (examples/glm53/phases.h). coord = (request, tile); cst = (tiles, linear, moe, state stride).
#include "examples/glm53/phases.h"
#define GLM_PR(i,type,stride) ((type)c->args[i]+(uint64_t)c->coord[0]*(stride))
#define GLM_PB(i) ((const uint8_t*)c->args[i])
#define GLM_PW(i) ((const glm_weight*)c->args[i])
#define GLM_PT c->coord[1],c->cst[0],(float*)c->lds
extern "C" __device__ __attribute__((noinline)) void glm53_hc(const etx_ctx* c) {
  const glm_act* in=GLM_PR(2,const glm_act*,GLM_HC*GLM_H);
  glm_ph_hc(GLM_PB(0),GLM_PW(1),A_FN,glm_src_act{in},GLM_PR(3,float*,GLM_S_HC),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_hcf(const etx_ctx* c) {
  const float* mid=GLM_PR(2,const float*,GLM_S_MID);
  glm_ph_hc(GLM_PB(0),GLM_PW(1),F_FN,glm_src_f32{mid},GLM_PR(3,float*,GLM_S_HC),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_p1(const etx_ctx* c) {
  glm_ph_p1(GLM_PB(0),GLM_PW(1),GLM_PR(2,const glm_act*,GLM_HC*GLM_H),GLM_PR(3,const float*,GLM_S_HC),GLM_PR(4,float*,GLM_S_P1),c->cst[1],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_p2(const etx_ctx* c) {
  glm_ph_p2(GLM_PB(1),GLM_PW(2),GLM_PR(3,const float*,GLM_S_P1),GLM_PR(4,float*,GLM_S_P2),GLM_PR(5,float*,(uint64_t)c->cst[3]),
            glm53_step_load((const int*)c->args[0]+2*c->coord[0]),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_mla(const etx_ctx* c) {
  glm_ph_mla(GLM_PB(1),GLM_PW(2),GLM_PR(3,const float*,GLM_S_P1),GLM_PR(4,const float*,GLM_S_P2),GLM_PR(5,const float*,(uint64_t)c->cst[3]),
             GLM_PR(6,float*,GLM_S_ATT),glm53_step_load((const int*)c->args[0]+2*c->coord[0]),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_kda(const etx_ctx* c) {
  glm_ph_kda(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_P1),GLM_PR(3,float*,(uint64_t)c->cst[3]),GLM_PR(4,float*,GLM_S_ATT),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_o(const etx_ctx* c) {
  glm_ph_o(GLM_PB(0),GLM_PW(1),GLM_PR(2,const glm_act*,GLM_HC*GLM_H),GLM_PR(3,const float*,GLM_S_HC),GLM_PR(4,const float*,GLM_S_ATT),
           GLM_PR(5,float*,GLM_S_MID),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_rt(const etx_ctx* c) {
  glm_ph_rt(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_MID),GLM_PR(3,const float*,GLM_S_HC),GLM_PR(4,float*,GLM_S_RT),c->cst[2],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_up(const etx_ctx* c) {
  glm_ph_up(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_RT),GLM_PR(3,float*,GLM_S_ACT),GLM_PR(4,float*,GLM_S_SEL),c->cst[2],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_down(const etx_ctx* c) {
  glm_ph_down(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_ACT),GLM_PR(3,const float*,GLM_S_SEL),GLM_PR(4,const float*,GLM_S_MID),
              GLM_PR(5,const float*,GLM_S_HC),GLM_PR(6,glm_act*,GLM_HC*GLM_H),c->cst[2],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_lm(const etx_ctx* c) {
  glm_ph_head(GLM_PR(0,const glm_act*,GLM_HC*GLM_H),GLM_PB(1),GLM_PW(2),GLM_PB(3),GLM_PW(4),GLM_PR(5,float*,2*GLM_H+GLM_VOCAB),GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_argmax(const etx_ctx* c) {
  glm_ph_argmax(GLM_PR(0,const float*,2*GLM_H+GLM_VOCAB)+2*GLM_H,(int*)c->args[1]+c->coord[0],(float*)c->lds);
}

// Actual FFN TP/EP: cst = (tiles, moe, rank, world); combine uses (tiles, moe, world, 0).
#include "examples/glm53/parallel_phases.h"
extern "C" __device__ __attribute__((noinline)) void glm53_parallel_route(const etx_ctx* c) {
  glm_ph_parallel_route(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_RT),
                        GLM_PR(3,float*,GLM_S_SEL),c->cst[1],(float*)c->lds);
}
extern "C" __device__ __attribute__((noinline)) void glm53_parallel_dispatch(const etx_ctx* c) {
  const float* rt=GLM_PR(0,const float*,GLM_S_RT);const float* sel=GLM_PR(1,const float*,GLM_S_SEL);
  float* packet=GLM_PR(2,float*,GLM_S_RT+GLM_S_SEL);
  for(int j=GLM_LANE;j<GLM_S_RT+GLM_S_SEL;j+=GLM_LANES)packet[j]=j<GLM_S_RT?rt[j]:sel[j-GLM_S_RT];
}
extern "C" __device__ __attribute__((noinline)) void glm53_parallel_up(const etx_ctx* c) {
  const float* packet=GLM_PR(2,const float*,GLM_S_RT+GLM_S_SEL);
  glm_ph_parallel_up(GLM_PB(0),GLM_PW(1),packet,packet+GLM_S_RT,GLM_PR(3,float*,GLM_S_ACT),
                     c->cst[1],ETX_GLM_PARALLEL==1,c->cst[2],c->cst[3],0,GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_parallel_down(const etx_ctx* c) {
  const float* packet=GLM_PR(2,const float*,GLM_S_RT+GLM_S_SEL);
  glm_ph_parallel_down(GLM_PB(0),GLM_PW(1),GLM_PR(3,const float*,GLM_S_ACT),packet+GLM_S_RT,
                       GLM_PR(4,float*,(GLM_TOPK+1)*GLM_H),c->cst[1],ETX_GLM_PARALLEL==1,
                       c->cst[2],c->cst[3],0,GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_parallel_combine(const etx_ctx* c) {
  const float* outputs[8];
  for(int rank=0;rank<c->cst[2];++rank)outputs[rank]=GLM_PR(6+rank,const float*,(GLM_TOPK+1)*GLM_H);
  glm_ph_parallel_combine(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_SEL),outputs,
                          GLM_PR(3,const float*,GLM_S_MID),GLM_PR(4,const float*,GLM_S_HC),
                          GLM_PR(5,glm_act*,GLM_HC*GLM_H),c->cst[1],ETX_GLM_PARALLEL==1,c->cst[2],0,GLM_PT);
}

// Head-sharded attention: cst = (tiles, linear, rank, world). All request
// strides stay global; only local head intervals and compact local weights run.
#include "examples/glm53/attention_phases.h"
extern "C" __device__ __attribute__((noinline)) void glm53_attention_input(const etx_ctx* c) {
  glm_ph_attention_input(GLM_PB(0),GLM_PW(1),GLM_PR(2,const glm_act*,GLM_HC*GLM_H),
                         GLM_PR(3,const float*,GLM_S_HC),GLM_PR(4,float*,GLM_H),(float*)c->lds);
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_dispatch(const etx_ctx* c) {
  const float* in=GLM_PR(0,const float*,GLM_H);float* out=GLM_PR(1,float*,GLM_H);
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)out[j]=in[j];
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_p1(const etx_ctx* c) {
  glm_ph_attention_p1(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_H),GLM_PR(3,float*,GLM_S_P1),
                      c->cst[1],c->cst[2],c->cst[3],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_p2(const etx_ctx* c) {
  glm_ph_attention_p2(GLM_PB(1),GLM_PW(2),GLM_PR(3,const float*,GLM_S_P1),GLM_PR(4,float*,GLM_S_P2),
                      GLM_PR(5,float*,GLM_STATE_MLA),glm53_step_load((const int*)c->args[0]+2*c->coord[0]),
                      c->cst[2],c->cst[3],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_heads(const etx_ctx* c) {
  glm_ph_attention_heads(GLM_PB(1),GLM_PW(2),GLM_PR(3,const float*,GLM_S_P1),
                         GLM_PR(4,const float*,c->cst[1]?GLM_S_P1:GLM_S_P2),
                         GLM_PR(5,float*,c->cst[1]?GLM_STATE_KDA:GLM_STATE_MLA),GLM_PR(6,float*,GLM_S_ATT),
                         glm53_step_load((const int*)c->args[0]+2*c->coord[0]),c->cst[1],c->cst[2],c->cst[3],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_o(const etx_ctx* c) {
  glm_ph_attention_o(GLM_PB(0),GLM_PW(1),GLM_PR(2,const float*,GLM_S_ATT),GLM_PR(3,float*,GLM_H),
                     c->cst[1],c->cst[2],c->cst[3],GLM_PT);
}
extern "C" __device__ __attribute__((noinline)) void glm53_attention_combine(const etx_ctx* c) {
  const float* partials[8];for(int rank=0;rank<c->cst[1];++rank)partials[rank]=GLM_PR(5+rank,const float*,GLM_H);
  glm_ph_attention_combine(GLM_PB(0),GLM_PW(1),GLM_PR(2,const glm_act*,GLM_HC*GLM_H),
                           GLM_PR(3,const float*,GLM_S_HC),partials,GLM_PR(4,float*,GLM_S_MID),c->cst[1],GLM_PT);
}
