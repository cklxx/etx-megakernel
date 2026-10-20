#include "etx/abi.h"
#include "examples/glm53/native.h"
// Host-updated per-step controls cross both GPUs and may be read by CUs with
// only DOMAIN predecessors. SYSTEM atomic loads must bypass stale XCD caches.
static __device__ __forceinline__ int glm53_step_load(const int* p) {
  return __hip_atomic_load(p,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
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
