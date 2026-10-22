// GLM-5.3 layer phases: each layer is a chain of ETX task grids whose tiles
// spread over every CU (native.h runs a whole layer in one workgroup).
//
// The same source runs single-lane in the CPU checker (GLM53_CPU). There every
// block reduction degenerates to the sequential order of native.h, so the CPU
// phases reproduce glm_layer bit for bit; on the GPU only summation order
// differs. Every rounding point of native.h is kept.
//
// Phase grids of one layer (t = tile, T = tiles of the grid):
//   hc   (B,T)  raw mHC rows of the attention streams          -> hca
//   p1   (B,T)  collapse+norm prologue, projections from x     -> p1
//   p2   (B,T)  MLA: q_b / kv_b (+KV state) / indexer q        -> p2, state
//   att  (B,T)  MLA heads, or KDA heads (conv+recurrence)      -> att (KDA: state)
//   o    (B,T)  O rows, attention mix fused per column          -> mid
//   hcf  (B,T)  raw mHC rows of the FFN streams                 -> hcf
//   rt   (B,T)  collapse+norm, router logits                   -> rt (logits | x)
//   up   (B,T)  top-k, gate/up rows of the selected experts    -> act, sel
//   down (B,T)  down rows of every selected expert, final mix   -> out
#pragma once
#include <limits.h>
#include "examples/glm53/native.h"

#define GLM_HCN (2*GLM_HC+GLM_HC*GLM_HC)
#define GLM_R4(n) (((n)+3)/4*4)
#define GLM_MAX(a,b) ((a)>(b)?(a):(b))
// Per-request float strides of the phase tensors (model.py mirrors them; tests compare).
#define GLM_S_HC GLM_R4(GLM_HCN)
#define GLM_S_P1 GLM_R4(GLM_MAX(3*GLM_QDIM+2*GLM_LD+GLM_LH,GLM_QR+GLM_KVR+2*GLM_ID+GLM_IH))
#define GLM_S_P2 GLM_R4(GLM_NH*GLM_AD+GLM_IH*GLM_ID)
#define GLM_S_ATT GLM_R4(GLM_MAX(GLM_QDIM,GLM_NH*GLM_AD))
#define GLM_S_MID (GLM_HC*GLM_H)
#define GLM_S_RT (GLM_R4(GLM_EXPERTS)+GLM_H)
#define GLM_S_ACT ((GLM_TOPK+1)*GLM_INTERMAX)
#define GLM_S_SEL GLM_R4(2*GLM_TOPK)
// Shared memory of one tile, in floats: 16 reduction words, 32 mHC coefficients, then the phase's own area.
#define GLM_LDS_HEAD 48
#define GLM_LDS_FLOATS (GLM_LDS_HEAD+GLM_MAX(GLM_H+2*GLM_R4(GLM_TOPK)+2*GLM_R4(GLM_EXPERTS), \
  GLM_MAX(GLM_R4(GLM_QR)+GLM_R4(GLM_KVR),GLM_MAX(GLM_R4(3*GLM_MAXLEN+GLM_ID)+5*GLM_R4(GLM_AD),10*GLM_R4(GLM_LD)))))
static_assert(GLM_HCN<=32,"mHC coefficients exceed the reserved shared words");

#ifdef GLM53_CPU
#define GLM_WLANE 0
#define GLM_WID 0
#define GLM_NW 1
#else
#define GLM_WLANE ((int)(threadIdx.x&63))
#define GLM_WID ((int)(threadIdx.x>>6))
#define GLM_NW ((int)(blockDim.x>>6))
#endif

// ---- reductions. Callers pass per-thread partials of a GLM_LANE-strided loop; on
// the CPU the single partial already is the sequential sum.
GLM_DEV float glm_xsum(float v,int width) {
#ifndef GLM53_CPU
  for(int o=width>>1;o>0;o>>=1)v+=__shfl_xor(v,o,64);
#endif
  return v;
}
GLM_DEV float glm_bsum(float v,float* red) {
#ifdef GLM53_CPU
  return v;
#else
  v=glm_xsum(v,64);__syncthreads();if(GLM_WLANE==0)red[GLM_WID]=v;__syncthreads();
  float s=0;for(int i=0;i<GLM_NW;++i)s+=red[i];return s;
#endif
}
GLM_DEV float glm_bmax(float v,float* red) {
#ifdef GLM53_CPU
  return v;
#else
  for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));
  __syncthreads();if(GLM_WLANE==0)red[GLM_WID]=v;__syncthreads();
  float m=red[0];for(int i=1;i<GLM_NW;++i)m=fmaxf(m,red[i]);return m;
#endif
}
// Index of the largest value, lowest index on ties: the sequential strict '>' scan.
GLM_DEV int glm_bargmax(float v,int i,float* red) {
#ifdef GLM53_CPU
  (void)v;(void)red;return i;
#else
  for(int o=32;o>0;o>>=1){float ov=__shfl_xor(v,o,64);int oi=__shfl_xor(i,o,64);if(ov>v||(ov==v&&oi<i)){v=ov;i=oi;}}
  int* ri=(int*)(red+8);__syncthreads();if(GLM_WLANE==0){red[GLM_WID]=v;ri[GLM_WID]=i;}__syncthreads();
  v=red[0];i=ri[0];for(int k=1;k<GLM_NW;++k)if(red[k]>v||(red[k]==v&&ri[k]<i)){v=red[k];i=ri[k];}
  return i;
#endif
}

// ---- activation sources: BF16 layer inputs and FP32 mixed streams, with 8-wide loads on the GPU.
struct glm_src_act { const glm_act* p; GLM_DEV float operator()(int j) const { return glm_get(p[j]); } };
struct glm_src_f32 { const float* p; GLM_DEV float operator()(int j) const { return p[j]; } };
#ifndef GLM53_CPU
// Phases issue every load of a thread before using any: latency, not bandwidth, bounds these small reads.
#define GLM_VT 256
GLM_DEV void glm_load8(const glm_src_act& s,int j,float* o){
#if GLM_BF16
  const uint4 u=*(const uint4*)(s.p+j);const uint32_t w[4]={u.x,u.y,u.z,u.w};
  for(int i=0;i<4;++i){o[2*i]=__uint_as_float(w[i]<<16);o[2*i+1]=__uint_as_float(w[i]&0xffff0000u);}
#else
  for(int i=0;i<8;++i)o[i]=s.p[j+i];
#endif
}
GLM_DEV void glm_load8(const glm_src_f32& s,int j,float* o){
  const float4 a=*(const float4*)(s.p+j),b=*(const float4*)(s.p+j+4);
  o[0]=a.x;o[1]=a.y;o[2]=a.z;o[3]=a.w;o[4]=b.x;o[5]=b.y;o[6]=b.z;o[7]=b.w;
}
template<class S> GLM_DEV bool glm_vec_src(const S& s){return blockDim.x==GLM_VT&&((uintptr_t)s.p&15)==0;}
#endif

// ---- dot products of weight rows with float vectors.
// Weight kinds a call site compiles the vector path for (bit per kind); other kinds take the
// scalar fallback. Each kind is a fully unrolled body: instantiating all three at every call site
// made the phase functions 30-65 KB, fetched cold by every tile (measured 2026-10-22).
#define GLM_KBF16 1
#define GLM_KF32 2
#define GLM_KFP8 4
#define GLM_KALL 7
GLM_DEV int glm_esize(const glm_weight& w){return w.kind==1?2:w.kind==2?4:1;}
// 16-byte vector loads need aligned rows and, for FP8, scale blocks that hold whole vectors.
GLM_DEV bool glm_vec(const glm_weight& w){
  return ((uint64_t)w.cols*glm_esize(w))%16==0 && w.offset%16==0 && (w.kind!=3 || (w.block>0 && w.block%16==0));
}
// Lanes that share one row: the whole wave for rows of 1 KiB and more, else the
// largest power of two whose 16-byte vectors fit the row.
template<int KM=GLM_KALL> GLM_DEV int glm_group(const glm_weight& w){
#ifdef GLM53_CPU
  (void)w;return 1;
#else
  if(!glm_vec(w)||!(KM&(w.kind==3?GLM_KFP8:w.kind==1?GLM_KBF16:GLM_KF32)))return 1;
  const uint64_t rb=(uint64_t)w.cols*glm_esize(w);int s=64;while(s>1&&(uint64_t)s*16>rb)s>>=1;return s;
#endif
}
#ifndef GLM53_CPU
// Bit-exact glm_fp8 (E4M3FN) without ldexpf.
GLM_DEV float glm_fp8f(uint32_t x){
  const uint32_t em=x&127u;
  const float v=em>=8u?__uint_as_float((em<<20)+(120u<<23)):(float)em*0x1p-9f;
  return __uint_as_float(__float_as_uint(v)|((x&128u)<<24));
}
template<int K> GLM_DEV float glm_dot16(uint4 v,float scale,const float* x,bool aligned){
  constexpr int E=K==3?16:K==1?8:4;float xv[E];
  if(aligned){for(int i=0;i<E/4;++i){float4 f=((const float4*)x)[i];xv[4*i]=f.x;xv[4*i+1]=f.y;xv[4*i+2]=f.z;xv[4*i+3]=f.w;}}
  else for(int i=0;i<E;++i)xv[i]=x[i];
  const uint32_t u[4]={v.x,v.y,v.z,v.w};float s=0;
  if(K==3){for(int i=0;i<4;++i)for(int k=0;k<4;++k)s+=glm_bf(glm_fp8f((u[i]>>(8*k))&255u)*scale)*xv[4*i+k];}
  else if(K==1){for(int i=0;i<4;++i){s+=__uint_as_float(u[i]<<16)*xv[2*i];s+=__uint_as_float(u[i]&0xffff0000u)*xv[2*i+1];}}
  else for(int i=0;i<4;++i)s+=__uint_as_float(u[i])*xv[i];
  return s;
}
template<int G,int K> GLM_DEV void glm_dots_vec(const uint8_t* b,const glm_weight* const* w,const int* row,const float* const* x,float* acc,int S,int sl){
  constexpr int ES=K==3?1:K==1?2:4;
  const int cols=w[0]->cols,block=w[0]->block;const uint32_t rb=(uint32_t)cols*ES;
  const uint8_t* p[G];const float* sc[G];bool al[G];
  // Every loop over the G rows is unrolled: a rolled loop indexes p/sc/v dynamically and puts them in scratch.
#pragma unroll
  for(int g=0;g<G;++g){acc[g]=0;p[g]=b+w[g]->offset+(uint64_t)row[g]*rb;al[g]=((uintptr_t)x[g]&15)==0;
    sc[g]=K==3?(const float*)(b+w[g]->scale)+(uint64_t)(row[g]/block)*((cols+block-1)/block):nullptr;}
  // Up to 16 vector loads per lane in flight: NB chunks of all G rows are loaded before any is used.
  constexpr int NB=G==3?2:G>=16?1:16/G;const uint32_t step=(uint32_t)S*16;   // G=3: expert down rows are 2 chunks
  for(uint32_t base=(uint32_t)sl*16;base<rb;base+=NB*step){
    uint4 v[NB][G];
#pragma unroll
    for(int n=0;n<NB;++n){const uint32_t off=base+n*step;
#pragma unroll
      for(int g=0;g<G;++g)v[n][g]=off<rb?*(const uint4*)(p[g]+off):make_uint4(0,0,0,0);}
#pragma unroll
    for(int n=0;n<NB;++n){const uint32_t off=base+n*step;if(off>=rb)break;const int c=(int)(off/ES);
#pragma unroll
      for(int g=0;g<G;++g)acc[g]+=glm_dot16<K>(v[n][g],K==3?sc[g][c/block]:1.0f,x[g]+c,al[g]);}
  }
#pragma unroll
  for(int g=0;g<G;++g)acc[g]=glm_xsum(acc[g],S);
}
#endif
// acc[g] = dot(row[g] of *w[g], x[g]) for G rows computed by the S lanes of one group
// (lane sl of S); every lane of the group receives the sums. Rows of one call share kind/cols.
template<int G,int KM=GLM_KALL> GLM_DEV void glm_dots(const uint8_t* b,const glm_weight* const* w,const int* row,const float* const* x,float* acc,int S,int sl){
#ifndef GLM53_CPU
  bool same=glm_vec(*w[0]);
  for(int g=1;g<G;++g)same=same&&w[g]->kind==w[0]->kind&&w[g]->cols==w[0]->cols&&w[g]->block==w[0]->block&&glm_vec(*w[g]);
  if(same){
    if constexpr((KM&GLM_KFP8)!=0)if(w[0]->kind==3){glm_dots_vec<G,3>(b,w,row,x,acc,S,sl);return;}
    if constexpr((KM&GLM_KBF16)!=0)if(w[0]->kind==1){glm_dots_vec<G,1>(b,w,row,x,acc,S,sl);return;}
    if constexpr((KM&GLM_KF32)!=0)if(w[0]->kind==2){glm_dots_vec<G,2>(b,w,row,x,acc,S,sl);return;}
  }
#endif
  // Scalar fallback (and the CPU): one lane walks the row in column order, as glm_mv.
  for(int g=0;g<G;++g){float s=0;if(sl==0)for(int c=0;c<w[g]->cols;++c)s+=glm_w(b,*w[g],row[g],c)*x[g][c];acc[g]=glm_xsum(s,S);}
}
// Units [u0,u1) round-robin over groups of S lanes; f(unit, lane-in-group).
template<class F> GLM_DEV void glm_each(int u0,int u1,int S,F f){
#ifdef GLM53_CPU
  (void)S;for(int u=u0;u<u1;++u)f(u,0);
#else
  const int per=64/S,slots=GLM_NW*per,slot=GLM_WID*per+GLM_WLANE/S,sl=GLM_WLANE%S;
  for(int u=u0+slot;u<u1;u+=slots)f(u,sl);
#endif
}
// out(r, dot(row r, x)) for rows [lo,hi) of w, four rows per group.
template<int KM=GLM_KALL,class Out> GLM_DEV void glm_rows(const uint8_t* b,const glm_weight& w,const float* x,int lo,int hi,Out out){
  if(hi<=lo)return;
  const int S=glm_group<KM>(w);const glm_weight* ws[4]={&w,&w,&w,&w};const float* xs[4]={x,x,x,x};
  glm_each(0,(hi-lo+3)/4,S,[&](int u,int sl){
    int row[4];for(int g=0;g<4;++g){int r=lo+4*u+g;row[g]=r<hi?r:lo+4*u;}
    float acc[4];glm_dots<4,KM>(b,ws,row,xs,acc,S,sl);
    if(sl==0)for(int g=0;g<4;++g)if(lo+4*u+g<hi)out(lo+4*u+g,acc[g]);
  });
}
// Rows [lo,hi) of segment k when tile t of T takes an equal share of the payload bytes of all n segments.
GLM_DEV void glm_share(const uint64_t* bytes,const int* rows,int n,int k,int t,int T,int& lo,int& hi){
  lo=hi=0;if(rows[k]<=0||bytes[k]==0)return;
  uint64_t total=0,start=0;for(int i=0;i<n;++i){if(i==k)start=total;total+=bytes[i]*(uint64_t)(rows[i]>0?rows[i]:0);}
  auto first=[&](uint64_t e)->int{if(e<=start)return 0;uint64_t r=(e-start+bytes[k]-1)/bytes[k];return r>(uint64_t)rows[k]?rows[k]:(int)r;};
  lo=first(total*(uint64_t)t/T);hi=first(total*(uint64_t)(t+1)/T);
}
struct glm_seg { int slot; const float* x; float* y; };
// Rounded projections (glm_mv) of several weights, split over the tiles by payload bytes.
template<int KM=GLM_KALL> GLM_DEV void glm_project(const uint8_t* b,const glm_weight* w,const glm_seg* seg,int n,int t,int T){
  uint64_t bytes[8];int rows[8];
  for(int i=0;i<n;++i){const glm_weight& m=w[seg[i].slot];rows[i]=m.rows;bytes[i]=(uint64_t)m.cols*glm_esize(m);}
  for(int k=0;k<n;++k){int lo,hi;glm_share(bytes,rows,n,k,t,T,lo,hi);float* y=seg[k].y;
    glm_rows<KM>(b,w[seg[k].slot],seg[k].x,lo,hi,[&](int r,float v){y[r]=glm_bf(v);});}
}

// ---- mHC, norms and mixing, with native.h's arithmetic.
// raw[r] = sum_c s[c]*inv*fn[r][c] for the mHC rows of tile t (glm_hc before its coefficients).
template<class S> GLM_DEV void glm_ph_hc(const uint8_t* b,const glm_weight* w,int base,S s,float* raw,int t,int T,float* lds){
#ifndef GLM53_CPU
  constexpr int N=GLM_HC*GLM_H/(8*GLM_VT);
  const glm_weight& f=w[base];
  if(N>0&&GLM_HC*GLM_H%(8*GLM_VT)==0&&glm_vec_src(s)&&f.kind==1&&f.offset%16==0&&f.cols==GLM_HC*GLM_H){
    // Each thread keeps its 8*N stream values in registers for both the norm and every row.
    float v[N][8];for(int k=0;k<N;++k)glm_load8(s,(k*GLM_VT+(int)threadIdx.x)*8,v[k]);
    float part=0;for(int k=0;k<N;++k)for(int i=0;i<8;++i)part+=v[k][i]*v[k][i];
    const float inv=1.0f/sqrtf(glm_bsum(part,lds)/(GLM_HC*GLM_H)+GLM_EPS);
    for(int r=t*GLM_HCN/T;r<(t+1)*GLM_HCN/T;++r){
      const uint4* row=(const uint4*)(b+f.offset+(uint64_t)r*f.cols*2);uint4 u[N];
      for(int k=0;k<N;++k)u[k]=row[k*GLM_VT+threadIdx.x];
      float a=0;
      for(int k=0;k<N;++k){const uint32_t q[4]={u[k].x,u[k].y,u[k].z,u[k].w};
        for(int i=0;i<4;++i){a+=v[k][2*i]*inv*__uint_as_float(q[i]<<16);a+=v[k][2*i+1]*inv*__uint_as_float(q[i]&0xffff0000u);}}
      a=glm_bsum(a,lds);if(GLM_LANE==0)raw[r]=a;}
    return;
  }
#endif
  float part=0;for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES){float v=s(j);part+=v*v;}
  const float inv=1.0f/sqrtf(glm_bsum(part,lds)/(GLM_HC*GLM_H)+GLM_EPS);
  for(int r=t*GLM_HCN/T;r<(t+1)*GLM_HCN/T;++r){float a=0;
    for(int c=GLM_LANE;c<GLM_HC*GLM_H;c+=GLM_LANES)a+=s(c)*inv*glm_w(b,w[base],r,c);
    a=glm_bsum(a,lds);if(GLM_LANE==0)raw[r]=a;}
}
// hc = post[HC] | comb[HC*HC] | pre[HC] from the raw rows (glm_hc's coefficient part).
// zs: GLM_HCN floats of scratch shared memory, free until the caller's own area is used.
GLM_DEV void glm_hc_coeffs(const uint8_t* b,const glm_weight* w,int base,const float* raw,float* hc,float* zs){
  // One lane per row loads its inputs: the row values arrive in one round trip instead of 3*GLM_HCN.
  for(int r=GLM_LANE;r<GLM_HCN;r+=GLM_LANES)zs[r]=raw[r]*glm_w(b,w[base+2],r<GLM_HC?0:r<2*GLM_HC?1:2)+glm_w(b,w[base+1],r);
  GLM_SYNC();
#ifndef GLM53_CPU
  if(GLM_HC*GLM_HC<=64){
    // One lane per comb entry of wave 0 (a single lane spent ~25 us on the dependent divisions).
    // Each lane reads its row/column through shuffles and sums them in the sequential order.
    if(threadIdx.x<64){
      const int l=threadIdx.x,r=l/GLM_HC%GLM_HC,c=l%GLM_HC;const bool on=l<GLM_HC*GLM_HC;
      const float z=on?zs[2*GLM_HC+l]:0.0f;
      float mx=-INFINITY;for(int cc=0;cc<GLM_HC;++cc)mx=fmaxf(mx,__shfl(z,r*GLM_HC+cc,64));
      const float e=expf(z-mx);float den=0;for(int cc=0;cc<GLM_HC;++cc)den+=__shfl(e,r*GLM_HC+cc,64);
      float m=e/den+GLM_HCEPS;
      for(int it=0;it<GLM_SINK;++it){
        if(it){den=GLM_HCEPS;for(int cc=0;cc<GLM_HC;++cc)den+=__shfl(m,r*GLM_HC+cc,64);m/=den;}
        den=GLM_HCEPS;for(int rr=0;rr<GLM_HC;++rr)den+=__shfl(m,rr*GLM_HC+c,64);m/=den;
      }
      if(on)hc[GLM_HC+l]=m;
      if(l<GLM_HC){hc[GLM_HC+GLM_HC*GLM_HC+l]=glm_sig(zs[l])+GLM_HCEPS;hc[l]=2*glm_sig(zs[GLM_HC+l]);}
    }
    GLM_SYNC();return;
  }
#endif
  if(GLM_LANE==0){
    float* post=hc;float* comb=post+GLM_HC;float* pre=comb+GLM_HC*GLM_HC;
    for(int r=0;r<GLM_HCN;++r){
      float z=zs[r];
      if(r<GLM_HC)pre[r]=glm_sig(z)+GLM_HCEPS;
      else if(r<2*GLM_HC)post[r-GLM_HC]=2*glm_sig(z);
      else comb[r-2*GLM_HC]=z;
    }
    for(int r=0;r<GLM_HC;++r){float mx=-INFINITY,den=0;
      for(int c=0;c<GLM_HC;++c)mx=fmaxf(mx,comb[r*GLM_HC+c]);
      for(int c=0;c<GLM_HC;++c){float v=expf(comb[r*GLM_HC+c]-mx);comb[r*GLM_HC+c]=v;den+=v;}
      for(int c=0;c<GLM_HC;++c)comb[r*GLM_HC+c]=comb[r*GLM_HC+c]/den+GLM_HCEPS;
    }
    for(int it=0;it<GLM_SINK;++it){
      if(it)for(int r=0;r<GLM_HC;++r){float den=GLM_HCEPS;for(int c=0;c<GLM_HC;++c)den+=comb[r*GLM_HC+c];for(int c=0;c<GLM_HC;++c)comb[r*GLM_HC+c]/=den;}
      for(int c=0;c<GLM_HC;++c){float den=GLM_HCEPS;for(int r=0;r<GLM_HC;++r)den+=comb[r*GLM_HC+c];for(int r=0;r<GLM_HC;++r)comb[r*GLM_HC+c]/=den;}
    }
  } GLM_SYNC();
}
// y = glm_norm(x) for n values (x may alias y).
GLM_DEV void glm_pnorm(const uint8_t* b,const glm_weight& w,const float* x,float* y,int n,float* red){
  float part=0;for(int i=GLM_LANE;i<n;i+=GLM_LANES)part+=x[i]*x[i];
  const float inv=1.0f/sqrtf(glm_bsum(part,red)/n+GLM_EPS);
#ifndef GLM53_CPU
  if(w.kind==1&&n%8==0&&w.offset%16==0){
    // BF16 weights, 8 per 16-byte load, all of a thread's loads issued before use.
    const uint4* wv=(const uint4*)(b+w.offset);
    for(int j=threadIdx.x*8;j<n;j+=blockDim.x*8){const uint4 u=wv[j/8];const uint32_t q[4]={u.x,u.y,u.z,u.w};
      for(int i=0;i<4;++i){y[j+2*i]=glm_bf(glm_bf(x[j+2*i]*inv)*__uint_as_float(q[i]<<16));
                           y[j+2*i+1]=glm_bf(glm_bf(x[j+2*i+1]*inv)*__uint_as_float(q[i]&0xffff0000u));}}
    GLM_SYNC();return;
  }
#endif
  for(int i=GLM_LANE;i<n;i+=GLM_LANES)y[i]=glm_bf(glm_bf(x[i]*inv)*glm_w(b,w,i));
  GLM_SYNC();
}
// x = glm_norm(collapse(s, pre)): the end of glm_hc followed by the layer norm.
template<class S> GLM_DEV void glm_collapse_norm(const uint8_t* b,const glm_weight& nw,S s,const float* pre,float* x,float* red){
#ifndef GLM53_CPU
  constexpr int N=GLM_H/(8*GLM_VT);
  if(N>0&&GLM_H%(8*GLM_VT)==0&&glm_vec_src(s)){
    float v[N][GLM_HC][8];
    for(int k=0;k<N;++k)for(int c=0;c<GLM_HC;++c)glm_load8(s,c*GLM_H+(k*GLM_VT+(int)threadIdx.x)*8,v[k][c]);
    for(int k=0;k<N;++k)for(int i=0;i<8;++i){float sum=0;for(int c=0;c<GLM_HC;++c)sum+=pre[c]*v[k][c][i];x[(k*GLM_VT+threadIdx.x)*8+i]=glm_bf(sum);}
    GLM_SYNC();glm_pnorm(b,nw,x,x,GLM_H,red);return;
  }
#endif
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){float sum=0;for(int c=0;c<GLM_HC;++c)sum+=pre[c]*s(c*GLM_H+j);x[j]=glm_bf(sum);}
  GLM_SYNC();glm_pnorm(b,nw,x,x,GLM_H,red);
}
// Mixed stream c at column k (glm_mix).
template<class S> GLM_DEV float glm_mix1(const float* hc,S s,float y,int c,int k){
  float a=0;for(int r=0;r<GLM_HC;++r)a+=glm_bf(hc[GLM_HC+r*GLM_HC+c])*s(r*GLM_H+k);
  return glm_bf(glm_bf(glm_bf(hc[c])*y)+glm_bf(a));
}

// ---- phases. lds: GLM_LDS_FLOATS floats; [0,16) reductions, [16,48) mHC coefficients.
// Projections from the attention input: KDA q|k|v|f_a|b|g_a, MLA q_a|kv_a|ik|ig|iw.
GLM_DEV void glm_ph_p1(const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,float* p1,bool linear,int t,int T,float* lds){
  float* hc=lds+16;float* x=lds+GLM_LDS_HEAD;
  glm_hc_coeffs(b,w,A_FN,hca,hc,x);
  glm_collapse_norm(b,w[N_ATTN],glm_src_act{in},hc+GLM_HC+GLM_HC*GLM_HC,x,lds);
  if(linear){
    const glm_seg s[6]={{Q,x,p1},{K,x,p1+GLM_QDIM},{V,x,p1+2*GLM_QDIM},{FA,x,p1+3*GLM_QDIM},
                        {BETA,x,p1+3*GLM_QDIM+GLM_LD},{GA,x,p1+3*GLM_QDIM+GLM_LD+GLM_LH}};
    glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,6,t,T);
  } else {
    const glm_seg s[5]={{QA,x,p1},{KVA,x,p1+GLM_QR},{IK,x,p1+GLM_QR+GLM_KVR},{IG,x,p1+GLM_QR+GLM_KVR+GLM_ID},
                        {IW,x,p1+GLM_QR+GLM_KVR+2*GLM_ID}};
    glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,5,t,T);
  }
}
// MLA second projections: q (q_b), the position's KV row (kv_b) and indexer row, indexer q.
GLM_DEV void glm_ph_p2(const uint8_t* b,const glm_weight* w,const float* p1,float* p2,float* state,int pos,int t,int T,float* lds){
  float* qr=lds+GLM_LDS_HEAD;float* kv=qr+GLM_R4(GLM_QR);
  glm_pnorm(b,w[QAN],p1,qr,GLM_QR,lds);glm_pnorm(b,w[KVAN],p1+GLM_QR,kv,GLM_KVR,lds);
  const int stride=2*GLM_NH*GLM_AD+2*GLM_ID;float* st=state+(uint64_t)pos*stride;
  const glm_seg s[3]={{QB,qr,p2},{KVB,kv,st},{IQ,qr,p2+GLM_NH*GLM_AD}};
  glm_project<GLM_KBF16|GLM_KFP8>(b,w,s,3,t,T);
  if(t==0){
    const float* ik=p1+GLM_QR+GLM_KVR;const float* ig=ik+GLM_ID;
    // Indexer key layernorm with block sums (one lane took 1.1-1.7 ms of this tile, measured).
    float part=0;for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=ik[i];
    const float mean=glm_bsum(part,lds)/GLM_ID;
    part=0;for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=(ik[i]-mean)*(ik[i]-mean);
    const float inv=1/sqrtf(glm_bsum(part,lds)/GLM_ID+1e-6f);
    for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)st[2*GLM_NH*GLM_AD+i]=glm_bf((ik[i]-mean)*inv*glm_w(b,w[IKN],i)+glm_w(b,w[IKB],i));
    for(int j=GLM_LANE;j<GLM_ID;j+=GLM_LANES)st[2*GLM_NH*GLM_AD+GLM_ID+j]=ig[j];
  }
}
// MLA attention for the heads of tile t (glm_mla after the projections).
GLM_DEV void glm_ph_mla(const uint8_t* b,const glm_weight* w,const float* p1,const float* p2,const float* state,float* att,int pos,int t,int T,float* lds,int head_begin=0,int head_end=GLM_NH){
  float* a=lds+GLM_LDS_HEAD;float* chosen=a+GLM_MAXLEN;float* score=chosen+GLM_MAXLEN;float* pool=score+GLM_MAXLEN;
  const int stride=2*GLM_NH*GLM_AD+2*GLM_ID;
  const int np=(pos+1)/GLM_POOL,take=np<GLM_INDEX_TOPK/GLM_POOL?np:GLM_INDEX_TOPK/GLM_POOL;
  if(take==np){
    // Every complete pool fits the budget and the tail is always kept: all positions are chosen.
    for(int j=GLM_LANE;j<=pos;j+=GLM_LANES)chosen[j]=1;
  } else if(GLM_LANE==0){
    const float* iq=p2+GLM_NH*GLM_AD;const float* iw=p1+GLM_QR+GLM_KVR+2*GLM_ID;
    for(int p=0;p<np;++p){
      for(int d=0;d<GLM_ID;++d){float mx=-INFINITY,den=0,num=0;
        for(int z=0;z<GLM_POOL;++z)mx=fmaxf(mx,state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d));
        for(int z=0;z<GLM_POOL;++z)den+=expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d)-mx);
        for(int z=0;z<GLM_POOL;++z){float prob=glm_bf(expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d)-mx)/den);num+=glm_bf(prob*state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+d]);}pool[d]=glm_bf(num);
      }
      float sum=0;for(int h=0;h<GLM_IH;++h){float dot=0;for(int d=0;d<GLM_ID;++d)dot+=iq[h*GLM_ID+d]*pool[d];sum+=iw[h]/sqrtf((float)GLM_IH)*fmaxf(0,dot/sqrtf((float)GLM_ID));}score[p]=sum;
    }
    for(int j=0;j<=pos;++j)chosen[j]=0;
    for(int z=0;z<take;++z){int best=-1;float mx=-INFINITY;for(int p=0;p<np;++p)if(score[p]>mx){mx=score[p];best=p;}score[best]=-INFINITY;for(int j=0;j<GLM_POOL;++j)chosen[best*GLM_POOL+j]=1;}
    for(int j=np*GLM_POOL;j<=pos;++j)chosen[j]=1;
  }
  GLM_SYNC();
#ifdef GLM53_CPU
  const int S=1;
#else
  int S=64;while(S>1&&S>GLM_AD)S>>=1;
#endif
#ifndef GLM53_CPU
  if(GLM_AD==4*64&&blockDim.x==GLM_VT){
    // Scores: one thread per key, the key's AD values loaded as float4s all at once. Output: a
    // thread owns 4 value dims for one of G key groups; the G partial sums are added in group order.
    float* qs=lds+GLM_LDS_HEAD+GLM_R4(3*GLM_MAXLEN+GLM_ID);float* part=qs+GLM_R4(GLM_AD);
    constexpr int DQ=GLM_AD/4,G=GLM_VT/DQ;
    for(int h=head_begin+t*(head_end-head_begin)/T;h<head_begin+(t+1)*(head_end-head_begin)/T;++h){
      for(int d=threadIdx.x;d<GLM_AD;d+=GLM_VT)qs[d]=p2[h*GLM_AD+d];
      GLM_SYNC();
      float m=-INFINITY;
      for(int j=threadIdx.x;j<=pos;j+=GLM_VT){
        const float4* kk=(const float4*)(state+(uint64_t)j*stride+h*2*GLM_AD);float dot=0;
        for(int i=0;i<DQ;++i){const float4 kv=kk[i];dot+=qs[4*i]*kv.x+qs[4*i+1]*kv.y+qs[4*i+2]*kv.z+qs[4*i+3]*kv.w;}
        a[j]=chosen[j]?glm_bf(glm_bf(dot)/sqrtf((float)GLM_AD)):-INFINITY;m=fmaxf(m,a[j]);
      }
      const float mx=glm_bmax(m,lds);
      float sum=0;for(int j=threadIdx.x;j<=pos;j+=GLM_VT){a[j]=expf(a[j]-mx);sum+=a[j];}
      const float den=glm_bsum(sum,lds);
      for(int j=threadIdx.x;j<=pos;j+=GLM_VT)a[j]=glm_bf(a[j]/den);
      GLM_SYNC();
      const int dq=threadIdx.x%DQ,grp=threadIdx.x/DQ;float4 acc=make_float4(0,0,0,0);
      const float* vbase=state+h*2*GLM_AD+GLM_AD+4*dq;
      int j=grp;
      for(;j+3*G<=pos;j+=4*G){float4 x[4];float p[4];
        for(int u=0;u<4;++u){x[u]=*(const float4*)(vbase+(uint64_t)(j+u*G)*stride);p[u]=a[j+u*G];}
        for(int u=0;u<4;++u){acc.x+=p[u]*x[u].x;acc.y+=p[u]*x[u].y;acc.z+=p[u]*x[u].z;acc.w+=p[u]*x[u].w;}}
      for(;j<=pos;j+=G){const float4 x=*(const float4*)(vbase+(uint64_t)j*stride);const float p=a[j];
        acc.x+=p*x.x;acc.y+=p*x.y;acc.z+=p*x.z;acc.w+=p*x.w;}
      ((float4*)part)[grp*DQ+dq]=acc;
      GLM_SYNC();
      for(int d=threadIdx.x;d<GLM_AD;d+=GLM_VT){float r=0;for(int gg=0;gg<G;++gg)r+=part[gg*GLM_AD+d];att[h*GLM_AD+d]=glm_bf(r);}
      GLM_SYNC();
    }
    return;
  }
#endif
  for(int h=head_begin+t*(head_end-head_begin)/T;h<head_begin+(t+1)*(head_end-head_begin)/T;++h){
    const float* q=p2+h*GLM_AD;
    // One group of lanes per key position; scores rounded like eager BF16 matmul.
    glm_each(0,pos+1,S,[&](int j,int sl){
      const float* k=state+(uint64_t)j*stride+h*2*GLM_AD;float dot=0;
      for(int d=sl;d<GLM_AD;d+=S)dot+=q[d]*k[d];
      dot=glm_xsum(dot,S);
      if(sl==0)a[j]=chosen[j]?glm_bf(glm_bf(dot)/sqrtf((float)GLM_AD)):-INFINITY;
    });
    GLM_SYNC();
    float m=-INFINITY;for(int j=GLM_LANE;j<=pos;j+=GLM_LANES)m=fmaxf(m,a[j]);
    const float mx=glm_bmax(m,lds);
    float part=0;for(int j=GLM_LANE;j<=pos;j+=GLM_LANES){a[j]=expf(a[j]-mx);part+=a[j];}
    const float den=glm_bsum(part,lds);
    for(int j=GLM_LANE;j<=pos;j+=GLM_LANES)a[j]=glm_bf(a[j]/den);
    GLM_SYNC();
    for(int d=GLM_LANE;d<GLM_AD;d+=GLM_LANES){float sum=0;
      for(int j=0;j<=pos;++j)sum+=a[j]*state[(uint64_t)j*stride+h*2*GLM_AD+GLM_AD+d];att[h*GLM_AD+d]=glm_bf(sum);}
    GLM_SYNC();
  }
}
// KDA heads of tile t: short conv, forget gate, delta-rule recurrence, gated output norm.
GLM_DEV void glm_ph_kda(const uint8_t* b,const glm_weight* w,const float* p1,float* state,float* att,int t,int T,float* lds,int head_begin=0,int head_end=GLM_LH,bool sharded=false){
  const int L=GLM_R4(GLM_LD);
  float* q=lds+GLM_LDS_HEAD;float* k=q+L;float* v=k+L;float* g=v+L;float* gate=g+L;float* o=gate+L;float* fs=o+L;float* gs=fs+L;
  const float* small_f=p1+3*GLM_QDIM;const float* braw=small_f+GLM_LD;const float* small_g=braw+GLM_LH;
  float* conv=state;float* rec=conv+3*GLM_QDIM*GLM_CONV;
  for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){fs[i]=small_f[i];gs[i]=small_g[i];}
  GLM_SYNC();
  // Bounds partition heads, never the full-global activation/state addressing.
  // sharded selects compact local weight rows; false also permits bounded full weights.
  for(int h=head_begin+t*(head_end-head_begin)/T;h<head_begin+(t+1)*(head_end-head_begin)/T;++h){const int off=h*GLM_LD;
    const int localoff=sharded?off-head_begin*GLM_LD:off;
    for(int i=GLM_LANE;i<3*GLM_LD;i+=GLM_LANES){const int part=i/GLM_LD,n=i%GLM_LD,j=part*GLM_QDIM+off+n;
      float* hist=conv+(uint64_t)j*GLM_CONV;for(int z=0;z<GLM_CONV-1;++z)hist[z]=hist[z+1];hist[GLM_CONV-1]=p1[j];
      const int row=sharded?part*((head_end-head_begin)*GLM_LD)+(j%GLM_QDIM-head_begin*GLM_LD):j;
      float sum=0;for(int z=0;z<GLM_CONV;++z)sum+=hist[z]*glm_w(b,w[CONV],row,z);
      // conv1d returns its weight dtype before SiLU (FP32 in production).
      if(w[CONV].kind==1)sum=glm_bf(sum);
      (part==0?q:part==1?k:v)[n]=glm_bf(sum*glm_sig(sum));
    }
    glm_rows<GLM_KBF16>(b,w[FB],fs,localoff,localoff+GLM_LD,[&](int r,float y){g[r-localoff]=glm_bf(y);});
    glm_rows<GLM_KBF16>(b,w[GB],gs,localoff,localoff+GLM_LD,[&](int r,float y){gate[r-localoff]=glm_bf(y);});
    GLM_SYNC();
    for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){const int j=off+i;
      g[i]=expf(GLM_LOWER*glm_sig(expf(glm_w(b,w[ALOG],sharded?h-head_begin:j/GLM_LD))*(g[i]+glm_w(b,w[DT],sharded?localoff+i:j))));}
    const float beta=glm_bf(glm_sig(braw[h]));
    float pq=GLM_LANE==0?1e-6f:0.0f,pk=pq;
    for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){pq+=q[i]*q[i];pk+=k[i]*k[i];}
    const float nq=sqrtf(glm_bsum(pq,lds))*sqrtf((float)GLM_LD),nk=sqrtf(glm_bsum(pk,lds));
    for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){q[i]/=nq;k[i]/=nk;}
    GLM_SYNC();
    float* s=rec+(uint64_t)h*GLM_LD*GLM_LD;
#ifndef GLM53_CPU
    if(GLM_LD%2==0&&blockDim.x==2*GLM_LD){
      // Two threads per value column d, each walking half of the column in batches of 16 loads in
      // flight (a whole half-column in registers spilled inside the megakernel). The first pass
      // stores the decayed state, the second re-reads it from L2. The memory and output sums add
      // the two halves in order.
      constexpr int HL=GLM_LD/2,NB=16;float* part=gs+L;
      const int d=threadIdx.x%GLM_LD,half=threadIdx.x/GLM_LD,i0=half*HL;
      float mem=0;
      for(int bb=0;bb<HL;bb+=NB){float col[NB];
#pragma unroll
        for(int i=0;i<NB;++i)col[i]=s[(i0+bb+i)*GLM_LD+d];
#pragma unroll
        for(int i=0;i<NB;++i){col[i]*=g[i0+bb+i];mem+=col[i]*k[i0+bb+i];s[(i0+bb+i)*GLM_LD+d]=col[i];}}
      part[half*GLM_LD+d]=mem;GLM_SYNC();mem=part[d]+part[GLM_LD+d];
      const float delta=(v[d]-mem)*beta;float sum=0;
      for(int bb=0;bb<HL;bb+=NB){float col[NB];
#pragma unroll
        for(int i=0;i<NB;++i)col[i]=s[(i0+bb+i)*GLM_LD+d];
#pragma unroll
        for(int i=0;i<NB;++i){col[i]+=k[i0+bb+i]*delta;sum+=col[i]*q[i0+bb+i];s[(i0+bb+i)*GLM_LD+d]=col[i];}}
      GLM_SYNC();part[half*GLM_LD+d]=sum;GLM_SYNC();
      if(half==0)o[d]=glm_bf(part[d]+part[GLM_LD+d]);
    } else
#endif
    // One lane per value column d: the native per-column order, no cross-lane sums.
    for(int d=GLM_LANE;d<GLM_LD;d+=GLM_LANES){float mem=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]*=g[i];mem+=s[i*GLM_LD+d]*k[i];}
      float delta=(v[d]-mem)*beta,sum=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]+=k[i]*delta;sum+=s[i*GLM_LD+d]*q[i];}
      o[d]=glm_bf(sum);
    }
    GLM_SYNC();
    float part=0;for(int j=GLM_LANE;j<GLM_LD;j+=GLM_LANES)part+=o[j]*o[j];
    const float inv=1/sqrtf(glm_bsum(part,lds)/GLM_LD+GLM_EPS);
    for(int j=GLM_LANE;j<GLM_LD;j+=GLM_LANES)att[off+j]=glm_bf(o[j]*inv*glm_w(b,w[ONORM],j)*glm_sig(gate[j]));
    GLM_SYNC();
  }
}
// O rows of tile t; each finished column k mixes into the FFN streams (glm_mix).
GLM_DEV void glm_ph_o(const uint8_t* b,const glm_weight* w,const glm_act* in,const float* hca,const float* att,float* mid,int t,int T,float* lds){
  float* hc=lds+16;glm_hc_coeffs(b,w,A_FN,hca,hc,lds+GLM_LDS_HEAD);
  const uint64_t bytes=1;const int rows=w[O].rows;int lo,hi;glm_share(&bytes,&rows,1,0,t,T,lo,hi);
  glm_rows<GLM_KBF16|GLM_KFP8>(b,w[O],att,lo,hi,[&](int k,float v){const float y=glm_bf(v);
    for(int c=0;c<GLM_HC;++c)mid[c*GLM_H+k]=glm_mix1(hc,[&](int j){return glm_get(in[j]);},y,c,k);});
}
// FFN input x (tile 0 publishes it after the logits) and the router logits of tile t.
GLM_DEV void glm_ph_rt(const uint8_t* b,const glm_weight* w,const float* mid,const float* hcf,float* rt,bool moe,int t,int T,float* lds){
  float* hc=lds+16;float* x=lds+GLM_LDS_HEAD;
  glm_hc_coeffs(b,w,F_FN,hcf,hc,x);
  glm_collapse_norm(b,w[N_FFN],glm_src_f32{mid},hc+GLM_HC+GLM_HC*GLM_HC,x,lds);
  if(t==0)for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)rt[GLM_R4(GLM_EXPERTS)+j]=x[j];
  if(moe){const uint64_t bytes=1;const int rows=w[ROUTER].rows;int lo,hi;glm_share(&bytes,&rows,1,0,t,T,lo,hi);
    glm_rows<GLM_KBF16>(b,w[ROUTER],x,lo,hi,[&](int r,float v){rt[r]=v;});}
}
// Routing of glm_ffn: sigmoid logits, top-k of logit+bias, renormalised and scaled, sorted by id.
// score, biased: GLM_EXPERTS floats each.
GLM_DEV void glm_topk(const uint8_t* b,const glm_weight* w,const float* logits,float* score,float* biased,int* ids,float* probs,float* red){
  for(int e=GLM_LANE;e<GLM_EXPERTS;e+=GLM_LANES){score[e]=glm_sig(logits[e]);biased[e]=score[e]+glm_w(b,w[BIAS],e);}
  GLM_SYNC();
  for(int k=0;k<GLM_TOPK;++k){float bv=-INFINITY;int bi=INT_MAX;
    for(int e=GLM_LANE;e<GLM_EXPERTS;e+=GLM_LANES){bool used=false;for(int j=0;j<k;++j)if(ids[j]==e)used=true;
      float s=biased[e];if(!used&&s>bv){bv=s;bi=e;}}
    const int best=glm_bargmax(bv,bi,red);
    if(GLM_LANE==0){ids[k]=best;probs[k]=score[best];}
    GLM_SYNC();
  }
  if(GLM_LANE==0){float den=1e-20f;for(int k=0;k<GLM_TOPK;++k)den+=probs[k];
    for(int k=0;k<GLM_TOPK;++k)probs[k]=probs[k]/den*GLM_ROUTE_SCALE;
    for(int i=0;i<GLM_TOPK;++i)for(int j=i+1;j<GLM_TOPK;++j)if(ids[j]<ids[i]){int t=ids[i];ids[i]=ids[j];ids[j]=t;float p=probs[i];probs[i]=probs[j];probs[j]=p;}
  }
  GLM_SYNC();
}
// Gate/up row pairs of every selected expert (shared expert last; dense: MG/MU), two pairs per group.
GLM_DEV void glm_ph_up(const uint8_t* b,const glm_weight* w,const float* rt,float* act,float* sel,bool moe,int t,int T,float* lds){
  float* x=lds+GLM_LDS_HEAD;float* probs=x+GLM_H;int* ids=(int*)(probs+GLM_R4(GLM_TOPK));float* score=(float*)(ids+GLM_R4(GLM_TOPK));
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)x[j]=rt[GLM_R4(GLM_EXPERTS)+j];
  GLM_SYNC();
  if(moe){glm_topk(b,w,rt,score,score+GLM_R4(GLM_EXPERTS),ids,probs,lds);
    if(t==0&&GLM_LANE==0)for(int k=0;k<GLM_TOPK;++k){sel[k]=(float)ids[k];sel[GLM_TOPK+k]=probs[k];}}
  const int ne=moe?GLM_TOPK+1:1,mi=w[moe?SG:MG].rows,pairs=ne*mi;
  const int lo=(int)((int64_t)pairs*t/T),hi=(int)((int64_t)pairs*(t+1)/T);
  if(hi<=lo)return;
  auto gslot=[&](int k){return !moe?MG:k<GLM_TOPK?EXPERTS+3*ids[k]:SG;};
  const int S=glm_group<GLM_KFP8>(w[gslot(0)]);
  glm_each(0,(hi-lo+1)/2,S,[&](int u,int sl){
    const glm_weight* ws[4];int row[4];const float* xs[4]={x,x,x,x};int pr[2];
    for(int i=0;i<2;++i){int p=lo+2*u+i;pr[i]=p<hi?p:lo+2*u;const int k=pr[i]/mi,gs=gslot(k);
      ws[2*i]=&w[gs];ws[2*i+1]=&w[gs+1];row[2*i]=row[2*i+1]=pr[i]%mi;}
    float acc[4];glm_dots<4,GLM_KFP8>(b,ws,row,xs,acc,S,sl);
    if(sl==0)for(int i=0;i<2;++i)if(lo+2*u+i<hi){
      const float gt=glm_bf(acc[2*i]),up=glm_bf(acc[2*i+1]);
      const float a=fminf(gt,GLM_CLIP),v=fmaxf(-GLM_CLIP,fminf(up,GLM_CLIP));
      act[(pr[i]/mi)*GLM_INTERMAX+pr[i]%mi]=glm_bf(glm_bf(a*glm_sig(a))*v);}
  });
}
// Down rows of tile t over all selected experts in id order, then the final mix of each column.
GLM_DEV void glm_ph_down(const uint8_t* b,const glm_weight* w,const float* act,const float* sel,const float* mid,const float* hcf,glm_act* out,bool moe,int t,int T,float* lds){
  float* hc=lds+16;glm_hc_coeffs(b,w,F_FN,hcf,hc,lds+GLM_LDS_HEAD);
  auto finish=[&](int j,float y){for(int c=0;c<GLM_HC;++c)out[c*GLM_H+j]=glm_put(glm_mix1(hc,[&](int i){return mid[i];},y,c,j));};
  const int lo=(int)((int64_t)GLM_H*t/T),hi=(int)((int64_t)GLM_H*(t+1)/T);
  if(!moe){glm_rows<GLM_KFP8>(b,w[MD],act,lo,hi,[&](int j,float v){finish(j,glm_bf(v));});return;}
  constexpr int G=GLM_TOPK+1;const glm_weight* ws[G];const float* xs[G];float probs[GLM_TOPK];
  for(int k=0;k<GLM_TOPK;++k){ws[k]=&w[EXPERTS+3*(int)sel[k]+2];xs[k]=act+k*GLM_INTERMAX;probs[k]=sel[GLM_TOPK+k];}
  ws[GLM_TOPK]=&w[SD];xs[GLM_TOPK]=act+GLM_TOPK*GLM_INTERMAX;
  const int S=glm_group<GLM_KFP8>(*ws[0]);
  glm_each(lo,hi,S,[&](int j,int sl){
    int row[G];for(int g=0;g<G;++g)row[g]=j;
    // Three rows per call: nine at once spill (measured); the sums are still combined in expert order.
    float acc[G];static_assert(G%3==0,"down groups three experts per call");
#pragma unroll
    for(int g=0;g<G;g+=3)glm_dots<3,GLM_KFP8>(b,ws+g,row+g,xs+g,acc+g,S,sl);
    if(sl==0){float y=0;for(int k=0;k<GLM_TOPK;++k)y=glm_bf(y+glm_bf(glm_bf(acc[k])*probs[k]));
      finish(j,glm_bf(y+glm_bf(acc[GLM_TOPK])));}
  });
}
// Final norm and LM-head rows of tile t (glm53_head without the argmax).
GLM_DEV void glm_ph_head(const glm_act* in,const uint8_t* nb,const glm_weight* nw,const uint8_t* hb,const glm_weight* hw,float* hs,int t,int T,float* lds){
  float* y=lds+GLM_LDS_HEAD;
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){float sum=0;for(int k=0;k<GLM_HC;++k)sum+=glm_get(in[k*GLM_H+j]);y[j]=glm_bf(sum/GLM_HC);if(t==0)hs[j]=y[j];}
  GLM_SYNC();glm_pnorm(nb,nw[0],y,y,GLM_H,lds);
  if(t==0)for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)hs[GLM_H+j]=y[j];
  const uint64_t bytes=1;const int rows=hw[0].rows;int lo,hi;glm_share(&bytes,&rows,1,0,t,T,lo,hi);
  float* logits=hs+2*GLM_H;glm_rows<GLM_KBF16>(hb,hw[0],y,lo,hi,[&](int r,float v){logits[r]=glm_bf(v);});
}
GLM_DEV void glm_ph_argmax(const float* logits,int* token,float* lds){
  float bv=-INFINITY;int bi=INT_MAX;
  for(int j=GLM_LANE;j<GLM_VOCAB;j+=GLM_LANES){const float v=logits[j];if(bi==INT_MAX||v>bv){bv=v;bi=j;}}
  const int best=glm_bargmax(bv,bi,lds);
  if(GLM_LANE==0)*token=best;
}
