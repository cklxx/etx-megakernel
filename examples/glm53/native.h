// GLM-5.3-Flash correctness backend. The same body runs in ETX HIP tiles and
// the single-thread CPU checker; no Transformers call is made by this backend.
#pragma once
#include <stdint.h>
#include <math.h>
#include "glm53_spec.h"
#ifdef GLM53_CPU
#define GLM_DEV inline
#define GLM_SYNC() ((void)0)
#define GLM_LANE 0
#define GLM_LANES 1
#else
#define GLM_DEV __device__ __forceinline__
#define GLM_SYNC() __syncthreads()
#define GLM_LANE ((int)threadIdx.x)
#define GLM_LANES ((int)blockDim.x)
#endif
struct glm_weight { uint64_t offset, scale; int32_t rows, cols, kind, block; };
static_assert(sizeof(glm_weight)==32, "staging ABI");
// Stable slot numbers shared by stage.py, fixture generation, CPU and HIP.
enum { A_FN,A_BASE,A_SCALE,F_FN,F_BASE,F_SCALE,N_ATTN,N_FFN,Q,K,V,CONV,
       FA,FB,DT,ALOG,BETA,GA,GB,ONORM,O,QA,QAN,QB,KVA,KVAN,KVB,IQ,IK,IKN,IKB,IW,IG,IAPE,
       MG,MU,MD,ROUTER,BIAS,SG,SU,SD,EXPERTS };
GLM_DEV float glm_bf(float x) {
#if GLM_BF16
  union {float f; uint32_t u;} v; v.f=x;
  if ((v.u & 0x7fffffffU) > 0x7f800000U) return x;
  v.u += 0x7fffU + ((v.u >> 16)&1); v.u &= 0xffff0000U; return v.f;
#else
  return x;
#endif
}
#if GLM_BF16
using glm_act=uint16_t;
GLM_DEV float glm_get(glm_act x) { union{float f;uint32_t u;} v; v.u=(uint32_t)x<<16;return v.f; }
GLM_DEV glm_act glm_put(float x) { union{float f;uint32_t u;} v; v.f=glm_bf(x);return v.u>>16; }
#else
using glm_act=float;
GLM_DEV float glm_get(glm_act x){return x;}
GLM_DEV glm_act glm_put(float x){return x;}
#endif
GLM_DEV float glm_sig(float x) { return 1.0f/(1.0f+expf(-x)); }
GLM_DEV float glm_fp8(uint8_t x) {
  int e=(x>>3)&15, m=x&7;
  float v=e ? ldexpf(1.0f+m/8.0f,e-7) : ldexpf(m/8.0f,-6);
  return x&128 ? -v : v;
}
GLM_DEV float glm_w(const uint8_t* blob,const glm_weight& w,int r,int c=0) {
  uint64_t i=(uint64_t)r*w.cols+c;
  if(w.kind==1) { union{float f;uint32_t u;}v;v.u=(uint32_t)((const uint16_t*)(blob+w.offset))[i]<<16;return v.f; }
  if(w.kind==2) return ((const float*)(blob+w.offset))[i];
  int scols=(w.cols+w.block-1)/w.block;
  float scale=((const float*)(blob+w.scale))[(r/w.block)*scols+c/w.block];
  // Trusted reference dequantizes FP8 blocks to BF16 before linear algebra.
  return glm_bf(glm_fp8(blob[w.offset+i])*scale);
}
GLM_DEV void glm_mv(const uint8_t* b,const glm_weight& w,const float* x,float* y,bool rounded=true) {
  for(int r=GLM_LANE;r<w.rows;r+=GLM_LANES){float sum=0;
    for(int c=0;c<w.cols;++c) sum += glm_w(b,w,r,c)*x[c];
    y[r]=rounded?glm_bf(sum):sum;
  } GLM_SYNC();
}
GLM_DEV void glm_norm(const uint8_t* b,const glm_weight& w,const float* x,float* y,int n,float eps=GLM_EPS) {
  if(GLM_LANE==0){float ss=0;for(int i=0;i<n;++i)ss+=x[i]*x[i];
    float inv=1.0f/sqrtf(ss/n+eps);
    for(int i=0;i<n;++i)y[i]=glm_bf(glm_bf(x[i]*inv)*glm_w(b,w,i));
  } GLM_SYNC();
}
// s: N*H streams; tmp has 2*HC+HC*HC+HC*H entries.
GLM_DEV void glm_hc(const uint8_t* b,const glm_weight* w,int base,const float* s,float* x,float* tmp) {
  if(GLM_LANE==0){
    float ss=0;for(int j=0;j<GLM_HC*GLM_H;++j)ss+=s[j]*s[j];
    float inv=1.0f/sqrtf(ss/(GLM_HC*GLM_H)+GLM_EPS);
    float* post=tmp;float* comb=post+GLM_HC;float* pre=comb+GLM_HC*GLM_HC;
    for(int r=0;r<GLM_HC*(2+GLM_HC);++r){float a=0;
      for(int c=0;c<GLM_HC*GLM_H;++c)a+=s[c]*inv*glm_w(b,w[base],r,c);
      float z=a*glm_w(b,w[base+2],r<GLM_HC?0:r<2*GLM_HC?1:2)+glm_w(b,w[base+1],r);
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
    for(int j=0;j<GLM_H;++j){float sum=0;for(int c=0;c<GLM_HC;++c)sum+=pre[c]*s[c*GLM_H+j];x[j]=glm_bf(sum);}
  } GLM_SYNC();
}
GLM_DEV void glm_mix(float* s,const float* y,const float* tmp) {
  // Separate output area: mixing in-place would overwrite another input stream.
  float* out=(float*)tmp+2*GLM_HC+GLM_HC*GLM_HC;
  for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES){int c=j/GLM_H,k=j%GLM_H;float a=0;
    for(int r=0;r<GLM_HC;++r)a+=glm_bf(tmp[GLM_HC+r*GLM_HC+c])*s[r*GLM_H+k];
    out[j]=glm_bf(glm_bf(glm_bf(tmp[c])*y[k])+glm_bf(a));
  } GLM_SYNC();
  for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES)s[j]=out[j];GLM_SYNC();
}
GLM_DEV void glm_mlp(const uint8_t* b,const glm_weight* w,int g,int u,int d,const float* x,float* y,float* scratch) {
  float* gate=scratch;float* up=scratch+GLM_INTERMAX;
  glm_mv(b,w[g],x,gate);glm_mv(b,w[u],x,up);
  for(int j=GLM_LANE;j<w[g].rows;j+=GLM_LANES){float a=fminf(gate[j],GLM_CLIP),v=fmaxf(-GLM_CLIP,fminf(up[j],GLM_CLIP));gate[j]=glm_bf(glm_bf(a*glm_sig(a))*v);}GLM_SYNC();
  glm_mv(b,w[d],gate,y);
}
GLM_DEV void glm_ffn(const uint8_t* b,const glm_weight* w,const float* x,float* y,float* a,bool moe) {
  if(!moe){glm_mlp(b,w,MG,MU,MD,x,y,a);return;}
  float* logits=a+2*GLM_INTERMAX;float* ids=logits+GLM_EXPERTS;float* probs=ids+GLM_TOPK;float* expert=probs+GLM_TOPK;
  glm_mv(b,w[ROUTER],x,logits,false);
  if(GLM_LANE==0){float den=1e-20f;
    for(int i=0;i<GLM_EXPERTS;++i)logits[i]=glm_sig(logits[i]);
    for(int k=0;k<GLM_TOPK;++k){int best=-1;float mx=-INFINITY;
      for(int e=0;e<GLM_EXPERTS;++e){bool used=false;for(int j=0;j<k;++j)if((int)ids[j]==e)used=true;
        float score=logits[e]+glm_w(b,w[BIAS],e);if(!used && score>mx){mx=score;best=e;}}
      ids[k]=best;probs[k]=logits[best];den+=probs[k];
    }
    for(int k=0;k<GLM_TOPK;++k)probs[k]=probs[k]/den*GLM_ROUTE_SCALE;
    // HF expert accumulation visits experts in ascending ID order.
    for(int i=0;i<GLM_TOPK;++i)for(int j=i+1;j<GLM_TOPK;++j)if(ids[j]<ids[i]){float t=ids[i];ids[i]=ids[j];ids[j]=t;t=probs[i];probs[i]=probs[j];probs[j]=t;}
    for(int j=0;j<GLM_H;++j)y[j]=0;
  }GLM_SYNC();
  for(int k=0;k<GLM_TOPK;++k){int slot=EXPERTS+3*(int)ids[k];glm_mlp(b,w,slot,slot+1,slot+2,x,expert,a);
    for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)y[j]=glm_bf(y[j]+glm_bf(expert[j]*probs[k]));GLM_SYNC();}
  glm_mlp(b,w,SG,SU,SD,x,expert,a);
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)y[j]=glm_bf(y[j]+expert[j]);GLM_SYNC();
}
GLM_DEV void glm_kda(const uint8_t* b,const glm_weight* w,const float* x,float* y,float* state,float* t) {
  float* q=t;float* k=q+GLM_QDIM;float* v=k+GLM_QDIM;float* small=v+GLM_QDIM;
  float* g=small+GLM_LD;float* beta=g+GLM_QDIM;float* gate=beta+GLM_LH;float* out=gate+GLM_QDIM;
  glm_mv(b,w[Q],x,q);glm_mv(b,w[K],x,k);glm_mv(b,w[V],x,v);
  float* conv=state;float* rec=conv+3*GLM_QDIM*GLM_CONV;
  for(int j=GLM_LANE;j<3*GLM_QDIM;j+=GLM_LANES){float* dst=j<GLM_QDIM?q:j<2*GLM_QDIM?k:v;int n=j%GLM_QDIM;
    float* hist=conv+j*GLM_CONV;for(int z=0;z<GLM_CONV-1;++z)hist[z]=hist[z+1];hist[GLM_CONV-1]=dst[n];
    float sum=0;for(int z=0;z<GLM_CONV;++z)sum+=hist[z]*glm_w(b,w[CONV],j,z);
    // conv1d returns its weight dtype before SiLU (FP32 in production).
    if(w[CONV].kind==1)sum=glm_bf(sum);
    dst[n]=glm_bf(sum*glm_sig(sum));
  }GLM_SYNC();
  glm_mv(b,w[FA],x,small);glm_mv(b,w[FB],small,g);glm_mv(b,w[BETA],x,beta);
  for(int j=GLM_LANE;j<GLM_QDIM;j+=GLM_LANES)g[j]=expf(GLM_LOWER*glm_sig(expf(glm_w(b,w[ALOG],j/GLM_LD))*(g[j]+glm_w(b,w[DT],j))));
  for(int j=GLM_LANE;j<GLM_LH;j+=GLM_LANES)beta[j]=glm_bf(glm_sig(beta[j]));GLM_SYNC();
  // One thread owns a complete head: no inter-thread reduction or state races.
  for(int h=GLM_LANE;h<GLM_LH;h+=GLM_LANES){int off=h*GLM_LD;float nq=1e-6f,nk=1e-6f;
    for(int i=0;i<GLM_LD;++i){nq+=q[off+i]*q[off+i];nk+=k[off+i]*k[off+i];}
    nq=sqrtf(nq)*sqrtf((float)GLM_LD);nk=sqrtf(nk);
    for(int i=0;i<GLM_LD;++i){q[off+i]/=nq;k[off+i]/=nk;}
    float* s=rec+h*GLM_LD*GLM_LD;
    for(int d=0;d<GLM_LD;++d){float mem=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]*=g[off+i];mem+=s[i*GLM_LD+d]*k[off+i];}
      float delta=(v[off+d]-mem)*beta[h],sum=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]+=k[off+i]*delta;sum+=s[i*GLM_LD+d]*q[off+i];}
      out[off+d]=glm_bf(sum);
    }
  }GLM_SYNC();
  glm_mv(b,w[GA],x,small);glm_mv(b,w[GB],small,gate);
  for(int h=GLM_LANE;h<GLM_LH;h+=GLM_LANES){float ss=0;for(int j=0;j<GLM_LD;++j)ss+=out[h*GLM_LD+j]*out[h*GLM_LD+j];float inv=1/sqrtf(ss/GLM_LD+GLM_EPS);
    for(int j=0;j<GLM_LD;++j){int n=h*GLM_LD+j;out[n]=glm_bf(out[n]*inv*glm_w(b,w[ONORM],j)*glm_sig(gate[n]));}
  }GLM_SYNC();glm_mv(b,w[O],out,y);
}
GLM_DEV void glm_mla(const uint8_t* b,const glm_weight* w,const float* x,float* y,float* state,float* t,int pos) {
  float* qr=t;float* q=qr+GLM_QR;float* kv=q+GLM_NH*GLM_AD;float* expanded=kv+GLM_KVR;
  float* iq=expanded+2*GLM_NH*GLM_AD;float* ik=iq+GLM_IH*GLM_ID;float* ig=ik+GLM_ID;
  float* iw=ig+GLM_ID;float* out=iw+GLM_IH;float* score=out+GLM_NH*GLM_AD;
  float* chosen=score+GLM_MAXLEN;float* pool=chosen+GLM_MAXLEN;
  const int stride=2*GLM_NH*GLM_AD+2*GLM_ID;
  glm_mv(b,w[QA],x,qr);glm_norm(b,w[QAN],qr,qr,GLM_QR);glm_mv(b,w[QB],qr,q);
  glm_mv(b,w[KVA],x,kv);glm_norm(b,w[KVAN],kv,kv,GLM_KVR);glm_mv(b,w[KVB],kv,expanded);
  for(int j=GLM_LANE;j<2*GLM_NH*GLM_AD;j+=GLM_LANES)state[(uint64_t)pos*stride+j]=expanded[j];GLM_SYNC();
  glm_mv(b,w[IQ],qr,iq);glm_mv(b,w[IK],x,ik);glm_mv(b,w[IG],x,ig);glm_mv(b,w[IW],x,iw);
  if(GLM_LANE==0){float mean=0,var=0;for(int i=0;i<GLM_ID;++i)mean+=ik[i];mean/=GLM_ID;
    for(int i=0;i<GLM_ID;++i)var+=(ik[i]-mean)*(ik[i]-mean);float inv=1/sqrtf(var/GLM_ID+1e-6f);
    for(int i=0;i<GLM_ID;++i)ik[i]=glm_bf((ik[i]-mean)*inv*glm_w(b,w[IKN],i)+glm_w(b,w[IKB],i));
  }GLM_SYNC();
  for(int j=GLM_LANE;j<GLM_ID;j+=GLM_LANES){state[(uint64_t)pos*stride+2*GLM_NH*GLM_AD+j]=ik[j];state[(uint64_t)pos*stride+2*GLM_NH*GLM_AD+GLM_ID+j]=ig[j];}GLM_SYNC();
  // kpool: only complete visible pools compete; append incomplete visible tail.
  if(GLM_LANE==0){int np=(pos+1)/GLM_POOL;
    for(int p=0;p<np;++p){
      for(int d=0;d<GLM_ID;++d){float mx=-INFINITY,den=0,num=0;
        for(int z=0;z<GLM_POOL;++z)mx=fmaxf(mx,state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d));
        for(int z=0;z<GLM_POOL;++z)den+=expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d)-mx);
        for(int z=0;z<GLM_POOL;++z){float prob=glm_bf(expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+glm_w(b,w[IAPE],z,d)-mx)/den);num+=glm_bf(prob*state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+d]);}pool[d]=glm_bf(num);
      }
      float sum=0;for(int h=0;h<GLM_IH;++h){float dot=0;for(int d=0;d<GLM_ID;++d)dot+=iq[h*GLM_ID+d]*pool[d];sum+=iw[h]/sqrtf((float)GLM_IH)*fmaxf(0,dot/sqrtf((float)GLM_ID));}score[p]=sum;
    }
    for(int j=0;j<=pos;++j)chosen[j]=0;
    int take=np<GLM_INDEX_TOPK/GLM_POOL?np:GLM_INDEX_TOPK/GLM_POOL;
    for(int z=0;z<take;++z){int best=-1;float mx=-INFINITY;for(int p=0;p<np;++p)if(score[p]>mx){mx=score[p];best=p;}score[best]=-INFINITY;for(int j=0;j<GLM_POOL;++j)chosen[best*GLM_POOL+j]=1;}
    for(int j=np*GLM_POOL;j<=pos;++j)chosen[j]=1;
  }GLM_SYNC();
  // Compute one attention head per thread; scores rounded like eager BF16 matmul.
  float* hs=pool+GLM_ID;
  for(int h=GLM_LANE;h<GLM_NH;h+=GLM_LANES){float* a=hs+(uint64_t)h*GLM_MAXLEN;float mx=-INFINITY,den=0;
    for(int j=0;j<=pos;++j){float dot=0;for(int d=0;d<GLM_AD;++d)dot+=q[h*GLM_AD+d]*state[(uint64_t)j*stride+h*2*GLM_AD+d];a[j]=chosen[j]?glm_bf(glm_bf(dot)/sqrtf((float)GLM_AD)):-INFINITY;mx=fmaxf(mx,a[j]);}
    for(int j=0;j<=pos;++j){a[j]=expf(a[j]-mx);den+=a[j];}
    for(int j=0;j<=pos;++j)a[j]=glm_bf(a[j]/den);
    for(int d=0;d<GLM_AD;++d){float sum=0;for(int j=0;j<=pos;++j)sum+=a[j]*state[(uint64_t)j*stride+h*2*GLM_AD+GLM_AD+d];out[h*GLM_AD+d]=glm_bf(sum);}
  }GLM_SYNC();glm_mv(b,w[O],out,y);
}
GLM_DEV void glm_layer(const uint8_t* b,const glm_weight* w,const glm_act* input,glm_act* output,float* state,float* scratch,int pos,bool linear,bool moe) {
  float* streams=scratch;float* collapsed=streams+GLM_HC*GLM_H;float* x=collapsed+GLM_H;float* y=x+GLM_H;
  float* hc=y+GLM_H;float* tmp=hc+2*GLM_HC+GLM_HC*GLM_HC+GLM_HC*GLM_H;
  for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES)streams[j]=glm_get(input[j]);GLM_SYNC();
  glm_hc(b,w,A_FN,streams,collapsed,hc);glm_norm(b,w[N_ATTN],collapsed,x,GLM_H);
  if(linear)glm_kda(b,w,x,y,state,tmp);else glm_mla(b,w,x,y,state,tmp,pos);
  glm_mix(streams,y,hc);glm_hc(b,w,F_FN,streams,collapsed,hc);glm_norm(b,w[N_FFN],collapsed,x,GLM_H);
  glm_ffn(b,w,x,y,tmp,moe);glm_mix(streams,y,hc);
  for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES)output[j]=glm_put(streams[j]);GLM_SYNC();
}
