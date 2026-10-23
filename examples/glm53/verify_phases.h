// Verify-only tiles. Include as a translation unit: the existing decode tiles
// remain the source of truth for rounding, DSA selection and cache layout.
#pragma once
#include "examples/glm53/tiles.hip"


// Verify rows use one owner per output row. Each scalar weight is consumed by
// all position accumulators before advancing the column; no full weight copy
// or position-private weight storage is needed. These are correctness-first
// scalar tiles, not a GPU throughput qualification.
#define GLM_VHC GLM_R4(GLM_HCN)
#define GLM_VX (2*GLM_VHC)
#define GLM_VA (GLM_VX+GLM_H)
#define GLM_VG (GLM_VA+3*GLM_QDIM)
#define GLM_VB (GLM_VG+GLM_QDIM)
#define GLM_VKV (GLM_VB+GLM_QDIM)
#define GLM_VNR (GLM_VKV+2*GLM_NH*GLM_AD+2*GLM_ID)
#define GLM_VWORK (GLM_VNR+GLM_QR+GLM_KVR)
#define GLM_VCOMMON (GLM_LD+GLM_LH+GLM_QDIM+GLM_POOL*GLM_ID)

template<int P,class Out> GLM_DEV void glm_verify_rows(
    const uint8_t* b,const glm_weight& w,const float* const* x,int lo,int hi,Out out){
  for(int r=lo+GLM_LANE;r<hi;r+=GLM_LANES){
    float acc[P]={};
    for(int j=0;j<w.cols;++j){const float v=glm_w(b,w,r,j);
      for(int p=0;p<P;++p)acc[p]+=v*x[p][j];}
    for(int p=0;p<P;++p)out(p,r,acc[p]);
  }
}
// The reductions and BF16 points match the decode scalar reference. Norm
// parameters are streamed outside the position loop, including in-place use.
template<int P> GLM_DEV void glm_verify_norm(const uint8_t* b,const glm_weight& w,
    const float* const* x,float* const* y,int n,float* red){
  float inv[P];
  for(int p=0;p<P;++p){float part=0;
    for(int i=GLM_LANE;i<n;i+=GLM_LANES)part+=x[p][i]*x[p][i];
    inv[p]=1.0f/sqrtf(glm_bsum(part,red)/n+GLM_EPS);}
  for(int i=GLM_LANE;i<n;i+=GLM_LANES){const float v=glm_w(b,w,i);
    for(int p=0;p<P;++p)y[p][i]=glm_bf(glm_bf(x[p][i]*inv[p])*v);}
  GLM_SYNC();
}
GLM_DEV void glm_verify_coeff(const float* z,float* hc){
  float* post=hc;float* comb=post+GLM_HC;float* pre=comb+GLM_HC*GLM_HC;
  for(int r=0;r<GLM_HCN;++r){
    if(r<GLM_HC)pre[r]=glm_sig(z[r])+GLM_HCEPS;
    else if(r<2*GLM_HC)post[r-GLM_HC]=2*glm_sig(z[r]);
    else comb[r-2*GLM_HC]=z[r];}
  for(int r=0;r<GLM_HC;++r){float mx=-INFINITY,den=0;
    for(int c=0;c<GLM_HC;++c)mx=fmaxf(mx,comb[r*GLM_HC+c]);
    for(int c=0;c<GLM_HC;++c){float v=expf(comb[r*GLM_HC+c]-mx);comb[r*GLM_HC+c]=v;den+=v;}
    for(int c=0;c<GLM_HC;++c)comb[r*GLM_HC+c]=comb[r*GLM_HC+c]/den+GLM_HCEPS;}
  for(int it=0;it<GLM_SINK;++it){
    if(it)for(int r=0;r<GLM_HC;++r){float den=GLM_HCEPS;
      for(int c=0;c<GLM_HC;++c)den+=comb[r*GLM_HC+c];
      for(int c=0;c<GLM_HC;++c)comb[r*GLM_HC+c]/=den;}
    for(int c=0;c<GLM_HC;++c){float den=GLM_HCEPS;
      for(int r=0;r<GLM_HC;++r)den+=comb[r*GLM_HC+c];
      for(int r=0;r<GLM_HC;++r)comb[r*GLM_HC+c]/=den;}}
}
// Compute all raw mHC rows and coefficients once, then collapse and normalize.
// Src selects BF16 attention streams or FP32 FFN streams without changing math.
template<int P,class Src> GLM_DEV void glm_verify_input(const uint8_t* b,
    const glm_weight* w,int base,int norm,const Src* src,float* const* raw,
    float* const* hc,float* const* x,float* red){
  float inv[P];
  for(int p=0;p<P;++p){float part=0;
    for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES){float v=src[p](j);part+=v*v;}
    inv[p]=1.0f/sqrtf(glm_bsum(part,red)/(GLM_HC*GLM_H)+GLM_EPS);}
  for(int r=0;r<GLM_HCN;++r){float acc[P]={};
    for(int j=GLM_LANE;j<GLM_HC*GLM_H;j+=GLM_LANES){const float v=glm_w(b,w[base],r,j);
      for(int p=0;p<P;++p)acc[p]+=src[p](j)*inv[p]*v;}
    for(int p=0;p<P;++p){const float v=glm_bsum(acc[p],red);if(GLM_LANE==0)raw[p][r]=v;}}
  // Three scale entries are shared by all rows, not loaded once per mHC row.
  if(GLM_LANE==0){float scale[3];for(int i=0;i<3;++i)scale[i]=glm_w(b,w[base+2],i);
    float z[P][GLM_HCN];
    for(int r=0;r<GLM_HCN;++r){const float bias=glm_w(b,w[base+1],r);
      for(int p=0;p<P;++p)z[p][r]=raw[p][r]*scale[r<GLM_HC?0:r<2*GLM_HC?1:2]+bias;}
    for(int p=0;p<P;++p)glm_verify_coeff(z[p],hc[p]);}
  GLM_SYNC();
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)for(int p=0;p<P;++p){float sum=0;
    for(int k=0;k<GLM_HC;++k)sum+=hc[p][GLM_HC+GLM_HC*GLM_HC+k]*src[p](k*GLM_H+j);
    x[p][j]=glm_bf(sum);}
  GLM_SYNC();const float* cx[P];for(int p=0;p<P;++p)cx[p]=x[p];
  glm_verify_norm<P>(b,w[norm],cx,x,GLM_H,red);
}
// args: weight/meta, (step, embedded) per row. Duplicate token IDs and mHC
// streams share one embedding lookup; different IDs read their distinct rows.
template<int P> GLM_DEV void glm_verify_embed(const etx_ctx* c){
  int token[P];glm_act* out[P];
  for(int p=0;p<P;++p){token[p]=glm53_step_load((const int*)c->args[2+2*p]+2*c->coord[0]+1);
    out[p]=GLM_PR(3+2*p,glm_act*,GLM_HC*GLM_H);}
  for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES)for(int p=0;p<P;++p){
    bool seen=false;for(int q=0;q<p;++q)seen=seen||token[q]==token[p];if(seen)continue;
    const glm_act v=glm_put(glm_w(GLM_PB(0),GLM_PW(1)[0],token[p],j));
    for(int q=p;q<P;++q)if(token[q]==token[p])for(int k=0;k<GLM_HC;++k)out[q][k*GLM_H+j]=v;}
}
// args: weight/meta, (input, hca, p1, work) per row.
template<int P> GLM_DEV void glm_verify_prologue(const etx_ctx* c){
  glm_src_act src[P];float *raw[P],*hc[P],*x[P],*p1[P];
  for(int p=0;p<P;++p){src[p]={GLM_PR(2+4*p,const glm_act*,GLM_HC*GLM_H)};
    raw[p]=GLM_PR(3+4*p,float*,GLM_S_HC);p1[p]=GLM_PR(4+4*p,float*,GLM_S_P1);
    hc[p]=GLM_PR(5+4*p,float*,GLM_VWORK);x[p]=hc[p]+GLM_VX;}
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  glm_verify_input<P>(b,w,A_FN,N_ATTN,src,raw,hc,x,(float*)c->lds);
  const int slots[6]={c->cst[1]?Q:QA,c->cst[1]?K:KVA,c->cst[1]?V:IK,
                     c->cst[1]?FA:IG,c->cst[1]?BETA:IW,GA};
  const int offsets[6]={0,c->cst[1]?GLM_QDIM:GLM_QR,
    c->cst[1]?2*GLM_QDIM:GLM_QR+GLM_KVR,
    c->cst[1]?3*GLM_QDIM:GLM_QR+GLM_KVR+GLM_ID,
    c->cst[1]?3*GLM_QDIM+GLM_LD:GLM_QR+GLM_KVR+2*GLM_ID,
    3*GLM_QDIM+GLM_LD+GLM_LH};
  const float* cx[P];for(int p=0;p<P;++p)cx[p]=x[p];
  for(int s=0;s<(c->cst[1]?6:5);++s)glm_verify_rows<P>(b,w[slots[s]],cx,0,w[slots[s]].rows,
    [&](int p,int r,float v){p1[p][offsets[s]+r]=glm_bf(v);});
}
// args: weight/meta, (input, work, att, mid) per row. Coefficients computed by
// prologue are reused here instead of fetching mHC parameters for every tile.
template<int P> GLM_DEV void glm_verify_output(const etx_ctx* c){
  const glm_act* in[P];const float *hc[P],*att[P];float* mid[P];
  for(int p=0;p<P;++p){in[p]=GLM_PR(2+4*p,const glm_act*,GLM_HC*GLM_H);
    hc[p]=GLM_PR(3+4*p,const float*,GLM_VWORK);att[p]=GLM_PR(4+4*p,const float*,GLM_S_ATT);
    mid[p]=GLM_PR(5+4*p,float*,GLM_S_MID);}
  const glm_weight& w=GLM_PW(1)[O];
  glm_verify_rows<P>(GLM_PB(0),w,att,0,w.rows,[&](int p,int j,float v){
    for(int k=0;k<GLM_HC;++k)mid[p][k*GLM_H+j]=glm_mix1(hc[p],[&](int i){return glm_get(in[p][i]);},glm_bf(v),k,j);});
}
// args: weight/meta, (mid, hcf, rt, work) per row.
template<int P> GLM_DEV void glm_verify_ffn_input(const etx_ctx* c){
  glm_src_f32 src[P];float *raw[P],*hc[P],*x[P],*rt[P];
  for(int p=0;p<P;++p){src[p]={GLM_PR(2+4*p,const float*,GLM_S_MID)};
    raw[p]=GLM_PR(3+4*p,float*,GLM_S_HC);rt[p]=GLM_PR(4+4*p,float*,GLM_S_RT);
    hc[p]=GLM_PR(5+4*p,float*,GLM_VWORK)+GLM_VHC;x[p]=rt[p]+GLM_R4(GLM_EXPERTS);}
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  glm_verify_input<P>(b,w,F_FN,N_FFN,src,raw,hc,x,(float*)c->lds);
  if(c->cst[1]){const float* cx[P];for(int p=0;p<P;++p)cx[p]=x[p];
    glm_verify_rows<P>(b,w[ROUTER],cx,0,w[ROUTER].rows,[&](int p,int r,float v){rt[p][r]=v;});}
}

// args: weights, meta, state, invariant scratch, (p1, att, work) per row.
// The convolution and recurrence still advance in position order. Only their
// fixed coefficients and f/g projections are shared across the row block.
template<int P> GLM_DEV void glm_verify_kda(const etx_ctx* c){
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  float* state=GLM_PR(2,float*,(uint64_t)c->cst[3]);
  float* common=GLM_PR(3,float*,GLM_VCOMMON);float* red=(float*)c->lds;
  const float* p1[P];float *att[P],*work[P];
  for(int p=0;p<P;++p){p1[p]=GLM_PR(4+3*p,const float*,GLM_S_P1);
    att[p]=GLM_PR(5+3*p,float*,GLM_S_ATT);work[p]=GLM_PR(6+3*p,float*,GLM_VWORK);}
  const float *fs[P],*gs[P];
  for(int p=0;p<P;++p){fs[p]=p1[p]+3*GLM_QDIM;gs[p]=fs[p]+GLM_LD+GLM_LH;}
  glm_verify_rows<P>(b,w[FB],fs,0,w[FB].rows,[&](int p,int r,float v){work[p][GLM_VG+r]=glm_bf(v);});
  glm_verify_rows<P>(b,w[GB],gs,0,w[GB].rows,[&](int p,int r,float v){work[p][GLM_VB+r]=glm_bf(v);});
  for(int j=GLM_LANE;j<GLM_LD;j+=GLM_LANES)common[j]=glm_w(b,w[ONORM],j);
  for(int h=GLM_LANE;h<GLM_LH;h+=GLM_LANES)common[GLM_LD+h]=glm_w(b,w[ALOG],h);
  GLM_SYNC();
  for(int j=GLM_LANE;j<GLM_QDIM;j+=GLM_LANES){const float dt=glm_w(b,w[DT],j);
    for(int p=0;p<P;++p)work[p][GLM_VG+j]=expf(GLM_LOWER*glm_sig(expf(common[GLM_LD+j/GLM_LD])*(work[p][GLM_VG+j]+dt)));}
  for(int j=GLM_LANE;j<3*GLM_QDIM;j+=GLM_LANES){
    float weight[GLM_CONV];for(int z=0;z<GLM_CONV;++z)weight[z]=glm_w(b,w[CONV],j,z);
    float* hist=state+(uint64_t)j*GLM_CONV;
    for(int p=0;p<P;++p){
      for(int z=0;z<GLM_CONV-1;++z)hist[z]=hist[z+1];hist[GLM_CONV-1]=p1[p][j];
      float sum=0;for(int z=0;z<GLM_CONV;++z)sum+=hist[z]*weight[z];
      if(w[CONV].kind==1)sum=glm_bf(sum);
      work[p][GLM_VA+j]=glm_bf(sum*glm_sig(sum));}}
  GLM_SYNC();
  float* rec=state+3*GLM_QDIM*GLM_CONV;
  for(int p=0;p<P;++p)for(int h=0;h<GLM_LH;++h){const int off=h*GLM_LD;
    float* q=work[p]+GLM_VA+off;float* k=q+GLM_QDIM;const float* v=k+GLM_QDIM;
    const float* g=work[p]+GLM_VG+off;const float* gate=work[p]+GLM_VB+off;
    const float beta=glm_bf(glm_sig(p1[p][3*GLM_QDIM+GLM_LD+h]));
    float pq=GLM_LANE==0?1e-6f:0.0f,pk=pq;
    for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){pq+=q[i]*q[i];pk+=k[i]*k[i];}
    const float nq=sqrtf(glm_bsum(pq,red))*sqrtf((float)GLM_LD),nk=sqrtf(glm_bsum(pk,red));
    for(int i=GLM_LANE;i<GLM_LD;i+=GLM_LANES){q[i]/=nq;k[i]/=nk;}
    GLM_SYNC();float* s=rec+(uint64_t)h*GLM_LD*GLM_LD;
    float* o=red+GLM_LDS_HEAD;
    for(int d=GLM_LANE;d<GLM_LD;d+=GLM_LANES){float mem=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]*=g[i];mem+=s[i*GLM_LD+d]*k[i];}
      const float delta=(v[d]-mem)*beta;float sum=0;
      for(int i=0;i<GLM_LD;++i){s[i*GLM_LD+d]+=k[i]*delta;sum+=s[i*GLM_LD+d]*q[i];}
      o[d]=glm_bf(sum);}
    GLM_SYNC();float part=0;
    for(int j=GLM_LANE;j<GLM_LD;j+=GLM_LANES)part+=o[j]*o[j];
    const float inv=1/sqrtf(glm_bsum(part,red)/GLM_LD+GLM_EPS);
    for(int j=GLM_LANE;j<GLM_LD;j+=GLM_LANES)att[p][off+j]=glm_bf(o[j]*inv*common[j]*glm_sig(gate[j]));
    GLM_SYNC();
  }
}
extern "C" __device__ void glm53_verify_kda_scan(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_kda<1>(c);break;
    case 2:glm_verify_kda<2>(c);break;
    case 3:glm_verify_kda<3>(c);break;
    case 4:glm_verify_kda<4>(c);break;
    case 5:glm_verify_kda<5>(c);break;
    case 6:glm_verify_kda<6>(c);break;
  }
}

GLM_DEV void glm_verify_mla(const float* p1,const float* p2,const float* state,float* att,int pos,const float* ape,float* lds){
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
        for(int z=0;z<GLM_POOL;++z)mx=fmaxf(mx,state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+ape[z*GLM_ID+d]);
        for(int z=0;z<GLM_POOL;++z)den+=expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+ape[z*GLM_ID+d]-mx);
        for(int z=0;z<GLM_POOL;++z){float prob=glm_bf(expf(state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+GLM_ID+d]+ape[z*GLM_ID+d]-mx)/den);num+=glm_bf(prob*state[(uint64_t)(p*GLM_POOL+z)*stride+2*GLM_NH*GLM_AD+d]);}pool[d]=glm_bf(num);
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
  int S=GLM_WAVE;while(S>1&&S>GLM_AD)S>>=1;
#endif
  for(int h=0;h<GLM_NH;++h){
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
// Fixed DSA transforms run over all rows before any cache publication. KV
// rows remain private scratch until their own causal scan iteration.
template<int P> GLM_DEV void glm_verify_dsa(const etx_ctx* c){
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  float* state=GLM_PR(2,float*,(uint64_t)c->cst[3]);
  float* common=GLM_PR(3,float*,GLM_VCOMMON);float* red=(float*)c->lds;
  const float* p1[P];float *p2[P],*att[P],*work[P],*qr[P],*kv[P];int pos[P];
  constexpr int stride=2*GLM_NH*GLM_AD+2*GLM_ID;
  for(int p=0;p<P;++p){
    pos[p]=glm53_step_load((const int*)c->args[4+5*p]+2*c->coord[0]);
    p1[p]=GLM_PR(5+5*p,const float*,GLM_S_P1);p2[p]=GLM_PR(6+5*p,float*,GLM_S_P2);
    att[p]=GLM_PR(7+5*p,float*,GLM_S_ATT);work[p]=GLM_PR(8+5*p,float*,GLM_VWORK);
    qr[p]=work[p]+GLM_VNR;kv[p]=qr[p]+GLM_QR;}
  const float* x[P];for(int p=0;p<P;++p)x[p]=p1[p];
  glm_verify_norm<P>(b,w[QAN],x,qr,GLM_QR,red);
  for(int p=0;p<P;++p)x[p]=p1[p]+GLM_QR;
  glm_verify_norm<P>(b,w[KVAN],x,kv,GLM_KVR,red);
  for(int p=0;p<P;++p)x[p]=qr[p];
  glm_verify_rows<P>(b,w[QB],x,0,w[QB].rows,[&](int p,int r,float v){p2[p][r]=glm_bf(v);});
  glm_verify_rows<P>(b,w[IQ],x,0,w[IQ].rows,[&](int p,int r,float v){p2[p][GLM_NH*GLM_AD+r]=glm_bf(v);});
  for(int p=0;p<P;++p)x[p]=kv[p];
  glm_verify_rows<P>(b,w[KVB],x,0,w[KVB].rows,[&](int p,int r,float v){work[p][GLM_VKV+r]=glm_bf(v);});
  float mean[P],inv[P];
  for(int p=0;p<P;++p){const float* ik=p1[p]+GLM_QR+GLM_KVR;float part=0;
    for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=ik[i];mean[p]=glm_bsum(part,red)/GLM_ID;
    part=0;for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES)part+=(ik[i]-mean[p])*(ik[i]-mean[p]);
    inv[p]=1/sqrtf(glm_bsum(part,red)/GLM_ID+1e-6f);}
  for(int i=GLM_LANE;i<GLM_ID;i+=GLM_LANES){const float n=glm_w(b,w[IKN],i),bias=glm_w(b,w[IKB],i);
    for(int p=0;p<P;++p){const float* ik=p1[p]+GLM_QR+GLM_KVR;
      work[p][GLM_VKV+2*GLM_NH*GLM_AD+i]=glm_bf((ik[i]-mean[p])*inv[p]*n+bias);
      work[p][GLM_VKV+2*GLM_NH*GLM_AD+GLM_ID+i]=ik[GLM_ID+i];}}
  // Pool parameters are needed only in the pruned-pool branch. They are small
  // invariant coefficients, unlike the projection matrices streamed above.
  bool pool=false;for(int p=0;p<P;++p)pool=pool||(pos[p]+1)/GLM_POOL>GLM_INDEX_TOPK/GLM_POOL;
  float* ape=common+GLM_LD+GLM_LH+GLM_QDIM;
  if(pool)for(int j=GLM_LANE;j<GLM_POOL*GLM_ID;j+=GLM_LANES)ape[j]=glm_w(b,w[IAPE],j/GLM_ID,j%GLM_ID);
  GLM_SYNC();
  for(int p=0;p<P;++p){
    float* st=state+(uint64_t)pos[p]*stride;
    for(int j=GLM_LANE;j<stride;j+=GLM_LANES)st[j]=work[p][GLM_VKV+j];
    GLM_SYNC();glm_verify_mla(p1[p],p2[p],state,att[p],pos[p],ape,red);GLM_SYNC();
  }
}
extern "C" __device__ void glm53_verify_dsa_scan(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_dsa<1>(c);break;
    case 2:glm_verify_dsa<2>(c);break;
    case 3:glm_verify_dsa<3>(c);break;
    case 4:glm_verify_dsa<4>(c);break;
    case 5:glm_verify_dsa<5>(c);break;
    case 6:glm_verify_dsa<6>(c);break;
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
  const float* hc=GLM_PR(5,const float*,GLM_VWORK)+GLM_VHC;
  glm_act* out=GLM_PR(6,glm_act*,GLM_HC*GLM_H);
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

extern "C" __device__ void glm53_verify_embed(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_embed<1>(c);break;
    case 2:glm_verify_embed<2>(c);break;
    case 3:glm_verify_embed<3>(c);break;
    case 4:glm_verify_embed<4>(c);break;
    case 5:glm_verify_embed<5>(c);break;
    case 6:glm_verify_embed<6>(c);break;
  }
}

extern "C" __device__ void glm53_verify_prologue(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_prologue<1>(c);break;
    case 2:glm_verify_prologue<2>(c);break;
    case 3:glm_verify_prologue<3>(c);break;
    case 4:glm_verify_prologue<4>(c);break;
    case 5:glm_verify_prologue<5>(c);break;
    case 6:glm_verify_prologue<6>(c);break;
  }
}

extern "C" __device__ void glm53_verify_output(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_output<1>(c);break;
    case 2:glm_verify_output<2>(c);break;
    case 3:glm_verify_output<3>(c);break;
    case 4:glm_verify_output<4>(c);break;
    case 5:glm_verify_output<5>(c);break;
    case 6:glm_verify_output<6>(c);break;
  }
}

extern "C" __device__ void glm53_verify_ffn_input(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_ffn_input<1>(c);break;
    case 2:glm_verify_ffn_input<2>(c);break;
    case 3:glm_verify_ffn_input<3>(c);break;
    case 4:glm_verify_ffn_input<4>(c);break;
    case 5:glm_verify_ffn_input<5>(c);break;
    case 6:glm_verify_ffn_input<6>(c);break;
  }
}

// args: weights/meta, (rt, sel) per position. Routing remains per row;
// correction bias is the only fixed input and is fetched once for the block.
template<int P> GLM_DEV void glm_verify_route(const etx_ctx* c){
  if(GLM_LANE!=0)return;
  float bias[GLM_EXPERTS];
  for(int e=0;e<GLM_EXPERTS;++e)bias[e]=glm_w(GLM_PB(0),GLM_PW(1)[BIAS],e);
  for(int p=0;p<P;++p){
    const float* rt=GLM_PR(2+2*p,const float*,GLM_S_RT);
    float* sel=GLM_PR(3+2*p,float*,GLM_S_SEL);
    float score[GLM_EXPERTS],biased[GLM_EXPERTS],prob[GLM_TOPK];int ids[GLM_TOPK];
    for(int e=0;e<GLM_EXPERTS;++e){score[e]=glm_sig(rt[e]);biased[e]=score[e]+bias[e];}
    for(int k=0;k<GLM_TOPK;++k){float best=-INFINITY;int id=INT_MAX;
      for(int e=0;e<GLM_EXPERTS;++e){bool used=false;for(int j=0;j<k;++j)used=used||ids[j]==e;
        if(!used&&biased[e]>best){best=biased[e];id=e;}}
      ids[k]=id;prob[k]=score[id];}
    float den=1e-20f;for(int k=0;k<GLM_TOPK;++k)den+=prob[k];
    for(int k=0;k<GLM_TOPK;++k)prob[k]=prob[k]/den*GLM_ROUTE_SCALE;
    for(int i=0;i<GLM_TOPK;++i)for(int j=i+1;j<GLM_TOPK;++j)if(ids[j]<ids[i]){
      int id=ids[i];ids[i]=ids[j];ids[j]=id;float v=prob[i];prob[i]=prob[j];prob[j]=v;}
    for(int k=0;k<GLM_TOPK;++k){sel[k]=(float)ids[k];sel[GLM_TOPK+k]=prob[k];}
  }
}
// Dense gate/up: the same rounding and clipping as glm_ph_up, with all rows
// consuming each gate/up weight before it is discarded.
template<int P> GLM_DEV void glm_verify_dense_up(const etx_ctx* c){
  const float* x[P];float* act[P];
  for(int p=0;p<P;++p){x[p]=GLM_PR(2+2*p,const float*,GLM_S_RT)+GLM_R4(GLM_EXPERTS);
    act[p]=GLM_PR(3+2*p,float*,GLM_S_ACT);}
  const uint8_t* b=GLM_PB(0);const glm_weight* w=GLM_PW(1);
  for(int r=GLM_LANE;r<w[MG].rows;r+=GLM_LANES){float gate[P]={},up[P]={};
    for(int j=0;j<w[MG].cols;++j){float g=glm_w(b,w[MG],r,j),u=glm_w(b,w[MU],r,j);
      for(int p=0;p<P;++p){gate[p]+=g*x[p][j];up[p]+=u*x[p][j];}}
    for(int p=0;p<P;++p){float a=fminf(glm_bf(gate[p]),GLM_CLIP);
      float v=fmaxf(-GLM_CLIP,fminf(glm_bf(up[p]),GLM_CLIP));
      act[p][r]=glm_bf(glm_bf(a*glm_sig(a))*v);}}
}
// args: weights/meta, (act, mid, work, output) per position.
template<int P> GLM_DEV void glm_verify_dense_down(const etx_ctx* c){
  const float *act[P],*mid[P],*hc[P];glm_act* out[P];
  for(int p=0;p<P;++p){act[p]=GLM_PR(2+4*p,const float*,GLM_S_ACT);
    mid[p]=GLM_PR(3+4*p,const float*,GLM_S_MID);
    hc[p]=GLM_PR(4+4*p,const float*,GLM_VWORK)+GLM_VHC;
    out[p]=GLM_PR(5+4*p,glm_act*,GLM_HC*GLM_H);}
  const glm_weight& w=GLM_PW(1)[MD];
  glm_verify_rows<P>(GLM_PB(0),w,act,0,w.rows,[&](int p,int j,float v){
    for(int k=0;k<GLM_HC;++k)out[p][k*GLM_H+j]=glm_put(glm_mix1(hc[p],[&](int i){return mid[p][i];},glm_bf(v),k,j));});
}

extern "C" __device__ void glm53_verify_route(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_route<1>(c);break;
    case 2:glm_verify_route<2>(c);break;
    case 3:glm_verify_route<3>(c);break;
    case 4:glm_verify_route<4>(c);break;
    case 5:glm_verify_route<5>(c);break;
    case 6:glm_verify_route<6>(c);break;
  }
}

extern "C" __device__ void glm53_verify_dense_up(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_dense_up<1>(c);break;
    case 2:glm_verify_dense_up<2>(c);break;
    case 3:glm_verify_dense_up<3>(c);break;
    case 4:glm_verify_dense_up<4>(c);break;
    case 5:glm_verify_dense_up<5>(c);break;
    case 6:glm_verify_dense_up<6>(c);break;
  }
}

extern "C" __device__ void glm53_verify_dense_down(const etx_ctx* c){
  switch(c->cst[2]){
    case 1:glm_verify_dense_down<1>(c);break;
    case 2:glm_verify_dense_down<2>(c);break;
    case 3:glm_verify_dense_down<3>(c);break;
    case 4:glm_verify_dense_down<4>(c);break;
    case 5:glm_verify_dense_down<5>(c);break;
    case 6:glm_verify_dense_down<6>(c);break;
  }
}
