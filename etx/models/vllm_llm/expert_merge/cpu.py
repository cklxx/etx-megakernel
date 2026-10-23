"""Compile and run extracted production arithmetic on synthetic CPU inputs.

No CPU rsqrt emulation can certify AMD device arithmetic or visibility.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .source import emit, parts

STUBS = r'''
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#define __device__
#define __forceinline__ inline
#define DSL_G
#define DSL_NB 1
#define DSL_RB 8
#define ETX_EMIT_DSL_EXT 0
#define ETX_DSL_VECNORM 1
#define ETX_DSL_RBATCH 1
#define ETX_DSL_NORM_HOIST 1
using __hip_bfloat16=uint16_t;
using dsl_u32x4=uint32_t __attribute__((ext_vector_type(4)));
struct Dim { int x; } threadIdx, blockDim{1024};
static float observed[1024], total=2048.f;
static unsigned __float_as_uint(float f) { unsigned u; memcpy(&u,&f,4); return u; }
static float __uint_as_float(unsigned u) { float f; memcpy(&f,&u,4); return f; }
static uint16_t __float2bfloat16(float f) { unsigned u=__float_as_uint(f); return (u+0x7fff+((u>>16)&1))>>16; }
static float __bfloat162float(uint16_t u) { return __uint_as_float(unsigned(u)<<16); }
static float dsl_bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
static float dsl_a_ldbf(const uint16_t* p) { return __bfloat162float(*p); }
static void dsl_a_stbf(uint16_t* p,uint16_t v) { *p=v; }
static float dsl_a_ldf(const float* p) { return *p; }
static void dsl_a_stf(float* p,float v) { *p=v; }
static dsl_u32x4 dsl_a_ld128(const uint16_t* p) { dsl_u32x4 v; memcpy(&v,p,16); return v; }
static void dsl_a_st128(uint16_t* p,dsl_u32x4 v) { memcpy(p,&v,16); }
static float etx_block_sum(float v,float*) { observed[threadIdx.x]=v; return total; }
static float rsqrtf(float x) { return 1.f/sqrtf(x); }
static void __syncthreads() {}
struct DslArgs { int K=2048,R=8,mode=1; float eps=1.e-6f; const uint16_t* ln=nullptr; };
struct etx_ctx { void** args; };
static float tree(const float* ss) {
  float waves[16]{};
  for(int w=0;w<16;++w) {
    float a[64],b[64];
    for(int i=0;i<64;++i) a[i]=ss[64*w+i];
    for(int off=32;off;off>>=1) {
      for(int i=0;i<64;++i) b[i]=a[i]+a[i^off];
      memcpy(a,b,sizeof(a));
    }
    waves[w]=a[0];
  }
  float s=0.f; for(float v:waves) s+=v; return s;
}
'''

MAIN = r'''
int main() {
  uint32_t rng=20261015; int cases=0,order_sensitive=0,boundary_sensitive=0;
  for(int sample=0;sample<32;++sample) for(int head=0;head<2;++head) {
    std::vector<uint16_t> rows(8*2048),rin(2048),ln(2048),rout(2048,0x7fc1),refout(2048),candidateout(2048),merged(2048);
    std::vector<float> ref(8192),candidate(8192),partial(1024),refpartial(1024);
    DslArgs d; d.ln=ln.data();
    for(int i=0;i<2048;++i) {
      rin[i]=__float2bfloat16(float(i%9-4)/8); ln[i]=__float2bfloat16(float(i%7+5)/8);
      for(int s=0;s<8;++s) { rng=rng*1664525+1013904223; rows[s*2048+i]=__float2bfloat16(float(int(rng%4001)-2000)/128); }
      if(i%3<2) {
        for(int s=0;s<8;++s) rows[s*2048+i]=0;
        if(i%3==0) { rows[i]=__float2bfloat16(1073741824.f); rows[2048+i]=__float2bfloat16(-1073741824.f); rows[4096+i]=__float2bfloat16(1.f); }
        else { rows[i]=__float2bfloat16(1.f); rows[2048+i]=__float2bfloat16(1.f/256); rin[i]=__float2bfloat16(-1.f); }
      }
      float f=0,b=0; for(int s=0;s<8;++s) f+=__bfloat162float(rows[s*2048+i]);
      for(int s=7;s>=0;--s) b+=__bfloat162float(rows[s*2048+i]);
      order_sensitive+=f!=b;
      boundary_sensitive+=dsl_bfr(dsl_bfr(f)+__bfloat162float(rin[i]))!=dsl_bfr(f+__bfloat162float(rin[i]));
    }
    float inverse=0;
    void* args[]={nullptr,merged.data(),partial.data(),&inverse}; etx_ctx c{args};
    // First collect the reference per-thread statistics. The block sum stub
    // records its input. Then run the real fold again with the actual CPU tree.
    for(int t=0;t<1024;++t) { threadIdx.x=t; reference_norm_at(d,rows.data(),rin.data(),nullptr,refout.data(),ref.data(),ref.data()+4096,true); refpartial[t]=observed[t]; }
    total=tree(refpartial.data()); inverse=rsqrtf(total/2048.f+d.eps);
    for(int t=0;t<1024;++t) { threadIdx.x=t; reference_norm_at(d,rows.data(),rin.data(),nullptr,refout.data(),ref.data(),ref.data()+4096,true); }
    // A permutation visits every row exactly once. Owner identity does not
    // enter the arithmetic. This tests both canonical residual roles.
    for(int k=0;k<32;++k) {
      const int g=(k*13+sample)%32;
      for(int t=0;t<1024;++t) { threadIdx.x=t; row_fold(d,&c,g,rows.data(),rin.data(),head?nullptr:rout.data(),candidate.data()); }
    }
    for(int t=0;t<256;++t) if(__float_as_uint(partial[t])!=__float_as_uint(refpartial[t])) return 10;
    if(__float_as_uint(tree(partial.data()))!=__float_as_uint(total)) return 11;
    for(int i=0;i<2048;++i) if(rout[i] != (head ? 0x7fc1 : merged[i])) return 12;
    for(int t=0;t<1024;++t) { threadIdx.x=t; candidate_norm(d,&c,rows.data(),rin.data(),nullptr,candidateout.data(),candidate.data(),candidate.data()+4096,true); }
    if(refout!=candidateout) return 13;
    ++cases;
  }
  if(!order_sensitive || !boundary_sensitive) return 14;
  printf("CPU-ARITHMETIC OK cases=%d ordered_counterexamples=%d bf16_boundary_counterexamples=%d; original per-thread statistics and normalized bf16 exact; CPU rsqrt proxy only\n",cases,order_sensitive,boundary_sensitive);
}
'''


def program():
    original,fold=parts()
    src=emit(True,False)
    start=src.index("template <int FixedK = 0, int FixedR = -1>\nstatic __device__ __forceinline__ void eem_stage_norm")
    end=src.index("static __device__ __forceinline__ void eem_append",start)
    norm=src[start:end].replace("eem_stage_norm(","candidate_norm(")
    start=src.index("  for (int i = group*8")
    end=src.index("  ETX_RELEASE_DOMAIN();",start)
    row=src[start:end]
    row="static void row_fold(const DslArgs& d,const etx_ctx* c,int group,const uint16_t* rows,const uint16_t* rin,uint16_t* rout,float* xs) { const int K=2048,R=8; const bool store=true; float ss=0.f;\n"+row+"}\n"
    return STUBS+original.replace("dsl_stage_norm_at(","reference_norm_at(")+row+norm+MAIN


def run(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    cpp=out/"arithmetic.cpp"; cpp.write_text(program())
    clang=os.environ.get("CXX") or ("/opt/homebrew/opt/llvm/bin/clang++" if sys.platform=="darwin"
                                    and Path("/opt/homebrew/opt/llvm/bin/clang++").is_file() else shutil.which("clang++"))
    if not clang: raise RuntimeError("CPU clang++ is missing")
    sdk=["-isysroot",subprocess.check_output(["xcrun","--show-sdk-path"],text=True).strip()] if sys.platform=="darwin" else []
    for contract in ("off","fast"):
        exe=out/("arithmetic-"+contract)
        subprocess.run([clang,"-std=c++17","-O2","-ffp-contract="+contract,*sdk,str(cpp),"-o",str(exe)],check=True)
        subprocess.run([str(exe.resolve())],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("out",type=Path)
    run(p.parse_args().out)


if __name__=="__main__": main()
