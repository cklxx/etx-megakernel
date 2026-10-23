"""Execute the real batched MoE on CPU; compile device bodies without a GPU.

The injected ledger counts dequantized weight-element reads, not hardware
transactions or FP8 scale-cache traffic. WP-S1 is not part of this checkout.
"""
import os
from pathlib import Path
import subprocess

import pytest

from examples.glm53 import model_build, stage

ROOT = Path(__file__).resolve().parents[1]

CPU = r'''
#define GLM53_CPU 1
#include "examples/glm53/verify_moe.h"
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdio>
struct Ledger {
  const glm_weight* weights; int* counts;
  float operator()(const uint8_t* b,const glm_weight& w,int r,int c) const {
    ++counts[(&w-weights)*GLM_H*GLM_INTERMAX+r*w.cols+c];
    return glm_w(b,w,r,c);
  }
};
int main(){
  constexpr int P=POSITIONS, N=EXPERTS+3*GLM_EXPERTS;
  std::vector<glm_weight> w(N);
  std::vector<uint8_t> b(65536);
  int offset=0;
  for(int e=0;e<=GLM_EXPERTS;++e){
    int gs=e==GLM_EXPERTS?SG:EXPERTS+3*e;
    for(int m=0;m<3;++m){
      const int width=e==GLM_EXPERTS?GLM_INTERMAX:GLM_INTERMAX-2;
      auto& d=w[gs+m];d.rows=m==2?GLM_H:width;
      d.cols=m==2?width:GLM_H;d.kind=KIND;d.block=3;
      d.offset=offset;offset+=d.rows*d.cols*(KIND==2?4:KIND==1?2:1);
      offset=(offset+3)/4*4;d.scale=offset;
      const int ns=((d.rows+2)/3)*((d.cols+2)/3);
      for(int s=0;s<ns;++s){float v=.125f*(1+s%3);memcpy(b.data()+offset,&v,4);offset+=4;}
      for(int i=0;i<d.rows*d.cols;++i){
        float v=((i*17+gs*7)%31-15)*.25f;
        if(KIND==2)memcpy(b.data()+d.offset+4*i,&v,4);
        if(KIND==1){union{float f;uint32_t u;} z;z.f=glm_bf(v);uint16_t h=z.u>>16;memcpy(b.data()+d.offset+2*i,&h,2);}
        if(KIND==3)b[d.offset+i]=(i%2?128:0)|(40+i%25);
      }
    }
  }
  float x[P][GLM_H],sel[P][2*GLM_TOPK];
  float act[P][(GLM_TOPK+1)*GLM_INTERMAX],part[P][(GLM_TOPK+1)*GLM_H];
  const float *xp[P],*sp[P],*ap[P];float *aw[P],*pw[P];
  for(int p=0;p<P;++p){
    xp[p]=x[p];sp[p]=sel[p];ap[p]=act[p];aw[p]=act[p];pw[p]=part[p];
    for(int c=0;c<GLM_H;++c)x[p][c]=glm_bf(((p*5+c*3)%19-9)*1.25f);
    // Same IDs, partially overlapping IDs, and a disjoint row in one union.
    int first=PATTERN==0?0:PATTERN==1?(p%2):2*(p%2);
    for(int k=0;k<GLM_TOPK;++k){sel[p][k]=first+k;sel[p][GLM_TOPK+k]=.25f*(k+1);}
    for(float& v:act[p])v=NAN;for(float& v:part[p])v=NAN;
  }
  std::vector<int> counts(N*GLM_H*GLM_INTERMAX);
  Ledger read{w.data(),counts.data()};
  for(int e=0;e<=GLM_EXPERTS;++e)for(int t=0;t<TILES;++t)
    glm_verify_moe_up<P>(b.data(),w.data(),xp,sp,aw,e,t,TILES,read);
  for(int e=0;e<=GLM_EXPERTS;++e)for(int t=0;t<TILES;++t)
    glm_verify_moe_down<P>(b.data(),w.data(),ap,sp,pw,e,t,TILES,read);
  for(int e=0;e<=GLM_EXPERTS;++e){
    bool used=e==GLM_EXPERTS;
    for(int p=0;p<P;++p)for(int k=0;k<GLM_TOPK;++k)used=used||(int)sel[p][k]==e;
    int gs=e==GLM_EXPERTS?SG:EXPERTS+3*e;
    for(int m=0;m<3;++m)for(int i=0;i<w[gs+m].rows*w[gs+m].cols;++i)
      assert(counts[(gs+m)*GLM_H*GLM_INTERMAX+i]==(used?1:0));
  }
  // Native decode MLP is the independent per-row oracle, including rounding.
  for(int p=0;p<P;++p){
    float expected[GLM_H]={},scratch[2*GLM_INTERMAX],out[GLM_H];
    for(int k=0;k<=GLM_TOPK;++k){
      int gs=k==GLM_TOPK?SG:EXPERTS+3*(int)sel[p][k];
      glm_mlp(b.data(),w.data(),gs,gs+1,gs+2,x[p],out,scratch);
      for(int c=0;c<w[gs].rows;++c)assert(scratch[c]==act[p][k*GLM_INTERMAX+c]);
      for(int c=0;c<GLM_H;++c){
        assert(out[c]==part[p][k*GLM_H+c]);
        expected[c]=glm_bf(expected[c]+(k==GLM_TOPK?out[c]:glm_bf(out[c]*sel[p][GLM_TOPK+k])));
      }
    }
    for(int c=0;c<GLM_H;++c)assert(expected[c]==glm_verify_moe_mix(part[p],sel[p],c));
  }
  puts("per-row equality and union read-once OK");
}
'''

DEVICE = r'''
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include "examples/glm53/verify_moe.h"
extern "C" __global__ void verify_moe_compile(const uint8_t* b,const glm_weight* w,
 const float* const* x,const float* const* sel,float* const* act,
 const float* const* ca,float* const* partial,float* out) {
 glm_verify_moe_up<6>(b,w,x,sel,act,blockIdx.x);
 // Compile-only entry; these calls require separate launches in execution.
 glm_verify_moe_down<6>(b,w,ca,sel,partial,blockIdx.x);
 if(threadIdx.x==0)out[0]=glm_verify_moe_mix(partial[0],sel[0],0);
}
'''


def run(cmd):
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    work = tmp_path_factory.mktemp('verify-moe')
    d = model_build.production_compile_spec()['dimensions'].copy()
    d.update(H=7, EXPERTS=5, TOPK=2, INTERMAX=5)
    (work / 'glm53_spec.h').write_text(stage.header(d))
    (work / 'cpu.cpp').write_text(CPU)
    (work / 'device.cu').write_text(DEVICE)
    return work


@pytest.mark.parametrize('positions', [1, 2, 4, 6])
@pytest.mark.parametrize('pattern', [0, 1, 2])
@pytest.mark.parametrize('kind,bf16', [(1, 1), (2, 0), (2, 1), (3, 1)])
def test_per_row_equality_and_read_once(sources, tmp_path, positions, pattern, kind, bf16):
    # More tiles than rows also exercises empty ranges and odd tails.
    for tiles in (1, 3, 9):
        spec = (sources / 'glm53_spec.h').read_text().replace('#define GLM_BF16 1', f'#define GLM_BF16 {bf16}')
        (tmp_path / 'glm53_spec.h').write_text(spec)
        exe = tmp_path / 'moe'
        run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-I', tmp_path,
             '-I', ROOT, f'-DPOSITIONS={positions}', f'-DPATTERN={pattern}',
             f'-DKIND={kind}', f'-DTILES={tiles}', sources / 'cpu.cpp', '-o', exe])
        assert 'union read-once OK' in run([exe])


@pytest.fixture(params=['small', 'production'])
def device_spec(request, sources, tmp_path):
    d = model_build.production_compile_spec()['dimensions']
    spec = stage.header(d) if request.param == 'production' else (sources / 'glm53_spec.h').read_text()
    (tmp_path / 'glm53_spec.h').write_text(spec)
    return tmp_path


def test_sm90_compiles(sources, tmp_path, device_spec):
    cuda = Path(os.environ.get('CUDA', '/opt/cuda'))
    nvcc = cuda / 'bin/nvcc'
    if not nvcc.is_file():
        pytest.skip('CUDA nvcc unavailable')
    obj = tmp_path / 'moe.o'
    run([nvcc, '-arch=sm_90', '-std=c++17', '--fmad=false', '-I', ROOT,
         '-I', device_spec, '-c', sources / 'device.cu', '-o', obj])
    assert obj.stat().st_size > 0


def test_gfx942_compiles(sources, tmp_path, device_spec):
    rocm = Path(os.environ.get('ROCM', '/data/rocm-headers/rocm-llvm/opt/rocm-7.2.4'))
    compiler = os.environ.get('ETX_CLANG') or str(rocm / 'lib/llvm/bin/clang++')
    headers = Path(os.environ.get('ETX_ROCM_HEADERS', '/data/rocm-headers'))
    devlibs = Path(os.environ.get('ETX_DEVLIBS', str(headers / 'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    if not Path(compiler).is_file() or not devlibs.is_dir():
        pytest.skip('ROCm clang and device libraries unavailable')
    obj = tmp_path / 'moe.o'
    run([compiler, '-x', 'hip', '--cuda-device-only', '--offload-arch=gfx942',
         '-std=c++17', '-O2', '-ffp-contract=off', '-nogpuinc', '-fgpu-rdc',
         f'--rocm-device-lib-path={devlibs}', '-isystem', headers / 'inc',
         '-include', '__clang_hip_runtime_wrapper.h', '-D__HIP_PLATFORM_AMD__',
         '-I', ROOT, '-I', device_spec, '-c', sources / 'device.cu', '-o', obj])
    assert obj.stat().st_size > 0
