"""FFN parallel helper contracts. Authored without executing tests/compilers.

The CPU test below compiles only when explicitly run. It compares EP to the
existing phased FFN, with nonowned descriptors removed and scratch poisoned.
It is not a GPU, compact-loader or full-model qualification. The TP check covers
ordered FP32 partial reduction and BF16 boundaries, not sharded matrix execution.
"""
import ctypes
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'examples/glm53/parallel_phases.h'


def test_parallel_helper_public_interfaces():
    source = HEADER.read_text()
    arguments = {
        'route': 'const uint8_t* b,const glm_weight* w,const float* rt,float* sel,bool moe,float* lds',
        'up': ('const uint8_t* b,const glm_weight* w,const float* rt,const float* sel,float* act,'
               'bool moe,bool tensor_parallel,int rank,int world,int shared_owner,int t,int T,float* lds'),
        'down': ('const uint8_t* b,const glm_weight* w,const float* act,const float* sel,float* expert_out,'
                 'bool moe,bool tensor_parallel,int rank,int world,int shared_owner,int t,int T,float* lds'),
        'combine': ('const uint8_t* b,const glm_weight* w,const float* sel,const float* const* rank_outputs,'
                    'const float* mid,const float* hcf,glm_act* out,bool moe,bool tensor_parallel,'
                    'int world,int shared_owner,int t,int T,float* lds'),
    }
    for name, want in arguments.items():
        match = re.search(r'GLM_DEV void glm_ph_parallel_' + name + r'\((.*?)\)\{', source, re.S)
        assert match, name
        assert re.sub(r'\s+', '', match.group(1)) == re.sub(r'\s+', '', want), name
    assert source.count('glm_topk(') == 1


CPU_SOURCE = r'''
#define GLM53_CPU 1
#include "examples/glm53/parallel_phases.h"
extern "C" int ep_check(const void* bp,const void* wp,const float* rt,
                        const float* mid,const float* hcf,int moe,int world,int shared_owner,int T){
  const uint8_t* b=(const uint8_t*)bp;const glm_weight* w=(const glm_weight*)wp;
  float lds[GLM_LDS_FLOATS],act[GLM_S_ACT],sel[GLM_S_SEL];
  glm_act want[GLM_HC*GLM_H],got[GLM_HC*GLM_H];
  for(int i=0;i<GLM_S_ACT;++i)act[i]=NAN;
  for(int t=0;t<T;++t)glm_ph_up(b,w,rt,act,sel,moe,t,T,lds);
  for(int t=0;t<T;++t)glm_ph_down(b,w,act,sel,mid,hcf,want,moe,t,T,lds);
  float ep_sel[GLM_S_SEL];glm_ph_parallel_route(b,w,rt,ep_sel,moe,lds);
  if(moe)for(int k=0;k<2*GLM_TOPK;++k)if(sel[k]!=ep_sel[k])return 10;
  float local_act[8][GLM_S_ACT],partial[8][(GLM_TOPK+1)*GLM_H];
  const float* outputs[8];
  for(int rank=0;rank<world;++rank){
    // Preserve global slot numbering but make any unowned access invalid. The
    // original blob is reused here; relocation/range copies belong to loader tests.
    glm_weight local[EXPERTS+3*GLM_EXPERTS];
    for(int s=0;s<EXPERTS+3*GLM_EXPERTS;++s)local[s]={};
    const int ne=moe?GLM_TOPK+1:1;
    for(int k=0;k<ne;++k)if(glm_parallel_owner(ep_sel,moe,k,world,shared_owner)==rank){
      const int slot=glm_parallel_slot(ep_sel,moe,k);
      for(int d=0;d<3;++d)local[slot+d]=w[slot+d];
    }
    for(int i=0;i<GLM_S_ACT;++i)local_act[rank][i]=NAN;
    for(int i=0;i<(GLM_TOPK+1)*GLM_H;++i)partial[rank][i]=NAN;
    for(int t=0;t<T;++t)glm_ph_parallel_up(b,local,rt,ep_sel,local_act[rank],moe,false,rank,world,shared_owner,t,T,lds);
    for(int t=0;t<T;++t)glm_ph_parallel_down(b,local,local_act[rank],ep_sel,partial[rank],moe,false,rank,world,shared_owner,t,T,lds);
    for(int k=0;k<GLM_TOPK+1;++k){
      const bool owned=k<ne&&glm_parallel_owner(ep_sel,moe,k,world,shared_owner)==rank;
      for(int j=0;j<GLM_H;++j){float v=partial[rank][k*GLM_H+j];
        if(!isfinite(v)||(!owned&&v!=0))return 20+rank;
      }
    }
    outputs[rank]=partial[rank];
  }
  for(int t=0;t<T;++t)glm_ph_parallel_combine(b,w,ep_sel,outputs,mid,hcf,got,moe,false,world,shared_owner,t,T,lds);
  for(int j=0;j<GLM_HC*GLM_H;++j)if(got[j]!=want[j])return 40;
  return 0;
}
extern "C" int tp_combine_check(const void* bp,const void* wp,const float* mid,const float* hcf,int moe,int world,int T){
  const uint8_t* b=(const uint8_t*)bp;const glm_weight* w=(const glm_weight*)wp;
  float lds[GLM_LDS_FLOATS],sel[GLM_S_SEL]={};
  float partial[8][(GLM_TOPK+1)*GLM_H];const float* outputs[8];
  for(int k=0;k<GLM_TOPK;++k){sel[k]=(float)k;sel[GLM_TOPK+k]=glm_bf(1.0f/(k+2));}
  glm_act got[GLM_HC*GLM_H],want[GLM_HC*GLM_H];
  for(int r=0;r<world;++r){outputs[r]=partial[r];
    for(int k=0;k<GLM_TOPK+1;++k)for(int j=0;j<GLM_H;++j){
      // Odd j: rank-order cancellation. Even j: rounding before reduction
      // loses a half-ULP which must survive the FP32 rank sum.
      float v=j%2?(r==0?1.0e8f:r==1?1.0f:r==2?-1.0e8f:0.0f):
                    (r==0?1.00390625f:r==1?0.00390625f:0.0f);
      partial[r][k*GLM_H+j]=v;
    }
  }
  for(int t=0;t<T;++t)glm_ph_parallel_combine(b,w,sel,outputs,mid,hcf,got,moe,true,world,0,t,T,lds);
  float hc[2*GLM_HC+GLM_HC*GLM_HC];glm_hc_coeffs(b,w,F_FN,hcf,hc,lds+GLM_LDS_HEAD);
  for(int j=0;j<GLM_H;++j){float y=0;
    for(int k=0;k<(moe?GLM_TOPK+1:1);++k){
      float dot=partial[0][k*GLM_H+j];for(int r=1;r<world;++r)dot+=partial[r][k*GLM_H+j];
      dot=glm_bf(dot);
      if(!moe)y=dot;
      else if(k<GLM_TOPK)y=glm_bf(y+glm_bf(dot*sel[GLM_TOPK+k]));
      else y=glm_bf(y+dot);
    }
    for(int c=0;c<GLM_HC;++c)want[c*GLM_H+j]=glm_put(glm_mix1(hc,[&](int i){return mid[i];},y,c,j));
  }
  for(int j=0;j<GLM_HC*GLM_H;++j)if(got[j]!=want[j])return 50;
  return 0;
}
'''


@pytest.mark.parametrize('bf16,fp8', [(False, False), (True, False), (True, True)])
def test_ep_cpu_matches_existing_phased_ffn(tmp_path, bf16, fp8):
    """Empty ranks, shared on last rank, dense fallback and nondividing tiles."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from examples.glm53 import cpu_check
    stage = tmp_path / 'fixture'
    spec, _, _ = cpu_check.fixture(stage, bf16=bf16, fp8=fp8)
    source = tmp_path / 'parallel.cpp'
    source.write_text(CPU_SOURCE)
    library = tmp_path / 'parallel.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC',
                    '-I', str(ROOT), '-I', str(stage), str(source), '-o', str(library)],
                   check=True, timeout=120)
    lib = ctypes.CDLL(str(library))
    lib.ep_check.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    lib.ep_check.restype = ctypes.c_int
    lib.tp_combine_check.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    lib.tp_combine_check.restype = ctypes.c_int
    d = spec['dimensions']
    rt = (ctypes.c_float * (d['EXPERTS'] + d['H']))()
    # Tiny EXPERTS=4 is already R4 aligned. Tie logits also exercise low-ID ties.
    assert d['EXPERTS'] % 4 == 0
    for j in range(d['EXPERTS'], len(rt)):
        rt[j] = (j % 5 - 2) * 0.125
    mid = (ctypes.c_float * (d['HC'] * d['H']))(*[(j % 7 - 3) * 0.0625 for j in range(d['HC'] * d['H'])])
    hcf = (ctypes.c_float * (2 * d['HC'] + d['HC'] ** 2))()
    for moe, layer in [(0, 'l0'), (1, 'l1')]:
        weights = spec['weights'][layer]
        blob = ctypes.create_string_buffer((stage / weights['blob']).read_bytes())
        meta = ctypes.create_string_buffer((stage / weights['meta']).read_bytes())
        for world in (1, 2, 4, 8):
            for tiles in (1,3,7,11):
                assert lib.tp_combine_check(blob,meta,mid,hcf,moe,world,tiles)==0, (moe,world,tiles)
            for shared in sorted({0, world - 1}):
                for tiles in (1, 3, 7, 11):
                    assert lib.ep_check(blob, meta, rt, mid, hcf, moe, world, shared, tiles) == 0, (moe, world, shared, tiles)
