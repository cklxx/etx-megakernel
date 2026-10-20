"""Native experiments: CPU bit equality and gfx942 code generation, never a GPU run."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'examples/glm53'
SPEC = '''
#define GLM_H 7
#define GLM_HC 2
#define GLM_BF16 BF16_MODE
#define GLM_EPS 1e-6f
#define GLM_HCEPS 1e-6f
#define GLM_SINK 2
#define GLM_INTERMAX 11
#define GLM_CLIP 7.0f
#define GLM_EXPERTS 3
#define GLM_TOPK 2
#define GLM_ROUTE_SCALE 1.25f
#define GLM_LD 2
#define GLM_LH 2
#define GLM_QDIM 4
#define GLM_CONV 3
#define GLM_LOWER -5.0f
#define GLM_QR 3
#define GLM_NH 2
#define GLM_AD 2
#define GLM_KVR 3
#define GLM_IH 2
#define GLM_ID 2
#define GLM_MAXLEN 16
#define GLM_POOL 2
#define GLM_INDEX_TOPK 4
'''
CPU = r'''
#include "native.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// C headers only: no host C++ standard-library installation is needed.
static struct {
  uint8_t bytes[1<<22];size_t used=0;
  size_t size()const{return used;}
  uint8_t* data(){return bytes;}
  void push_back(uint8_t v){if(used==sizeof(bytes))abort();bytes[used++]=v;}
  void insert(const uint8_t* b,const uint8_t* e){while(b!=e)push_back(*b++);}
} blob;
static glm_weight make(int rows,int cols,int kind,int salt) {
  while(blob.size()%4)blob.push_back(0);
  glm_weight w={(uint64_t)blob.size(),0,rows,cols,kind,3};
  for(int i=0;i<rows*cols;++i){
    float x=((i*17+salt*7)%61-30)/128.0f;
    if(kind==2){uint8_t bytes[4];memcpy(bytes,&x,4);blob.insert(bytes,bytes+4);}
    else if(kind==1){union{float f;uint32_t u;}v;v.f=x;uint16_t b=v.u>>16;
      blob.push_back(b&255);blob.push_back(b>>8);}
    else blob.push_back((uint8_t)((i*13+salt)%120 | ((i&1)<<7)));
  }
  while(blob.size()%4)blob.push_back(0);
  w.scale=blob.size();
  for(int i=0;i<((rows+2)/3)*((cols+2)/3);++i){float f=(i%7+1)/128.0f;
    uint8_t bytes[4];memcpy(bytes,&f,4);blob.insert(bytes,bytes+4);}
  return w;
}
static void emit(const float* x,int n){if(fwrite(x,sizeof(float),n,stdout)!=(size_t)n)abort();}
int main(){
  // Independent original scalar-row loop catches summation-order changes,
  // tail rows/K chunks, all payload kinds and both rounded output modes.
  const int widths[]={1,7,13},heights[]={1,9,33};const bool rounds[]={false,true};
  for(int kind=0;kind<3;++kind)for(int cols: widths)for(int rows: heights){
    glm_weight w=make(rows,cols,kind,5);float x[13],got[33],ref[33];
    for(int i=0;i<cols;++i)x[i]=(i%5-2)*0.27f;
    for(bool rounded: rounds){
      for(int r=0;r<rows;++r){float sum=0;for(int c=0;c<cols;++c)sum+=glm_w(blob.data(),w,r,c)*x[c];ref[r]=rounded?glm_bf(sum):sum;}
      glm_mv(blob.data(),w,x,got,rounded);if(memcmp(got,ref,rows*sizeof(float)))return 2;
      glm_mv<true>(blob.data(),w,x,got,rounded);if(memcmp(got,ref,rows*sizeof(float)))return 3;
      emit(got,rows);
    }
  }
  for(int kind=0;kind<3;++kind){
    glm_weight w[EXPERTS+3*GLM_EXPERTS]={};
    const int gates[]={MG,MU,SG,SU},downs[]={MD,SD};
    for(int slot: gates)w[slot]=make(11,7,kind,slot);
    for(int slot: downs)w[slot]=make(7,11,kind,slot);
    w[ROUTER]=make(GLM_EXPERTS,7,kind,ROUTER);w[BIAS]=make(GLM_EXPERTS,1,2,BIAS);
    for(int e=0;e<GLM_EXPERTS;++e){int s=EXPERTS+3*e;w[s]=make(11,7,kind,s);w[s+1]=make(11,7,kind,s+1);w[s+2]=make(7,11,kind,s+2);}
    float x[7],y[7],a[128]={};for(int i=0;i<7;++i)x[i]=(i-3)*0.17f;
    glm_ffn(blob.data(),w,x,y,a,false);emit(y,7);
    glm_ffn(blob.data(),w,x,y,a,true);emit(y,7);
  }
}
'''
DEVICE = r'''
#include <hip/hip_runtime.h>
#include "native.h"
// Independent routed/shared/default payload paths, each with a compile-time
// dtype, make SLC placement and ordinary FP8 scale loads inspectable.
#define READER(NAME,POLICY,EXPLICIT,KIND) \
extern "C" __global__ void NAME(const uint8_t* b,float* y) { \
  glm_weight w={0,4096,16,16,KIND,4}; \
  y[threadIdx.x]=glm_w<POLICY,EXPLICIT>(b,w,threadIdx.x%16,threadIdx.x%16); }
READER(routed8, (ETX_GLM_ROUTED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 0)
READER(routed16,(ETX_GLM_ROUTED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 1)
READER(routed32,(ETX_GLM_ROUTED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 2)
READER(shared8, (ETX_GLM_SHARED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 0)
READER(shared16,(ETX_GLM_SHARED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 1)
READER(shared32,(ETX_GLM_SHARED_SLC!=0), (ETX_GLM_CACHE_AB!=0), 2)
READER(dense8, false, false, 0)
READER(dense16,false, false, 1)
READER(dense32,false, false, 2)
extern "C" __global__ void matvec(const uint8_t* b,const glm_weight* w,const float* x,float* y) {
  glm_mv(b,*w,x,y);
}
extern "C" __global__ void matvec_small(const uint8_t* b,const float* x,float* y) {
  glm_weight w={0,4096,16,16,2,4};glm_mv(b,w,x,y);
}
extern "C" __global__ void matvec_large(const uint8_t* b,const float* x,float* y) {
  glm_weight w={0,4096,16,17,2,4};glm_mv(b,w,x,y);
}
extern "C" __global__ void ffn(const uint8_t* b,const glm_weight* w,const float* x,float* y,float* a) {
  glm_ffn(b,w,x,y,a,true);
}
'''


def _cpu_compiler():
    compiler = shutil.which('c++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('a C++ compiler is required')
    return compiler


def _run(cmd):
    result = subprocess.run([str(x) for x in cmd], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    return result


def _defines(options):
    return [f'-DETX_GLM_{k}={v}' for k, v in options.items()]


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    out = tmp_path_factory.mktemp('glm-native-options')
    (out/'glm53_spec.h').write_text(SPEC)
    (out/'cpu.cpp').write_text(CPU)
    (out/'device.hip').write_text(DEVICE)
    return out


@pytest.mark.parametrize('bf16', [0, 1])
def test_cpu_default_and_opt_in_policies_are_bit_identical(sources, bf16):
    variants = [
        {},
        {'ROUTED_SLC': 1},
        {'SHARED_SLC': 1},
        {'ROUTED_SLC': 1, 'SHARED_SLC': 1},
        {'CACHE_AB': 1},
        {'CACHE_AB': 1, 'ROUTED_SLC': 1},
        {'CACHE_AB': 1, 'SHARED_SLC': 1},
        {'CACHE_AB': 1, 'ROUTED_SLC': 1, 'SHARED_SLC': 1},
        {'ROW_TILE': 2, 'K_TILE': 3},
        {'ROW_TILE': 4, 'K_TILE': 1},
        {'ROW_TILE': 32, 'K_TILE': 2147483647},
        {'K_TILE': 7},
        {'ROW_TILE': 3},
        {'LDS_BYTES': 4},  # CPU direct fallback; not a GPU barrier test.
        {'ROUTED_SLC': 1, 'SHARED_SLC': 1, 'ROW_TILE': 3, 'K_TILE': 5, 'LDS_BYTES': 64},
    ]
    baseline = None
    for i, options in enumerate(variants):
        exe = sources/f'cpu-{bf16}-{i}'
        _run([_cpu_compiler(), '-std=c++17', '-O2', '-ffp-contract=off', '-DGLM53_CPU',
              f'-DBF16_MODE={bf16}', *_defines(options), '-I', NATIVE, '-I', sources,
              sources/'cpu.cpp', '-o', exe])
        output = _run([exe]).stdout
        assert output
        if baseline is None:
            baseline = output
        assert output == baseline, options


@pytest.mark.parametrize('option,value', [
    ('ROUTED_SLC', -1), ('ROUTED_SLC', 2), ('SHARED_SLC', 2), ('CACHE_AB', 2),
    ('ROW_TILE', 0), ('ROW_TILE', 33), ('K_TILE', -1), ('K_TILE', 2147483648),
    ('LDS_BYTES', -4), ('LDS_BYTES', 3), ('LDS_BYTES', 65540),
])
def test_invalid_macros_are_compile_time_errors(sources, option, value):
    p = subprocess.run([_cpu_compiler(), '-std=c++17', '-DGLM53_CPU', '-DBF16_MODE=0',
                        *_defines({option: value}), '-I', str(NATIVE), '-I', str(sources),
                        '-fsyntax-only', str(sources/'cpu.cpp')], capture_output=True, timeout=30)
    assert p.returncode != 0
    assert f'ETX_GLM_{option} must be'.encode() in p.stderr


def _gfx942_compiler():
    compiler = os.environ.get('ETX_CLANG') or '/opt/homebrew/opt/llvm/bin/clang++'
    rocm = Path(os.environ.get('ETX_ROCM_HEADERS', str(ROOT/'build/external/rocm-headers')))
    devlibs = Path(os.environ.get('ETX_DEVLIBS', str(rocm/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    if not Path(compiler).is_file() or not (rocm/'inc/hip/hip_runtime.h').is_file() or not devlibs.is_dir():
        pytest.skip('local gfx942 clang, ROCm headers and device libraries required')
    return [compiler, '-x', 'hip', '--cuda-device-only', '--offload-arch=gfx942',
            '-O3', '-std=c++17', '-nogpuinc', f'--rocm-device-lib-path={devlibs}',
            '-isystem', str(rocm/'inc'), '-include', '__clang_hip_runtime_wrapper.h',
            '-D__HIP_PLATFORM_AMD__', '-DBF16_MODE=1']


def _kernel_asm(assembly, name):
    return assembly.split(f'{name}:', 1)[1].split(f'.Lfunc_end', 1)[0]


@pytest.mark.parametrize('cache_ab', [0, 1])
@pytest.mark.parametrize('routed,shared', [(0,0), (1,0), (0,1), (1,1)])
def test_gfx942_payload_cache_bits_are_isolated(sources, routed, shared, cache_ab):
    out = sources/f'cache-{cache_ab}-{routed}-{shared}.s'
    _run([*_gfx942_compiler(), *_defines({'ROUTED_SLC': routed, 'SHARED_SLC': shared, 'CACHE_AB': cache_ab}),
          '-I', NATIVE, '-I', sources, '-S', sources/'device.hip', '-o', out])
    assembly = out.read_text()
    for group, enabled in [('routed', routed), ('shared', shared), ('dense', 0)]:
        for bits, op in [(8, 'ubyte'), (16, 'ushort'), (32, 'dword')]:
            body = _kernel_asm(assembly, f'{group}{bits}')
            slc = [line for line in body.splitlines() if re.search(r'\bslc\b', line)]
            assert len(slc) == enabled, (group, bits, slc)
            if enabled:
                assert f'global_load_{op}' in slc[0]
                assert 's_waitcnt vmcnt(0)' in body
                if bits == 8:
                    # The dequantization scale is deliberately ordinary cached.
                    assert any('load_dword' in line and not re.search(r'\bslc\b', line)
                               for line in body.splitlines())
            if group != 'dense' and cache_ab and not enabled:
                assert f'global_load_{op}' in body and 's_waitcnt vmcnt(0)' in body
    # Exercise the real call sites too, not just isolated load wrappers. The
    # top-k2 routed loop contributes twice the three MLP projections of shared.
    ffn_slc = len(re.findall(r'\bslc\b', _kernel_asm(assembly, 'ffn')))
    assert bool(ffn_slc) == bool(routed or shared)
    if not routed and not shared:
        assert not re.search(r'\bslc\b', assembly)


def test_gfx942_matched_cache_ab_differs_only_in_cache_bits(sources):
    assemblies = []
    for enabled in (0, 1):
        out = sources/f'matched-{enabled}.s'
        _run([*_gfx942_compiler(), *_defines({'CACHE_AB': 1, 'ROUTED_SLC': enabled, 'SHARED_SLC': enabled}),
              '-I', NATIVE, '-I', sources, '-S', sources/'device.hip', '-o', out])
        assemblies.append(out.read_text())
    for group in ('routed', 'shared', 'dense'):
        for bits in (8, 16, 32):
            control = _kernel_asm(assemblies[0], f'{group}{bits}')
            treatment = _kernel_asm(assemblies[1], f'{group}{bits}')
            # Exact instructions/registers/scheduling, including waitcnts,
            # not merely matching instruction counts.
            assert control == re.sub(r'\s+slc\b', '', treatment)
            if group != 'dense':
                assert 's_waitcnt vmcnt(0)' in control


def test_hip_host_pass_accepts_explicit_cache_options(sources):
    flags = [flag for flag in _gfx942_compiler() if flag != '--cuda-device-only']
    _run([*flags, '--cuda-host-only', *_defines({'CACHE_AB': 1, 'ROUTED_SLC': 1, 'LDS_BYTES': 64}),
          '-I', NATIVE, '-I', sources, '-fsyntax-only', sources/'device.hip'])


def test_gfx942_nonzero_lds_and_tile_variant_compiles(sources):
    out = sources/'lds-tiles.s'
    _run([*_gfx942_compiler(), *_defines({'ROW_TILE': 3, 'K_TILE': 5, 'LDS_BYTES': 64}),
          '-I', NATIVE, '-I', sources, '-S', sources/'device.hip', '-o', out])
    assembly = out.read_text()
    body = _kernel_asm(assembly, 'matvec')
    # Dynamic direct/shared pointers use FLAT loads on gfx942; known-small K
    # below proves the same slab is read through dedicated LDS instructions.
    assert 'ds_write' in body and ('ds_read' in body or 'flat_load_dword' in body)
    small = _kernel_asm(assembly, 'matvec_small')
    large = _kernel_asm(assembly, 'matvec_large')
    assert 'ds_write' in small and 'ds_read' in small
    assert 'ds_write' not in large and 'ds_read' not in large
    assert 's_barrier' in body
    sizes = re.findall(r'\.amdhsa_group_segment_fixed_size\s+(\d+)\b', assembly)
    assert '64' in sizes and set(sizes) == {'0', '64'}  # no per-template duplication
    assert not re.search(r'\bslc\b', assembly)
    # Both staged and capacity-exceeded paths survive the dynamic K compile.
    assert re.search(r'\b(?:global|flat)_load_dword', body)
