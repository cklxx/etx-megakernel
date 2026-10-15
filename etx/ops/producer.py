"""Experimental o_proj-owned router scoring. Default off. No hardware claims."""
from __future__ import annotations
import os
from pathlib import Path

NAME = 'dsl_gemv_producer'
PARTIALS = 256 * 129
TAGS = PARTIALS
COUNTER = TAGS + 256
EPOCH = COUNTER + 1
WORDS = EPOCH + 63


def enabled():
    value = os.environ.get('ETX_VL_PRODUCER_SCORE', '0')
    if value not in ('0', '1'):
        raise ValueError('ETX_VL_PRODUCER_SCORE accepts only 0 or 1')
    return value == '1'


def validate(d, *, fold, gemv, topk, fold_topk, cu, requests, batch, coh, uc):
    from etx.ops import local_fold
    if not (d.L == 48 and d.MI == 768 and d.qk_norm and d.norm_topk and not d.sparsemixer and d.H == 2048 and d.E == 128 and d.TOPK == 8
            and d.NH * d.HD == 4096 and fold and gemv and topk and not fold_topk
            and cu == 256 and requests == 1 and not batch and coh and not uc
            and not (d.ln_norm or d.bias_o or d.bias_lm) and not local_fold.enabled()
            and os.environ.get('ETX_COHERENCE', 'default') == 'default'):
        raise ValueError('producer scoring requires coherent default-cache Qwen3-30B-A3B bf16 B1, '
                         'CU=256, norm folds, standalone DSL topk, no UC/local fold/bias')


def import_entry(base):
    import copy
    item = copy.deepcopy(base)
    item.update(export='etx_tile_' + NAME, kernel=NAME, dsl='producer', arg_bytes=192)
    for off in (160, 168, 176):
        item['params'].append(dict(ty='ptr addrspace(1)', name=f'%{len(item["params"])}',
                                   size=8, align=8, offset=off))
    return item


TILE = r'''
// The original hypothesis: completed o_proj owners score BEFORE the global scale.
#if !defined(ETX_DSL_COH) || defined(ETX_DSL_COH_L1) || DSL_NB != 1 || ETX_EMIT_DSL_EXT
#error "producer scoring requires coherent default-cache bf16 batch one"
#endif
#define ETX_PRODUCER_INLINE static __device__ __forceinline__
#include "etx/ops/producer_math.h"
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_gemv_producer(const etx_ctx* c) {
  const unsigned char* a = etx_imp_argblock(c->args, c->cst[0]);
  const DslArgs d = dsl_args(a);
  const int blk = etx_uniform(c->coord[0]);
  float* xs = (float*)etx_lds_arena;
  float* part = xs + DSL_K_MAX;
  const __hip_bfloat16* wr = *(const __hip_bfloat16* const*)(a + 160);
  __hip_bfloat16* normout = *(__hip_bfloat16* const*)(a + 168);
  int* tei = *(int* const*)(a + 176);
  float* scratch = (float*)c->args[1];
  auto* tags = (unsigned*)scratch + @TAGS@;
  auto* count = (unsigned*)scratch + @COUNTER@;
  auto* epochp = (unsigned*)scratch + @EPOCH@;
  const unsigned epoch = __hip_atomic_load(epochp, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  // Same GEMV primitive, row ownership and output rounding as the baseline.
  dsl_stage(d.x, 4096, xs);
  dsl_gemv_chunked_nb<1>(d.W, 4096, DSL_K_MAX, xs, blk*8, blk*8+8, part,
      [&](int r, const float* v) { dsl_a_stbf(d.out+r, __float2bfloat16(v[0])); });
  // All GEMV waves joined. The owner's eight residual elements now exist.
  if (threadIdx.x < 8) {
    const int j = threadIdx.x, h = blk*8+j;
    const float z = dsl_bfr(dsl_a_ldbf(d.out+h) + dsl_a_ldbf(d.resid_in+h));
    xs[j] = z; xs[8+j] = __bfloat162float(d.ln[h]);
    dsl_a_stbf(d.resid_out+h, __float2bfloat16(z));
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    const int e = threadIdx.x;
    float w[8];
    for (int j=0;j<8;++j) w[j] = __bfloat162float(wr[e*2048+blk*8+j]);
    dsl_a_stf(scratch+blk*129+e, etx_producer_dot8(w, xs, xs+8));
  }
  if (threadIdx.x == 0) {
    dsl_a_stf(scratch+blk*129+128, etx_producer_square8(xs));
    __hip_atomic_store(tags+blk, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // Every payload writer fences. A leader-only fence would miss other waves.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned old = __hip_atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
    if (old >= 256u) __builtin_trap();
    xs[16] = old == 255u ? 1.f : 0.f;
  }
  __syncthreads();
  if (xs[16] == 0.f) return; // Never spin. Other workers can retire.
  const int tid = threadIdx.x;
  if (tid < 256 && __hip_atomic_load(tags+tid, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != epoch)
    __builtin_trap();
  // Identical norm ownership: 256 dot8 square sums, wave butterfly, wave order.
  const float ss = etx_block_sum(tid < 256 ? dsl_a_ldf(scratch+tid*129+128) : 0.f, part);
  const float scale = rsqrtf(ss/2048.f+d.eps);
  if (tid < 256) {
    for (int j=0;j<8;++j) {
      const int h=tid*8+j;
      const float z=dsl_a_ldbf(d.resid_out+h);
      const float n=dsl_bfr((z*scale)*__bfloat162float(d.ln[h]));
      dsl_a_stbf(normout+h, __float2bfloat16(n));
    }
  }
  if (tid < 128) {
    float score=0.f;
    for (int p=0;p<256;++p) score += dsl_a_ldf(scratch+p*129+tid);
    dsl_a_stbf((__hip_bfloat16*)d.logits+tid, __float2bfloat16(score*scale));
  }
  __syncthreads();
  // Use the kept implementation, including KEY32. No new selector.
  int* ids=(int*)xs; float* weights=xs+16;
  dsl_topk_lds(d.logits,128,8,1,ids,weights);
  if (tid < 8) for (int x=0;x<8;++x) {
    dsl_a_sti((int*)d.ids+x*8+tid,ids[tid]);
    dsl_a_sti(tei+x*8+tid,tid);
    dsl_a_stf((float*)d.wts+x*8+tid,weights[tid]);
  }
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    if (epoch == 0xffffffffu) __builtin_trap(); // Restart the process before wrap.
    __hip_atomic_store(epochp,epoch+1,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(count,0u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  // Runtime completion includes this finalizer, the normalized input and routing fanout.
}
extern "C" __device__ __attribute__((always_inline)) void etx_pre_dsl_gemv_producer(const etx_ctx* c) {
  etx_imp_touch(etx_imp_argblock(c->args,c->cst[0]),192);
}
extern "C" __device__ __attribute__((always_inline)) void etx_touch_dsl_gemv_producer(const etx_ctx* c) {
  dsl_touch_plain(etx_imp_argblock(c->args,c->cst[0]),c->coord[0]);
}
'''


def emit():
    source = TILE.replace('#include "etx/ops/producer_math.h"',
                          Path(__file__).with_name("producer_math.h").read_text().replace("#pragma once", ""))
    for key in ('TAGS', 'COUNTER', 'EPOCH'):
        source = source.replace('@'+key+'@', str(globals()[key]))
    return source


def apply(graph, dims):
    targets = [g for g in graph.grids if g.name.startswith('o_proj_')]
    if len(targets) != dims.L:
        raise ValueError('producer plan missing o_proj owners')
    graph.tensor('producer_scratch', (WORDS,), role='scratch', dtype='fp32', bytes_per_elem=4)
    for g in targets:
        if g.grid != (256,) or g.body.symbol != 'etx_tile_'+NAME:
            raise ValueError('producer plan must have 256 global o_proj owners')
        g.args = ['impargs', 'producer_scratch']
    graph.codegen['producer_score'] = True
