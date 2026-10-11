"""Diagnostic-only GEMV load-group overlay. Ordinary generated source is untouched."""
import re

HELPERS = r'''
#include "etx/inbody.h"
struct etx_load_trace {
  uint64_t* records;
  unsigned count = 0;
  uint64_t begin = 0;
  __device__ explicit etx_load_trace(uint64_t* r) : records(r) {}
  __device__ __forceinline__ void start() {
    if (records) begin = etx_inbody_tick();
  }
  __device__ __forceinline__ void stop(unsigned bytes) {
    if (!records) return;
    // Input operands at the caller force every loaded value ready before this
    // timestamp. This is a diagnostic completion boundary, not an issue tick.
    const uint64_t end = etx_inbody_tick();
    if (etx_lane() == 0 && count < ETX_LOAD_GROUP_CAPACITY) {
      uint64_t* r = records + 1 + count * 3;
      r[0] = begin; r[1] = end; r[2] = bytes;
    }
    ++count;
  }
  __device__ __forceinline__ void finish() {
    if (records && etx_lane() == 0) records[0] = count;
  }
};
'''


def instrument(source):
    # Propagate a per-wave trace through the helper call chain, including the
    # GU1 direct call. Unsupported split/big-K paths cannot silently give rates:
    # their missing byte coverage is rejected by the CPU reader.
    functions = ('dsl_gemv_rows_nb', 'dsl_gemv_rows', 'dsl_gemv', 'dsl_gemv_nb',
                 'dsl_gemv_chunked', 'dsl_gemv_chunked_nb', 'dsl_gemv_bigk',
                 'dsl_gemv_split', 'dsl_gemv_accum')
    source = source.replace('// ---- the stream loop', HELPERS+'\n// ---- the stream loop', 1)
    for name in functions:
        source = re.sub(r'\b'+name+r'(<[^<>]*>)?\(', lambda m: m[0]+'lt, ', source)
        source = source.replace('void '+name+'(lt, ', 'void '+name+'(etx_load_trace& lt, ')
    source = re.sub(r'(void dsl_run_\w+\()(const unsigned char\* a)', r'\1etx_load_trace& lt, \2', source)
    # Entry points own one trace per wave and task; never a device global TLS.
    source = re.sub(r'(  dsl_run_\w+\()([^\n]+;)',
                    r'  etx_load_trace lt(c->inbody_load ? c->inbody_load + etx_wave() * ETX_LOAD_RECORD_WORDS : nullptr);\n\1lt, \2\n  lt.finish();', source)
    # Only bracket the real streamed weight loads, not prologue, dots, row
    # reduction, rounding or SiLU. Both the full groups and partial tail count.
    source = source.replace('  for (; s0 + U <= se; s0 += U) {',
                            '  for (; s0 + U <= se; s0 += U) {\n    lt.start();', 1)
    source = source.replace('  if (s0 < se) {', '  if (s0 < se) {\n    lt.start();', 1)
    fence = '''
    // Empty asm uses every dword: LLVM must resolve VMEM before the end tick.
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se)
#pragma unroll
      for (int j = 0; j < EPL / 2; ++j) asm volatile("" :: "v"(w[u][j]) : "memory");
    lt.stop(min(U, se - s0) * ETX_WAVE_SIZE * EPL * 2);
'''
    anchor = '#pragma unroll\n    for (int u = 0; u < U; ++u)'
    for suffix in (' consume(s0 + u, w[u]);', ' if (s0 + u < se) consume(s0 + u, w[u]);'):
        full = anchor+suffix
        if source.count(full) != 1: raise ValueError('load stamp consume anchor changed')
        source = source.replace(full, fence+full)
    return source
