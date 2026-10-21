"""Private static-dispatch LDS lifetime alias, preserving native operand storage.

Bootstrap and descriptor bytes are used before input staging. Every lane first
copies domain/worker and the descriptor by value, then crosses a block barrier.
The original all-wave release/barrier ends body lifetime before dispatch writes
its next descriptor. No numerical state or simultaneously live state aliases.
"""
from .protocol import require

def alias_dispatch(text,batch):
    require(batch in (2,4),'registered native batch')
    require(f'#define ETX_LDS_USED {batch*8192}' in text,'unexpected numerical arena size')
    require('every task on this device is static' in text and '#define ETX_RELAY 1' not in text,'only fixed no-relay micro queues are covered')
    require('#define ETX_EMIT_ARGPF 0' in text and '#define ETX_EMIT_BAR2 0' in text,'LDS lifetime proof excludes argument prefetch and two-parity dispatcher')
    decl='__shared__ __align__(16) unsigned char etx_lds[ETX_LDS_USED > 0 ? ETX_LDS_USED : 16];'
    require(text.count(decl)==2,'native task/unfused numerical arena anchors changed')
    marker='static __device__ __forceinline__ bool etx_deps_ready('
    require(text.count(marker)==1,'dispatch definition boundary changed')
    text=text.replace(marker,f'__shared__ __align__(16) unsigned char column_micro_arena[{batch*8192}];\n\n'+marker,1)
    text=text.replace(decl,'unsigned char* const etx_lds=column_micro_arena;')
    old='__shared__ uint32_t s_domain, s_worker; __shared__ int32_t s_tid, s_slot;'
    require(text.count(old)==1,'bootstrap dispatch state changed')
    text=text.replace(old,'uint32_t& s_domain=*(uint32_t*)(column_micro_arena+0); uint32_t& s_worker=*(uint32_t*)(column_micro_arena+4);\n  int32_t& s_tid=*(int32_t*)(column_micro_arena+8); int32_t& s_slot=*(int32_t*)(column_micro_arena+12);')
    old='const uint32_t domain = s_domain, worker = s_worker;'
    require(text.count(old)==1,'bootstrap by-value capture changed')
    text=text.replace(old,old+'\n  __syncthreads(); // ALIAS PROOF: every lane has persistent worker/domain values')
    old='__shared__ etx_task s_task;'
    require(text.count(old)==1,'static descriptor state changed')
    text=text.replace(old,'static_assert(sizeof(etx_task)==20,"descriptor alias bound");\n  etx_task& s_task=*(etx_task*)(column_micro_arena+16);')
    old='const etx_task cur = s_task;'
    require(text.count(old)==1,'descriptor by-value capture changed')
    text=text.replace(old,old+'\n    __syncthreads(); // ALIAS PROOF: every lane owns its descriptor before any staging write')
    require('etx_run_task(p, cur, worker, domain, tid);\n    __syncthreads();' in text,'post-body lifetime barrier changed')
    proof=dict(arena_bytes=batch*8192,dispatcher_live_bytes=36,phases=['bootstrap -> per-lane worker/domain copy -> block barrier','descriptor -> per-lane by-value copy -> block barrier','ready/owner polls use arena before numerical inputs exist','numerical staging/dot/publication -> all-wave drain -> block barrier','next descriptor may overwrite dead numerical inputs'],numeric_storage_unchanged=True,alias_live_state=False)
    return text,proof
