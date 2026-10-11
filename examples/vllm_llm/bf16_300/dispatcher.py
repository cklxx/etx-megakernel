"""Sparse ETX machinery probes on a TWO-TASK graph of the current MoE tiles.

This reuses the production kernel emitter. No new full-model system. The graph
is a harness: args contains the two original DSL blocks at offsets 0 and 160.
trace_time uses its existing four-u64/task allocation. tr[0:2] are ticks,
tr[2:4] task/worker. Saving happens after the slice, never within it.
Descriptor includes FIFO fetch and broadcast; wait includes existing argument
setup/pre-wait/acquire; exit includes original publication/arrivals. They are
labelled separately and cannot be silently reassigned to tile-loop work.
"""
from etx.codegen import emit_kernel
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.passes import compile_graph, PassOptions
from .source import replace_one
from .protocol import require

PAIRS = ('descriptor', 'wait', 'exit', 'empty')


def graph(tile_file):
    g = Graph('bf16_300_dispatcher')
    g.codegen.update(coh=1, fold_topk=1)
    g.tensor('args', (320,), bytes_per_elem=1)
    g.etensor('up_done', (1,), wait_count=1)
    resource = Resource(threads=1024, vgpr=128, lds_bytes=0)
    g.call_device('up', (1,), hip_link(str(tile_file), 'etx_tile_dsl_moe_up'), args=['args'], resource=resource,
                  consts=[0], out_edges={'up_done': 'i->(0)'}, duration_us=20)
    g.call_device('down', (1,), hip_link(str(tile_file), 'etx_tile_dsl_moe_down'), args=['args'], resource=resource,
                  consts=[160], in_edges={'up_done': 'i->(0)'}, duration_us=12)
    return compile_graph(g, 'gfx942', {}, options=PassOptions(inline_tiles=True))


def instrument(plan, pair, task=1):
    require(pair in PAIRS and task >= 0, 'dispatcher_selection')
    s = emit_kernel(plan)
    # Disable the general trace's stores in this clone; retain the SAME payload,
    # queue, waits, barriers, releases, abort handling and imported calls.
    s = replace_one(s, '  if (tr && threadIdx.x == 0) tr[0] = (uint64_t)ETX_TIMER();', '')
    s = replace_one(s, '  if (tr && threadIdx.x == 0) tr[1] = (uint64_t)ETX_TIMER();', '')
    s = replace_one(s, '  if (tr && threadIdx.x == 0) { tr[2] = (uint64_t)ETX_TIMER(); tr[3] = worker; }', '')
    selected = f'tid == {task} && tr && threadIdx.x == 0'
    save = f'  if ({selected}) {{ tr[0] = bf_begin; tr[1] = bf_end; tr[2] = tid; tr[3] = worker; }}'
    if pair == 'descriptor':
        start = '  for (;;) {\n    if (threadIdx.x == 0) {'
        s = replace_one(s, start, '  for (;;) {\n    uint64_t bf_begin = 0, bf_end = 0;\n    if (threadIdx.x == 0) bf_begin = (uint64_t)ETX_TIMER();\n    if (threadIdx.x == 0) {')
        anchor = '    const etx_task cur = s_task;'
        s = replace_one(s, anchor, anchor + '\n    if (threadIdx.x == 0) bf_end = (uint64_t)ETX_TIMER();')
        anchor = '    etx_run_task(p, cur, worker, domain, tid);\n    __syncthreads();'
        s = replace_one(s, anchor, anchor + f'\n    if (tid == {task} && p.trace_time && threadIdx.x == 0) {{\n'
                        '      uint64_t* q = p.trace_time + (size_t)tid * 4; q[0] = bf_begin; q[1] = bf_end; q[2] = tid; q[3] = worker; }')
    else:
        start = s.index('static __device__ __forceinline__ void etx_run_task(')
        finish = s.index('#ifndef ETX_MIN_WG_PER_CU', start) + len('#ifndef ETX_MIN_WG_PER_CU')
        prefix, suffix = s[:start], s[finish:]
        s = s[start:finish]
        anchor = '  int sym = -1;'
        begin = '  uint64_t bf_begin = 0, bf_end = 0;\n'
        if pair in ('wait', 'empty'):
            begin += f'  if ({selected}) bf_begin = (uint64_t)ETX_TIMER();\n'
        if pair == 'empty':
            begin += f'  if ({selected}) bf_end = (uint64_t)ETX_TIMER();\n'
        s = replace_one(s, anchor, begin + anchor)
        if pair == 'wait':
            anchor = '  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body'
            s = replace_one(s, anchor, f'  if ({selected}) bf_end = (uint64_t)ETX_TIMER();\n' + anchor)
        if pair == 'exit':
            anchor = '  if (threadIdx.x != 0) return;'
            s = replace_one(s, anchor, f'  if ({selected}) bf_begin = (uint64_t)ETX_TIMER();\n' + anchor)
        anchor = '  }\n}\n\n#ifndef ETX_MIN_WG_PER_CU'
        before = f'  if ({selected}) bf_end = (uint64_t)ETX_TIMER();\n' if pair == 'exit' else ''
        s = replace_one(s, anchor, '  }\n' + before + save + '\n}\n\n#ifndef ETX_MIN_WG_PER_CU')
        s = prefix + s + suffix
    return s
