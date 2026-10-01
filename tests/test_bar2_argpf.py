"""ETX_VL_BAR2 / ETX_VL_ARGPF contracts in the emitted kernel: off = byte-identical emission; BAR2 = the static
loop body holds no __syncthreads (two per task, both inside etx_run_task, the queue advanced before the post-drain
barrier); ARGPF = wave 1 prefetches the next head's argument block after etx_run_task, per-type offset/stride/bytes
tables emitted."""
import re

from etx.codegen import emit_kernel
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.passes import compile_graph

RES = Resource(threads=256, lds_bytes=0, vgpr=64)


def _build(argpf: bool = False, bar2: bool = False) -> Graph:
    g = Graph("bp")
    g.tensor("t", (64,), role="activation", bytes_per_elem=2)
    g.etensor("E1", (1,), wait_count=4)
    g.call_device("a", (4,), hip_link("a.hip", "etx_tile_a", arg_bytes=136), resource=RES, args=["t"],
                  consts=(96, 160), out_edges={"E1": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    g.call_device("b", (4,), hip_link("b.hip", "etx_tile_b"), resource=RES, args=["t"],
                  in_edges={"E1": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    if argpf:
        g.codegen["argpf"] = True
    if bar2:
        g.codegen["bar2"] = True
    return g


def _no_reset_tail(src: str) -> str:
    """Drop the ETX_DEV_RESET tail. It runs once per STEP, after the task loop; these tests count the barriers a
    TASK costs, so leaving it in would make them fail for a reason they are not about."""
    return re.sub(r"#ifdef ETX_DEV_RESET.*?#endif\n", "", src, flags=re.S)


def _static_loop(src: str) -> str:
    m = re.search(r"// every task on this device is static.*?^\}", src, re.S | re.M)
    assert m, "static loop"
    return m.group(0)


def _run_task(src: str) -> str:
    m = re.search(r"static __device__ __forceinline__ void etx_run_task.*?^\}", src, re.S | re.M)
    assert m, "etx_run_task"
    return m.group(0)


def test_switches_off_is_the_old_emission():
    base = emit_kernel(compile_graph(_build(), "gfx942", {}, {}))
    assert "etx_argpf" not in base and "BAR2" not in base and "s_task2" not in base
    on = emit_kernel(compile_graph(_build(argpf=True, bar2=True), "gfx942", {}, {}))
    assert base != on


def test_bar2_two_barriers_per_task():
    src = _no_reset_tail(emit_kernel(compile_graph(_build(bar2=True), "gfx942", {}, {})))
    loop = _static_loop(src)
    body = loop.split("for (;;) {", 1)[1]
    assert body.count("__syncthreads()") == 0            # none in the loop body: both live in etx_run_task
    assert loop.count("__syncthreads()") == 1            # the preamble broadcast only
    rt = _run_task(src)
    assert rt.count("__syncthreads();") == 2
    # the next head is READ before the body (its latency hides there) and only PUBLISHED, from registers, between
    # the drain and the post-drain barrier: a global load on the publication path measured +0.117 ms (2026-10-01)
    read = rt.index("nx_tid = p.static_queue[cursor]")
    publish = rt.index("s_tid2[par ^ 1] = nx_tid")
    drain = rt.index("ETX_RELEASE_DOMAIN();   // every wave drains")
    assert read < rt.index("etx_call_body(sym, &ctx)") < drain < publish < rt.rindex("__syncthreads();")
    between = rt[drain:rt.rindex("__syncthreads();")]
    assert "p.static_queue[" not in between and "p.static_descs[" not in between
    assert "etx_run_task(p, cur, worker, domain, tid, cursor, cend, s_tid2, s_task2, par);" in loop


def test_default_loop_unchanged_barrier_count():
    src = _no_reset_tail(emit_kernel(compile_graph(_build(), "gfx942", {}, {})))
    loop = _static_loop(src)
    assert loop.split("for (;;) {", 1)[1].count("__syncthreads()") == 2   # loop-top + loop-end
    assert _run_task(src).count("__syncthreads();") == 2


def test_argpf_tables_and_wave1_prefetch():
    src = _no_reset_tail(emit_kernel(compile_graph(_build(argpf=True), "gfx942", {}, {})))
    # a: arg_bytes 136, consts (96, 160) -> off 96, stride 160; b: no arg_bytes -> all zero
    assert re.search(r"etx_argpf_off\[2\] = \{96, 0\}|etx_argpf_off\[2\] = \{0, 96\}", src)
    assert re.search(r"etx_argpf_str\[2\] = \{160, 0\}|etx_argpf_str\[2\] = \{0, 160\}", src)
    assert re.search(r"etx_argpf_bytes\[2\] = \{136, 0\}|etx_argpf_bytes\[2\] = \{0, 136\}", src)
    loop = _static_loop(src)
    tail = loop.split("etx_run_task(p, cur, worker, domain, tid);", 1)[1]
    assert "threadIdx.x - 64u < 64u && s_ncur < cend" in tail and "etx_imp_touch(etx_imp_argblock(" in tail
    assert "s_ncur = cursor;" in loop                     # thread 0 publishes the next queue position


def test_argpf_with_bar2_reads_the_parity_slot():
    src = _no_reset_tail(emit_kernel(compile_graph(_build(argpf=True, bar2=True), "gfx942", {}, {})))
    loop = _static_loop(src)
    assert "s_tid2[par ^ 1] >= 0" in loop and "s_task2[par ^ 1].type" in loop
