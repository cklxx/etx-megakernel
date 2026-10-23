"""D5 CPU gates: default identity, compiler eligibility, and executed weight addresses.

The host harness runs the exact pinned planner and D5 wrapper/address code.
Only its device load and AMD register sink are replaced with an observer;
the weight buffers have PROT_NONE, and no activation fields exist in the
host ABI. This tests addresses/counts, not GPU cache behavior or timing.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from etx.codegen import emit_kernel, emit_plan_json
from etx.machine import load_machine
from etx.passes import PassOptions, compile_graph
from examples.dsv2lite import graph as dsv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "09011c207b1e92febf29a2c2efc2d944ef9ad7ac8a0803633579c8444d821f8a"
DEFAULT_KERNEL = "751293fad5f5c98ec6dbf9eeca3c73319b1d4ded4b6ca73996a65757c0ffae5d"
TASKS = {"qkv": 7992, "merge": 6912, "oproj": 7992, "router": 7696,
         "shared_gate_up": 1924, "shared_down": 1924, "lm_head": 296}


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in os.environ:
        if key.startswith("ETX_DSV2_") or key == "ETX_PREFETCH_ISOLATE_POLLER":
            monkeypatch.delenv(key)


def _plan(layers=2, prefetch=True, machine="gfx942"):
    return compile_graph(dsv.build(layers=layers), machine, {}, {},
                         PassOptions(prefetch=prefetch, inline_tiles=True, relay="on"))


def test_default_plan_and_kernel_are_byte_identical(monkeypatch):
    for value in (None, "0"):
        if value is not None:
            monkeypatch.setenv("ETX_DSV2_D5", value)
        plan = _plan(layers=27)
        assert all(g.body.prefetch is None and g.body.source == dsv.T for g in plan.graph.grids)
        assert hashlib.sha256(emit_plan_json(plan).encode()).hexdigest() == DEFAULT_PLAN
        source = emit_kernel(plan)
        assert hashlib.sha256(source.encode()).hexdigest() == DEFAULT_KERNEL
        assert "fleet_d5" not in source


def test_only_selected_bodies_get_the_hook_and_it_precedes_the_wait(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_D5", "1")
    plan = _plan()
    source = emit_kernel(plan)
    assert '#include "examples/dsv2lite/fleet_d5.hip"' in source
    symbols = [int(x) for x in re.search(r"etx_prefetch_symbol\[\d+\] = \{([^}]+)\}", source)[1].split(",")]
    consts = re.search(r"etx_prefetch_consts\[\d+\]\[4\] = \{(.+)\};", source)[1]
    consts = [tuple(map(int, row.split(","))) for row in re.findall(r"\{([^}]+)\}", consts)]
    dispatch = source.split("void etx_call_prefetch(", 1)[1].split("    default: break;", 1)[0]
    calls = {int(i): s for i, s in re.findall(r"case (\d+): (fleet_d5_\w+)\(ctx\); break;", dispatch)}
    task_source = source.split("void etx_run_task(", 1)[1]
    assert task_source.index("ctx.cst[i] = etx_prefetch_consts[t.type][i]") < task_source.index("etx_call_prefetch(pf_sym, &ctx)")
    assert task_source.index("etx_call_prefetch(pf_sym, &ctx)") < task_source.index("// phase 1: immediates, waits, acquire")
    for g in plan.graph.grids:
        expected = dsv.D5_PREFETCH.get(g.body.symbol)
        assert g.body.prefetch == expected
        symbol = symbols[plan.type_ids[g.name]]
        if expected:
            assert calls[symbol] == expected
            assert consts[plan.type_ids[g.name]] == tuple(g.consts) + (0,) * (4-len(g.consts))
        else:
            assert symbol == -1
    assert len(calls) == len(set(calls.values()))  # one call site per sibling, across all layers
    assert {g.body.prefetch for g in plan.graph.grids if g.body.prefetch} == set(dsv.D5_PREFETCH.values())


@pytest.mark.parametrize("disabled", ["option", "capability"])
def test_compiler_gates_suppress_prefetch_calls(monkeypatch, disabled):
    monkeypatch.setenv("ETX_DSV2_D5", "1")
    machine = load_machine("gfx942")
    if disabled == "capability":
        machine.capabilities.update(async_copy_to_lds="none", tma=False)
    plan = _plan(prefetch=disabled != "option", machine=machine)
    assert plan.prefetch == []
    assert not re.search(r"fleet_d5_\w+\(&ctx\)", emit_kernel(plan))
    assert "etx_call_prefetch" not in emit_kernel(plan)


def test_p7_records_l2_warm_weight_arguments_but_is_advisory(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_D5", "1")
    plan = _plan()
    assert plan.options.prefetch and plan.machine.capabilities["async_copy_to_lds"] == "dword_only"
    assert plan.prefetch and {p["method"] for p in plan.prefetch} == {"l2_warm"}
    recorded = {p["task"]: p for p in plan.prefetch}
    first_tasks = {q[0] for q in plan.static_queues.values() if q}
    for task in plan.tasks:
        g = plan.graph.grid(task.grid)
        if g.body.prefetch and task.id not in first_tasks:
            assert recorded[task.id]["weights"] == ["fleet"]
    # The argument is an opaque FleetParams pointer, not itself an address
    # proof. P7 excludes queue heads; actual hooks still run there. The
    # host harness below establishes the accessed matrix/row ownership.
    assert all(plan.graph.grid(plan.tasks[p["task"]].grid).body.prefetch for p in plan.prefetch)
    assert any(t.id in first_tasks and plan.graph.grid(t.grid).body.prefetch for t in plan.tasks)


def test_warm_source_has_one_discarded_vector_and_no_mutable_payload():
    source = (ROOT / dsv.D5_T).read_text()
    code = re.sub(r"//[^\n]*", "", source)
    assert code.count("*reinterpret_cast<const fleet::fleet_u32x4*>") == 1
    assert 'asm volatile("" :: "v"(v.x), "v"(v.y), "v"(v.z), "v"(v.w) : "memory");' in code
    assert not re.search(r"\b(?:act|kv|lds|args|events|prefetch_issue|resolve_unit|__syncthreads)\b", code)
    assert "threadIdx.x < 64 || p.rows <= 0" in code
    assert "if (row >= p.rows) return;" in code
    assert not re.search(r"\b(?:for|while)\s*\(", code)


def test_source_load_counts_cover_the_full_realised_graph(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_D5", "1")
    active, zero = Counter(), Counter()
    for grid in dsv.build().grids:
        if not grid.body.prefetch:
            continue
        kind = grid.body.prefetch.removeprefix("fleet_d5_")
        assert grid.grid == ((16, 16) if kind == "merge" else (8, 37))
        if kind == "router" and grid.consts[0] == 0:
            zero[kind] += math.prod(grid.grid)
        elif kind.startswith("shared_"):
            active[kind] += 2 * 37
            zero[kind] += 6 * 37
        else:
            active[kind] += math.prod(grid.grid)
    assert dict(active) == TASKS
    assert dict(zero) == {"router": 296, "shared_gate_up": 5772, "shared_down": 5772}


HOST_PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
template<class T> T min(T a,T b) { return a<b?a:b; }
template<class T> T max(T a,T b) { return a>b?a:b; }
#define __device__
#define __forceinline__ inline
struct { int x; } threadIdx;
namespace fleet {
constexpr int kNumXCDs=8, kWorkersPerXCD=37;
enum { TASK_QKV_FUSED, TASK_ATTENTION, TASK_MERGE_UV, TASK_O_PROJ,
       TASK_NORM_ROUTER, TASK_EXPERT_GATE_UP, TASK_EXPERT_DOWN,
       TASK_DENSE_GATE_UP, TASK_DENSE_DOWN, TASK_LM_HEAD };
constexpr int FLAG_OPROJ_ROW_SPLIT=1;
struct fleet_u32x4 { uint32_t x,y,z,w; };
struct TaskDescriptor { int kind,layer,head,kv_chunk,n_split,expert_slot,flags; };
struct ModelDims {
 int hidden=2048, heads=16, q_head_dim=192, qk_nope=128, qk_rope=64, v_head=128;
 int kv_lora=512, moe_inter=1408, dense_inter=10944, n_routed=64, top_k=6, vocab=102400, first_k_dense=1;
};
struct Weights {
 const char *layer_qkv[27], *layer_kv_b[27], *layer_o_proj[27], *layer_router[27],
            *layer_shared[27], *layer_dense[27], *embed=nullptr, *lm_head=nullptr;
};
'''

HOST_OBSERVER = r'''
} // namespace fleet
// Intentionally no Activations, KVCache, routing, LDS or generic args.
struct FleetParams { fleet::ModelDims d; fleet::Weights w; } params;
struct etx_ctx { int coord[4]={}, cst[4]={}, domain=0; } current;
#define FP(c) (&params)
#define LAYER(c) ((c)->cst[0])
static int kind, expected_loads, loads;
static const char* base;
static int64_t size_bytes;
static uintptr_t previous_address;
static fleet::fleet_u32x4 observe(const char* p) {
 const auto& d=params.d;
 const uintptr_t address=reinterpret_cast<uintptr_t>(p), begin=reinterpret_cast<uintptr_t>(base);
 assert(threadIdx.x>=64 && threadIdx.x<256);
 assert(address>=begin && address+16<=begin+size_bytes && address%16==0);
 const int64_t off=address-begin, hb=2*d.hidden;
 const int x=current.domain, worker=current.coord[1];
 if (kind==0) { // Q rows, owned by this XCD/worker; no kv_a or activations
   const int row=off/hb, first=x*(d.heads/8)*d.q_head_dim;
   assert(row>=first && row<first+(d.heads/8)*d.q_head_dim && (row-first)%37==worker);
 } else if (kind==1) { // only this head/chunk's W_UV rows
   const int64_t head_bytes=(d.qk_nope+d.v_head)*d.kv_lora*2;
   const int h=off/head_bytes, row=(off%head_bytes)/(d.kv_lora*2), first=d.qk_nope+current.coord[1]*(d.v_head/16);
   assert(h==current.coord[0] && row>=first && row<first+d.v_head/16);
 } else if (kind==2) { // this worker's output rows, this XCD's K slice
   const int64_t stride=d.heads*d.v_head*2, kc=stride/8;
   assert((off/stride)%37==worker && off%stride>=x*kc && off%stride+16<=(x+1)*kc);
 } else if (kind==3) {
   assert(current.cst[0]>=1 && (off/hb)%37==worker && off/hb<d.n_routed);
 } else if (kind==4) {
   const int row=off/hb, half=x-d.top_k;
   assert(half>=0 && half<2 && row/2>=half*d.moe_inter && row/2<(half+1)*d.moe_inter);
   assert((row/2-half*d.moe_inter)%37==worker);
 } else if (kind==5) {
   const int64_t down=2*(2*d.moe_inter)*hb, stride=2*d.moe_inter*2;
   const int half=x-d.top_k;
   assert(half>=0 && half<2 && off>=down);
   const int64_t at=off-down;
   assert((at/stride)%37==worker && at%stride>=half*d.moe_inter*2 && at%stride+16<=(half+1)*d.moe_inter*2);
 } else if (kind==6) {
   const int per=(d.vocab+7)/8, row=off/hb;
   assert(row>=x*per && row<min(d.vocab,(x+1)*per) && (row-x*per)%37==worker);
 } else assert(false);
 assert(address>previous_address);
 previous_address=address;
 ++loads;
 return {};
}
'''

HOST_MAIN = r'''
static const char* allocation(int64_t n) {
 void* p=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
 assert(p!=MAP_FAILED);
 return static_cast<const char*>(p);
}
int main() {
 auto& d=params.d; auto& w=params.w;
 const int64_t hb=d.hidden*2;
 const int64_t sizes[]={ (d.heads*d.q_head_dim+d.kv_lora+d.qk_rope)*hb,
   int64_t(d.heads)*(d.qk_nope+d.v_head)*d.kv_lora*2,
   int64_t(d.hidden)*d.heads*d.v_head*2, d.n_routed*hb,
   3*(2*d.moe_inter)*hb, 3*(2*d.moe_inter)*hb, d.vocab*hb };
 for (int L=0;L<27;++L) {
   w.layer_qkv[L]=allocation(sizes[0]); w.layer_kv_b[L]=allocation(sizes[1]);
   w.layer_o_proj[L]=allocation(sizes[2]); w.layer_router[L]=allocation(sizes[3]);
   w.layer_shared[L]=allocation(sizes[4]); w.layer_dense[L]=nullptr;
 }
 w.lm_head=allocation(sizes[6]);
 void (*hooks[])(const etx_ctx*)={fleet_d5_qkv,fleet_d5_merge,fleet_d5_oproj,
   fleet_d5_router,fleet_d5_shared_gate_up,fleet_d5_shared_down,fleet_d5_lm_head};
 const char* names[]={"qkv","merge","oproj","router","shared_gate_up","shared_down","lm_head"};
 for (kind=0;kind<7;++kind) {
   int64_t active_tasks=0, total_loads=0, zero_tasks=0;
   const int layer_begin=kind==6 ? -1 : (kind==4 || kind==5 ? 1 : 0);
   const int layer_end=kind==6 ? 0 : 27;
   for (int L=layer_begin;L<layer_end;++L) for (int a=0;a<(kind==1?16:8);++a)
     for (int b=0;b<(kind==1?16:37);++b) {
       current.cst[0]=L; current.coord[0]=a; current.coord[1]=b; current.domain=kind==1?a/2:a;
       base=kind==0?w.layer_qkv[L]:kind==1?w.layer_kv_b[L]:kind==2?w.layer_o_proj[L]:
            kind==3?w.layer_router[L]:(kind==4 || kind==5)?w.layer_shared[L]:w.lm_head;
       size_bytes=sizes[kind];
       expected_loads=(kind==3 && L==0) || ((kind==4 || kind==5) && a<d.top_k) ? 0 : 192;
       loads=0; previous_address=0;
       for (threadIdx.x=0;threadIdx.x<256;++threadIdx.x) hooks[kind](&current);
       assert(loads==expected_loads);
       if (loads) ++active_tasks; else ++zero_tasks;
       total_loads+=loads;
     }
   printf("%s %lld %lld %lld\n",names[kind],(long long)active_tasks,(long long)total_loads,(long long)zero_tasks);
 }
}
'''


def test_exact_cpp_planner_and_wrapper_addresses_are_only_owned_weights(tmp_path, pinned_src):
    compiler = shutil.which("c++")
    assert compiler, "D5 address proof requires a host C++ compiler"
    fleet = (pinned_src / "kernels/fleet_kernel.hip").read_text()
    planner = fleet[fleet.index("struct PrefetchPlan"):fleet.index("// Issue this thread's 8 touch loads")]
    gemv = (pinned_src / "kernels/gemv.h").read_text()
    row_slice = gemv[gemv.index("struct RowSlice"):gemv.index("// bf16 boundaries")]
    wrapper = (ROOT / dsv.D5_T).read_text()
    wrapper = wrapper.replace('#include "examples/dsv2lite/fleet_shim.hip"', "")
    load = "*reinterpret_cast<const fleet::fleet_u32x4*>(p.base + row * p.stride + col * 16)"
    sink = 'asm volatile("" :: "v"(v.x), "v"(v.y), "v"(v.z), "v"(v.w) : "memory");'
    assert wrapper.count(load) == wrapper.count(sink) == 1
    wrapper = wrapper.replace(load, "observe(p.base + row * p.stride + col * 16)").replace(sink, "(void)v;")
    cpp, exe = tmp_path / "addresses.cpp", tmp_path / "addresses"
    cpp.write_text(HOST_PREFIX + row_slice + planner + HOST_OBSERVER + wrapper + HOST_MAIN)
    built = subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    checked = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    counts = {name: tuple(map(int, nums)) for name, *nums in map(str.split, checked.stdout.splitlines())}
    assert counts == {name: (n, n * 192, 296 if name == "router" else 5772 if name.startswith("shared_") else 0)
                      for name, n in TASKS.items()}
    assert sum(n * 192 * 16 for n in TASKS.values()) == 106_708_992
    print("D5 exact source address/count proof (kind, active tasks, vector loads, zero-load tasks):\n" + checked.stdout)


def test_verify_default_strips_d5_and_arm_sets_it(monkeypatch, tmp_path):
    from etx.models.dsv2lite import verify_lever as verify
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        out = Path(cmd[-1]) / cmd[-2]
        out.mkdir(parents=True)
        for name in ("resource.log", "plan.json", "megakernel_d0.hip", "isa.bin", "mk.s"):
            (out / name).write_text("")
        return ""

    monkeypatch.setenv("ETX_DSV2_D5", "1")
    monkeypatch.setenv("ETX_DSV2_COH", "l1")
    monkeypatch.setattr(verify, "run", fake_run)
    monkeypatch.setattr(verify, "fleet_export", lambda: tmp_path / "pinned")
    monkeypatch.setattr(verify, "parse", lambda _: {})
    verify.build("default", tmp_path)
    verify.build("alld5", tmp_path)
    assert "ETX_DSV2_D5" not in calls[0][1] and "ETX_DSV2_COH" not in calls[0][1]
    assert calls[1][1]["ETX_DSV2_D5"] == "1" and "ETX_DSV2_COH" not in calls[1][1]
    assert calls[1][0][-2] == "gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live"
