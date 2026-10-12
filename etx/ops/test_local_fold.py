"""CPU regression gates for the opt-in local expert fold; no GPU/cache claims.

Run explicitly: .venv/bin/python -m pytest -q etx/ops/test_local_fold.py
The payload test compiles extracted production arithmetic, with CPU memory and
bf16 stubs; the graph test compiles the actual 48-layer Qwen plan.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from etx.ops import gemv, local_fold


ROOT = Path(__file__).resolve().parents[2]
BASE_IMPORTS = ROOT / "build/rocm22/foldlocal_off/imp/imports.json"
BASE_REVISION = "b1cb7f85fa7e1f43aa9ead481e727554dd5c6589"


@pytest.fixture(autouse=True)
def clean_switches(monkeypatch):
    for key in ("ETX_CFLAGS", "CFLAGS", "ETX_HIPCC_FLAGS", "ETX_DSL_LOCAL_FOLD",
                "ETX_INBODY_EMIT"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("variable", ["ETX_CFLAGS", "CFLAGS"])
@pytest.mark.parametrize("flags,want", [
    ("", False), ("-O3", False), ("-DETX_DSL_LOCAL_FOLD", True),
    ("-DETX_DSL_LOCAL_FOLD=1", True), ("-DETX_DSL_LOCAL_FOLD=0", False),
    ("-DETX_DSL_LOCAL_FOLD -UETX_DSL_LOCAL_FOLD", False),
    ("-UETX_DSL_LOCAL_FOLD -DETX_DSL_LOCAL_FOLD=1", True),
])
def test_compiler_switch_reaches_python(variable, flags, want, monkeypatch):
    monkeypatch.setenv(variable, flags)
    assert local_fold.enabled() is want


def test_conflicting_or_invalid_compiler_switch_fails(monkeypatch):
    monkeypatch.setenv("ETX_CFLAGS", "-DETX_DSL_LOCAL_FOLD=1")
    monkeypatch.setenv("CFLAGS", "-DETX_DSL_LOCAL_FOLD=0")
    with pytest.raises(ValueError, match="disagree"):
        local_fold.enabled()
    monkeypatch.delenv("CFLAGS")
    monkeypatch.setenv("ETX_CFLAGS", "-DETX_DSL_LOCAL_FOLD=2")
    with pytest.raises(ValueError, match="only 0 or 1"):
        local_fold.enabled()


@pytest.mark.parametrize("flags", ["", "-DETX_DSL_LOCAL_FOLD=0"])
def test_off_emitted_source_is_byte_identical_to_git_head(flags):
    """Execute both emitters in one clean process; compare complete bytes."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ETX_", "DSL_")) and k != "CFLAGS"}
    env["ETX_CFLAGS"] = flags
    env["LOCAL_FOLD_TEST_BASE_REVISION"] = BASE_REVISION
    script = '''
import hashlib, os, subprocess
from pathlib import Path
from etx.ops import gemv
source = subprocess.check_output(["git", "show", os.environ["LOCAL_FOLD_TEST_BASE_REVISION"]+":etx/ops/gemv.py"], text=True)
legacy = {"__name__": "frozen_legacy_gemv", "__file__": gemv.__file__}
exec(compile(source, gemv.__file__, "exec"), legacy)
old, new = legacy["emit_tiles"]().encode(), gemv.emit_tiles().encode()
assert old == new, "default-off emitted tile bytes changed"
print("OFF SOURCE SHA256", hashlib.sha256(new).hexdigest())
'''
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", script],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OFF SOURCE SHA256" in result.stdout


def _source_parts():
    stream = gemv.STREAM
    start = stream.index("template <int FixedK = 0, int FixedR = -1>")
    end = stream.index("static __device__ __forceinline__ void dsl_stage_norm(", start)
    original = stream[start:end]
    generated = local_fold.emit(stream, gemv.BODIES["plain"])
    norm_start = generated.index("template <int FixedK = 0, int FixedR = -1>")
    assembly_start = generated.index('extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_local_assemble')
    consumer_start = generated.index('extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_local_fold_gemv')
    return original, generated, generated[norm_start:assembly_start], generated[assembly_start:consumer_start]


def test_original_fold_and_norm_boundaries_are_reused():
    original, generated, norm, assembly = _source_parts()
    vector_start = original.index("  for (int i = (int)threadIdx.x; i < K / 8;")
    vector_end = original.index("#ifdef ETX_DSL_NORM_HOIST", vector_start)
    vector = original[vector_start:vector_end]
    scalar_start = original.index("#else\n  for (int i = (int)threadIdx.x; i < K;") + len("#else\n")
    scalar_end = original.index("#if ETX_EMIT_DSL_EXT", scalar_start)
    scalar = original[scalar_start:scalar_end]
    assert vector in assembly and scalar in assembly
    # The norm after each original stage is verbatim, including hoist, sums,
    # bf16 boundaries and the final barrier. The proof block ends before it.
    vector_norm = original[vector_end:scalar_start - len("#else\n")]
    scalar_norm = original[scalar_end:]
    assert vector_norm in norm and scalar_norm in norm
    assert "ETX_ACQUIRE_DOMAIN();" in generated
    assert 'requires the validated sc1 domain acquire' in generated
    assert 'requires the all-wave store drain' in generated
    assert 'local-fold graph/source requires -DETX_DSL_LOCAL_FOLD' in generated
    assert "dsl_stage_local_norm_at(d, local, d.rows, d.resid_in, nullptr, nullptr" in generated


CPP_STUBS = r'''
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#define __device__
#define __forceinline__ inline
#define DSL_G
#define DSL_RB 8
#define DSL_NB 1
#define ETX_EMIT_DSL_EXT 0
#define ETX_DSL_COH 1
#define ETX_DSL_LOCAL_FOLD 1
using __hip_bfloat16 = uint16_t;
using dsl_u32x4 = uint32_t __attribute__((ext_vector_type(4)));
struct Dim { int x; } threadIdx, blockDim{1024};
static float etx_lds_arena[8192], observed_ss[1024];
static uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static __hip_bfloat16 __float2bfloat16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffff) > 0x7f800000) return (u >> 16) | 0x40;
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
static float __bfloat162float(__hip_bfloat16 b) { return __uint_as_float(uint32_t(b) << 16); }
static float dsl_bfr(float f) { return __bfloat162float(__float2bfloat16(f)); }
static float dsl_a_ldbf(const __hip_bfloat16* p) { return __bfloat162float(*p); }
static void dsl_a_stbf(__hip_bfloat16* p, __hip_bfloat16 v) { *p = v; }
static dsl_u32x4 dsl_a_ld128(const __hip_bfloat16* p) { dsl_u32x4 v; memcpy(&v, p, 16); return v; }
static void dsl_a_st128(__hip_bfloat16* p, dsl_u32x4 v) { memcpy(p, &v, 16); }
static float etx_block_sum(float v, float*) { observed_ss[threadIdx.x] = v; return 2048.f; }
static float rsqrtf(float v) { return 1.f / sqrtf(v); }
static void __syncthreads() {}
struct DslArgs {
  int K=2048, R=8, mode=1;
  float eps=1e-6f;
  const __hip_bfloat16 *rows=nullptr, *resid_in=nullptr, *ln=nullptr;
  __hip_bfloat16* resid_out=nullptr;
};
struct etx_ctx { int coord[4]{}, cst[4]{}; void** args; };
static const unsigned char* etx_imp_argblock(void* const* args, int offset) {
  return (const unsigned char*)args[0] + offset;
}
static DslArgs dsl_args(const unsigned char* p) { return *(const DslArgs*)p; }
'''


CPP_MAIN = r'''
int main() {
  constexpr int K=2048, NX=8;
  std::vector<uint16_t> rows(8*K), rin(K), ln(K), rout(K), oldout(K), localout(K);
  std::vector<float> original(K), local(8*K+8), expected(K);
  DslArgs d; d.rows=rows.data(); d.resid_in=rin.data(); d.ln=ln.data();
  // volatile keeps the sensitivity scans opaque: with fully constant inputs at -O2 the compiler folds
  // both summation orders to one value and the precondition below (exit 16) fires on valid code.
  volatile uint32_t rng_seed = 7u;
  uint32_t rng=rng_seed; int tests=0, order_sensitive=0, round_sensitive=0;
  for (int sample=0; sample<16; ++sample) {
    for (int i=0;i<K;++i) {
      rin[i]=__float2bfloat16(float((i%11)-5)/8);
      ln[i]=__float2bfloat16(float((i%7)+1)/8);
      for (int r=0;r<8;++r) {
        rng=rng*1664525u+1013904223u;
        rows[r*K+i]=__float2bfloat16(float(int(rng%4001)-2000)/128);
      }
      if (i%3==0) { // ordered fp32 cancellation: the BIG terms must be adjacent in the forward order,
        // otherwise (+2^30, +1, -2^30) and its reverse both round to 0 and the probe is not sensitive.
        for(int r=0;r<8;++r) rows[r*K+i]=0;
        rows[i]=__float2bfloat16(1073741824.f);
        rows[K+i]=__float2bfloat16(-1073741824.f);
        rows[2*K+i]=__float2bfloat16(1.f);
      }
      if (i%3==1) { // first bf16 boundary must occur before the residual add
        for(int r=0;r<8;++r) rows[r*K+i]=0;
        rows[i]=__float2bfloat16(1.f);
        rows[K+i]=__float2bfloat16(1.f/256);
        rin[i]=__float2bfloat16(-1.f);
      }
      float s=0.f, reverse=0.f;
      for(int r=0;r<8;++r) s+=__bfloat162float(rows[r*K+i]);
      for(int r=7;r>=0;--r) reverse+=__bfloat162float(rows[r*K+i]);
      expected[i]=dsl_bfr(dsl_bfr(s)+__bfloat162float(rin[i]));
      order_sensitive += __float_as_uint(s)!=__float_as_uint(reverse);
      round_sensitive += __float_as_uint(expected[i])!=__float_as_uint(dsl_bfr(s+__bfloat162float(rin[i])));
    }
    // Each assembler independently writes only its 2048-float slice. Canonical
    // residual publication belongs exclusively to XCD 0, and head uses nullptr.
    for (int head=0;head<2;++head) {
      std::fill(local.begin(),local.end(),-999.f);
      for(int x=7;x>=0;--x) {
        std::fill(rout.begin(),rout.end(),0x7fc1);
        d.resid_out=head?nullptr:rout.data();
        void* args[]={&d,local.data()}; etx_ctx c; c.args=args; c.coord[0]=x;
        for(int t=0;t<1024;++t) { threadIdx.x=t; etx_tile_dsl_local_assemble(&c); }
        for(int i=0;i<K;++i) {
          if(__float_as_uint(local[x*K+i])!=__float_as_uint(expected[i])) return 10;
          if(rout[i] != (!head && x==0 ? __float2bfloat16(expected[i]) : 0x7fc1)) return 11;
        }
        for(int i=NX*K;i<NX*K+8;++i) if(local[i]!=-999.f) return 12;
      }
      d.resid_out=nullptr;
      float red[1024]{};
      std::array<float,1024> reference_ss{};
      for(int t=0;t<1024;++t) {
        threadIdx.x=t;
        reference_norm_at(d, d.rows, d.resid_in, nullptr, oldout.data(), original.data(), red, true);
        reference_ss[t]=observed_ss[t];
      }
      for(int x=0;x<NX;++x) {
        std::fill(etx_lds_arena,etx_lds_arena+K,-999.f);
        for(int t=0;t<1024;++t) {
          threadIdx.x=t;
          dsl_stage_local_norm_at(d, local.data()+x*K, d.rows, d.resid_in,
                                  nullptr, localout.data(), etx_lds_arena, red, true);
          if(__float_as_uint(observed_ss[t])!=__float_as_uint(reference_ss[t])) return 13;
        }
        for(int i=0;i<K;++i) {
          if(__float_as_uint(etx_lds_arena[i])!=__float_as_uint(original[i])) return 14;
          if(localout[i]!=oldout[i]) return 15;
        }
        ++tests;
      }
    }
  }
  if(!order_sensitive || !round_sensitive) return 16;
  printf("PAYLOAD OK %d XCD/sample/head cases; ordered=%d boundary=%d; ss lane partition exact\n",
         tests,order_sensitive,round_sensitive);
}
'''


@pytest.mark.parametrize("defines", [[], ["-DETX_DSL_VECNORM"],
    ["-DETX_DSL_VECNORM", "-DETX_DSL_RBATCH", "-DETX_DSL_NORM_HOIST"],
    ["-DETX_DSL_VECNORM", "-DETX_DSL_RBATCH", "-DETX_DSL_NORM_HOIST",
     "-DETX_DSL_LOCAL_FOLD_VERIFY=1"]])
def test_extracted_production_payload_and_square_partition(tmp_path, defines):
    # Prefer a toolchain that ships the full C++ headers: Apple clang alone cannot find <cstdint>
    # without an SDK sysroot. Same choice the sibling fleet payload test makes.
    clang = None
    for candidate in ("/opt/homebrew/opt/llvm/bin/clang++", shutil.which("clang++"),
                      "/usr/bin/clang++"):
        if candidate and Path(candidate).is_file():
            clang = candidate
            break
    if clang is None:
        pytest.skip("CPU clang++ required for ext_vector_type production payload gate")
    sdk = (["-isysroot", subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()]
           if sys.platform == "darwin" and Path("/usr/bin/xcrun").is_file() else [])
    original, generated, norm, assembly = _source_parts()
    end = generated.index('extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_local_fold_gemv')
    source = CPP_STUBS + original.replace("dsl_stage_norm_at(", "reference_norm_at(", 1)
    source += generated[:end] + CPP_MAIN
    cpp = tmp_path / "payload.cpp"
    exe = tmp_path / "payload"
    cpp.write_text(source)
    result = subprocess.run([clang, *sdk, "-std=c++17", "-O2", "-ffp-contract=off", *defines,
                             str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, f"payload exit {result.returncode}: {result.stdout}{result.stderr}"
    assert "PAYLOAD OK 256" in result.stdout


@pytest.fixture(scope="module")
def qwen_graphs():
    if not BASE_IMPORTS.is_file():
        pytest.skip("run the required foldlocal_off ROCm build to provide its real import manifest")
    from etx.passes import compile_graph
    from examples.vllm_llm.kept_build import KEPT_ENV
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith(("ETX_", "DSL_")) or key == "CFLAGS":
                mp.delenv(key)
        for key, value in KEPT_ENV.items():
            mp.setenv(key, value)
        mp.setenv("ETX_LLM_CONFIG", str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
        mp.setenv("ETX_VL_IMPORTS", str(BASE_IMPORTS))
        spec = importlib.util.spec_from_file_location("local_fold_test_model", ROOT / "examples/vllm_llm/model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = module.build()
        dims = module.load_dims()
        instances = module.instances(dims)
        module.plan_offsets(instances, module.imports())
        enabled = copy.deepcopy(base)
        local_fold.apply(enabled, dims)
        yield base, enabled, compile_graph(copy.deepcopy(base), "gfx942", {}, {}), \
            compile_graph(copy.deepcopy(enabled), "gfx942", {}, {}), instances


def test_actual_qwen_graph_preserves_old_tasks_and_adds_local_publishers(qwen_graphs):
    from etx.ir.types import Scope
    from etx.passes import verify_plan
    base, enabled, old_plan, new_plan, instances = qwen_graphs
    assert not verify_plan(old_plan) and not verify_plan(new_plan)
    old_tasks = {(t.grid, t.coord): t for t in old_plan.tasks}
    old_grids = {g.name: g for g in base.grids}
    new_grids = {g.name: g for g in enabled.grids}
    added = [g for g in enabled.grids if g.name.startswith("local_fold_")]
    assert len(added) == 48
    assert len(new_plan.tasks) == len(old_plan.tasks) + 384
    assert new_plan.workers_per_device() == old_plan.workers_per_device()
    for task in new_plan.tasks:
        if task.grid.startswith("local_fold_"):
            assert task.domain == task.coord[0]
            assert task.worker % new_plan.workers_per_domain == 0
            continue
        old = old_tasks[(task.grid, task.coord)]
        assert (task.domain, task.worker, task.mode) == (old.domain, old.worker, old.mode)
        g, old_g = new_grids[task.grid], old_grids[task.grid]
        assert g.consts == old_g.consts and g.resource == old_g.resource
        assert g.out_edges == old_g.out_edges
        assert g.body.source == old_g.body.source
        assert g.body.touch == old_g.body.touch and g.body.pre_wait == old_g.body.pre_wait
        if g.name not in {"lm_head", *(f"qkv_{i}" for i in range(1,48))}:
            assert g.args == old_g.args and g.body == old_g.body and g.in_edges == old_g.in_edges
    # Insertion must not advance the old round-robin cursor or reorder a queue.
    for worker, queue in new_plan.static_queues.items():
        kept = [(new_plan.tasks[t].grid, new_plan.tasks[t].coord) for t in queue
                if not new_plan.tasks[t].grid.startswith("local_fold_")]
        old = [(old_plan.tasks[t].grid, old_plan.tasks[t].coord) for t in old_plan.static_queues[worker]]
        assert kept == old
    for assembler in added:
        consumer = new_grids[assembler.name.removeprefix("local_fold_")]
        event = next(iter(assembler.out_edges))
        assert new_plan.events[event].scope == Scope.DOMAIN
        assert new_plan.events[event].counts == [1]*8
        assert set(consumer.in_edges) == {event}
        assert set(assembler.in_edges) == set(old_grids[consumer.name].in_edges)
        assert next(iter(assembler.in_edges.values())).items == (("expr", "0"),)
        assert next(iter(assembler.in_edges)) in base.events
        assert new_plan.events[next(iter(assembler.in_edges))].scope == Scope.DEVICE
    assert enabled.tensors["dsl_local_fold"].shape == (8,2048)


def test_scratch_and_residual_reuse_wait_for_all_old_readers(qwen_graphs):
    """Follow actual event ancestry, not textual ordering of inserted grids."""
    base, graph, old_plan, new_plan, instances = qwen_graphs
    grids = {g.name:g for g in graph.grids}
    producers = {e:{g.name for g in graph.producers_of(e)} for e in graph.events}
    memo = {}
    def ancestors(name):
        if name not in memo:
            direct = set().union(*(producers[e] for e in grids[name].in_edges)) if grids[name].in_edges else set()
            memo[name] = direct | set().union(*(ancestors(p) for p in direct)) if direct else set()
        return memo[name]
    targets = [f"qkv_{i}" for i in range(1,48)] + ["lm_head"]
    for before, after in zip(targets, targets[1:]):
        # One global completion covers every old local-copy reader before
        # any XCD writes its next generation of that copy.
        assert before in ancestors("local_fold_"+after)
    for layer in range(1,48):
        assert f"router_{layer-1}" in ancestors(f"local_fold_qkv_{layer}")
        assert f"qkv_{layer}" in ancestors(f"router_{layer}")
    assert "router_47" in ancestors("local_fold_lm_head")
    assert "lm_head" in ancestors("argmax")


def test_host_alias_rows_match_actual_launch_ids_and_arguments(qwen_graphs):
    base, graph, old_plan, new_plan, instances = qwen_graphs
    rows = local_fold.plan_instances(instances)
    assert len(rows) == len(instances)+48
    # Native prologue precedes all host instance rows; native argmax follows.
    assert len(graph.grids) == len(rows)+2
    recovered = []
    for number, row in enumerate(rows):
        launch = graph.grids[number+1]
        assert launch.consts[0] == row["offset"]
        if row.get("local_fold_argument_alias"):
            assert launch.name.startswith("local_fold_")
            assert launch.grid == (8,)
            assert {k:v for k,v in row.items() if k != "local_fold_argument_alias"} == rows[number+1]
            assert row["tasks"] == 256 and row["grid"] == (256,1,1)
        else:
            recovered.append(row)
            assert not launch.name.startswith("local_fold_")
    assert recovered == instances


def test_unsupported_graph_is_rejected_before_mutation(qwen_graphs):
    base = qwen_graphs[0]
    from examples.llm.model import load_dims
    # The actual fixture keeps its model environment live through teardown.
    dims = load_dims()
    bad = copy.deepcopy(base)
    bad.codegen["requests"] = 2
    before = copy.deepcopy(bad)
    with pytest.raises(ValueError, match="batch 1"):
        local_fold.apply(bad, dims)
    assert bad == before
    bad = copy.deepcopy(base)
    next(g for g in bad.grids if g.name == "qkv_1").grid = (128,)
    before = copy.deepcopy(bad)
    with pytest.raises(ValueError, match="unsupported local fold consumer"):
        local_fold.apply(bad, dims)
    assert bad == before
