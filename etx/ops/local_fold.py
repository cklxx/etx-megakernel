"""Opt-in per-XCD expert-fold assembly for the batch-one Qwen graph.

Only -DETX_DSL_LOCAL_FOLD enables emission and graph wiring. Read CFLAGS as
well as ETX_CFLAGS because rocm_build_check.sh renames the latter in Docker.
No default-path source text, argument ABI, or resource declaration is changed.
"""
from __future__ import annotations

from dataclasses import replace
import os
import shlex


def enabled() -> bool:
    values = []
    for var in ("ETX_CFLAGS", "CFLAGS"):
        value = None
        for token in shlex.split(os.environ.get(var, "")):
            if token in ("-DETX_DSL_LOCAL_FOLD", "-DETX_DSL_LOCAL_FOLD=1"):
                value = True
            elif token in ("-UETX_DSL_LOCAL_FOLD", "-DETX_DSL_LOCAL_FOLD=0"):
                value = False
            elif token.startswith("-DETX_DSL_LOCAL_FOLD="):
                raise ValueError("ETX_DSL_LOCAL_FOLD accepts only 0 or 1")
        if value is not None:
            values.append(value)
    if len(set(values)) > 1:
        raise ValueError("ETX_CFLAGS and CFLAGS disagree on ETX_DSL_LOCAL_FOLD")
    return bool(values and values[0])


def _once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("local fold source anchor changed: " + old[:80])
    return source.replace(old, new, 1)


def emit(stream: str, plain: str) -> str:
    """Reuse the actual ordered fold and norm text, not a second arithmetic implementation.

    Both vector and scalar paths are retained. The stage replacement preserves
    which lane owns each square and its accumulation order. A source edit that
    moves a boundary fails emission instead of silently using an old fold.
    """
    start = stream.index("template <int FixedK = 0, int FixedR = -1>")
    end = stream.index("static __device__ __forceinline__ void dsl_stage_norm(", start)
    original = stream[start:end]
    vstart = original.index("  for (int i = (int)threadIdx.x; i < K / 8;")
    vend = original.index("#ifdef ETX_DSL_NORM_HOIST", vstart)
    vector_fold = original[vstart:vend]
    sstart = original.index("  for (int i = (int)threadIdx.x; i < K;", original.index("#else\n  for (int i"))
    send = original.index("#if ETX_EMIT_DSL_EXT", sstart)
    scalar_fold = original[sstart:send]
    vector_stage = '''  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    const dsl_local_f4 lo = ((const DSL_G dsl_local_f4*)local)[2 * i];
    const dsl_local_f4 hi = ((const DSL_G dsl_local_f4*)local)[2 * i + 1];
    const float z[8] = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
#pragma unroll
    for (int j = 0; j < 8; ++j) { xs[8 * i + j] = z[j]; ss += z[j] * z[j]; }
  }
'''
    scalar_stage = '''  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) {
    const float z = ((const DSL_G float*)local)[i];
    xs[i] = z; ss += z * z;
  }
'''
    # The proof reads through the real local stage FIRST, then compares every
    # float bit to the original coherent fold. It does not repair bad payloads.
    vproof = _once(vector_fold,
                   "{ xs[8 * i + j] = z[j]; ss += z[j] * z[j]; }",
                   "{ if (__float_as_uint(xs[8 * i + j]) != __float_as_uint(z[j])) __builtin_trap(); }")
    sproof = _once(scalar_fold, "xs[i] = z; ss += z * z;",
                   "if (__float_as_uint(xs[i]) != __float_as_uint(z)) __builtin_trap();")
    proof = "#if defined(ETX_DSL_LOCAL_FOLD_VERIFY) && ETX_DSL_LOCAL_FOLD_VERIFY\n"
    proof += "  { const bool store = false;\n#ifdef ETX_DSL_VECNORM\n" + vproof
    proof += "#else\n" + sproof + "#endif\n  }\n#endif\n"
    norm = _once(original, "dsl_stage_norm_at(", "dsl_stage_local_norm_at(")
    norm = _once(norm, "const DslArgs& d,", "const DslArgs& d, const float* local,")
    norm = _once(norm, vector_fold, vector_stage + proof)
    norm = _once(norm, scalar_fold, scalar_stage + proof)
    # Exact original prefix in each assembler. Only XCD 0 publishes the
    # canonical bf16 residual; the final head's rout is already null.
    assembly = '''
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_local_assemble(const etx_ctx* c) {
  const DslArgs d = dsl_args(etx_imp_argblock(c->args, c->cst[0]));
  const int K = d.K, R = d.R;
  const __hip_bfloat16 *rows = d.rows, *rin = d.resid_in;
  __hip_bfloat16* rout = d.resid_out;
  const bool store = c->coord[0] == 0;
  float* xs = (float*)etx_lds_arena;
  float ss = 0.f;
#ifdef ETX_DSL_VECNORM
''' + vector_fold + "#else\n" + scalar_fold + '''#endif
  // Same lanes read their own LDS cells: no extra inter-lane join is needed.
  float* local = (float*)c->args[1] + c->coord[0] * 2048;
#ifdef ETX_DSL_VECNORM
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    ((DSL_G dsl_local_f4*)local)[2 * i] = {xs[8*i], xs[8*i+1], xs[8*i+2], xs[8*i+3]};
    ((DSL_G dsl_local_f4*)local)[2 * i + 1] = {xs[8*i+4], xs[8*i+5], xs[8*i+6], xs[8*i+7]};
  }
#else
  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) ((DSL_G float*)local)[i] = xs[i];
#endif
  // Runtime: all-wave store drain + barrier, then a count-one DOMAIN arrive.
}
'''
    body = _once(plain,
                 "dsl_stage_norm_nb(d, DSL_NB, KS, xs, part + DSL_NB * 16 * DSL_RCH, blk == 0);",
                 "dsl_stage_local_norm_at(d, local, d.rows, d.resid_in, nullptr, nullptr, xs, part + DSL_NB * 16 * DSL_RCH, false);")
    consumer = '''
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_local_fold_gemv(const etx_ctx* c) {
  const unsigned char* a = etx_imp_argblock(c->args, c->cst[0]);
  const int blk = etx_uniform(c->coord[0]);
  const DslArgs d = dsl_args(a);
  float* xs = (float*)etx_lds_arena;
  const float* local = (const float*)c->args[1] + (blk / 32) * 2048;
  // Coherent inputs/outputs retain the ordinary DSL event protocol. The ONE
  // cached exception is the private payload, acquired here before staging.
  // Keep ETX's tested sc1 DOMAIN acquire, not fleet's sc0: ETX measured stale
  // reused 8-KiB payloads with sc0 (bench/sync/sync_bench.hip:13-17,31-33).
  if (threadIdx.x == 0) ETX_ACQUIRE_DOMAIN();
  __syncthreads();
''' + body + "\n}\n"
    return '''
// ETX_DSL_LOCAL_FOLD: emitted only when requested at Python planning/emission.
#if !defined(ETX_DSL_LOCAL_FOLD) || !ETX_DSL_LOCAL_FOLD
#error "local-fold graph/source requires -DETX_DSL_LOCAL_FOLD in the device compiler"
#endif
#if !defined(ETX_DSL_COH) || DSL_NB != 1 || ETX_EMIT_DSL_EXT
#error "local fold requires coherent batch-one Qwen bf16 tiles"
#endif
#ifdef ETX_ACQUIRE_DOMAIN_RELAXED
#error "local fold requires the validated sc1 domain acquire, not relaxed sc0"
#endif
#ifdef ETX_NO_ALLWAVE_DRAIN
#error "local fold publication requires the all-wave store drain"
#endif
typedef float dsl_local_f4 __attribute__((ext_vector_type(4)));
''' + norm + assembly + consumer


def apply(graph, dims) -> None:
    """Insert eight pinned publishers without advancing P5's round-robin cursor.

    Existing launch instances and argument blocks stay unchanged: publishers
    read the consumer's block, and the generic host allocates the new scratch
    tensor. No new host dispatch export or imported-launch identity is needed.
    """
    from etx.frontends import hip_link
    from etx.ir.edgemap import EdgeMap
    if (dims.name != "qwen3-30b-a3b" or dims.H != 2048 or dims.TOPK != 8
            or graph.codegen.get("requests") != 1 or graph.codegen.get("nb") != 1
            or not graph.codegen.get("coh") or graph.codegen.get("dsl_ext")):
        raise ValueError("ETX_DSL_LOCAL_FOLD supports coherent Qwen3-30B-A3B bf16 batch 1 only")
    targets = [g for g in graph.grids if g.name == "lm_head"
               or (g.name.startswith("qkv_") and g.name != "qkv_0")]
    if len(targets) != dims.L:
        raise ValueError("local fold requires every expert fold, including lm_head")
    for g in targets:
        if (g.body.symbol != "etx_tile_dsl_gemv_fold" or g.grid != (256,)
                or len(g.consts) != 1 or g.args != ["impargs"] or len(g.in_edges) != 1):
            raise ValueError(f"unsupported local fold consumer {g.name}")
        ev = next(iter(g.in_edges))
        if graph.events[ev].shape != (1,) or graph.events[ev].wait_count != 256:
            raise ValueError(f"{g.name}: expected the original global expert completion")
        producers = graph.producers_of(ev)
        if len(producers) != 1 or not producers[0].name.startswith("moe_down_"):
            raise ValueError(f"{g.name}: expected an expert down producer")
    graph.tensor("dsl_local_fold", (8, 2048), role="scratch", dtype="fp32", bytes_per_elem=4)
    originals, graph.grids = graph.grids, []
    target_names = {g.name for g in targets}
    for g in originals:
        if g.name in target_names:
            old_event = next(iter(g.in_edges))
            local_event = "E_local_fold_" + g.name
            graph.etensor(local_event, (8,), wait_count=1)
            graph.call_device("local_fold_" + g.name, (8,),
                              hip_link(g.body.source, "etx_tile_dsl_local_assemble", coherence="coherent"),
                              resource=g.resource, args=["impargs", "dsl_local_fold"], consts=g.consts,
                              in_edges={old_event: "x->(0)"}, out_edges={local_event: "x->x"},
                              domain_map="x->x", worker_map="x->(0)", duration_us=1.0)
            g.in_edges = {local_event: EdgeMap.parse("i->(i/32)")}
            g.domain_map = "i->(i/32)"
            g.args = ["impargs", "dsl_local_fold"]
            g.body = replace(g.body, symbol="etx_tile_dsl_local_fold_gemv")
        graph.grids.append(g)


def plan_instances(instances: list[dict]) -> list[dict]:
    """Host --check launches row n at n+1; include each inserted publisher.

    A publisher row aliases the following fold's *argument layout*: same
    export, role, geometry and byte offset. Host argument setup is idempotent
    and fills that block twice. The actual eight-task publisher launch is in
    plan.json/plan_data.h, not this argument-layout alias. Keeping the original
    geometry is necessary for rpt and for --check-dsl's name-based consumer
    lookup. Imported instances retain their arguments but get correct new IDs.
    """
    result = []
    for instance in instances:
        if (instance["role"] == "lm_head"
                or (instance["role"] == "qkv" and instance["layer"] > 0)):
            if instance["export"] != "dsl_gemv_fold":
                raise ValueError("local fold plan alias requires dsl_gemv_fold")
            result.append(dict(instance, local_fold_argument_alias=True))
        result.append(instance)
    return result
