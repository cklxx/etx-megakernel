"""The attn_reduce -> wvsplitk_y2 memory contract: a minimal build passes, and every mutation of it fails."""
import json
import os
from pathlib import Path

import pytest

from etx.codegen import emit_lowering_header
from etx.machine.model import load_machine
from etx.tools import acquire_contract as ac

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "etx/runtime/include/etx"
CFLAGS = "-O3 -DETX_DSL_NT -DETX_DSL_VECNORM"


def wrapper(export, params):
    """params: (offset, llvm type) in the argument block, as etx.importer.adapter emits them."""
    out = [f"define void @etx_imp_{export}(ptr noundef %args, i32 noundef %blk, i32 noundef %tid, i32 noundef %slot) #0 {{",
           "  %args4 = addrspacecast ptr %args to ptr addrspace(4)"]
    for k, (off, ty) in enumerate(params):
        out += [f"  %a{k}.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {off}",
                f"  %a{k} = load {ty}, ptr addrspace(4) %a{k}.p, align {8 if ty.startswith('ptr') else 4}"]
    for g, off in (("gx", 64), ("gy", 68), ("gz", 72)):
        out += [f"  %{g}.p = getelementptr inbounds i8, ptr addrspace(4) %args4, i64 {off}",
                f"  %{g} = load i32, ptr addrspace(4) %{g}.p, align 4"]
    out += ["  %bx = urem i32 %blk, %gx", "  %b1 = udiv i32 %blk, %gx", "  %by = urem i32 %b1, %gy",
            "  %bz = udiv i32 %b1, %gy"]
    args = ", ".join([f"{ty} %a{k}" for k, (_, ty) in enumerate(params)]
                     + [f"i32 %{v}" for v in ("bx", "by", "bz", "gx", "gy", "gz", "tid", "slot")])
    return "\n".join(out + [f"  call void @etx_imp_{export}.body.s0({args})", "  ret void", "}"])


G = "ptr addrspace(1)"
ETXP = "i32 %etx.bx, i32 %etx.by, i32 %etx.bz, i32 %etx.gx, i32 %etx.gy, i32 %etx.gz, i32 %etx.tid, i32 %etx.slot"
REDUCE_BODY = f"""
define internal void @etx_imp_attn_reduce.body.s0({G} noalias noundef writeonly captures(none) %0, {G} noalias noundef readonly captures(none) %1, {G} noalias noundef readonly captures(none) %2, {G} noalias noundef readonly %3, {G} noalias noundef readonly captures(none) %4, {G} noalias noundef readonly captures(address_is_null) %5, i32 noundef %6, {G} noalias noundef readonly captures(address_is_null) %7, {ETXP}) local_unnamed_addr #7 {{
  %9 = zext i32 %etx.bx to i64
  %10 = getelementptr inbounds nuw float, {G} %1, i64 %9
  %11 = load float, {G} %10, align 4, !tbaa !1
  %12 = getelementptr inbounds nuw float, {G} %2, i64 %9
  %13 = load float, {G} %12, align 4, !tbaa !1
  %14 = getelementptr inbounds nuw i32, {G} %4, i64 %9
  %15 = load i32, {G} %14, align 4, !tbaa !1
  %16 = icmp eq {G} %5, null
  br i1 %16, label %20, label %17

17:                                               ; preds = %8
  %18 = getelementptr inbounds nuw i32, {G} %5, i64 %9
  %19 = load i32, {G} %18, align 4
  br label %20

20:                                               ; preds = %17, %8
  br label %21

21:                                               ; preds = %21, %20
  %22 = phi {G} [ %3, %20 ], [ %26, %21 ]
  %23 = phi float [ 0.000000e+00, %20 ], [ %25, %21 ]
  %24 = load float, {G} %22, align 4, !tbaa !1
  %25 = fadd float %23, %24
  %26 = getelementptr inbounds nuw float, {G} %22, i64 1
  %27 = icmp ult i32 %15, 64
  br i1 %27, label %21, label %28

28:                                               ; preds = %21
  fence syncscope("workgroup") release
  call void @etx_imp_slot_barrier(i32 %etx.slot, i32 2)
  fence syncscope("workgroup") acquire
  store float %25, ptr addrspace(3) @etx_lds_arena, align 4, !tbaa !1
  %29 = icmp eq {G} %7, null
  br i1 %29, label %32, label %30

30:                                               ; preds = %28
  %31 = load float, {G} %7, align 4, !tbaa !1
  br label %32

32:                                               ; preds = %30, %28
  %33 = fptrunc float %25 to bfloat
  %34 = getelementptr inbounds nuw i8, {G} %0, i64 %9
  store bfloat %33, {G} %34, align 2, !tbaa !2
  ret void
}}"""
GEMV_BODY = f"""
define internal void @etx_imp_wvsplitk_y2.body.s0(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, {G} noundef readonly captures(none) %6, {G} noalias noundef readonly captures(none) %7, {G} noalias noundef readonly captures(address_is_null) %8, {G} noundef writeonly captures(none) %9, i32 noundef %10, i32 noundef %11, {ETXP}) local_unnamed_addr #1 {{
  %13 = zext i32 %etx.tid to i64
  %14 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %7, i64 %13
  %15 = getelementptr inbounds nuw i8, ptr addrspace(3) @etx_lds_arena, i64 %13
  tail call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %15, {G} noundef align 16 dereferenceable(16) %14, i64 16, i1 false), !tbaa.struct !7
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %16 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %6, i64 %13
  %17 = load <4 x float>, {G} %16, align 16, !tbaa !8, !nontemporal !13
  %18 = load <4 x float>, ptr addrspace(3) %15, align 16
  %19 = fmul <4 x float> %17, %18
  %20 = extractelement <4 x float> %19, i64 0
  %21 = icmp eq {G} %8, null
  br i1 %21, label %25, label %22

22:                                               ; preds = %12
  %23 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %8, i64 %13
  %24 = load bfloat, {G} %23, align 2, !tbaa !8
  br label %25

25:                                               ; preds = %22, %12
  %26 = fptrunc float %20 to bfloat
  %27 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %9, i64 %13
  store bfloat %26, {G} %27, align 2, !tbaa !8
  ret void
}}"""
ATTN_BODY = ("define internal void @etx_imp_attn_mfma16_g8.body.s0("
             + ", ".join([f"{G} noalias noundef readonly captures(none) %{k}" for k in range(16)]
                         + [f"{G} noalias noundef readnone captures(none) %16"]
                         + [f"i32 noundef %{k}" for k in (17, 18, 19)] + [ETXP])
             + ") local_unnamed_addr #7 {\n  ret void\n}")
P8, I32 = "ptr addrspace(1)", "i32"
IR = "\n".join([
    "target triple = \"amdgcn-amd-amdhsa\"",
    "%struct.__hip_bfloat16 = type { i16 }",
    "@etx_lds_arena = external addrspace(3) global [0 x i8], align 16",
    wrapper("attn_reduce", [(0, P8), (8, P8), (16, P8), (24, P8), (32, P8), (40, P8), (48, I32), (56, P8)]),
    REDUCE_BODY,
    wrapper("wvsplitk_y2", [(0, I32), (4, I32), (8, I32), (12, I32), (16, I32), (20, I32), (24, P8), (32, P8),
                            (40, P8), (48, P8), (56, I32), (60, I32)]),
    GEMV_BODY, ATTN_BODY,
    "declare void @etx_imp_slot_barrier(i32, i32) #0",
    "declare void @llvm.memcpy.p3.p1.i64(ptr addrspace(3), ptr addrspace(1), i64, i1 immarg)", ""])


def consumer_case(word=12, scope="DEVICE", acquire="        ETX_ACQUIRE_DEVICE();"):
    return f"""    case 6: {{ // o_proj_0 [static] grid=(304,)
      ctx.cst[0] = 80; ctx.cst[1] = 0; ctx.cst[2] = 0; ctx.cst[3] = 0;
#ifndef ETX_NO_PRE_WAIT
      if (threadIdx.x < 64) etx_pre_wvsplitk_y2(&ctx);   // before the wait: does not depend on the event
#endif
      if (threadIdx.x == 0) {{
      // wait E_4 via 'i->(0)' ({scope})
#define ETX_TARGET(idx) etx_wait_{scope}(p.events + ETX_EVW({word} + (idx)), p.ctrl_abort)
  ETX_TARGET(0);
#undef ETX_TARGET
{acquire}
      }}
      sym = 6;   // etx_tile_wvsplitk_y2
      break; }}"""


PHASE1 = """    case 5: { // reduce_0 [static] grid=(4,)
      ctx.cst[0] = 0; ctx.cst[1] = 0; ctx.cst[2] = 0; ctx.cst[3] = 0;
      if (threadIdx.x == 0) {
      // wait E_3 via 'i->(0)' (DEVICE)
#define ETX_TARGET(idx) etx_wait_DEVICE(p.events + ETX_EVW(11 + (idx)), p.ctrl_abort)
  ETX_TARGET(0);
#undef ETX_TARGET
        ETX_ACQUIRE_DEVICE();
      }
      sym = 5;   // etx_tile_attn_reduce
      break; }
@CONSUMER@
    default: break;
  }"""
MIDDLE = """  __syncthreads();
  if (tr && threadIdx.x == 0) tr[1] = (uint64_t)ETX_TIMER();
  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body
#ifndef ETX_NO_ALLWAVE_DRAIN
  ETX_RELEASE_DOMAIN();   // every wave drains its own stores before the barrier: thread 0's release only covers wave 0
#endif
  __syncthreads();
  if (tr && threadIdx.x == 0) { tr[2] = (uint64_t)ETX_TIMER(); tr[3] = worker; }
  if (threadIdx.x != 0) return;"""
PHASE3 = """  switch (t.type) {   // phase 3: releases, arrives, pushes
    case 5: { // reduce_0
      // arrive E_4 via 'i->(0)' (DEVICE)
#define ETX_TARGET(idx) (void)etx_arrive_flush_DEVICE_s(p, 12 + (idx), domain, 1)
  ETX_TARGET(0);
#undef ETX_TARGET
      break; }
    case 6: { // o_proj_0
      // arrive E_5 via 'i->(0)' (DEVICE)
#define ETX_TARGET(idx) (void)etx_arrive_flush_DEVICE_s(p, 13 + (idx), domain, 38)
  ETX_TARGET(0);
#undef ETX_TARGET
      break; }
    default: break;
  }
}"""


def source(consumer=None):
    return "\n".join(["// generated by etx: graph fixture, arch gfx942, device 0", "#include \"etx_lowering.h\"",
                      "__global__ void etx_megakernel(etx_params p) {",
                      "  switch (t.type) {   // phase 1: immediates, waits, acquire",
                      PHASE1.replace("@CONSUMER@", consumer or consumer_case()), MIDDLE, PHASE3, ""])


def plan():
    reduce_domains = [0, 1, 3, 4]
    descs = [dict(id=k, type=5, coord=[k], domain=d) for k, d in enumerate(reduce_domains)]
    descs += [dict(id=4 + k, type=6, coord=[k], domain=k % 8) for k in range(304)]
    return dict(
        types={"reduce_0": dict(id=5, symbol="etx_tile_attn_reduce", mode="static", grid=[4]),
               "o_proj_0": dict(id=6, symbol="etx_tile_wvsplitk_y2", mode="static", grid=[304])},
        events=[dict(name="E_3", id=3, offset=11, shape=[1], scope="DEVICE", counts=[5], runtime_init=False),
                dict(name="E_4", id=4, offset=12, shape=[1], scope="DEVICE", counts=[4], runtime_init=False)],
        descs=descs, task_in=[[11]] * 4 + [[12]] * 304, ev_inst=[["E_4", [0], [0, 1, 2, 3]]])


IMP_TILES = "\n".join(["// fixture", ac.PRE_WAIT, ""])


def build(tmp_path, ir=IR, src=None, pl=None, lowering=None, imp_tiles=IMP_TILES):
    d = tmp_path / "build"
    d.mkdir(exist_ok=True)
    (d / "imports.ll").write_text(ir)
    (d / "megakernel_d0.hip").write_text(src if src is not None else source())
    (d / "plan.json").write_text(json.dumps(pl if pl is not None else plan()))
    (d / "etx_lowering.h").write_text(lowering if lowering is not None else emit_lowering_header(load_machine("gfx942")))
    (d / "imp_tiles.hip").write_text(imp_tiles)
    return d


def errors_of(tmp_path, cflags=CFLAGS, runtime=RUNTIME, **kw):
    d = build(tmp_path, **kw)
    errors, report = ac.run(d, d / "imports.ll", cflags, runtime)
    return errors, report


def violates(tmp_path, needle, **kw):
    errors, _ = errors_of(tmp_path, **kw)
    assert errors, "the mutation passed the contract check"
    assert any(needle in e for e in errors), errors
    return errors


def test_minimal_build_passes(tmp_path, capsys):
    errors, report = errors_of(tmp_path)
    assert errors == []
    inv = report["ir"]
    assert inv["etx_imp_attn_reduce.body.s0"] == {0: {"store": 1}, 1: {"plain": 1}, 2: {"plain": 1}, 3: {"plain": 1},
                                                  4: {"plain": 1}, 5: {"plain": 1}, 7: {"plain": 1}}
    assert inv["etx_imp_wvsplitk_y2.body.s0"] == {6: {"nontemporal": 1}, 7: {"memcpy": 1}, 8: {"plain": 1},
                                                  9: {"store": 1}}
    (p,) = report["pairs"]
    assert (p["event"], p["word"], p["share"], p["crosses_xcd"]) == ("E_4", 12, 1, True)
    d = tmp_path / "build"
    assert ac.main(["--build", str(d), "--ir", str(d / "imports.ll"), "--cflags", CFLAGS]) == 0
    assert capsys.readouterr().out.startswith("PASS: attn_reduce -> wvsplitk_y2")


IR_MUTATIONS = {
    "wrapper offset": (("ptr addrspace(4) %args4, i64 32\n  %a7 = load ptr", "ptr addrspace(4) %args4, i64 40\n  %a7 = load ptr"),
                       "argument load"),
    "consumer input not readonly": ((f"{G} noalias noundef readonly captures(none) %7, {G} noalias noundef readonly captures(address_is_null) %8",
                                     f"{G} noalias noundef captures(none) %7, {G} noalias noundef readonly captures(address_is_null) %8"),
                                    "parameter 7 (A) is readwrite"),
    "producer output not noalias": ((f"{G} noalias noundef writeonly captures(none) %0", f"{G} noundef writeonly captures(none) %0"),
                                    "parameter 0 (out) noalias"),
    "atomic": (("  %26 = fptrunc float %20 to bfloat", "  %26 = fptrunc float %20 to bfloat\n  %x = atomicrmw add ptr addrspace(3) %15, i32 1 monotonic"),
               "opcode 'atomicrmw'"),
    "volatile": (("%11 = load float, ", "%11 = load volatile float, "), "volatile"),
    "inline asm": (("  %33 = fptrunc", "  call void asm sideeffect \"buffer_wbl2\", \"\"()\n  %33 = fptrunc"), "inline asm"),
    "inttoptr": (("  %33 = fptrunc", "  %p = inttoptr i64 %9 to ptr addrspace(1)\n  %v = load float, ptr addrspace(1) %p, align 4\n  %33 = fptrunc"),
                 "opcode 'inttoptr'"),
    "unknown callee": (("  tail call void @llvm.amdgcn.s.barrier()", "  tail call void @llvm.amdgcn.s.barrier()\n  call void @llvm.amdgcn.buffer.wbinvl1()"),
                       "llvm.amdgcn.buffer.wbinvl1 is not in the allowlist"),
    "agent fence": (('fence syncscope("workgroup") acquire\n  store float', 'fence syncscope("agent") acquire\n  store float'),
                    "fence outside the workgroup scope"),
    "pointer loaded": (("%11 = load float, ", "%11 = load ptr addrspace(1), "), "pointer is loaded from memory"),
    "store to a read input": (("%13 = load float, {G} %12, align 4, !tbaa !1".format(G=G), f"store float %11, {G} %12, align 4"),
                              "store through parameter 2 (max_logits)"),
    "select of two parameters": (("%27 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %9, i64 %13".format(G=G),
                                  f"%s = select i1 %21, {G} %7, {G} %9\n  %27 = getelementptr inbounds nuw %struct.__hip_bfloat16, {G} %s, i64 %13"),
                                 "may address parameters [7, 9]"),
    "flat access": (("  %33 = fptrunc", "  %f = load float, ptr %x, align 4\n  %33 = fptrunc"), "outside the global (1) and LDS (3)"),
    "uncalled body": ((ATTN_BODY, ATTN_BODY + "\n" + REDUCE_BODY.replace("body.s0", "body.s1")), "bodies called"),
    "attention touches attn_out": ((f"{G} noalias noundef readnone captures(none) %16", f"{G} noalias noundef writeonly captures(none) %16"),
                                   "parameter 16 (bound to attn_out) is not readnone"),
    "activation read without memcpy": ((
        "  tail call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %15, "
        f"{G} noundef align 16 dereferenceable(16) %14, i64 16, i1 false), !tbaa.struct !7",
        f"  %m = load <4 x float>, {G} %14, align 16"), "parameter 7 (A) is accessed as ['plain']"),
    "consumer input non-temporal": ((
        "  tail call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) noundef align 16 dereferenceable(16) %15, "
        f"{G} noundef align 16 dereferenceable(16) %14, i64 16, i1 false), !tbaa.struct !7",
        f"  %m = load <4 x float>, {G} %14, align 16, !nontemporal !13"), "parameter 7 (A) is accessed as ['nontemporal']"),
    "global pointer escapes": (("  %33 = fptrunc", f"  %i = ptrtoint {G} %0 to i64\n  %33 = fptrunc"), "opcode 'ptrtoint'"),
    "missing export": (("@etx_imp_wvsplitk_y2(", "@etx_imp_wvsplitk_y3("), "export etx_imp_wvsplitk_y2 is not defined"),
}


@pytest.mark.parametrize("name", sorted(IR_MUTATIONS))
def test_ir_mutation_fails(tmp_path, name):
    (old, new), needle = IR_MUTATIONS[name]
    assert IR.count(old) == 1, f"fixture text for {name!r} not unique"
    violates(tmp_path, needle, ir=IR.replace(old, new))


SRC_MUTATIONS = {
    "no acquire": (lambda: source(consumer_case(acquire="")), "expected 'ETX_ACQUIRE_DEVICE"),
    "coherent tile, no acquire": (lambda: source(consumer_case(acquire="        // coherent tile: no acquire invalidate")),
                                  "expected 'ETX_ACQUIRE_DEVICE"),
    "DOMAIN wait": (lambda: source(consumer_case(scope="DOMAIN")), "source o_proj_0 (phase 1)"),
    "event word mismatch": (lambda: source(consumer_case(word=13)), "arrives on E_4 word 12"),
    "no-flush arrive": (lambda: source().replace("etx_arrive_flush_DEVICE_s(p, 12", "etx_arrive_flush_DEVICE_s_nf(p, 12"),
                        "source reduce_0 (phase 3)"),
    "second producer": (lambda: source().replace("(p, 13 + (idx), domain, 38)", "(p, 12 + (idx), domain, 38)"),
                        "also arrive on word 12"),
    "no all-wave drain": (lambda: source().replace("#ifndef ETX_NO_ALLWAVE_DRAIN\n  ETX_RELEASE_DOMAIN();", "#ifndef ETX_NO_ALLWAVE_DRAIN\n"),
                          "all-wave drain"),
    "arrive before the barrier": (lambda: source().replace("  __syncthreads();\n  if (tr && threadIdx.x == 0) { tr[2]",
                                                           "  if (tr && threadIdx.x == 0) { tr[2]"), "all-wave drain"),
    "body before the wait barrier": (lambda: source().replace("  __syncthreads();\n  if (tr && threadIdx.x == 0) tr[1]",
                                                              "  if (tr && threadIdx.x == 0) tr[1]"), "all-wave drain"),
    "coherent define": (lambda: source().replace("#include \"etx_lowering.h\"", "#define ETX_COHERENT_DEVICE_DATA 1\n#include \"etx_lowering.h\""),
                        "#define ETX_COHERENT_DEVICE_DATA"),
    "consumer touches activations before the wait": (
        lambda: source().replace("etx_pre_wvsplitk_y2(&ctx);", "etx_pre_x_wvsplitk_y2(&ctx);"), "source o_proj_0 (phase 1)"),
    "no imported pair": (lambda: source().replace("// etx_tile_wvsplitk_y2", "// etx_tile_dsl_gemv"), "no attn_reduce -> wvsplitk_y2 pair"),
}


@pytest.mark.parametrize("name", sorted(SRC_MUTATIONS))
def test_source_mutation_fails(tmp_path, name):
    make, needle = SRC_MUTATIONS[name]
    src = make()
    assert src != source()
    violates(tmp_path, needle, src=src)


def mutate_plan(change):
    p = plan()
    change(p)
    return p


PLAN_MUTATIONS = {
    "event counts fewer producers": (lambda p: p["events"][1].update(counts=[3]), "counting 4 producers"),
    "event scope DOMAIN": (lambda p: p["events"][1].update(scope="DOMAIN"), "is not a DEVICE event"),
    "consumer waits elsewhere": (lambda p: p["task_in"].__setitem__(10, [11]), "not every o_proj_0 task waits"),
    "foreign producer": (lambda p: p["ev_inst"][0][2].__setitem__(3, 99), "not the reduce_0 tasks"),
    "dynamic consumer": (lambda p: p["types"]["o_proj_0"].update(mode="dynamic"), "static tasks only"),
    "type id differs": (lambda p: p["types"]["o_proj_0"].update(id=7), "type o_proj_0 id 7"),
}


@pytest.mark.parametrize("name", sorted(PLAN_MUTATIONS))
def test_plan_mutation_fails(tmp_path, name):
    change, needle = PLAN_MUTATIONS[name]
    violates(tmp_path, needle, pl=mutate_plan(change))


def test_share_must_match_producers_per_domain(tmp_path):
    src = source().replace("(p, 12 + (idx), domain, 1)", "(p, 12 + (idx), domain, 2)")
    violates(tmp_path, "arrive share 2 differs from producers per domain", src=src)
    two = plan()
    for d, domain in zip(two["descs"][:4], (0, 0, 1, 1)):
        d["domain"] = domain
    errors, _ = errors_of(tmp_path, src=src, pl=two)
    assert errors == []


LOWERING = emit_lowering_header(load_machine("gfx942"))
LOWERING_MUTATIONS = {
    "acquire empty": (('#define ETX_ACQUIRE_DEVICE() do { asm volatile("buffer_inv sc1" ::: "memory"); } while (0)\n#define ETX_POLL_DEVICE',
                       '#define ETX_ACQUIRE_DEVICE() do { /* none */; } while (0)\n#define ETX_POLL_DEVICE'), "ETX_ACQUIRE_DEVICE()"),
    "release without writeback": (('#define ETX_RELEASE_DEVICE() do { asm volatile("buffer_wbl2 sc1\\n\\ts_waitcnt vmcnt(0)" ::: "memory"); } while (0)\n#define ETX_ACQUIRE_DEVICE',
                                   '#define ETX_RELEASE_DEVICE() do { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); } while (0)\n#define ETX_ACQUIRE_DEVICE'),
                                  "ETX_RELEASE_DEVICE()"),
    "relaxed device acquire": (("#ifdef ETX_COHERENT_DEVICE_DATA", "#ifdef ETX_ACQUIRE_DEVICE_RELAXED\n#endif\n#ifdef ETX_COHERENT_DEVICE_DATA"),
                               "relaxed DEVICE acquire"),
    "other arch": (('#define ETX_ARCH "gfx942"', '#define ETX_ARCH "gfx950"'), "ETX_ARCH"),
}


@pytest.mark.parametrize("name", sorted(LOWERING_MUTATIONS))
def test_lowering_mutation_fails(tmp_path, name):
    (old, new), needle = LOWERING_MUTATIONS[name]
    assert LOWERING.count(old) == 1
    violates(tmp_path, needle, lowering=LOWERING.replace(old, new))


def runtime_copy(tmp_path, file, old, new):
    d = tmp_path / "runtime"
    d.mkdir(exist_ok=True)
    for name in ("primitives.h", "import.h"):
        text = (RUNTIME / name).read_text()
        if name == file:
            assert text.count(old) == 1
            text = text.replace(old, new)
        (d / name).write_text(text)
    return d


def test_runtime_changes_fail(tmp_path):
    rt = runtime_copy(tmp_path, "primitives.h", "  if (old + 1 != share) return 1;\n  ETX_RELEASE_DEVICE();\n",
                      "  if (old + 1 != share) return 1;\n")
    violates(tmp_path, "etx_arrive_flush_DEVICE_s differs", runtime=rt)
    rt = runtime_copy(tmp_path, "primitives.h", "while (ETX_POLL_##S(e) > 0)", "while (ETX_POLL_##S(e) > 1)")
    violates(tmp_path, "etx_wait_<scope> differs", runtime=rt)
    rt = runtime_copy(tmp_path, "import.h", "for (int o = 0; o < bytes; o += 64)", "for (int o = 0; o < 2 * bytes; o += 64)")
    violates(tmp_path, "etx_imp_touch differs", runtime=rt)
    violates(tmp_path, "etx_pre_wvsplitk_y2 is not the argument-block touch",
             imp_tiles=IMP_TILES.replace("c->cst[0]), 128)", "c->cst[0]), 4096)"))


@pytest.mark.parametrize("flags, needle", [
    ("-DETX_COHERENT_DEVICE_DATA", "-DETX_COHERENT_DEVICE_DATA is not in the contract's allowlist"),
    ("-O3 -DETX_NO_ALLWAVE_DRAIN", "-DETX_NO_ALLWAVE_DRAIN"),
    ("-D ETX_VL_UC", "-DETX_VL_UC"),
    ("-DSOMETHING_NEW=1", "-DSOMETHING_NEW"),
    ("-include extra.h", "-include is not allowed"),
    ("-UETX_DSL_NT", "-U is not allowed"),
])
def test_cflags_fail(tmp_path, flags, needle):
    violates(tmp_path, needle, cflags=flags)


def test_bindings_reject_a_new_reader_of_attn_out():
    from etx.importer import spec

    class Spec:
        ARGUMENTS = dict(spec.ARGUMENTS, extra=[("ptr", "attn_out")])
        HOST_DISPATCH = spec.HOST_DISPATCH
    errors = []
    ac.check_bindings(errors, Spec)
    assert any("arguments naming attn_out" in e for e in errors), errors
    errors = []
    ac.check_bindings(errors)
    assert errors == []


def test_refusals(tmp_path, capsys):
    d = build(tmp_path)
    assert ac.main(["--build", str(d), "--ir", str(d / "imports.ll")]) == 2
    assert "missing input: --cflags" in capsys.readouterr().out
    (d / "plan.json").unlink()
    assert ac.main(["--build", str(d), "--ir", str(d / "imports.ll"), "--cflags", CFLAGS]) == 2
    assert "missing input:" in capsys.readouterr().out
    d = build(tmp_path)
    assert ac.main(["--build", str(d), "--ir", str(d / "absent.ll"), "--cflags", CFLAGS]) == 2
    assert "missing input: IR" in capsys.readouterr().out


def test_violation_exit_code(tmp_path, capsys):
    d = build(tmp_path, src=source(consumer_case(acquire="")))
    assert ac.main(["--build", str(d), "--ir", str(d / "imports.ll"), "--cflags", CFLAGS]) == 1
    out = capsys.readouterr().out
    assert "VIOLATION:" in out and "FAIL:" in out and "PASS" not in out


# The imported builds of the main checkout (examples/vllm_llm/build.sh, ROCm 7.2.4 device libraries). Built without
# ETX_VL_CHECK, so the device flags were build.sh's defaults.
REAL = Path(os.environ.get("ETX_CONTRACT_BUILDS", Path.home() / "code/etx/build/rocm22"))


@pytest.mark.parametrize("name, pairs", [("qwen3-30b-a3b_xcd_qknr", 48), ("mixtral-8x7b_imported", 32)])
def test_real_imported_build(name, pairs):
    d = REAL / name
    if not (d / "imports.bc").is_file():
        pytest.skip(f"real build not present: {d} (set ETX_CONTRACT_BUILDS)")
    from etx.importer.bundle import llvm_tool
    if not Path(llvm_tool("llvm-dis")).is_file():
        pytest.skip("llvm-dis not found (set ETX_LLVM)")
    errors, report = ac.run(d, None, CFLAGS, RUNTIME)
    assert errors == []
    assert len(report["pairs"]) == pairs and all(p["crosses_xcd"] for p in report["pairs"])
