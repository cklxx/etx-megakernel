"""Operator specifications for the imported bf16 decode path."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BF = "__hip_bfloat16"
CB = "c10::BFloat16"


@dataclass
class Kernel:
    export: str                   # etx_imp_<export>
    match: str                    # substring of the mangled name, unique among the file's specialisations
    spec: str                     # the specialisation, as written in the (void)&... keep line
    block: tuple[int, int, int]   # the launch's block shape (from vLLM's host code)
    lds_cap: int | None = None
    note: str = ""
    inline: bool = True           # False: an out-of-line call (own register allocation), see ir_import


@dataclass
class Source:
    path: str                     # relative to the vLLM checkout
    kernels: list[Kernel]
    drop: str | None = None       # extra definitions to drop (slice_device --drop)
    includes: list[str] = field(default_factory=list)   # extra -I (relative to etx/importer)
    defines: list[str] = field(default_factory=list)


SOURCES = [
    Source("csrc/rocm/skinny_gemms.cu", drop=r"wvSplitKQ|fp8|Fp8|FP8", includes=["shim_nofp8"], kernels=[
        # wvSplitK(): gfx9 -> WVSPLIT_TILE_CFG(THRDS=64, WVPRGRP=16, sYT, N); sYT = ceil(M / (CuCount*4));
        # N == 1 -> YTILE 2, UNRL 2 unless sYT <= 1 (YTILE 1, UNRL 4); Kbp*N <= LDS/2 -> the _sml_ variant.
        Kernel("wvsplitk_y2", "wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi2ELi16ELi8ELi2ELi1E",
               f"wvSplitK_hf_sml_<{BF}, 64, 2, 16, 8, 2, 1>", (64, 16, 1), lds_cap=32 * 1024,
               note="M > 1216 (every projection and lm_head); LDS holds the K-vector, K <= 16384"),
        Kernel("wvsplitk_y1", "wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi1E",
               f"wvSplitK_hf_sml_<{BF}, 64, 1, 16, 8, 4, 1>", (64, 16, 1), lds_cap=32 * 1024,
               note="M <= 1216 (the MoE router, 128 rows)"),
    ]),
    Source("csrc/rocm/attention.cu", defines=["ENABLE_FP8"], kernels=[
        # paged_attention_custom_launcher: grid (num_seqs, max_num_partitions, num_kv_heads), block 256;
        # gqa <= 4 -> mfma4, else mfma16; then the reduction, grid (num_heads, num_seqs), block head_size.
        Kernel("attn_mfma4_g4", "QKV_mfma4_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi4E",
               f"paged_attention_ll4mi_QKV_mfma4_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 4>", (256, 1, 1),
               note="Qwen3-8B (32/8 heads)"),
        Kernel("attn_mfma16_g6", "QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi6E",
               f"paged_attention_ll4mi_QKV_mfma16_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 6, MFMAType::F16>", (256, 1, 1),
               note="Qwen2.5-1.5B (12/2 heads)"),
        Kernel("attn_mfma16_g8", "QKV_mfma16_kernelI14__hip_bfloat16S0_LN4vllm18Fp8KVCacheDataTypeE0ES0_Li16ELi128ELi256ELb0ELi8E",
               f"paged_attention_ll4mi_QKV_mfma16_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto, {BF}, 16, 128, 256, false, 8, MFMAType::F16>", (256, 1, 1),
               note="Qwen3-30B-A3B (32/4 heads)"),
        Kernel("attn_reduce", "reduce_kernelI14__hip_bfloat16S0_Li128ELi128ELi256ELi1E",
               f"paged_attention_ll4mi_reduce_kernel<{BF}, {BF}, 128, 128, 256, 1>", (128, 1, 1),
               note="npar_loops = ceil(partitions / 64) = 1 up to 16K tokens"),
    ]),
    Source("csrc/libtorch_stable/layernorm_kernels.cu", kernels=[
        # rms_norm: grid num_tokens, block min(hidden / vec, 1024), vec = gcd(8, hidden); 2-D for a vector,
        # 3-D for q/k norm over [tokens, heads, head_dim].
        Kernel("rms_norm_2d", "rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1E", f"vllm::rms_norm_kernel<{CB}, 8, 2, true>", (0, 1, 1),
               note="block = hidden/8 (512 for 4096, 192 for 1536, 256 for 2048): set per model"),
        Kernel("rms_norm_3d", "rms_norm_kernelIN3c108BFloat16ELi8ELi3ELb1E", f"vllm::rms_norm_kernel<{CB}, 8, 3, true>", (16, 1, 1),
               note="q/k norm: head_dim 128 / 8 = 16 threads per head"),
        # fused_add_rms_norm: grid num_tokens, block min(hidden, 1024), width 8 when aligned
        Kernel("fused_add_rms_norm", "fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb1E", f"vllm::fused_add_rms_norm_kernel<{CB}, 8, true>", (1024, 1, 1)),
    ]),
    Source("csrc/libtorch_stable/pos_encoding_kernels.cu", kernels=[
        # rotary_embedding: grid num_tokens, block min(num_heads * rot_dim / 2, 512); neox style for Qwen
        Kernel("rope_neox", "rotary_embedding_kernelIN3c108BFloat16ES2_Lb1E", f"vllm::rotary_embedding_kernel<{CB}, {CB}, true>", (512, 1, 1),
               note="block min(NH*64, 512): 512 for every model here"),
    ]),
    Source("csrc/libtorch_stable/fused_qknorm_rope_kernel.cu", kernels=[
        # fused_qk_norm_rope (vLLM's QKNormRoPEFusionPass, compilation_config enable_qk_norm_rope_fusion): batch 1 ->
        # token_heads_per_warp 1 (auto-select only raises it on SM 9.0) -> launchFusedQKNormRope: block 256 (8 warps of
        # 32 lanes), grid ceil((NH + NKV) / 8), one warp per head; bf16 qkv and cos/sin cache, neox -> interleave false
        Kernel("qknorm_rope", "fusedQKNormRopeKernelIN3c108BFloat16E",
               f"tensorrt_llm::kernels::fusedQKNormRopeKernel<{CB}, {CB}, 128, false>", (256, 1, 1),
               note="head_dim 128 (Qwen3); grid 5 for 32 + 4 heads"),
    ]),
    Source("csrc/libtorch_stable/cache_kernels.cu", defines=["ENABLE_FP8"], kernels=[
        # reshape_and_cache: grid num_tokens, block min(num_kv_heads * head_size / x, 512), x = 16 / 2
        Kernel("reshape_and_cache", "reshape_and_cache_kernelI14__hip_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0E",
               f"vllm::reshape_and_cache_kernel<{BF}, {BF}, vllm::Fp8KVCacheDataType::kAuto>", (0, 1, 1),
               note="block = NKV * 16: 128 (8B), 32 (1.5B), 64 (30B-A3B): set per model"),
    ]),
    Source("csrc/libtorch_stable/activation_kernels.cu", kernels=[
        # silu_and_mul: grid num_tokens, block min(d / vec, 1024), vec = 16 / 2 (d % 8 == 0)
        Kernel("silu_and_mul", "act_and_mul_kernelIN3c108BFloat16E",
               f"vllm::act_and_mul_kernel<{CB}, typename vllm::PackedTypeConverter<{CB}>::Type, vllm::silu_kernel<{CB}>, "
               f"vllm::packed_silu_kernel<typename vllm::PackedTypeConverter<{CB}>::Type>, true, true, false, false>", (1024, 1, 1),
               note="dense: 1 row, block 1024; MoE (fused_experts): 8 rows of 768, block 768/8 = 96: set per model"),
    ]),
    # Qwen3-30B-A3B MoE at batch 1 (fused_moe/fused_moe.py, TritonExperts; docs/RESEARCH-moe-import.md): router
    # (wvsplitk_y1), top-k, two Triton fused_moe_kernel launches (etx.importer.triton_import), silu, moe_sum
    Source("csrc/libtorch_stable/moe/topk_softmax_kernels.cu", kernels=[
        # topkGatingLauncherHelper: BYTES_PER_LDG = min(16, 2 * 128) = 16, VPT = 128 / (64 / (256 / 16)) ... = 8,
        # ROWS_PER_WARP = 64 / 16 = 4; block (64, WARPS_PER_TB = 4); grid ceil(ceil(rows / 4) / 4) = 1 at batch 1
        Kernel("topk_softmax_e128", "topkGatingILi8ELi128ELi4ELi16ELi64Ei14__hip_bfloat16LNS0_11ScoringFuncE0E",
               f"vllm::moe::topkGating<8, 128, 4, 16, 64, int, {BF}, vllm::moe::ScoringFunc::SCORING_SOFTMAX>", (64, 4, 1),
               note="128 experts, softmax scoring, int32 ids (Qwen3-30B-A3B)"),
        # topkGating is instantiated per expert count by the launcher switch; BYTES = min(16, 2E), ELTS = BYTES/2,
        # VPT = max(1, E/(ELTS*64))*ELTS -- 8 for both E=8 and E=128. Slicing one expert count and running another
        # computes garbage deterministically (Mixtral top-2-of-8 through the e128 slice, 2026-10-02): shape-keyed.
        Kernel("topk_softmax_e8", "topkGatingILi8ELi8ELi4ELi16ELi64Ei14__hip_bfloat16LNS0_11ScoringFuncE0E",
               f"vllm::moe::topkGating<8, 8, 4, 16, 64, int, {BF}, vllm::moe::ScoringFunc::SCORING_SOFTMAX>", (64, 4, 1),
               note="8 experts (Mixtral-8x7B)"),
    ]),
    Source("csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", kernels=[
        # moe_sum, vectorised path (hidden contiguous, 16 B aligned): block 256, grid min(ceil(tokens * d / 8 / 256), 65535)
        Kernel("moe_sum_k8", "moe_sum_vec_kernelIN3c108BFloat16EiLi8ELb0E",
               f"vllm::moe::moe_sum_vec_kernel<{CB}, int32_t, 8, false>", (256, 1, 1),
               note="top-8, not pad-aware (no expert map)"),
        # moe_sum_vec_kernel bakes TOPK into the template: the k8 slice at top-2 sums six rows nobody wrote
        # (the Mixtral failure root cause). Shape-keyed like the topk entries.
        Kernel("moe_sum_k2", "moe_sum_vec_kernelIN3c108BFloat16EiLi2ELb0E",
               f"vllm::moe::moe_sum_vec_kernel<{CB}, int32_t, 2, false>", (256, 1, 1),
               note="top-2 (Mixtral-8x7B), not pad-aware"),
    ]),
]

# Triton kernels, taken from the Triton cache vLLM wrote on the VM (vLLM compiles them at run time, so the exact kernel
# is the cached one; examples/vllm_llm/triton/<export>/CACHE_DIR names the cache entry). fused_moe_kernel at batch 1
# (naive block assignment, default config BLOCK_SIZE_M/N/K = 16/64/128, 4 warps, 20 KB LDS): num_valid_tokens = 8
# is not a multiple of 16, which selects the entries without that divisibility specialisation.
TRITON_DIR = Path(__file__).resolve().parents[2] / "examples/vllm_llm/triton"
TRITON = {
    # The fused_moe kernels are decode-shape binaries harvested from the Triton cache vLLM wrote for THIS model
    # (top_k, MUL_ROUTED_WEIGHT and the block config are constexprs baked at compile): keyed by expert count so a
    # model whose entries were not harvested fails at import with a name, not at EOS (Mixtral, 2026-10-02).
    "fused_moe_up": ("fused_moe_kernel", "Qwen3-30B-A3B gate_up: E=128, MUL_ROUTED_WEIGHT=False"),
    "fused_moe_down": ("fused_moe_kernel", "Qwen3-30B-A3B down: E=128, MUL_ROUTED_WEIGHT=True"),
    "fused_moe_up_e8": ("fused_moe_kernel", "Mixtral-8x7B gate_up: E=8, BN=16 BK=256 nw=2, MUL_ROUTED_WEIGHT=False"),
    "fused_moe_down_e8": ("fused_moe_kernel", "Mixtral-8x7B down: E=8, BN=16 BK=256 nw=2, MUL_ROUTED_WEIGHT=True"),
}


HOST_DISPATCH = r'''

    else if (r == "in_norm" && x.layer == 0) @in_norm@;
    else if (r == "in_norm") { copy(dn, d_norm, H * 2); fused_add(d_norm, W(pl + "ln1")); }
    else if (r == "qkv") gemv(W(pl + "wqkv"), qkv_in, qkv, bias ? W(pl + "bqkv") : nullptr);
    else if (r == "q_norm") @q_norm@;
    else if (r == "k_norm") @k_norm@;
    // fused_qk_norm_rope(qkv, nq, nk, nv, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, positions): kernel
    // arguments (qkv, nq, nk, nv, eps, q_w, k_w, cos_sin, positions, num_tokens, rotary_dim = cos_sin.size(1) = HD)
    else if (r == "qknr") @qknr@;
    else if (r == "rope") @rope@;
    else if (r == "cache") @cache@;
    else if (r == "attn") @attn@;
    else if (r == "reduce") @reduce@;
    else if (r == "o_proj") gemv(W(pl + "wo"), attn_out, o, nullptr);
    else if (r == "post_norm") { copy(o, o_norm, H * 2); fused_add(o_norm, W(pl + "ln2")); }
    else if (r == "gate_up") gemv(W(pl + "wgu"), o_norm, gu, nullptr);
    else if (r == "moe_act") @moe_act@;    // one row per expert
    else if (r == "act") @act@;
    else if (r == "router") gemv(W(pl + "wr"), o_norm, rlog, nullptr);
    // topk_softmax -> topkGatingLauncherHelper(gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,
    // num_tokens, topk, 0, num_experts, renormalize, bias, 1.0, stream, is_padding)
    else if (r == "topk") @topk@;
    // invoke_fused_moe_triton_kernel, naive block assignment: EM = M * top_k * BLOCK_SIZE_M, num_valid_tokens = M * top_k;
    // Triton dropped the None pointers and the strides equal to 1 (stride_ak, stride_bk, stride_cn) from the signature
    // and appended its global / profile scratch pointers (both sizes 0)
    else if (r == "moe_up") @moe_up@;
    else if (r == "moe_down") @moe_down@;
    // moe_sum, vectorised: (out, input, num_tokens, hidden, stride_token, stride_topk, nullptr, nullptr, 0, 0); the output
    // is the buffer the next fused_add_rms_norm reads (vLLM copies it there)
    else if (r == "moe_sum") @moe_sum@;
    else if (r == "down") gemv(W(pl + "wd"), act_out, dn, nullptr);
    else if (r == "final_norm") { copy(dn, d_norm, H * 2); fused_add(d_norm, W("final_norm")); }
    else if (r == "lm_head") gemv(tie ? W("embed") : W("lm_head"), d_norm, logits, nullptr);
    else die("unknown role " + r);
'''
HOST_HELPERS = r'''

    auto gemv = [&](const void* w, const void* in, void* out, const void* b) {   // ops.wvSplitK(weight, x, cu_count, bias)
      // CuCount must be the GRID SIZE actually launched, not the device's CU count: the kernel's row loop strides by
      // CuCount * _WvPrGrp * YTILE, so a larger value than the grid leaves rows [grid*wv*YT, CuCount*wv*YT) of every
      // pass unwritten. Only lm_head (M > grid*wv*YT) loops; its holes hit any token id in them (round 42: Mixtral).
      @gemv@;
    };
    auto fused_add = [&](bf* in, const void* w) { @fused_add@; };
'''

@dataclass(frozen=True)
class Operator:
    name: str
    template: str
    grid_rule: tuple[str, str, str]
    supported_shapes: str
    constants: dict
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...] = ()
    layouts: str = "contiguous row-major tensors"
    dtypes: str = "bf16 tensors, fp32 reductions, int32 indices, int64 positions"
    # Parameter index, parameter name, required expression. The ABI supplies offsets.
    checks: tuple[tuple[int, str, str], ...] = ()
    arguments: dict = field(default_factory=dict)
    # Inclusive bounds per axis. Equal bounds require an exact value.
    # Expressions use integer constants, model names, and multiplication.
    block_rule: tuple[tuple[str, str], ...] = (("1", "1024"), ("1", "1"), ("1", "1"))

    def grid(self, **context):
        return tuple(int(eval(v, {"__builtins__": {}}, context)) for v in self.grid_rule)


ARGUMENTS = {
    'in_norm': [('ptr', 'hn_r'), ('ptr', 'res_r'), ('i64', 'H'), ('i64', '0'), ('i64', '0'), ('i64', '0'), ('i64', '0'), ('ptr', 'W(pl + "ln1")'), ('i64', '0'), ('f32', 'eps'), ('i32', '1'), ('i32', 'H')],
    'q_norm': [('ptr', 'qn'), ('ptr', 'qkv'), ('i64', 'HD'), ('i64', 'NQKV'), ('i64', '0'), ('i64', 'NH'), ('i64', '0'), ('ptr', 'W(pl + "qn")'), ('i64', '0'), ('f32', 'eps'), ('i32', 'NH'), ('i32', 'HD')],
    'k_norm': [('ptr', 'kn'), ('ptr', 'qkv + NH * HD'), ('i64', 'HD'), ('i64', 'NQKV'), ('i64', '0'), ('i64', 'NKV'), ('i64', '0'), ('ptr', 'W(pl + "kn")'), ('i64', '0'), ('f32', 'eps'), ('i32', 'NKV'), ('i32', 'HD')],
    'qknr': [('ptr', 'qkv'), ('i32', 'NH'), ('i32', 'NKV'), ('i32', 'NKV'), ('f32', 'eps'), ('ptr', 'W(pl + "qn")'), ('ptr', 'W(pl + "kn")'), ('ptr', 'W("cos_sin")'), ('ptr', 'positions'), ('i32', '1'), ('i32', 'HD')],
    'rope': [('ptr', 'positions'), ('ptr', 'qbuf'), ('ptr', 'kbuf'), ('ptr', 'W("cos_sin")'), ('i32', 'HD'), ('i64', 'qstr'), ('i64', 'kstr'), ('i64', 'HD'), ('i32', 'NH'), ('i32', 'NKV'), ('i32', 'HD'), ('i64', '0'), ('b8', 'false')],
    'cache': [('ptr', 'kbuf'), ('ptr', 'qkv + (NH + NKV) * HD'), ('ptr', 'kc[x.layer]'), ('ptr', 'vc[x.layer]'), ('ptr', 'slot_mapping'), ('i32', '(int)kstr'), ('i32', 'NQKV'), ('i32', 'NKV'), ('i32', 'HD'), ('i32', 'BS'), ('i32', 'X'), ('ptr', 'kv_scale'), ('ptr', 'kv_scale')],
    'attn': [('ptr', 'qbuf'), ('ptr', 'kc[x.layer]'), ('ptr', 'vc[x.layer]'), ('i32', 'NKV'), ('f32', 'scale'), ('ptr', 'block_tables'), ('ptr', 'seq_lens'), ('ptr', 'nullptr'), ('i32', 'MAXBLK'), ('ptr', 'nullptr'), ('i32', '(int)qstr'), ('i32', 'NKV * HD * BS'), ('i32', 'HD * BS'), ('ptr', 'exp_sums'), ('ptr', 'max_logits'), ('ptr', 'tmp_out'), ('ptr', 'attn_out'), ('i32', 'MAXBLK'), ('ptr', 'kv_scale'), ('ptr', 'kv_scale')],
    'reduce': [('ptr', 'attn_out'), ('ptr', 'exp_sums'), ('ptr', 'max_logits'), ('ptr', 'tmp_out'), ('ptr', 'seq_lens'), ('ptr', 'nullptr'), ('i32', 'parts'), ('ptr', 'nullptr')],
    'moe_act': [('ptr', 'mc2'), ('ptr', 'mc1'), ('i32', 'MI'), ('f32', '0.0f'), ('f32', '1.0f'), ('f32', '0.0f')],
    'act': [('ptr', 'act_out'), ('ptr', 'gu'), ('i32', 'I'), ('f32', '0.0f'), ('f32', '1.0f'), ('f32', '0.0f')],
    'topk': [('ptr', 'rlog'), ('ptr', 'nullptr'), ('ptr', 'tw_r'), ('i32', '1'), ('ptr', 'ti_r'), ('ptr', 'tei + (size_t)rr * TOPK'), ('i32', 'TOPK'), ('i32', '0'), ('i32', 'E'), ('b8', 'norm_topk'), ('ptr', 'nullptr'), ('f64', '1.0'), ('ptr', 'nullptr')],
    'moe_up': [('ptr', 'o_norm'), ('ptr', 'W(pl + "w13")'), ('ptr', 'mc1'), ('ptr', 'ti_r'), ('ptr', 'ntpp'), ('i32', '2 * MI'), ('i32', 'H'), ('i32', 'TOPK * MOE_BM'), ('i32', 'TOPK'), ('i32', 'H'), ('i32', '2 * MI * H'), ('i32', 'H'), ('i32', '2 * MI'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('ptr', 'nullptr'), ('ptr', 'nullptr')],
    'moe_down': [('ptr', 'mc2'), ('ptr', 'W(pl + "w2")'), ('ptr', 'mc3'), ('ptr', 'tw_r'), ('ptr', 'ti_r'), ('ptr', 'ntpp'), ('i32', 'H'), ('i32', 'MI'), ('i32', 'TOPK * MOE_BM'), ('i32', 'TOPK'), ('i32', 'MI'), ('i32', 'H * MI'), ('i32', 'MI'), ('i32', 'H'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('i32', '0'), ('ptr', 'nullptr'), ('ptr', 'nullptr')],
    'moe_sum': [('ptr', 'd_norm'), ('ptr', 'mc3'), ('i64', '1'), ('i32', 'H'), ('i64', '(long long)TOPK * H'), ('i64', 'H'), ('ptr', 'nullptr'), ('ptr', 'nullptr'), ('i64', '0'), ('i64', '0')],
    'gemv': [('i32', 'x.K'), ('i32', 'x.K'), ('i32', 'x.K'), ('i32', 'xM'), ('i32', 'b ? xM : 1'), ('i32', '1'), ('ptr', 'w'), ('ptr', 'in'), ('ptr', 'b'), ('ptr', 'out'), ('i32', 'x.wv'), ('i32', 'x.gx * x.gy * x.gz')],
    'fused_add': [('ptr', 'in'), ('i64', 'H'), ('ptr', 'res_r'), ('ptr', 'w'), ('f32', 'eps'), ('i32', '1'), ('i32', 'H'), ('i64', 'H')],
}

OPERATORS = {}
for source in SOURCES:
    for kernel in source.kernels:
        e = kernel.export
        grid = ("1", "1", "1")
        shape = "batch=1, bf16"
        constants = {}
        reads, writes, aliases = ("input", "weights"), ("output",), ()
        checks = ()
        layout = "contiguous row-major tensors"
        if e.startswith("wvsplitk"):
            y = int(e[-1])
            grid = ("CU", "1", "1")
            shape += ", N=1, M>=YTILE, K<=16384"
            constants = dict(YTILE=y, THRDS=64, WVPRGRP=16, N=1)
            checks = ((0, "K", "p1"), (2, "K", "p1"), (5, "N", "1"), (11, "CuCount", "gx"))
            reads = ("weights", "input", "bias")
        elif e.startswith("attn_mfma"):
            g = int(e[-1])
            grid = ("1", "parts", "NKV")
            shape += f", HD=128, G={g}, BS=16, PART=256"
            constants = dict(HD=128, G=g, BS=16, PART=256)
            reads = ("query", "key_cache", "value_cache", "block_tables", "seq_lens")
            writes = ("exp_sums", "max_logits", "tmp_out", "output")
            layout = "query row-major, key cache [block, head, HD/8, BS, 8], value cache [block, head, HD, BS]"
        elif e == "attn_reduce":
            grid = ("NH", "1", "1")
            constants = dict(HD=128, PART=256, MAX_PARTS=64)
            reads = ("exp_sums", "max_logits", "tmp_out", "seq_lens")
        elif e == "rms_norm_3d":
            grid = ("heads", "1", "1")
            constants = dict(HD=128, VEC=8)
            checks = ((11, "HD", "128"),)
        elif e == "rms_norm_2d":
            constants = dict(VEC=8)
        elif e == "fused_add_rms_norm":
            reads = ("input", "residual", "weights")
            writes = ("input", "residual")
            aliases = (("input", "output"),)
            constants = dict(VEC=8)
        elif e == "rope_neox":
            reads = ("query", "key", "positions", "cos_sin")
            writes = ("query", "key")
            aliases = (("query", "query_output"), ("key", "key_output"))
            constants = dict(NEOX=True)
        elif e == "qknorm_rope":
            grid = ("(NH+NKV+7)//8", "1", "1")
            constants = dict(HD=128, INTERLEAVE=False)
            checks = ((10, "HD", "128"),)
            reads = ("qkv", "q_weight", "k_weight", "positions", "cos_sin")
            writes = ("qkv",)
            aliases = (("qkv", "output"),)
        elif e == "reshape_and_cache":
            reads = ("key", "value", "slot_mapping")
            writes = ("key_cache", "value_cache")
            layout = "key cache [block, head, HD/8, BS, 8], value cache [block, head, HD, BS]"
            constants = dict(CACHE_DTYPE="auto", X=8)
        elif e == "silu_and_mul":
            grid = ("rows", "1", "1")
            constants = dict(VEC=8)
            reads = ("gate_up",)
        elif e.startswith("topk_softmax"):
            experts = int(e.split('_e')[1])
            constants = dict(E=experts, VPT=8, WARPS=4)
            checks = ((3, "num_tokens", "1"), (8, "E", str(experts)))
            reads = ("logits",)
            writes = ("routing_weights", "expert_ids", "token_expert_indices")
        elif e.startswith("moe_sum"):
            topk = int(e.split('_k')[1])
            constants = dict(TOPK=topk)
            checks = ((2, "num_tokens", "1"), (4, "stride_token/TOPK", f"{topk}*p5"))
            reads = ("expert_output",)
        block_rule = tuple((str(value), str(value)) for value in kernel.block)
        if e in ('rms_norm_2d', 'rms_norm_3d', 'fused_add_rms_norm',
                 'rope_neox', 'silu_and_mul', 'moe_sum_k8', 'moe_sum_k2'):
            block_rule = (('1', '1024'), ('1', '1'), ('1', '1'))
        if e == 'reshape_and_cache':
            # Each thread copies one head segment. Extra threads return.
            block_rule = (('NKV*16', '1024'), ('1', '1'), ('1', '1'))
        OPERATORS[e] = Operator(e, kernel.spec, grid, shape, constants, reads, writes, aliases, layout,
                                checks=checks, block_rule=block_rule)

from dataclasses import replace

for e, op in list(OPERATORS.items()):
    if e.startswith('wvsplitk'): roles = ('gemv',)
    elif e.startswith('attn_mfma'): roles = ('attn',)
    elif e.startswith('topk_softmax'): roles = ('topk',)
    elif e.startswith('moe_sum'): roles = ('moe_sum',)
    else:
        roles = {
            'rms_norm_2d': ('in_norm',), 'rms_norm_3d': ('q_norm', 'k_norm'),
            'qknorm_rope': ('qknr',), 'rope_neox': ('rope',),
            'reshape_and_cache': ('cache',), 'attn_reduce': ('reduce',),
            'silu_and_mul': ('act', 'moe_act'), 'fused_add_rms_norm': ('fused_add',),
        }[e]
    OPERATORS[e] = replace(op, arguments={role: ARGUMENTS[role] for role in roles})

from etx.importer.launch_spec import registry, verify_capture, launch_for, validate_wvsplitk, capture_dimensions, capture_grid

for e, capture in registry().items():
    c, s = capture['constants'], capture['shape']
    up = capture['role'] == 'up'
    n, k = capture_dimensions(capture)
    first = 5 if up else 6
    expected = (n, k, s['TOPK']*c['BLOCK_SIZE_M'], s['TOPK'], k, n*k, k, n)
    names = ('N', 'K', 'EM/BLOCK_SIZE_M', 'num_valid_tokens/TOPK', 'stride_am', 'stride_be', 'stride_bn', 'stride_cm')
    checks = tuple((first+j, name, str(value)) for j, (name, value) in enumerate(zip(names, expected)))
    OPERATORS[e] = Operator(e, 'fused_moe_kernel',
        tuple(map(str, capture_grid(capture))),
        f"batch=1, {s}", c | {'MOE_BN': c['BLOCK_SIZE_N']},
        ('input', 'weights', 'expert_ids', 'num_tokens_post_padded') + (() if up else ('routing_weights',)),
        ('output',), checks=checks, block_rule=((str(capture['num_warps']*64),)*2, ('1', '1'), ('1', '1')), arguments={('moe_up' if up else 'moe_down'): ARGUMENTS['moe_up' if up else 'moe_down']})


def grid_for(export, dims, *, role, cu, parts):
    spec = OPERATORS[export]
    if export.startswith('fused_moe'):
        launch = launch_for(dims, 'up' if role == 'moe_up' else 'down')
        if launch.export != export:
            raise ValueError(f'{export}: shape conflicts with {launch.export}')
    if export.startswith('attn_') and dims.HD != spec.constants['HD']:
        raise ValueError(f'{export}: HD conflicts with compiled HD={spec.constants["HD"]}')
    if export.startswith('attn_mfma') and dims.G != spec.constants['G']:
        raise ValueError(f'{export}: G conflicts with compiled G={spec.constants["G"]}')
    if export == 'attn_reduce' and parts > spec.constants['MAX_PARTS']:
        raise ValueError(f'{export}: parts exceeds MAX_PARTS')
    return spec.grid(CU=cu, parts=parts, NKV=dims.NKV, NH=dims.NH,
                     heads=dims.NH if role == 'q_norm' else dims.NKV,
                     rows=dims.TOPK if role == 'moe_act' else 1)


def validate_values(export, values, grid):
    """Check arguments before Python packs the block."""
    spec = OPERATORS.get(export.removeprefix('etx_imp_'))
    if spec is None:
        return
    gx, gy, gz = grid
    env = dict(gx=gx, gy=gy, gz=gz) | {f'p{i}': v for i, v in enumerate(values)}
    for index, name, expr in spec.checks:
        expected = eval(expr, {'__builtins__': {}}, env)
        if values[index] != expected:
            raise ValueError(f'{spec.name}: {name} conflicts: expected {expected}, got {values[index]}')
    if spec.name.startswith('wvsplitk'):
        validate_wvsplitk(grid=grid, cu_count=values[11], rows=values[3],
                         ytile=spec.constants['YTILE'], wave_groups=values[10])
        if not 0 < values[1] <= 16384:
            raise ValueError(f'{spec.name}: K exceeds supported shape')
    if spec.name.startswith('fused_moe') and tuple(grid) != spec.grid():
        raise ValueError(f'{spec.name}: grid conflicts with BLOCK_SIZE_N/MOE_BN={spec.constants["MOE_BN"]}')


def emit_validation(imports):
    """Emit the same scalar checks for the host loader and its reference checker."""
    attention = OPERATORS['attn_mfma4_g4'].constants
    cache = OPERATORS['reshape_and_cache'].constants
    out = [f'#define ETX_SPEC_BS {attention["BS"]}', f'#define ETX_SPEC_PART {attention["PART"]}',
           f'#define ETX_SPEC_CACHE_X {cache["X"]}', '#include <cstring>', '#include <cstdint>',
           'static const char* etx_spec_validate(const etx_imp_desc* d, const unsigned char* data, int gx, int gy, int gz) {']
    for e, info in imports.items():
        spec = OPERATORS.get(e)
        if spec is None:
            continue
        out.append(f'  if (!strcmp(d->name, "{e}")) {{')
        for j, param in enumerate(info['params']):
            if param['ty'] in ('i32', 'i64'):
                ty = 'int32_t' if param['ty'] == 'i32' else 'int64_t'
                out.append(f'    {ty} p{j}; memcpy(&p{j}, data+d->off[{j}], sizeof(p{j}));')
        for j, name, expr in spec.checks:
            out.append(f'    if (p{j} != {expr}) return "{name} conflicts with operator specification";')
        if e.startswith('wvsplitk'):
            out += ['    if (gx <= 0 || gy != 1 || gz != 1) return "grid conflicts with one-dimensional wvSplitK";',
                    f'    if (p3 < {spec.constants["YTILE"]}) return "M conflicts with YTILE";',
                    '    if (p10 < 1 || p10 > 16) return "wave_groups conflicts with WVPRGRP";',
                    '    if (p1 <= 0 || p1 > 16384) return "K exceeds supported shape";']
        if e.startswith('fused_moe'):
            gx, gy, gz = spec.grid()
            out.append(f'    if (gx != {gx} || gy != {gy} || gz != {gz}) return "grid conflicts with BLOCK_SIZE_N/MOE_BN={spec.constants["MOE_BN"]}";')
        out.append('  }')
    return '\n'.join(out + ['  return nullptr;', '}'])


def emit_host_files(directory):
    directory = Path(directory)
    recipes = {role: args for op in OPERATORS.values() for role, args in op.arguments.items()}
    def expand(source):
        for role, args in recipes.items():
            chain = 'a' + ''.join(f'.{ty}({expr})' for ty, expr in args)
            source = source.replace('@'+role+'@', chain)
        return source
    (directory/'operator_dispatch.inc').write_text(expand(HOST_DISPATCH))
    (directory/'operator_helpers.inc').write_text(expand(HOST_HELPERS))


def emit_model_validation(imports):
    out = ['static const char* etx_spec_model(const char* name, const char* role, int H, int MI, int E, int TOPK, int NH, int NKV, int HD, int BS, int parts, int gx, int gy, int gz) {',
           '  const int CU = gx, heads = !strcmp(role, "q_norm") ? NH : NKV;',
           '  const int rows = !strcmp(role, "moe_act") ? TOPK : 1;']
    for e in imports:
        spec = OPERATORS.get(e)
        if spec is None:
            continue
        out.append(f'  if (!strcmp(name, "{e}")) {{')
        constraints = {}
        if e.startswith('fused_moe'):
            constraints = registry()[e]['shape']
        for key in ('HD', 'BS', 'E', 'TOPK'):
            if key in spec.constants:
                constraints[key] = spec.constants[key]
        for key, value in constraints.items():
            out.append(f'    if ({key} != {value}) return "{key} conflicts with compiled shape";')
        if 'G' in spec.constants:
            out.append(f'    if (NH != NKV*{spec.constants["G"]}) return "G conflicts with compiled shape";')
        if 'MAX_PARTS' in spec.constants:
            out.append(f'    if (parts < 1 || parts > {spec.constants["MAX_PARTS"]}) return "parts exceeds MAX_PARTS";')
        if e == 'rms_norm_2d':
            out.append('    if (H <= 0 || H % 8) return "H conflicts with VEC=8";')
        for axis, (lower, upper) in enumerate(spec.block_rule):
            block = imports[e]['block'][axis]
            out.append(f'    if ({block} < ({lower}) || {block} > ({upper})) return "block conflicts with operator specification";')
        condition = ' || '.join(f'{axis} != ({expr.replace("//", "/")})' for axis, expr in zip(('gx','gy','gz'), spec.grid_rule))
        parameter = 'BLOCK_SIZE_N/MOE_BN' if e.startswith('fused_moe') else 'grid'
        out.append(f'    if ({condition}) return "{parameter} conflicts with grid";')
        out.append('  }')
    return '\n'.join(out + ['  return nullptr;', '}'])


def validate_import(export, info):
    """Reject a stale ABI or a changed compiled specification."""
    from dataclasses import asdict
    spec = OPERATORS[export]
    recorded = info.get('operator_spec')
    if recorded is not None:
        import json
        current = json.loads(json.dumps(asdict(spec)))
        recorded = json.loads(json.dumps(recorded))
        # Block guidance is not part of the ABI or the compiled constants.
        # Check the actual descriptor against the current bounds below.
        current.pop('block_rule')
        recorded.pop('block_rule', None)
        if recorded != current:
            keys = [key for key in current if recorded.get(key) != current[key]]
            if 'constants' in keys:
                keys += [key for key, value in current['constants'].items()
                         if recorded.get('constants', {}).get(key) != value]
            raise ValueError(f'{export}: operator specification conflicts: {keys}')
    if 'spec' in info and info['spec'] != spec.template:
        raise ValueError(f'{export}: template conflicts with compiled constants {list(spec.constants)}')
    validate_block(spec, info['block'])
    types = {'ptr': 'ptr', 'i32': 'i32', 'i64': 'i64', 'f32': 'float', 'f64': 'double', 'b8': 'i1'}
    for role, args in spec.arguments.items():
        actual = [p['ty'].split()[0] for p in info['params']]
        expected = [types[ty] for ty, _ in args]
        if actual != expected:
            raise ValueError(f'{export}: argument ABI conflicts for {role}: expected {expected}, got {actual}')
    if export.startswith('fused_moe'):
        verify_capture(export)
        if info.get('capture_spec') != registry()[export]:
            raise ValueError(f'{export}: BLOCK_SIZE_N/MOE_BN or capture constants conflict')


def validate_block(spec, block, context=None):
    """Check the shared bounds. Defer model bounds until the plan exists."""
    if len(block) != 3 or any(type(value) is not int for value in block):
        raise ValueError(f'{spec.name}: block must contain three integers')
    for value, (lower, upper) in zip(block, spec.block_rule):
        for expression, is_lower in ((lower, True), (upper, False)):
            if context is None and not expression.isdigit():
                continue
            bound = eval(expression, {'__builtins__': {}}, context or {})
            if (value < bound if is_lower else value > bound):
                raise ValueError(f'{spec.name}: block conflicts with bounds {spec.block_rule}: got {block}')
    if any(value < 1 for value in block) or block[0]*block[1]*block[2] > 1024:
        raise ValueError(f'{spec.name}: block exceeds hardware limits')


def validate_model_imports(imports, dims, exports=None):
    """Reject faults in used exports. Return separate checks for unused exports."""
    used = set(imports) if exports is None else set(exports)
    missing = used - imports.keys()
    if missing:
        raise ValueError(f'missing imported exports: {sorted(missing)}')
    context = dict(NKV=dims.NKV)
    report = {'used': {}, 'unused': {}}
    for export, info in imports.items():
        status = 'ok'
        try:
            if export not in OPERATORS:
                status = 'outside operator specification'
            elif 'params' not in info:
                status = 'resource-only simulation record'
            else:
                validate_import(export, info)
                validate_block(OPERATORS[export], info['block'], context)
        except ValueError as error:
            if export in used:
                raise
            status = str(error)
        report['used' if export in used else 'unused'][export] = status
    return report
