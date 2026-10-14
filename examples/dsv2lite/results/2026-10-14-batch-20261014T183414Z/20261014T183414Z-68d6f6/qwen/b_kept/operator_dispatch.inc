

    else if (r == "in_norm" && x.layer == 0) a.ptr(hn_r).ptr(res_r).i64(H).i64(0).i64(0).i64(0).i64(0).ptr(W(pl + "ln1")).i64(0).f32(eps).i32(1).i32(H);
    else if (r == "in_norm") { copy(dn, d_norm, H * 2); fused_add(d_norm, W(pl + "ln1")); }
    else if (r == "qkv") gemv(W(pl + "wqkv"), qkv_in, qkv, bias ? W(pl + "bqkv") : nullptr);
    else if (r == "q_norm") a.ptr(qn).ptr(qkv).i64(HD).i64(NQKV).i64(0).i64(NH).i64(0).ptr(W(pl + "qn")).i64(0).f32(eps).i32(NH).i32(HD);
    else if (r == "k_norm") a.ptr(kn).ptr(qkv + NH * HD).i64(HD).i64(NQKV).i64(0).i64(NKV).i64(0).ptr(W(pl + "kn")).i64(0).f32(eps).i32(NKV).i32(HD);
    // fused_qk_norm_rope(qkv, nq, nk, nv, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, positions): kernel
    // arguments (qkv, nq, nk, nv, eps, q_w, k_w, cos_sin, positions, num_tokens, rotary_dim = cos_sin.size(1) = HD)
    else if (r == "qknr") a.ptr(qkv).i32(NH).i32(NKV).i32(NKV).f32(eps).ptr(W(pl + "qn")).ptr(W(pl + "kn")).ptr(W("cos_sin")).ptr(positions).i32(1).i32(HD);
    else if (r == "rope") a.ptr(positions).ptr(qbuf).ptr(kbuf).ptr(W("cos_sin")).i32(HD).i64(qstr).i64(kstr).i64(HD).i32(NH).i32(NKV).i32(HD).i64(0).b8(false);
    else if (r == "cache") a.ptr(kbuf).ptr(qkv + (NH + NKV) * HD).ptr(kc[x.layer]).ptr(vc[x.layer]).ptr(slot_mapping).i32((int)kstr).i32(NQKV).i32(NKV).i32(HD).i32(BS).i32(X).ptr(kv_scale).ptr(kv_scale);
    else if (r == "attn") a.ptr(qbuf).ptr(kc[x.layer]).ptr(vc[x.layer]).i32(NKV).f32(scale).ptr(block_tables).ptr(seq_lens).ptr(nullptr).i32(MAXBLK).ptr(nullptr).i32((int)qstr).i32(NKV * HD * BS).i32(HD * BS).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(attn_out).i32(MAXBLK).ptr(kv_scale).ptr(kv_scale);
    else if (r == "reduce") a.ptr(attn_out).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(seq_lens).ptr(nullptr).i32(parts).ptr(nullptr);
    else if (r == "o_proj") gemv(W(pl + "wo"), attn_out, o, nullptr);
    else if (r == "post_norm") { copy(o, o_norm, H * 2); fused_add(o_norm, W(pl + "ln2")); }
    else if (r == "gate_up") gemv(W(pl + "wgu"), o_norm, gu, nullptr);
    else if (r == "moe_act") a.ptr(mc2).ptr(mc1).i32(MI).f32(0.0f).f32(1.0f).f32(0.0f);    // one row per expert
    else if (r == "act") a.ptr(act_out).ptr(gu).i32(I).f32(0.0f).f32(1.0f).f32(0.0f);
    else if (r == "router") gemv(W(pl + "wr"), o_norm, rlog, nullptr);
    // topk_softmax -> topkGatingLauncherHelper(gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,
    // num_tokens, topk, 0, num_experts, renormalize, bias, 1.0, stream, is_padding)
    else if (r == "topk") a.ptr(rlog).ptr(nullptr).ptr(tw_r).i32(1).ptr(ti_r).ptr(tei + (size_t)rr * TOPK).i32(TOPK).i32(0).i32(E).b8(norm_topk).ptr(nullptr).f64(1.0).ptr(nullptr);
    // invoke_fused_moe_triton_kernel, naive block assignment: EM = M * top_k * BLOCK_SIZE_M, num_valid_tokens = M * top_k;
    // Triton dropped the None pointers and the strides equal to 1 (stride_ak, stride_bk, stride_cn) from the signature
    // and appended its global / profile scratch pointers (both sizes 0)
    else if (r == "moe_up") a.ptr(o_norm).ptr(W(pl + "w13")).ptr(mc1).ptr(ti_r).ptr(ntpp).i32(2 * MI).i32(H).i32(TOPK * MOE_BM).i32(TOPK).i32(H).i32(2 * MI * H).i32(H).i32(2 * MI).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).ptr(nullptr).ptr(nullptr);
    else if (r == "moe_down") a.ptr(mc2).ptr(W(pl + "w2")).ptr(mc3).ptr(tw_r).ptr(ti_r).ptr(ntpp).i32(H).i32(MI).i32(TOPK * MOE_BM).i32(TOPK).i32(MI).i32(H * MI).i32(MI).i32(H).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).ptr(nullptr).ptr(nullptr);
    // moe_sum, vectorised: (out, input, num_tokens, hidden, stride_token, stride_topk, nullptr, nullptr, 0, 0); the output
    // is the buffer the next fused_add_rms_norm reads (vLLM copies it there)
    else if (r == "moe_sum") a.ptr(d_norm).ptr(mc3).i64(1).i32(H).i64((long long)TOPK * H).i64(H).ptr(nullptr).ptr(nullptr).i64(0).i64(0);
    else if (r == "down") gemv(W(pl + "wd"), act_out, dn, nullptr);
    else if (r == "final_norm") { copy(dn, d_norm, H * 2); fused_add(d_norm, W("final_norm")); }
    else if (r == "lm_head") gemv(tie ? W("embed") : W("lm_head"), d_norm, logits, nullptr);
    else die("unknown role " + r);
