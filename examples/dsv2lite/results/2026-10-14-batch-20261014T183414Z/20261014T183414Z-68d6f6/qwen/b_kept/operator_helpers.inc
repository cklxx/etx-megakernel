

    auto gemv = [&](const void* w, const void* in, void* out, const void* b) {   // ops.wvSplitK(weight, x, cu_count, bias)
      // CuCount must be the GRID SIZE actually launched, not the device's CU count: the kernel's row loop strides by
      // CuCount * _WvPrGrp * YTILE, so a larger value than the grid leaves rows [grid*wv*YT, CuCount*wv*YT) of every
      // pass unwritten. Only lm_head (M > grid*wv*YT) loops; its holes hit any token id in them (round 42: Mixtral).
      a.i32(x.K).i32(x.K).i32(x.K).i32(xM).i32(b ? xM : 1).i32(1).ptr(w).ptr(in).ptr(b).ptr(out).i32(x.wv).i32(x.gx * x.gy * x.gz);
    };
    auto fused_add = [&](bf* in, const void* w) { a.ptr(in).i64(H).ptr(res_r).ptr(w).f32(eps).i32(1).i32(H).i64(H); };
