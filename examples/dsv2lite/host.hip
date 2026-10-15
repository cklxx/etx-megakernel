// Host for the ETX port of fleet-mi300x (milestone M2): fleet's model loading,
// KV cache and golden tokens, ETX's kernel and step protocol, fleet's checks
// (tokens vs HuggingFace greedy, per-layer gate on the first step, on-device
// embed->argmax timing). One ETX step per token.
//
// Build (from the etx repo, FLEET = path to fleet-mi300x):
//   python -m etx compile examples/dsv2lite/graph.py --arch gfx942 --out build/dsv2lite
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 -fgpu-rdc -DFLEET_NT_WEIGHTS=1 \
//     -I $FLEET/src -I etx/runtime/include -I build/dsv2lite \
//     build/dsv2lite/megakernel_d0.hip examples/dsv2lite/fleet_shim.hip examples/dsv2lite/host.hip -o build/dsv2lite/run
// Run from the fleet-mi300x directory (its build/ holds weights, cache, golden tokens):
//   /path/to/etx/build/dsv2lite/run --tokens 32 --context 1024
#define main fleet_launch_main_unused
#include "host/fleet_launch.hip"       // fleet's loaders (manifest, blob, YaRN tables, cache, tokens) and checks
#undef main
#include "plan_data.h"
#include "etx/host.h"

using namespace fleet;

struct FleetParams {                    // must match fleet_shim.hip
  RuntimeState rt; ModelDims d; Weights w; Activations act; KVCache kv;
  int32_t tok; uint32_t epoch; int32_t pos; int32_t dump;
  uint64_t* pro_trace;                  // ETX_DS_PROLOGUE=<file>: per-task prologue stamps of the last step
};
extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_unfused_d0(etx_params, int32_t);
extern "C" void fleet_set_params(const void* host_fp, hipStream_t s, int r);   // fleet_shim.hip: refresh request r's __constant__ copy
#ifndef ETX_DS_REQUESTS            // requests decoded together (the graph built with ETX_DSV2_REQUESTS)
#define ETX_DS_REQUESTS 1
#endif
extern "C" size_t fleet_scratch_bytes();
#ifdef ETX_DSV2_SNAPSHOT
#include "examples/dsv2lite/snapshot.h"
#endif

int main(int argc, char** argv) {
  std::string weights_path = "build/weights.bin", cache_path = "build/fleet_cache.bin";
  std::string golden_path = "build/golden_tokens.txt", hidden_path = "build/golden_hidden.bin", json_path;
  std::string r1_dir = "build_r1";     // request 1's cache, golden tokens and hidden states (reference_rotated.py)
  std::vector<std::string> req_dirs;   // --req-dir, once per request 1..R-1 in order (docs/BATCH-2026-10-14.md section 4)
  int tokens = 32, context = 1024, first_token = -1, repeat = 1; bool unfused = false;
  bool teacher = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value after " + a); return argv[++i]; };
    if (a == "--weights") weights_path = next();
    else if (a == "--cache") cache_path = next();
    else if (a == "--golden") golden_path = next();
    else if (a == "--hidden") hidden_path = next();
    else if (a == "--json") json_path = next();
    else if (a == "--tokens") tokens = std::atoi(next().c_str());
    else if (a == "--context") context = std::atoi(next().c_str());
    else if (a == "--first-token") first_token = std::atoi(next().c_str());
    else if (a == "--repeat") repeat = std::atoi(next().c_str());
    else if (a == "--unfused") unfused = true;     // baseline: fleet's tiles, one launch per grid (HIP graph)
    else if (a == "--teacher-force") teacher = true;
    else if (a == "--r1-dir") r1_dir = next();
    else if (a == "--req-dir") req_dirs.push_back(next());
    else die("unknown argument " + a);
  }
  hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s  arch %s  CUs %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
  
  // ---- model (fleet's loaders, verbatim)
  const Manifest m = load_manifest(weights_path + ".manifest");
  ModelDims d{}; Weights w{};
  d.hidden = (int)m.scalar("hidden"); d.heads = (int)m.scalar("heads"); d.qk_nope = (int)m.scalar("qk_nope");
  d.qk_rope = (int)m.scalar("qk_rope"); d.q_head_dim = d.qk_nope + d.qk_rope; d.v_head = (int)m.scalar("v_head");
  d.kv_lora = (int)m.scalar("kv_lora"); d.moe_inter = (int)m.scalar("moe_inter"); d.dense_inter = (int)m.scalar("dense_inter");
  d.n_routed = (int)m.scalar("n_routed"); d.top_k = (int)m.scalar("top_k"); d.vocab = (int)m.scalar("vocab");
  d.layers = (int)m.scalar("layers"); d.first_k_dense = (int)m.scalar("first_k_dense");
  d.rms_eps = (float)m.scalar("rms_eps"); d.routed_scaling = (float)m.scalar("routed_scaling"); d.context = context;
  RopeParams rp{m.scalar("rope_theta"), m.scalar("rope_factor"), m.scalar("beta_fast"), m.scalar("beta_slow"),
                m.scalar("mscale"), m.scalar("mscale_all_dim"), d.qk_rope, (int)m.scalar("rope_original_max")};
  { double s = std::pow((double)d.q_head_dim, -0.5);
    if (rp.mscale_all_dim != 0.0) { const double ms = yarn_get_mscale(rp.factor, rp.mscale_all_dim); s = s * ms * ms; }
    d.softmax_scale = (float)s; }
  if (d.hidden > kMaxHidden || d.kv_lora != kMaxKvLora || d.qk_rope != kMaxQkRope || d.moe_inter > kMaxMoeInter ||
      d.n_routed > kMaxRouted || d.top_k > kMaxTopK || d.heads % kNumXCDs != 0) die("model dims exceed the kernel's compile-time bounds");
  std::printf("model: hidden %d heads %d kv_lora %d experts %d top-%d layers %d (dense < %d) vocab %d\n",
              d.hidden, d.heads, d.kv_lora, d.n_routed, d.top_k, d.layers, d.first_k_dense, d.vocab);
  const size_t blob_bytes = (size_t)m.scalar("bytes");
  __hip_bfloat16* blob = nullptr; HIP_CHECK(hipMalloc(&blob, blob_bytes)); upload_file(weights_path, blob, blob_bytes);
  w.layer_qkv = layer_table(blob, m, d.layers, "qkv"); w.layer_kv_b = layer_table(blob, m, d.layers, "kv_b");
  w.layer_o_proj = layer_table(blob, m, d.layers, "o_proj"); w.layer_norm_in = layer_table(blob, m, d.layers, "norm_in");
  w.layer_norm_post = layer_table(blob, m, d.layers, "norm_post"); w.layer_kv_a_norm = layer_table(blob, m, d.layers, "kv_a_norm");
  w.layer_router = layer_table(blob, m, d.layers, "router"); w.layer_experts = layer_table(blob, m, d.layers, "experts");
  w.layer_shared = layer_table(blob, m, d.layers, "shared"); w.layer_dense = layer_table(blob, m, d.layers, "dense");
  w.embed = tensor_ptr(blob, m, "embed"); w.lm_head = tensor_ptr(blob, m, "lm_head"); w.final_norm = tensor_ptr(blob, m, "final_norm");
  w.expert_stride = (int64_t)m.scalar("expert_stride");
  const int max_pos = context + tokens;
  const int R = ETX_DS_REQUESTS;
  // ---- per request: activations, KV cache, reference, fleet runtime state (request 0 uses --cache/--golden/--hidden,
  // request 1 the same files in --r1-dir: another prompt, reference_rotated.py)
  struct Req {
    Activations act{}; KVCache kv{}; RuntimeState rt{}; FleetParams fp{}; FleetParams* d_fp = nullptr;
    std::vector<int> golden; std::vector<float> golden_hidden; int first_token = -1;
    std::vector<int> produced; std::vector<uint64_t> times; std::vector<float> ms;
    int matched = 0, compared = 0, first_div = -1, layers_ok = -1, warm_matched = -1, warm_compared = -1;
  };
  std::vector<Req> reqs(R);
  // request r >= 1 reads its own prompt's reference: --req-dir (one per request, in order), or --r1-dir for two requests
  if (!req_dirs.empty() && (int)req_dirs.size() != R - 1)
    die("--req-dir given " + std::to_string(req_dirs.size()) + " times for a " + std::to_string(R) + "-request build (need " +
        std::to_string(R - 1) + ")");
  if (req_dirs.empty() && R > 2) die("a " + std::to_string(R) + "-request build needs --req-dir for requests 1.." + std::to_string(R - 1));
  auto req_dir = [&](int r) { return req_dirs.empty() ? r1_dir : req_dirs[r - 1]; };
  std::vector<float> cos_h, sin_h; yarn_tables(rp, max_pos, cos_h, sin_h);
  for (int r = 0; r < R; ++r) {
    Req& q = reqs[r]; Activations& act = q.act; KVCache& kv = q.kv;
    const std::string cp = r == 0 ? cache_path : req_dir(r) + "/fleet_cache.bin";
    const std::string gp = r == 0 ? golden_path : req_dir(r) + "/golden_tokens.txt";
    const std::string hp = r == 0 ? hidden_path : req_dir(r) + "/golden_hidden.bin";
    act.token_in = dev_alloc<int32_t>(tokens + 1); act.token_out = dev_alloc<int32_t>(tokens); act.token_time = dev_alloc<uint64_t>((size_t)tokens * 2);
    act.x[0] = dev_alloc<float>(d.hidden); act.x[1] = dev_alloc<float>(d.hidden);
    act.x_norm = dev_alloc<float>((size_t)kNumXCDs * d.hidden); act.q = dev_alloc<float>((size_t)d.heads * d.q_head_dim);
    act.q_absorbed = dev_alloc<float>((size_t)d.heads * kQAbsSlices * d.kv_lora); act.kv_a = dev_alloc<float>((size_t)kNumXCDs * (d.kv_lora + d.qk_rope));
    act.attn_partial = dev_alloc<float>((size_t)d.heads * kMaxKvChunks * kPartialStride); act.o = dev_alloc<float>((size_t)d.heads * d.v_head);
    act.expert_h = dev_alloc<float>(std::max((size_t)kNumXCDs * d.moe_inter, (size_t)d.dense_inter));
    act.expert_out = dev_alloc<__hip_bfloat16>((size_t)kNumXCDs * d.hidden); act.oproj_partial = dev_alloc<float>((size_t)kNumXCDs * d.hidden);
    act.logits = dev_alloc<float>(d.vocab); act.argmax_val = dev_alloc<float>((size_t)kNumXCDs * kWorkersPerXCD);
    act.argmax_idx = dev_alloc<int32_t>((size_t)kNumXCDs * kWorkersPerXCD); act.router_logits = dev_alloc<float>((size_t)kNumXCDs * d.n_routed);
    act.topk_ids = dev_alloc<int32_t>((size_t)kNumXCDs * kMaxTopK); act.topk_w = dev_alloc<float>((size_t)kNumXCDs * kMaxTopK);
    act.layer_dump = dev_alloc<float>((size_t)d.layers * d.hidden);
    act.cos = dev_upload(cos_h); act.sin = dev_upload(sin_h);
    kv.row = d.kv_lora + d.qk_rope;
    { std::ifstream f(cp, std::ios::binary | std::ios::ate); if (!f) die("cannot open " + cp);
      const size_t bytes = (size_t)f.tellg(); const size_t per_row = (size_t)d.layers * kv.row * sizeof(__hip_bfloat16);
      kv.max_pos = (int)(bytes / per_row); }
    if (max_pos > kv.max_pos) die("cache too short for context + tokens: " + cp);
    { const size_t cache_bytes = (size_t)d.layers * kv.max_pos * kv.row * sizeof(__hip_bfloat16);
      __hip_bfloat16* cache_dev = nullptr; HIP_CHECK(hipMalloc(&cache_dev, cache_bytes)); upload_file(cp, cache_dev, cache_bytes); kv.data = cache_dev; }
    q.golden = load_tokens(gp);
    for (int o = 0; o < r; ++o)   // prompt distinctness is checked by examples/dsv2lite/batch_goldens.py before the run;
                                  // two rotations may still share their short greedy output, which only weakens the
                                  // mix-up check for that pair (docs/BATCH-2026-10-14.md 4a), so print and continue
      if (q.golden.size() > (size_t)tokens && reqs[o].golden.size() > (size_t)tokens && std::equal(q.golden.begin() + 1, q.golden.begin() + tokens + 1, reqs[o].golden.begin() + 1))
        std::printf("BATCHCOLLISION requests %d and %d: identical %d checked greedy tokens [1:tokens+1]; prompts are distinct, "
                    "the request mix-up check is unavailable for this pair\n",
                    o, r, tokens);
    q.first_token = first_token;
    if (q.golden.empty() && q.first_token < 0) die("no golden tokens and no --first-token for request " + std::to_string(r));
    if (q.first_token < 0) q.first_token = q.golden[0];
    { std::ifstream f(hp, std::ios::binary | std::ios::ate);
      if (f) { const size_t bytes = (size_t)f.tellg(); if (bytes == (size_t)d.layers * d.hidden * sizeof(float)) { q.golden_hidden.resize(bytes / 4); f.seekg(0); f.read((char*)q.golden_hidden.data(), bytes); } } }
    // fleet runtime state as the bodies see it: dummy counters saturate every in-body wait (one set per request)
    RuntimeState& rt = q.rt;
    { uint32_t* sat = dev_alloc<uint32_t>((size_t)kXCDs * kMaxEvents);
      std::vector<uint32_t> h((size_t)kXCDs * kMaxEvents, 0u);   // epoch-counted up-counters (qabs publication); never reset
      HIP_CHECK(hipMemcpy(sat, h.data(), h.size() * 4, hipMemcpyHostToDevice));
      rt.xcd_counters = sat; rt.global_events = sat; rt.xcd_flags = sat;
      uint32_t* misc = dev_alloc<uint32_t>(16); rt.xcd_arrivals = misc; rt.grid_arrivals = misc + kXCDs; rt.abort = misc + kXCDs + 1;
      rt.teacher = teacher ? 1 : 0; rt.spin_limit = 1u << 22; rt.coherent_acts = 0; rt.smoke = 0; rt.direct_poll = 1;
      for (int i = 0; i < 4; ++i) rt.fenced_events[i] = -1; rt.epoch0 = 1; rt.n_tokens = 1; rt.trace = nullptr; rt.trace_token = -1; }
    q.fp.rt = rt; q.fp.d = d; q.fp.w = w; q.fp.kv = kv;
    std::memcpy(&q.fp.act, &act, sizeof(act));    // Activations has __restrict__ array members: no implicit copy-assignment
    HIP_CHECK(hipMalloc(&q.d_fp, sizeof(FleetParams)));
    q.produced.assign(tokens, -1); q.times.assign((size_t)tokens * 2, 0); q.ms.assign(tokens, 0.f);
  }
  const char* pro_path = getenv("ETX_DS_PROLOGUE");   // prologue stamps: request 0 only (the slots carry no request)
#if defined(ETX_DSV2_STAMPS2) && ETX_DSV2_STAMPS2
  // Fixed little-endian u64 header {magic, version, words/slot, slots};
  // the first three words of each row retain the legacy meanings.
  const size_t pro_stride = 16;
  if (pro_path) for (int ty = 0; ty < ETX_N_TYPES; ++ty)
    if (!strncmp(etx_type_names[ty], "iso", 3)) die("STAMPS2 cannot overwrite native slots with isolate tasks");
#else
  const size_t pro_stride = 3;
#endif
  const size_t pro_words = (size_t)28 * 32 * 1024 * pro_stride;
  if (pro_path) { HIP_CHECK(hipMalloc(&reqs[0].fp.pro_trace, pro_words * 8)); HIP_CHECK(hipMemset(reqs[0].fp.pro_trace, 0, pro_words * 8)); }

  // ---- ETX host: argument table = FleetParams pointer + per-grid layer ids
  etx_host hst(0);
  hst.kernel = (const void*)etx_megakernel_d0;
  hst.unfused_kernel = (const void*)etx_unfused_d0;
  hst.alloc_args();
  for (int r = 0; r < R; ++r) hst.args[etx_host::arg_index(R == 1 ? "fleet" : ("fleet" + std::to_string(r)).c_str())] = reqs[r].d_fp;
  for (int a = 0; a < ETX_N_ARGS; ++a) {
    const char* n = etx_arg_names[a];
    if (!strncmp(n, "layer_", 6)) { int32_t L = std::atoi(n + 6); HIP_CHECK(hipMemcpy(hst.args[a], &L, 4, hipMemcpyHostToDevice)); }
  }
  hst.setup();
  std::printf("etx: %d tasks, %d event words, %d workers (%d domains x %d), lds %d B (fleet scratch %zu B)\n",
              ETX_N_TASKS, ETX_EVENT_WORDS, ETX_N_WORKERS, ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, (int)ETX_LDS_BYTES, fleet_scratch_bytes());
  if (fleet_scratch_bytes() > (size_t)ETX_LDS_BYTES) die("graph declares less LDS than fleet::Scratch needs");

#ifdef ETX_DSV2_SNAPSHOT
  return ds_snapshot_run(hst, reqs[0].fp, reqs[0].d_fp, reqs[0].first_token, tokens, repeat, unfused);
#endif

  double wall_total = 0; std::vector<double> step_ms;
  for (int run = 0; run < repeat; ++run) {
    const bool last = run == repeat - 1;
    for (int r = 0; r < R; ++r) {
      Req& q = reqs[r];
      std::vector<int32_t> tin(tokens + 1, 0); tin[0] = q.first_token;
      if (teacher) for (int i = 0; i < tokens && i < (int)q.golden.size(); ++i) tin[i] = q.golden[i];
      HIP_CHECK(hipMemcpy(q.act.token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(q.act.token_time, 0, (size_t)tokens * 16));
    }
    wall_total = 0; step_ms.clear();
    for (int t = 0; t < tokens; ++t) {
      for (int r = 0; r < R; ++r) {
        FleetParams& fp = reqs[r].fp;
        fp.tok = t; fp.epoch = (uint32_t)(run * tokens + t + 1); fp.pos = context + t; fp.dump = (last && t == 0) ? 1 : 0;
        HIP_CHECK(hipMemcpy(reqs[r].d_fp, &fp, sizeof(fp), hipMemcpyHostToDevice));
        fleet_set_params(&fp, hst.stream, r);
      }
      if (unfused) { if (!hst.run_unfused()) { std::printf("token %d: unfused step failed\n", t); return 2; } }
      else if (!hst.run(20.0)) { std::printf("token %d: ETX step failed\n", t); return 2; }
      wall_total += hst.last_ms; step_ms.push_back(hst.last_ms);
    }
    for (int r = 0; r < R; ++r) {
      Req& q = reqs[r];
      HIP_CHECK(hipMemcpy(q.produced.data(), q.act.token_out, tokens * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(q.times.data(), q.act.token_time, q.times.size() * 8, hipMemcpyDeviceToHost));
      q.matched = q.compared = 0; q.first_div = -1;
      for (int i = 0; i < tokens; ++i) {
        q.ms[i] = (q.times[2 * i + 1] - q.times[2 * i]) * 0.01f / 1e3f;
        const bool have_ref = (size_t)(i + 1) < q.golden.size();
        std::string verdict;
        if (have_ref) { ++q.compared; if (q.produced[i] == q.golden[i + 1]) { ++q.matched; verdict = "ok"; } else { verdict = "MISMATCH (HF " + std::to_string(q.golden[i + 1]) + ")"; if (q.first_div < 0) q.first_div = i; } }
        if (last) std::printf("%stoken %2d: %8.3f ms (device)  step %7.3f ms  out %6d  %s\n", R > 1 ? ("r" + std::to_string(r) + " ").c_str() : "",
                              i, q.ms[i], step_ms[i], q.produced[i], verdict.c_str());
      }
      if (!last) std::printf("run %d (warm-up)%s: %d/%d tokens match\n", run, R > 1 ? (" r" + std::to_string(r)).c_str() : "", q.matched, q.compared);
      if (!last) { q.warm_matched = q.matched; q.warm_compared = q.compared; }
    }
  }
  for (int r = 0; r < R; ++r) {
    Req& q = reqs[r];
    if (q.golden_hidden.empty()) continue;
    // Opt-in visibility only: retain the consecutive-prefix gate and every legacy line when disabled.
    const char* layer_metrics_env = getenv("ETX_DS_LAYER_METRICS");
    const bool layer_metrics = layer_metrics_env && !strcmp(layer_metrics_env, "1");
    std::vector<float> ours((size_t)d.layers * d.hidden);
    HIP_CHECK(hipMemcpy(ours.data(), q.act.layer_dump, ours.size() * 4, hipMemcpyDeviceToHost));
    q.layers_ok = 0; bool chain = true;
    for (int L = 0; L < d.layers; ++L) {
      const float* a = ours.data() + (size_t)L * d.hidden; const float* b = q.golden_hidden.data() + (size_t)L * d.hidden;
      double dot = 0, na = 0, nb = 0, max_abs = 0, max_ref = 0;
      for (int j = 0; j < d.hidden; ++j) { dot += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j]; max_abs = std::max(max_abs, (double)std::fabs(a[j] - b[j])); max_ref = std::max(max_ref, (double)std::fabs(b[j])); }
      const double rel = max_abs / std::max(max_ref, 1e-12), cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
      const bool ok = rel <= 2e-2 && cos >= 0.999; if (chain && ok) ++q.layers_ok; else chain = false;
      if (R == 1) std::printf("  layer %2d %s: max rel %9.2e  cosine %.6f  [%s]\n", L, L < d.first_k_dense ? "dense" : "moe  ", rel, cos, ok ? "ok" : "FAIL");
      if (layer_metrics) std::printf("LAYERGATE r=%d layer=%d rel=%.9e cosine=%.9f pass=%d\n", r, L, rel, cos, ok ? 1 : 0);
    }
    std::printf("  %sconsecutive layers within the gate: %d of %d\n", R > 1 ? ("r" + std::to_string(r) + " ").c_str() : "", q.layers_ok, d.layers);
  }
  hst.report_phases(etx_type_names, ETX_N_TYPES);     // ETX_TRACE=1: per-type wait / body / span of the last token
  if (pro_path) {                                      // raw u64 x 3 per slot, slots as PRO_SLOT; zero = not run
    std::vector<uint64_t> pt(pro_words); HIP_CHECK(hipMemcpy(pt.data(), reqs[0].fp.pro_trace, pro_words * 8, hipMemcpyDeviceToHost));
    std::ofstream f(pro_path, std::ios::binary);
#if defined(ETX_DSV2_STAMPS2) && ETX_DSV2_STAMPS2
    const uint64_t header[4] = {0x32504d4154535445ull, 2, 16, (uint64_t)28 * 32 * 1024}; // "ETSTAMP2"
    f.write((const char*)header, sizeof(header));
#endif
    f.write((const char*)pt.data(), pt.size() * 8);
    std::printf("wrote prologue stamps to %s\n", pro_path);
  }
  std::vector<double> ss(step_ms); std::sort(ss.begin(), ss.end());
  const double step_med = ss[ss.size() / 2];
  bool all_ok = true; int matched = 0, compared = 0;
  for (int r = 0; r < R; ++r) {
    Req& q = reqs[r];
    std::vector<float> sorted(q.ms); std::sort(sorted.begin(), sorted.end());
    double sum = 0; for (float v : q.ms) sum += v;
    std::printf("\n%sper-token latency (embed -> argmax on device): median %.3f ms  mean %.3f ms; step wall (launch to end) %.3f ms/token\n",
                R > 1 ? ("r" + std::to_string(r) + " ").c_str() : "", sorted[sorted.size() / 2], sum / tokens, wall_total / tokens);
    std::printf("%stokens: %d/%d match HF greedy%s\n", R > 1 ? ("r" + std::to_string(r) + " ").c_str() : "", q.matched, q.compared,
                q.first_div >= 0 ? (", first divergence at step " + std::to_string(q.first_div)).c_str() : "");
    all_ok = all_ok && q.compared == tokens && q.matched == q.compared;
    matched += q.matched; compared += q.compared;
  }
  if (R > 1)
    std::printf("\nRESULT requests=%d step median %.3f ms -> %.1f tokens/s (%d tokens per step); tokens %d/%d match\n",
                R, step_med, R * 1000.0 / step_med, R, matched, compared);
  // docs/BATCH-2026-10-14.md section 1: throughput from the host's step time, per-request latency from each request's own
  // device timestamps (token_time: s_memrealtime, 10 ns ticks), the median time from one token's end to the next's.
  // Exact decimals (ms with 6 places = integer ns) for etx/session/batch.py.
  for (int r = 0; r < R; ++r) {
    Req& q = reqs[r];
    std::vector<uint64_t> iv;
    for (int i = 1; i < tokens; ++i) iv.push_back(q.times[2 * i + 1] - q.times[2 * i - 1]);
    std::sort(iv.begin(), iv.end());
    std::vector<float> dev(q.ms); std::sort(dev.begin(), dev.end());
    const uint64_t ticks = iv.empty() ? 0 : iv[iv.size() / 2];
    std::printf("BATCHREQ r=%d tokens=%d/%d layers=%d warmup=%d/%d interval_median_ms=%llu.%05llu device_median_ms=%.6f\n",
                r, q.matched, q.compared, q.layers_ok, q.warm_matched, q.warm_compared,
                (unsigned long long)(ticks / 100000), (unsigned long long)(ticks % 100000), (double)dev[dev.size() / 2]);
  }
  std::printf("BATCH requests=%d tokens=%d step_median_ms=%.6f\n", R, tokens, step_med);
  if (!json_path.empty()) {
    std::ofstream j(json_path);
    j << "{\"mode\": \"" << (teacher ? "teacher" : "free") << "\", \"requests\": " << R << ", \"tokens\": " << tokens
      << ", \"step_median_ms\": " << step_med << ", \"tokens_per_s\": " << R * 1000.0 / step_med
      << ", \"wall_ms_per_step\": " << wall_total / tokens << ", \"matched\": " << matched << ", \"compared\": " << compared
      << ", \"layers_within_gate\": [";
    for (int r = 0; r < R; ++r) j << (r ? ", " : "") << reqs[r].layers_ok;
    j << "]}\n";
  }
  return all_ok ? 0 : 3;
}
