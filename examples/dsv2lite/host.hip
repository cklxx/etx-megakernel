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
};
extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" void fleet_set_params(const void* host_fp, hipStream_t s);   // fleet_shim.hip: refresh the __constant__ copy
extern "C" size_t fleet_scratch_bytes();

int main(int argc, char** argv) {
  std::string weights_path = "build/weights.bin", cache_path = "build/fleet_cache.bin";
  std::string golden_path = "build/golden_tokens.txt", hidden_path = "build/golden_hidden.bin", json_path;
  int tokens = 32, context = 1024, first_token = -1, repeat = 1;
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
    else if (a == "--teacher-force") teacher = true;
    else die("unknown argument " + a);
  }
  hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s  arch %s  CUs %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
  
  // ---- model (fleet's loaders, verbatim)
  const Manifest m = load_manifest(weights_path + ".manifest");
  ModelDims d{}; Weights w{}; Activations act{}; KVCache kv{};
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
  { std::vector<float> cos_h, sin_h; yarn_tables(rp, max_pos, cos_h, sin_h); act.cos = dev_upload(cos_h); act.sin = dev_upload(sin_h); }
  kv.row = d.kv_lora + d.qk_rope;
  { std::ifstream f(cache_path, std::ios::binary | std::ios::ate); if (!f) die("cannot open " + cache_path);
    const size_t bytes = (size_t)f.tellg(); const size_t per_row = (size_t)d.layers * kv.row * sizeof(__hip_bfloat16);
    kv.max_pos = (int)(bytes / per_row); }
  if (max_pos > kv.max_pos) die("cache too short for context + tokens");
  { const size_t cache_bytes = (size_t)d.layers * kv.max_pos * kv.row * sizeof(__hip_bfloat16);
    __hip_bfloat16* cache_dev = nullptr; HIP_CHECK(hipMalloc(&cache_dev, cache_bytes)); upload_file(cache_path, cache_dev, cache_bytes); kv.data = cache_dev; }
  std::vector<int> golden = load_tokens(golden_path);
  if (golden.empty() && first_token < 0) die("no golden tokens and no --first-token");
  if (first_token < 0) first_token = golden[0];
  std::vector<float> golden_hidden;
  { std::ifstream f(hidden_path, std::ios::binary | std::ios::ate);
    if (f) { const size_t bytes = (size_t)f.tellg(); if (bytes == (size_t)d.layers * d.hidden * sizeof(float)) { golden_hidden.resize(bytes / 4); f.seekg(0); f.read((char*)golden_hidden.data(), bytes); } } }

  // ---- fleet runtime state as the bodies see it: dummy counters saturate every in-body wait
  RuntimeState rt{};
  { uint32_t* sat = dev_alloc<uint32_t>((size_t)kXCDs * kMaxEvents);
    std::vector<uint32_t> h((size_t)kXCDs * kMaxEvents, 0u);   // epoch-counted up-counters (qabs publication); never reset
    HIP_CHECK(hipMemcpy(sat, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    rt.xcd_counters = sat; rt.global_events = sat; rt.xcd_flags = sat;
    uint32_t* misc = dev_alloc<uint32_t>(16); rt.xcd_arrivals = misc; rt.grid_arrivals = misc + kXCDs; rt.abort = misc + kXCDs + 1;
    rt.teacher = teacher ? 1 : 0; rt.spin_limit = 1u << 22; rt.coherent_acts = 0; rt.smoke = 0; rt.direct_poll = 1;
    for (int i = 0; i < 4; ++i) rt.fenced_events[i] = -1; rt.epoch0 = 1; rt.n_tokens = 1; rt.trace = nullptr; rt.trace_token = -1; }
  FleetParams fp{}; fp.rt = rt; fp.d = d; fp.w = w; fp.kv = kv;
  std::memcpy(&fp.act, &act, sizeof(act));    // Activations has __restrict__ array members: no implicit copy-assignment
  FleetParams* d_fp = nullptr; HIP_CHECK(hipMalloc(&d_fp, sizeof(FleetParams)));

  // ---- ETX host: argument table = FleetParams pointer + per-grid layer ids
  etx_host hst(0);
  hst.kernel = (const void*)etx_megakernel_d0;
  hst.alloc_args();
  hst.args[etx_host::arg_index("fleet")] = d_fp;
  for (int a = 0; a < ETX_N_ARGS; ++a) {
    const char* n = etx_arg_names[a];
    if (!strncmp(n, "layer_", 6)) { int32_t L = std::atoi(n + 6); HIP_CHECK(hipMemcpy(hst.args[a], &L, 4, hipMemcpyHostToDevice)); }
  }
  hst.setup();
  std::printf("etx: %d tasks, %d event words, %d workers (%d domains x %d), lds %d B (fleet scratch %zu B)\n",
              ETX_N_TASKS, ETX_EVENT_WORDS, ETX_N_WORKERS, ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, (int)ETX_LDS_BYTES, fleet_scratch_bytes());
  if (fleet_scratch_bytes() > (size_t)ETX_LDS_BYTES) die("graph declares less LDS than fleet::Scratch needs");

  std::vector<float> ms(tokens); std::vector<int> produced(tokens, -1); std::vector<uint64_t> times((size_t)tokens * 2);
  int matched = 0, compared = 0, first_div = -1; double wall_total = 0;
  for (int run = 0; run < repeat; ++run) {
    const bool last = run == repeat - 1;
    std::vector<int32_t> tin(tokens + 1, 0); tin[0] = first_token;
    if (teacher) for (int i = 0; i < tokens; ++i) tin[i] = golden[i];
    HIP_CHECK(hipMemcpy(act.token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(act.token_time, 0, (size_t)tokens * 16));
    matched = compared = 0; first_div = -1; wall_total = 0;
    for (int t = 0; t < tokens; ++t) {
      fp.tok = t; fp.epoch = (uint32_t)(run * tokens + t + 1); fp.pos = context + t; fp.dump = (last && t == 0) ? 1 : 0;
      HIP_CHECK(hipMemcpy(d_fp, &fp, sizeof(fp), hipMemcpyHostToDevice));
      fleet_set_params(&fp, hst.stream);
      if (!hst.run(20.0)) { std::printf("token %d: ETX step failed\n", t); return 2; }
      wall_total += hst.last_ms;
    }
    HIP_CHECK(hipMemcpy(produced.data(), act.token_out, tokens * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(times.data(), act.token_time, times.size() * 8, hipMemcpyDeviceToHost));
    for (int i = 0; i < tokens; ++i) {
      ms[i] = (times[2 * i + 1] - times[2 * i]) * 0.01f / 1e3f;
      const bool have_ref = (size_t)(i + 1) < golden.size();
      std::string verdict;
      if (have_ref) { ++compared; if (produced[i] == golden[i + 1]) { ++matched; verdict = "ok"; } else { verdict = "MISMATCH (HF " + std::to_string(golden[i + 1]) + ")"; if (first_div < 0) first_div = i; } }
      if (last) std::printf("token %2d: %8.3f ms (device)  step %7.3f ms  out %6d  %s\n", i, ms[i], hst.last_ms, produced[i], verdict.c_str());
    }
    if (!last) std::printf("run %d (warm-up): %d/%d tokens match\n", run, matched, compared);
  }
  int layers_ok = -1;
  if (!golden_hidden.empty()) {
    std::vector<float> ours((size_t)d.layers * d.hidden);
    HIP_CHECK(hipMemcpy(ours.data(), act.layer_dump, ours.size() * 4, hipMemcpyDeviceToHost));
    layers_ok = 0; bool chain = true;
    for (int L = 0; L < d.layers; ++L) {
      const float* a = ours.data() + (size_t)L * d.hidden; const float* b = golden_hidden.data() + (size_t)L * d.hidden;
      double dot = 0, na = 0, nb = 0, max_abs = 0, max_ref = 0;
      for (int j = 0; j < d.hidden; ++j) { dot += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j]; max_abs = std::max(max_abs, (double)std::fabs(a[j] - b[j])); max_ref = std::max(max_ref, (double)std::fabs(b[j])); }
      const double rel = max_abs / std::max(max_ref, 1e-12), cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
      const bool ok = rel <= 2e-2 && cos >= 0.999; if (chain && ok) ++layers_ok; else chain = false;
      std::printf("  layer %2d %s: max rel %9.2e  cosine %.6f  [%s]\n", L, L < d.first_k_dense ? "dense" : "moe  ", rel, cos, ok ? "ok" : "FAIL");
    }
    std::printf("  consecutive layers within the gate: %d of %d\n", layers_ok, d.layers);
  }
  hst.report_phases(etx_type_names, ETX_N_TYPES);     // ETX_TRACE=1: per-type wait / body / span of the last token
  std::vector<float> sorted(ms); std::sort(sorted.begin(), sorted.end());
  double sum = 0; for (float v : ms) sum += v;
  std::printf("\nper-token latency (embed -> argmax on device): median %.3f ms  mean %.3f ms; step wall (launch to end) %.3f ms/token\n",
              sorted[sorted.size() / 2], sum / tokens, wall_total / tokens);
  std::printf("tokens: %d/%d match HF greedy%s\n", matched, compared, first_div >= 0 ? (", first divergence at step " + std::to_string(first_div)).c_str() : "");
  if (!json_path.empty()) {
    std::ofstream j(json_path);
    j << "{\"mode\": \"" << (teacher ? "teacher" : "free") << "\", \"tokens\": " << tokens << ", \"median_ms\": " << sorted[sorted.size() / 2]
      << ", \"wall_ms_per_token\": " << wall_total / tokens << ", \"matched\": " << matched << ", \"compared\": " << compared
      << ", \"layers_within_gate\": " << layers_ok << "}\n";
  }
  return (compared == tokens && matched == compared) ? 0 : 3;
}
