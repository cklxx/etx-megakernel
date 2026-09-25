// Host for examples/llm: loads a model packed by examples/llm/prep.py, runs one ETX step per token
// (the prompt is fed token by token, then greedy generation), and compares with Hugging Face greedy.
//
//   run <packed-dir> [--gen N] [--repeat R] [--teacher]
// <packed-dir> holds weights.bin, llm.manifest (cfg / tensor lines) and golden.txt (prompt ids,
// HF greedy ids, HF top-1 margins). --teacher feeds HF's tokens and checks every argmax instead.
#include "plan_data.h"
#include "etx/host.h"
#include "examples/llm/llm_params.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_unfused_d0(etx_params, int32_t);
extern "C" void llm_set_params(const void* host_params, hipStream_t s);
extern "C" size_t llm_params_bytes();

static void die(const std::string& m) { fprintf(stderr, "error: %s\n", m.c_str()); exit(1); }

struct Manifest {
  std::map<std::string, double> cfg;
  std::map<std::string, std::pair<size_t, size_t>> t;     // name -> (byte offset, bf16 count)
  int geti(const char* k) const { auto it = cfg.find(k); if (it == cfg.end()) die(std::string("manifest lacks cfg ") + k); return (int)it->second; }
};

static Manifest read_manifest(const std::string& path) {
  Manifest m; std::ifstream f(path); if (!f) die("cannot open " + path);
  std::string kind;
  while (f >> kind) {
    if (kind == "cfg") { std::string k; double v; f >> k >> v; m.cfg[k] = v; }
    else if (kind == "t") { std::string n; size_t off, cnt; f >> n >> off >> cnt; m.t[n] = {off, cnt}; }
    else { std::string rest; std::getline(f, rest); }
  }
  return m;
}

static std::vector<int> read_ints(std::istream& in) {
  std::string line; std::getline(in, line); std::istringstream s(line); std::string head; s >> head;
  std::vector<int> v; int x; while (s >> x) v.push_back(x); return v;
}

// same rule as examples/llm/model_sliced.py::split: n/unit units split evenly, the remainder to the first parts
static void split8(int n, int unit, int32_t* start, int32_t* count) {
  const int u = n / unit, base = u / 8, rem = u % 8; int s = 0;
  for (int k = 0; k < 8; ++k) { const int c = base + (k < rem ? 1 : 0); start[k] = s * unit; count[k] = c * unit; s += c; }
}

static inline float bf16_round(float v) { return __bfloat162float(__float2bfloat16(v)); }

template <class T> static T* dalloc(size_t n) { T* p; ETX_CHECK(hipMalloc(&p, n * sizeof(T))); ETX_CHECK(hipMemset(p, 0, n * sizeof(T))); return p; }

int main(int argc, char** argv) {
  if (argc < 2) die("usage: run <packed-dir> [--gen N] [--repeat R] [--teacher]");
  const std::string dir = argv[1];
  int gen = -1, repeat = 2; bool teacher = false, unfused = false; std::string golden_name = "golden.txt", dump_path;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gen" && i + 1 < argc) gen = atoi(argv[++i]);
    else if (a == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
    else if (a == "--teacher") teacher = true;
    else if (a == "--unfused") unfused = true;       // baseline: same tiles, one launch per grid (HIP graph)
    else if (a == "--golden" && i + 1 < argc) golden_name = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];   // per-layer residuals of the last prompt step (sliced graph)
  }
  const Manifest m = read_manifest(dir + "/llm.manifest");
  std::ifstream gf(dir + "/" + golden_name); if (!gf) die("no " + golden_name);
  const std::vector<int> prompt = read_ints(gf), golden = read_ints(gf);
  std::vector<float> margin; { std::string line; std::getline(gf, line); std::istringstream s(line); std::string h; s >> h; float x; while (s >> x) margin.push_back(x); }
  if (gen < 0 || gen > (int)golden.size()) gen = (int)golden.size();

  LlmParams hp{};
  hp.H = m.geti("H"); hp.NH = m.geti("NH"); hp.NKV = m.geti("NKV"); hp.HD = m.geti("HD"); hp.G = hp.NH / hp.NKV;
  hp.INTER = m.geti("INTER"); hp.L = m.geti("L"); hp.V = m.geti("V"); hp.E = m.geti("E"); hp.TOPK = m.geti("TOPK"); hp.MI = m.geti("MI");
  hp.CTX = m.geti("CTX"); hp.CHUNK = m.geti("CHUNK"); hp.NC = hp.CTX / hp.CHUNK;
  hp.qk_norm = m.geti("qk_norm"); hp.has_bias = m.geti("bias"); hp.moe = m.geti("moe"); hp.norm_topk = m.geti("norm_topk"); hp.tie = m.geti("tie");
  hp.eps = (float)m.cfg.at("eps");
  hp.rpt_qkv = m.geti("rpt_qkv"); hp.rpt_o = m.geti("rpt_o"); hp.rpt_lm = m.geti("rpt_lm");
  if (hp.moe) { hp.rpt_r = m.geti("rpt_r"); hp.rpt_egu = m.geti("rpt_egu"); hp.rpt_edn = m.geti("rpt_edn"); }
  else { hp.rpt_gu = m.geti("rpt_gu"); hp.rpt_dn = m.geti("rpt_dn"); }
  if (hp.L > LLM_MAX_LAYERS) die("too many layers");
  const int steps = (int)prompt.size() - 1 + gen;
  if (steps + 1 > hp.CTX) die("prompt + generation exceed CTX");
  printf("model: H=%d NH=%d NKV=%d HD=%d L=%d V=%d %s  steps=%d (prompt %zu, gen %d)\n", hp.H, hp.NH, hp.NKV, hp.HD, hp.L, hp.V,
         hp.moe ? "MoE" : "dense", steps, prompt.size(), gen);

  // ---- weights: one blob
  const std::string wpath = dir + "/weights.bin";
  FILE* wf = fopen(wpath.c_str(), "rb"); if (!wf) die("cannot open " + wpath);
  fseek(wf, 0, SEEK_END); const size_t wbytes = (size_t)ftell(wf); fseek(wf, 0, SEEK_SET);
  char* dW; ETX_CHECK(hipMalloc(&dW, wbytes));
  {
    const size_t chunk = 256ull << 20; void* pinned; ETX_CHECK(hipHostMalloc(&pinned, chunk));
    for (size_t off = 0; off < wbytes; off += chunk) {
      const size_t n = std::min(chunk, wbytes - off);
      if (fread(pinned, 1, n, wf) != n) die("short read on weights.bin");
      ETX_CHECK(hipMemcpy(dW + off, pinned, n, hipMemcpyHostToDevice));
    }
    ETX_CHECK(hipHostFree(pinned)); fclose(wf);
  }
  printf("weights: %.2f GB\n", wbytes / 1e9);
  auto W = [&](const std::string& n, bool required = true) -> const __hip_bfloat16* {
    auto it = m.t.find(n); if (it == m.t.end()) { if (required) die("manifest lacks tensor " + n); return nullptr; }
    return (const __hip_bfloat16*)(dW + it->second.first);
  };
  hp.embed = W("embed"); hp.final_norm = W("final_norm"); hp.lm_head = hp.tie ? nullptr : W("lm_head");
  const size_t kv_elems = (size_t)hp.CTX * hp.NKV * hp.HD;
  for (int L = 0; L < hp.L; ++L) {
    const std::string p = "L" + std::to_string(L) + ".";
    LlmLayer& y = hp.layer[L];
    y.ln1 = W(p + "ln1"); y.ln2 = W(p + "ln2"); y.wqkv = W(p + "wqkv"); y.bqkv = W(p + "bqkv", false);
    y.qn = W(p + "qn", false); y.kn = W(p + "kn", false); y.wo = W(p + "wo");
    if (hp.moe) { y.wr = W(p + "wr"); y.eg = W(p + "eg"); y.eu = W(p + "eu"); y.ed = W(p + "ed"); }
    else { y.wg = W(p + "wg"); y.wu = W(p + "wu"); y.wd = W(p + "wd"); }
    y.kc = dalloc<__hip_bfloat16>(kv_elems); y.vc = dalloc<__hip_bfloat16>(kv_elems);
  }
  // ---- sliced layout (model_sliced.py): per-XCD tables and the slice blob, when prep.py --sliced ran
  hp.X = 8; hp.W = ETX_WORKERS_PER_DOMAIN;
  split8(hp.NH, 1, hp.qh0, hp.nqh);
  for (int k = 0; k < 8; ++k) {
    if (hp.nqh[k] == 0) { hp.kv0[k] = 0; hp.nkv[k] = 0; }
    else { const int kv0 = hp.qh0[k] / hp.G, kv1 = (hp.qh0[k] + hp.nqh[k] - 1) / hp.G; hp.kv0[k] = kv0; hp.nkv[k] = kv1 - kv0 + 1; }
  }
  if (hp.moe) for (int k = 0; k < 8; ++k) { hp.i0[k] = 0; hp.ni[k] = 0; } else split8(hp.INTER, 128, hp.i0, hp.ni);
  split8(hp.H, 8, hp.h0, hp.nh);
  bool sliced = false;
  {
    std::ifstream sm(dir + "/sliced.manifest");
    if (sm) {
      Manifest ms; std::string kind;
      while (sm >> kind) { if (kind == "t") { std::string n; size_t off, cnt; sm >> n >> off >> cnt; ms.t[n] = {off, cnt}; } else { std::string rest; std::getline(sm, rest); } }
      const std::string spath = dir + "/sliced.bin";
      FILE* sf = fopen(spath.c_str(), "rb"); if (!sf) die("cannot open " + spath);
      fseek(sf, 0, SEEK_END); const size_t sbytes = (size_t)ftell(sf); fseek(sf, 0, SEEK_SET);
      char* dS; ETX_CHECK(hipMalloc(&dS, sbytes));
      { const size_t chunk = 256ull << 20; void* pinned; ETX_CHECK(hipHostMalloc(&pinned, chunk));
        for (size_t off = 0; off < sbytes; off += chunk) { const size_t n = std::min(chunk, sbytes - off);
          if (fread(pinned, 1, n, sf) != n) die("short read on sliced.bin"); ETX_CHECK(hipMemcpy(dS + off, pinned, n, hipMemcpyHostToDevice)); }
        ETX_CHECK(hipHostFree(pinned)); fclose(sf); }
      auto WS = [&](const std::string& n) -> const __hip_bfloat16* { auto it = ms.t.find(n); return it == ms.t.end() ? nullptr : (const __hip_bfloat16*)(dS + it->second.first); };
      for (int L = 0; L < hp.L; ++L) for (int k = 0; k < 8; ++k) {
        const std::string p = "L" + std::to_string(L) + ".", x = ".x" + std::to_string(k);
        hp.layer[L].wqkv_x[k] = WS(p + "wqkv" + x); hp.layer[L].bqkv_x[k] = WS(p + "bqkv" + x);
        hp.layer[L].wo_x[k] = WS(p + "wo" + x); hp.layer[L].wd_x[k] = WS(p + "wd" + x);
      }
      sliced = true;
      printf("sliced weights: %.2f GB\n", sbytes / 1e9);
    }
  }
  hp.layer_dump = dump_path.empty() ? nullptr : dalloc<float>((size_t)(hp.L + 1) * hp.H); hp.dump_step = (int)prompt.size() - 1;
  hp.ctr_qkv = dalloc<int32_t>((size_t)hp.L * 8); hp.ctr_attn = dalloc<int32_t>((size_t)hp.L * 8); hp.step_seq = 0; hp.unfused = unfused ? 1 : 0;
  hp.xa = dalloc<float>(hp.H); hp.xin_x = dalloc<float>((size_t)8 * hp.H); hp.xa_x = dalloc<float>((size_t)8 * hp.H); hp.xfin = dalloc<float>(hp.H); hp.o_part = dalloc<float>((size_t)8 * hp.H); hp.d_part = dalloc<float>((size_t)std::max(8, hp.TOPK) * hp.H);
  hp.rlog_x = dalloc<float>((size_t)8 * std::max(1, hp.E)); hp.eid = dalloc<int32_t>(std::max(8, hp.TOPK));
  const bool sliced_graph = strncmp(ETX_GRAPH_NAME, "llm_sliced_", 11) == 0;
  if (sliced_graph && !sliced) die("this binary was built from model_sliced.py: run prep.py --sliced for " + dir);
  // ---- RoPE tables, computed like HF (fp32 inv_freq and angles, cos/sin rounded to bf16)
  {
    const int half = hp.HD / 2; const float theta = (float)m.cfg.at("theta");
    std::vector<float> cs((size_t)hp.CTX * half), sn((size_t)hp.CTX * half);
    for (int i = 0; i < half; ++i) {
      const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)hp.HD);
      for (int pos = 0; pos < hp.CTX; ++pos) {
        const float f = (float)pos * inv;
        cs[(size_t)pos * half + i] = bf16_round(cosf(f)); sn[(size_t)pos * half + i] = bf16_round(sinf(f));
      }
    }
    float* dc = dalloc<float>(cs.size()); float* ds = dalloc<float>(sn.size());
    ETX_CHECK(hipMemcpy(dc, cs.data(), cs.size() * 4, hipMemcpyHostToDevice)); ETX_CHECK(hipMemcpy(ds, sn.data(), sn.size() * 4, hipMemcpyHostToDevice));
    hp.rope_cos = dc; hp.rope_sin = ds;
  }
  // ---- activations
  const int nqkv = (hp.NH + 2 * hp.NKV) * hp.HD;
  hp.x = dalloc<float>(hp.H); hp.qkv = dalloc<float>(nqkv); hp.q = dalloc<float>((size_t)hp.NH * hp.HD);
  hp.attn_part = dalloc<float>((size_t)hp.NH * hp.NC * (hp.HD + 2)); hp.o = dalloc<float>((size_t)hp.NH * hp.HD);
  hp.h = dalloc<float>(std::max(1, hp.INTER)); hp.rlogits = dalloc<float>(std::max(1, hp.E)); hp.hexp = dalloc<float>(std::max(1, hp.TOPK * hp.MI));
  const int n_lm = (hp.V + hp.rpt_lm - 1) / hp.rpt_lm;
  hp.lm_val = dalloc<float>(n_lm); hp.lm_idx = dalloc<int32_t>(n_lm);
  hp.token_in = dalloc<int32_t>(steps + 2); hp.token_out = dalloc<int32_t>(steps + 1); hp.token_time = dalloc<uint64_t>(2 * (steps + 1));
  if (llm_params_bytes() != sizeof(LlmParams)) die("LlmParams layout differs between host and device");

  // ---- ETX host
  etx_host hst(0);
  hst.kernel = (const void*)etx_megakernel_d0;
  hst.unfused_kernel = (const void*)etx_unfused_d0;
  hst.setup();
  printf("etx: %d tasks, %d event words, %d workers (%d domains x %d), lds %d B\n", ETX_N_TASKS, ETX_EVENT_WORDS, ETX_N_WORKERS,
         ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, (int)ETX_LDS_BYTES);
  if (!hst.check_residency()) die("the persistent grid does not fit");

  std::vector<int32_t> seq(prompt);                          // prompt + HF's continuation (teacher input)
  seq.insert(seq.end(), golden.begin(), golden.begin() + gen);
  std::vector<int32_t> out(steps + 1); std::vector<uint64_t> tt(2 * (steps + 1));
  int matched = 0, first_div = -1;
  std::vector<double> step_ms;
  for (int run = 0; run < repeat; ++run) {
    std::vector<int32_t> tin(steps + 2, 0);
    for (int i = 0; i < (int)(teacher ? seq.size() : prompt.size()); ++i) tin[i] = seq[i];
    ETX_CHECK(hipMemcpy(hp.token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
    for (int t = 0; t < steps; ++t) {
      hp.tok = t; hp.pos = t; hp.write_next = (!teacher && t + 1 >= (int)prompt.size()) ? 1 : 0;
      llm_set_params(&hp, hst.stream);
      hp.step_seq++;
      if (unfused) { if (!hst.run_unfused()) { printf("step %d: unfused step failed\n", t); return 2; } }
      else if (!hst.run(60.0)) { printf("step %d: ETX step failed\n", t); return 2; }
    }
    ETX_CHECK(hipMemcpy(out.data(), hp.token_out, steps * 4, hipMemcpyDeviceToHost));
    ETX_CHECK(hipMemcpy(tt.data(), hp.token_time, 2 * steps * 8, hipMemcpyDeviceToHost));
    matched = 0; first_div = -1; step_ms.clear();
    for (int i = 0; i < gen; ++i) {
      const int t = (int)prompt.size() - 1 + i;
      if (out[t] == golden[i]) ++matched; else if (first_div < 0) first_div = i;
      step_ms.push_back((tt[2 * t + 1] - tt[2 * t]) * 1e-5);     // 100 MHz ticks -> ms
    }
    if (getenv("ETX_LLM_DUMP")) {        // per-position: ours / HF / margin
      for (int i = 0; i < gen; ++i) { const int t = (int)prompt.size() - 1 + i; printf("  %2d: %6d %6d %s %.3f\n", i, out[t], golden[i], out[t] == golden[i] ? "  " : "XX", i < (int)margin.size() ? margin[i] : -1.f); }
      std::vector<int32_t> all(steps); ETX_CHECK(hipMemcpy(all.data(), hp.token_out, steps * 4, hipMemcpyDeviceToHost));
      printf("  prompt-step argmax:"); for (int t = 0; t + 1 < (int)prompt.size(); ++t) printf(" %d", all[t]); printf("\n");
    }
    printf("run %d: %d/%d generated tokens match HF greedy%s", run, matched, gen, teacher ? " (teacher-forced argmax)" : "");
    if (first_div >= 0) printf("; first divergence at %d (HF top-1 margin there %.4f)", first_div, first_div < (int)margin.size() ? margin[first_div] : -1.f);
    printf("\n");
  }
  if (hp.layer_dump) {
    std::vector<float> ld((size_t)(hp.L + 1) * hp.H); ETX_CHECK(hipMemcpy(ld.data(), hp.layer_dump, ld.size() * 4, hipMemcpyDeviceToHost));
    FILE* f = fopen(dump_path.c_str(), "wb"); if (f) { const int32_t hdr[2] = {hp.L + 1, hp.H}; fwrite(hdr, 4, 2, f); fwrite(ld.data(), 4, ld.size(), f); fclose(f); printf("wrote %s (%d x %d residuals: layer inputs 0..L-1 and the final residual)\n", dump_path.c_str(), hp.L + 1, hp.H); }
  }
  std::vector<double> s(step_ms); std::sort(s.begin(), s.end());
  const double med = s.empty() ? 0 : s[s.size() / 2];
  printf("per-token latency (embed -> argmax on device): median %.3f ms  min %.3f  max %.3f; weights %.2f GB -> %.0f GB/s effective\n",
         med, s.empty() ? 0 : s.front(), s.empty() ? 0 : s.back(), wbytes / 1e9, med > 0 ? (m.cfg.at("active_bytes") / 1e9) / (med / 1e3) : 0.0);
  printf("graph: %s\n", sliced_graph ? "sliced (one slice per XCD, two device-wide barriers per layer)" : "per-operator barriers");
  printf("mode: %s\n", unfused ? (getenv("ETX_UNFUSED_NOGRAPH") ? "unfused (stream launches)" : "unfused (HIP graph)") : "ETX megakernel");
  printf("RESULT model=%s matched=%d gen=%d median_ms=%.3f active_GB=%.3f\n", dir.c_str(), matched, gen, med, m.cfg.at("active_bytes") / 1e9);
  if (getenv("ETX_TRACE")) hst.report_phases(etx_type_names, ETX_N_TYPES);
  return matched == gen ? 0 : 1;
}
