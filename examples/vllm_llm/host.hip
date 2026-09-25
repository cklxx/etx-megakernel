// Host for examples/vllm_llm: Qwen decode with vLLM's own kernels inside one ETX megakernel.
//
//   run <packed-dir> <plan.txt> [--golden golden_c1024.txt] [--gen N] [--repeat R] [--teacher] [--unfused]
// <packed-dir> holds vl.bin / vl.manifest (examples/vllm_llm/prep.py) and the golden file of
// examples/llm/prep.py (prompt ids, HF greedy ids). plan.txt lists the grid instances of one step in
// order (examples/vllm_llm/model.py); for each the host builds the kernel's argument block exactly as
// vLLM's host code would launch it (argument-for-argument, cited in model.py), with the buffers laid out
// the way vLLM lays out its tensors (KV cache in ROCM_ATTN's layout, identity block table).
// --unfused launches every grid as its own kernel (HIP graph): the same imported code, kernel boundaries
// instead of ETX events.
#include "plan_data.h"
#include "etx/host.h"
#include "imports.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

struct VlParams {                        // must match vl_tiles.hip
  int32_t tok, pos, write_next, H, V, pad;
  const void* embed; void* resid; int64_t* positions; int32_t* seq_lens; int64_t* slot_mapping;
  const void* logits; int32_t* token_in; int32_t* token_out; uint64_t* token_time;
};
extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_unfused_d0(etx_params, int32_t);
void vl_set_params(const VlParams* p, hipStream_t s);
extern "C" int vl_params_bytes();

static void die(const std::string& m) { fprintf(stderr, "error: %s\n", m.c_str()); exit(1); }
template <class T> static T* dalloc(size_t n) { T* p; ETX_CHECK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); ETX_CHECK(hipMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); return p; }

struct Manifest {
  std::map<std::string, double> cfg;
  std::map<std::string, std::pair<size_t, size_t>> t;
  int geti(const char* k) const { auto it = cfg.find(k); if (it == cfg.end()) die(std::string("manifest lacks cfg ") + k); return (int)it->second; }
};
static Manifest read_manifest(const std::string& path) {
  Manifest m; std::ifstream f(path); if (!f) die("cannot open " + path);
  std::string kind;
  while (f >> kind) {
    if (kind == "cfg" || kind == "cfgf") { std::string k; double v; f >> k >> v; m.cfg[k] = v; }
    else if (kind == "t") { std::string n; size_t off, cnt; f >> n >> off >> cnt; m.t[n] = {off, cnt}; }
    else { std::string rest; std::getline(f, rest); }
  }
  return m;
}
static std::vector<int> read_ints(std::istream& in) {
  std::string line; std::getline(in, line); std::istringstream s(line); std::string head; s >> head;
  std::vector<int> v; int x; while (s >> x) v.push_back(x); return v;
}

struct Inst { int n; std::string exp; int layer; std::string role; int gx, gy, gz; size_t off; int tasks; int M, K, wv; };

// fills one argument block in declaration order, checking every value's type against the import's layout
struct Args {
  const etx_imp_desc* d; uint8_t* p; int i = 0;
  Args(const char* name, uint8_t* base) : d(etx_imp_find(name)), p(base) { if (!d) die(std::string("no import ") + name); }
  void chk(char t) { if (i >= d->nparams || d->ty[i] != t) die(std::string(d->name) + ": argument " + std::to_string(i) + " has type " + d->ty[i] + ", given " + t); }
  Args& ptr(const void* v) { chk('p'); memcpy(p + d->off[i++], &v, 8); return *this; }
  Args& i32(int v) { chk('i'); memcpy(p + d->off[i++], &v, 4); return *this; }
  Args& i64(long long v) { chk('l'); memcpy(p + d->off[i++], &v, 8); return *this; }
  Args& f32(float v) { chk('f'); memcpy(p + d->off[i++], &v, 4); return *this; }
  Args& b8(bool v) { chk('b'); p[d->off[i++]] = v ? 1 : 0; return *this; }
  void grid(int gx, int gy, int gz) {
    if (i != d->nparams) die(std::string(d->name) + ": " + std::to_string(i) + " of " + std::to_string(d->nparams) + " arguments given");
    const uint32_t g[4] = {(uint32_t)gx, (uint32_t)gy, (uint32_t)gz, (uint32_t)(gx * gy * gz)};
    memcpy(p + d->grid_off, g, 16);
  }
};

int main(int argc, char** argv) {
  if (argc < 3) die("usage: run <packed-dir> <plan.txt> [--golden F] [--gen N] [--repeat R] [--teacher] [--unfused]");
  const std::string dir = argv[1], plan_path = argv[2];
  int gen = -1, repeat = 2; bool teacher = false, unfused = false; std::string golden_name = "golden_c1024.txt";
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gen" && i + 1 < argc) gen = atoi(argv[++i]);
    else if (a == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
    else if (a == "--teacher") teacher = true;
    else if (a == "--unfused") unfused = true;
    else if (a == "--golden" && i + 1 < argc) golden_name = argv[++i];
  }
  const Manifest m = read_manifest(dir + "/vl.manifest");
  const int H = m.geti("H"), NH = m.geti("NH"), NKV = m.geti("NKV"), HD = m.geti("HD"), I = m.geti("INTER"), L = m.geti("L"), V = m.geti("V");
  const bool tie = m.geti("tie"), bias = m.geti("bias"), qk_norm = m.geti("qk_norm");
  const int MAXLEN = m.geti("maxlen"), NQKV = (NH + 2 * NKV) * HD;
  const float eps = (float)m.cfg.at("eps"), scale = 1.0f / sqrtf((float)HD);
  const int BS = 16, X = 8, MAXBLK = (MAXLEN + BS - 1) / BS;

  // ---- plan
  std::vector<Inst> insts; size_t total = 0; int parts = 0;
  {
    std::ifstream f(plan_path); if (!f) die("cannot open " + plan_path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream s(line);
      if (line.rfind("total", 0) == 0) { std::string k; int ml; s >> k >> total >> k >> parts >> k >> ml; if (ml != MAXLEN) die("plan maxlen differs from the packed model's"); continue; }
      Inst x; s >> x.n >> x.exp >> x.layer >> x.role >> x.gx >> x.gy >> x.gz >> x.off >> x.tasks >> x.M >> x.K >> x.wv; insts.push_back(x);
    }
  }

  // ---- golden
  std::ifstream gf(dir + "/" + golden_name); if (!gf) die("cannot open " + dir + "/" + golden_name);
  const std::vector<int> prompt = read_ints(gf), golden_all = read_ints(gf);
  if (gen < 0 || gen > (int)golden_all.size()) gen = (int)golden_all.size();
  const std::vector<int> golden(golden_all.begin(), golden_all.begin() + gen);
  const int steps = (int)prompt.size() + gen - 1;
  if (steps + 1 > MAXLEN) die("prompt + generation exceeds maxlen");

  // ---- weights
  std::ifstream wf(dir + "/vl.bin", std::ios::binary); if (!wf) die("cannot open vl.bin");
  const size_t wbytes = (size_t)m.cfg.at("bytes");
  std::vector<char> blob(wbytes); wf.read(blob.data(), wbytes);
  char* dw; ETX_CHECK(hipMalloc(&dw, wbytes)); ETX_CHECK(hipMemcpy(dw, blob.data(), wbytes, hipMemcpyHostToDevice));
  blob.clear(); blob.shrink_to_fit();
  auto W = [&](const std::string& n, bool required = true) -> void* {
    auto it = m.t.find(n); if (it == m.t.end()) { if (required) die("manifest lacks " + n); return nullptr; }
    return dw + it->second.first;
  };

  // ---- activations and caches (bf16 unless noted), shaped as vLLM's tensors for one token
  typedef uint16_t bf;
  bf *resid = dalloc<bf>(H), *hn = dalloc<bf>(H), *qkv = dalloc<bf>(NQKV), *qn = dalloc<bf>(NH * HD), *kn = dalloc<bf>(NKV * HD);
  bf *attn_out = dalloc<bf>(NH * HD), *o = dalloc<bf>(H), *gu = dalloc<bf>(2 * I), *act = dalloc<bf>(I), *dn = dalloc<bf>(H), *logits = dalloc<bf>(V);
  float *exp_sums = dalloc<float>(NH * parts), *max_logits = dalloc<float>(NH * parts); bf* tmp_out = dalloc<bf>((size_t)NH * parts * HD);
  int64_t *positions = dalloc<int64_t>(1), *slot_mapping = dalloc<int64_t>(1); int32_t* seq_lens = dalloc<int32_t>(1);
  std::vector<int32_t> bt(MAXBLK); for (int i = 0; i < MAXBLK; ++i) bt[i] = i;
  int32_t* block_tables = etx_host::upload(bt.data(), bt.size());
  const float one = 1.0f; float* kv_scale = etx_host::upload(&one, 1);
  std::vector<bf*> kc(L), vc(L);
  const size_t cache_elems = (size_t)MAXBLK * NKV * HD * BS;        // [blocks][NKV][HD/x][BS][x] and [blocks][NKV][HD][BS]
  for (int l = 0; l < L; ++l) { kc[l] = dalloc<bf>(cache_elems); vc[l] = dalloc<bf>(cache_elems); }
  int32_t *token_in = dalloc<int32_t>(steps + 2), *token_out = dalloc<int32_t>(steps + 1); uint64_t* token_time = dalloc<uint64_t>(2 * (steps + 1));

  // ---- argument blocks, instance by instance (see model.py for the vLLM call each role mirrors)
  std::vector<uint8_t> ab(total, 0);
  for (const Inst& x : insts) {
    Args a(x.exp.c_str(), ab.data() + x.off);
    const std::string r = x.role; const std::string pl = "L" + std::to_string(x.layer) + ".";
    bf* qkv_in = x.layer == 0 ? hn : dn;                                  // the normed hidden state entering qkv
    bf* qbuf = qk_norm ? qn : qkv; bf* kbuf = qk_norm ? kn : qkv + NH * HD;
    const long long qstr = qk_norm ? NH * HD : NQKV, kstr = qk_norm ? NKV * HD : NQKV;
    auto gemv = [&](const void* w, const void* in, void* out, const void* b) {   // ops.wvSplitK(weight, x, cu_count, bias)
      a.i32(x.K).i32(x.K).i32(x.K).i32(x.M).i32(b ? x.M : 1).i32(1).ptr(w).ptr(in).ptr(b).ptr(out).i32(x.wv).i32(304);
    };
    auto fused_add = [&](bf* in, const void* w) { a.ptr(in).i64(H).ptr(resid).ptr(w).f32(eps).i32(1).i32(H).i64(H); };
    if (r == "in_norm" && x.layer == 0) a.ptr(hn).ptr(resid).i64(H).i64(0).i64(0).i64(0).i64(0).ptr(W(pl + "ln1")).i64(0).f32(eps).i32(1).i32(H);
    else if (r == "in_norm") fused_add(dn, W(pl + "ln1"));
    else if (r == "qkv") gemv(W(pl + "wqkv"), qkv_in, qkv, bias ? W(pl + "bqkv") : nullptr);
    else if (r == "q_norm") a.ptr(qn).ptr(qkv).i64(HD).i64(NQKV).i64(0).i64(NH).i64(0).ptr(W(pl + "qn")).i64(0).f32(eps).i32(NH).i32(HD);
    else if (r == "k_norm") a.ptr(kn).ptr(qkv + NH * HD).i64(HD).i64(NQKV).i64(0).i64(NKV).i64(0).ptr(W(pl + "kn")).i64(0).f32(eps).i32(NKV).i32(HD);
    else if (r == "rope") a.ptr(positions).ptr(qbuf).ptr(kbuf).ptr(W("cos_sin")).i32(HD).i64(qstr).i64(kstr).i64(HD).i32(NH).i32(NKV).i32(HD).i64(0).b8(false);
    else if (r == "cache") a.ptr(kbuf).ptr(qkv + (NH + NKV) * HD).ptr(kc[x.layer]).ptr(vc[x.layer]).ptr(slot_mapping).i32((int)kstr).i32(NQKV).i32(NKV).i32(HD).i32(BS).i32(X).ptr(kv_scale).ptr(kv_scale);
    else if (r == "attn") a.ptr(qbuf).ptr(kc[x.layer]).ptr(vc[x.layer]).i32(NKV).f32(scale).ptr(block_tables).ptr(seq_lens).ptr(nullptr).i32(MAXBLK).ptr(nullptr)
                            .i32((int)qstr).i32(NKV * HD * BS).i32(HD * BS).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(attn_out).i32(MAXBLK).ptr(kv_scale).ptr(kv_scale);
    else if (r == "reduce") a.ptr(attn_out).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(seq_lens).ptr(nullptr).i32(parts).ptr(nullptr);
    else if (r == "o_proj") gemv(W(pl + "wo"), attn_out, o, nullptr);
    else if (r == "post_norm") fused_add(o, W(pl + "ln2"));
    else if (r == "gate_up") gemv(W(pl + "wgu"), o, gu, nullptr);
    else if (r == "act") a.ptr(act).ptr(gu).i32(I).f32(0.0f).f32(1.0f).f32(0.0f);
    else if (r == "down") gemv(W(pl + "wd"), act, dn, nullptr);
    else if (r == "final_norm") fused_add(dn, W("final_norm"));
    else if (r == "lm_head") gemv(tie ? W("embed") : W("lm_head"), dn, logits, nullptr);
    else die("unknown role " + r);
    a.grid(x.gx, x.gy, x.gz);
  }

  // ---- ETX host
  etx_host hst(0);
  hst.kernel = (const void*)etx_megakernel_d0;
  hst.unfused_kernel = (const void*)etx_unfused_d0;
  hst.alloc_args();
  {
    const int ia = etx_host::arg_index("impargs");
    ETX_CHECK(hipFree(hst.args[ia])); hst.args[ia] = etx_host::upload(ab.data(), ab.size());
  }
  hst.setup();
  printf("etx: %d tasks, %d event words, %d workers (%d domains x %d), %zu grid instances, %zu B of argument blocks\n",
         ETX_N_TASKS, ETX_EVENT_WORDS, ETX_N_WORKERS, ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, insts.size(), total);
  if (vl_params_bytes() != (int)sizeof(VlParams)) die("VlParams layout differs between host and device");
  if (!hst.check_residency()) die("the persistent grid does not fit");

  VlParams vp{}; vp.H = H; vp.V = V; vp.embed = W("embed"); vp.resid = resid; vp.positions = positions; vp.seq_lens = seq_lens;
  vp.slot_mapping = slot_mapping; vp.logits = logits; vp.token_in = token_in; vp.token_out = token_out; vp.token_time = token_time;
  std::vector<int32_t> seq(prompt); seq.insert(seq.end(), golden.begin(), golden.end());
  std::vector<int32_t> out(steps + 1); std::vector<uint64_t> tt(2 * (steps + 1));
  std::vector<double> step_ms, wall_ms;
  int matched = 0, first_div = -1;
  for (int run = 0; run < repeat; ++run) {
    std::vector<int32_t> tin(steps + 2, 0);
    for (int i = 0; i < (int)(teacher ? seq.size() : prompt.size()); ++i) tin[i] = seq[i];
    ETX_CHECK(hipMemcpy(token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
    wall_ms.clear();
    for (int t = 0; t < steps; ++t) {
      vp.tok = t; vp.pos = t; vp.write_next = (!teacher && t + 1 >= (int)prompt.size()) ? 1 : 0;
      const auto w0 = std::chrono::steady_clock::now();
      vl_set_params(&vp, hst.stream);
      if (unfused) { if (!hst.run_unfused()) { printf("step %d: unfused step failed\n", t); return 2; } }
      else if (!hst.run(60.0)) { printf("step %d: ETX step failed\n", t); return 2; }
      if (t + 1 >= (int)prompt.size()) wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count());
    }
    ETX_CHECK(hipMemcpy(out.data(), token_out, steps * 4, hipMemcpyDeviceToHost));
    ETX_CHECK(hipMemcpy(tt.data(), token_time, 2 * steps * 8, hipMemcpyDeviceToHost));
    matched = 0; first_div = -1; step_ms.clear();
    for (int i = 0; i < gen; ++i) {
      const int t = (int)prompt.size() - 1 + i;
      if (out[t] == golden[i]) ++matched; else if (first_div < 0) first_div = i;
      step_ms.push_back((tt[2 * t + 1] - tt[2 * t]) * 1e-5);
    }
    printf("run %d: %d/%d generated tokens match HF greedy%s", run, matched, gen, teacher ? " (teacher-forced argmax)" : "");
    if (first_div >= 0) printf("; first divergence at %d (ours %d, HF %d)", first_div, out[(int)prompt.size() - 1 + first_div], golden[first_div]);
    printf("\n");
  }
  std::vector<double> s(step_ms), w(wall_ms); std::sort(s.begin(), s.end()); std::sort(w.begin(), w.end());
  const double med = s.empty() ? 0 : s[s.size() / 2], wmed = w.empty() ? 0 : w[w.size() / 2];
  printf("per-token latency on device (prologue -> argmax): median %.3f ms  min %.3f  max %.3f; host wall per step median %.3f ms\n",
         med, s.empty() ? 0 : s.front(), s.empty() ? 0 : s.back(), wmed);
  printf("mode: %s\n", unfused ? "unfused (same imported kernels, one launch per grid, HIP graph)" : "ETX megakernel (vLLM kernels imported)");
  printf("RESULT model=%s matched=%d gen=%d median_ms=%.3f wall_ms=%.3f\n", dir.c_str(), matched, gen, med, wmed);
  return 0;
}
