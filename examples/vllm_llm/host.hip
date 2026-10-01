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
#include <cctype>
#include <string>

struct VlParams {                        // must match vl_tiles.hip
  int32_t tok, pos, write_next, H, V, pad;
  const void* embed; void* resid; int64_t* positions; int32_t* seq_lens; int64_t* slot_mapping;
  const void* logits; int32_t* token_in; int32_t* token_out; uint64_t* token_time;
};
extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_unfused_d0(etx_params, int32_t);
void vl_set_params(const VlParams* p, int n, hipStream_t s);
extern "C" int vl_params_bytes();
#ifdef ETX_VL_CHECK
extern "C" const void* etx_orig_find(const char* name);    // orig_table.hip: vLLM's kernels compiled as vLLM compiles them
// Triton kernels: vLLM's own code object from its Triton cache (etx.importer.vllm_kernels TRITON)
extern "C" bool etx_orig_triton(const char* name, const char** hsaco, const char** fn, int* shared);
#endif

// wait for the stream with a deadline: a hung kernel is reported by name instead of hanging the run
static bool wait_stream(hipStream_t s, double seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const hipError_t e = hipStreamQuery(s);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) { fprintf(stderr, "stream error %s\n", hipGetErrorString(e)); return false; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) return false;
  }
}

static void die(const std::string& m) { fprintf(stderr, "error: %s\n", m.c_str()); exit(1); }
// ETX_VL_UC builds: the tensors that cross XCDs inside a step live in uncached memory (MTYPE UC), which is what
// lets the build use ETX_COHERENT_DEVICE_DATA (DOMAIN fences for DEVICE events). The KV cache stays cached: its
// writer and its readers are pinned to one XCD (the build sets ETX_VL_PIN).
template <class T> static T* ualloc(size_t n) {
  T* p;
#ifdef ETX_VL_UC
  ETX_CHECK(hipExtMallocWithFlags((void**)&p, std::max<size_t>(n, 1) * sizeof(T), hipDeviceMallocUncached));
#else
  ETX_CHECK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
#endif
  ETX_CHECK(hipMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
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

struct Inst { int n; std::string exp; int layer; std::string role; int gx, gy, gz; size_t off; int tasks; int M, K, wv; int rep = 1; size_t stride = 0; long long copyoff = -1; int req = 0; int nb = 1; };

// fills one argument block in declaration order, checking every value's type against the import's layout
struct Args {
  const etx_imp_desc* d; uint8_t* p; int i = 0;
  Args(const char* name, uint8_t* base) : d(etx_imp_find(name)), p(base) { if (!d) die(std::string("no import ") + name); }
  void chk(char t) { if (i >= d->nparams || d->ty[i] != t) die(std::string(d->name) + ": argument " + std::to_string(i) + " has type " + d->ty[i] + ", given " + t); }
  Args& ptr(const void* v) { chk('p'); memcpy(p + d->off[i++], &v, 8); return *this; }
  Args& i32(int v) { chk('i'); memcpy(p + d->off[i++], &v, 4); return *this; }
  Args& i64(long long v) { chk('l'); memcpy(p + d->off[i++], &v, 8); return *this; }
  Args& f32(float v) { chk('f'); memcpy(p + d->off[i++], &v, 4); return *this; }
  Args& f64(double v) { chk('d'); memcpy(p + d->off[i++], &v, 8); return *this; }
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
  int gen = -1, repeat = 2, check_layers = 0, max_steps = 0; bool teacher = false, unfused = false, check_dsl = false; std::string golden_name = "golden_c1024.txt"; std::vector<std::string> extra_goldens;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gen" && i + 1 < argc) gen = atoi(argv[++i]);
    else if (a == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
    else if (a == "--teacher") teacher = true;
    else if (a == "--unfused") unfused = true;
    else if (a == "--golden" && i + 1 < argc) golden_name = argv[++i];
    // ETX_VL_REQUESTS>1: request r's prompt and HF greedy tokens. --golden1 F [--golden2 F ...], one per extra request;
    // each must be a DIFFERENT prompt (the same one in every request would make the experts identical and flatter the MoE)
    else if (a.rfind("--golden", 0) == 0 && a.size() > 8 && isdigit((unsigned char)a[8]) && i + 1 < argc) {
      const size_t r = (size_t)atoi(a.c_str() + 8);
      if (r < 1) die("--golden<r> needs r >= 1");
      if (extra_goldens.size() < r) extra_goldens.resize(r);
      extra_goldens[r - 1] = argv[++i];
    }
    else if (a == "--max-steps" && i + 1 < argc) max_steps = atoi(argv[++i]);
    else if (a == "--check-dsl") check_dsl = true;   // the DSL GEMVs of the last layer against a CPU reference (bf16 tolerance)   // traces: stop after N steps (no token check)
    else if (a == "--check" && i + 1 < argc) check_layers = atoi(argv[++i]);   // compare each imported launch with vLLM's original, layers < N
  }
  const Manifest m = read_manifest(dir + "/vl.manifest");
  const int H = m.geti("H"), NH = m.geti("NH"), NKV = m.geti("NKV"), HD = m.geti("HD"), I = m.geti("INTER"), L = m.geti("L"), V = m.geti("V");
  const bool tie = m.geti("tie"), bias = m.geti("bias"), qk_norm = m.geti("qk_norm");
  const int MAXLEN = m.geti("maxlen"), NQKV = (NH + 2 * NKV) * HD;
  const float eps = (float)m.cfg.at("eps"), scale = 1.0f / sqrtf((float)HD);
  auto cfg0 = [&](const char* k) { auto it = m.cfg.find(k); return it == m.cfg.end() ? 0 : (int)it->second; };
  const bool moe = cfg0("moe"); const int E = cfg0("E"), TOPK = cfg0("TOPK"), MI = cfg0("MI"); const bool norm_topk = cfg0("norm_topk");
  const int MOE_BM = 16;                  // fused_moe_kernel BLOCK_SIZE_M (vLLM's default config at batch 1)
  const int BS = 16, X = 8, MAXBLK = (MAXLEN + BS - 1) / BS;

  // ---- plan
  std::vector<Inst> insts; size_t total = 0; int parts = 0, nreq = 1;
  {
    std::ifstream f(plan_path); if (!f) die("cannot open " + plan_path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream s(line);
      if (line.rfind("total", 0) == 0) {
        std::string k; int ml; s >> k >> total >> k >> parts >> k >> ml; if (ml != MAXLEN) die("plan maxlen differs from the packed model's");
        if (s >> k) { if (k != "requests" || !(s >> nreq) || nreq < 1) die("plan header: expected requests <n >= 1>"); }
        continue;
      }
      Inst x; s >> x.n >> x.exp >> x.layer >> x.role >> x.gx >> x.gy >> x.gz >> x.off >> x.tasks >> x.M >> x.K >> x.wv;
      if (!(s >> x.rep >> x.stride >> x.copyoff)) { x.rep = 1; x.stride = 0; x.copyoff = -1; }
      else if (!(s >> x.req)) x.req = 0;
      if (!(s >> x.nb)) x.nb = 1;                      // ETX_VL_BATCH: activation columns this instance serves
#ifndef ETX_VL_NB
#define ETX_VL_NB 1
#endif
      // the tile is emitted for a fixed column count (etx/ops/gemv.py DSL_NB); a plan that asks for more would have
      // the extra requests silently read column 0 -- refuse instead (ETX_VL_BATCH sets both from ETX_VL_REQUESTS)
      if (x.nb > ETX_VL_NB) die("plan instance nb=" + std::to_string(x.nb) + " but the tiles were built with ETX_VL_NB=" + std::to_string(ETX_VL_NB));
      insts.push_back(x);
    }
  }
  bool qknr = false;                                                        // the plan runs the fused q/k norm + rope (in place)
  for (const Inst& x : insts) qknr = qknr || x.role == "qknr" || x.role == "prep";

  // ---- golden: request 0 from --golden, request r from --golden<r>; the prompts have one length
  std::vector<std::vector<int>> prompts, goldens;
  for (int r = 0; r < nreq; ++r) {
    const std::string gname = r == 0 ? golden_name : ((size_t)r <= extra_goldens.size() ? extra_goldens[r - 1] : std::string());
    if (gname.empty()) die("a " + std::to_string(nreq) + "-request plan needs --golden" + std::to_string(r) + " (request " + std::to_string(r) + "'s golden file)");
    std::ifstream gf(dir + "/" + gname); if (!gf) die("cannot open " + dir + "/" + gname);
    std::vector<int> pr = read_ints(gf), ga = read_ints(gf);
    if (gen < 0 || gen > (int)ga.size()) gen = (int)ga.size();
    prompts.push_back(pr); goldens.push_back(ga);
  }
  for (int r = 1; r < nreq; ++r) if (prompts[r].size() != prompts[0].size()) die("the requests' prompts differ in length");
  for (auto& gv : goldens) gv.resize(gen);
  const std::vector<int>& prompt = prompts[0];
  const std::vector<int>& golden = goldens[0];
  const int steps = (int)prompt.size() + gen - 1;
  if (steps + 1 > MAXLEN) die("prompt + generation exceeds maxlen");

  // ---- weights
  std::ifstream wf(dir + "/vl.bin", std::ios::binary); if (!wf) die("cannot open vl.bin");
  const size_t wbytes = (size_t)m.cfg.at("bytes");
  // in 1 GB pieces: the MoE model is 61 GB, more than a host copy should hold
  char* dw; ETX_CHECK(hipMalloc(&dw, wbytes));
  {
    const size_t piece = (size_t)1 << 30; std::vector<char> blob(std::min(piece, wbytes));
    for (size_t o = 0; o < wbytes; o += piece) {
      const size_t n = std::min(piece, wbytes - o);
      wf.read(blob.data(), n); if ((size_t)wf.gcount() != n) die("vl.bin is shorter than the manifest says");
      ETX_CHECK(hipMemcpy(dw + o, blob.data(), n, hipMemcpyHostToDevice));
    }
  }
  auto W = [&](const std::string& n, bool required = true) -> void* {
    auto it = m.t.find(n); if (it == m.t.end()) { if (required) die("manifest lacks " + n); return nullptr; }
    return dw + it->second.first;
  };

  // ---- activations and caches (bf16 unless noted), shaped as vLLM's tensors for one token
  typedef uint16_t bf;
  // ETX_VL_XCD plans replicate the one-block launches per XCD: residual, normed input, o and activation then
  // have one copy per XCD (NR = 8); everything else is shared
  int NR = 1; for (const Inst& x : insts) NR = std::max(NR, x.rep);
  std::vector<int32_t> bt(MAXBLK); for (int i = 0; i < MAXBLK; ++i) bt[i] = i;
  int32_t* block_tables = etx_host::upload(bt.data(), bt.size());
  const float one = 1.0f; float* kv_scale = etx_host::upload(&one, 1);
  const size_t cache_elems = (size_t)MAXBLK * NKV * HD * BS;        // [blocks][NKV][HD/x][BS][x] and [blocks][NKV][HD][BS]
  // MoE (fused_experts at batch 1): router logits, top-k weights / ids / token_expert_indices, num_tokens_post_padded
  // (vLLM's torch.full: TOPK * BLOCK_SIZE_M, a constant, written here once), intermediate caches 1-3 (vLLM aliases 1
  // and 3 in one buffer; separate here, same values); with ETX_VL_XCD the top-k outputs have one copy per XCD
  const int32_t ntpp_val = TOPK * MOE_BM; int32_t* ntpp = etx_host::upload(&ntpp_val, 1);
  // everything a request owns (ETX_VL_REQUESTS: one set each; the weights are shared)
  struct Req {
    bf *resid, *resid2, *hn, *qkv, *qn, *kn, *dx, *ox, *actx, *attn_out, *o, *gu, *act, *dn, *logits, *tmp_out, *rlog, *mc1, *mc2, *mc3;
    float *exp_sums, *max_logits, *topk_w; int64_t *positions, *slot_mapping; int32_t *seq_lens, *token_in, *token_out, *topk_ids, *tei;
    uint64_t* token_time; std::vector<bf*> kc, vc;
  };
  std::vector<Req> reqs(nreq);
  // ETX_VL_BATCH: a batched dense task reads request b's vector at `ptr + b * stride`, so every per-request tensor is
  // ONE allocation across the requests and request r's pointer is base + r * (its element count). The element counts
  // below are also the strides the dense argument blocks carry (n_* used again at the dsl_gemv call sites).
  const size_t n_resid = (size_t)NR * H, n_resid2 = H, n_hn = (size_t)NR * H, n_qkv = NQKV, n_qn = (size_t)NH * HD, n_kn = (size_t)NKV * HD;
  const size_t n_dx = (size_t)NR * H, n_ox = (size_t)NR * H, n_actx = (size_t)NR * I;
  const size_t n_attn_out = (size_t)NH * HD, n_o = H, n_gu = (size_t)2 * I, n_act = I, n_dn = H, n_logits = V;
  const size_t n_es = (size_t)NH * parts, n_ml = (size_t)NH * parts, n_tmp = (size_t)NH * parts * HD;
  const size_t n_rlog = std::max(E, 1), n_tk = (size_t)NR * std::max(TOPK, 1);
  const size_t n_mc1 = (size_t)std::max(TOPK, 1) * 2 * MI, n_mc2 = (size_t)std::max(TOPK, 1) * MI, n_mc3 = (size_t)std::max(TOPK, 1) * H;
#define ETX_VL_SLICE(F, N, T) { T* base_ = ualloc<T>((size_t)nreq * (N)); for (int r_ = 0; r_ < nreq; ++r_) reqs[r_].F = base_ + (size_t)r_ * (N); }
  ETX_VL_SLICE(resid, n_resid, bf) ETX_VL_SLICE(resid2, n_resid2, bf) ETX_VL_SLICE(hn, n_hn, bf) ETX_VL_SLICE(qkv, n_qkv, bf)
  ETX_VL_SLICE(qn, n_qn, bf) ETX_VL_SLICE(kn, n_kn, bf) ETX_VL_SLICE(dx, n_dx, bf) ETX_VL_SLICE(ox, n_ox, bf) ETX_VL_SLICE(actx, n_actx, bf)
  ETX_VL_SLICE(attn_out, n_attn_out, bf) ETX_VL_SLICE(o, n_o, bf) ETX_VL_SLICE(gu, n_gu, bf) ETX_VL_SLICE(act, n_act, bf)
  ETX_VL_SLICE(dn, n_dn, bf) ETX_VL_SLICE(logits, n_logits, bf) ETX_VL_SLICE(tmp_out, n_tmp, bf) ETX_VL_SLICE(rlog, n_rlog, bf)
  ETX_VL_SLICE(mc1, n_mc1, bf) ETX_VL_SLICE(mc2, n_mc2, bf) ETX_VL_SLICE(mc3, n_mc3, bf)
  ETX_VL_SLICE(exp_sums, n_es, float) ETX_VL_SLICE(max_logits, n_ml, float) ETX_VL_SLICE(topk_w, n_tk, float)
  ETX_VL_SLICE(positions, 1, int64_t) ETX_VL_SLICE(slot_mapping, 1, int64_t) ETX_VL_SLICE(seq_lens, 1, int32_t)
  ETX_VL_SLICE(topk_ids, n_tk, int32_t) ETX_VL_SLICE(tei, n_tk, int32_t)
#undef ETX_VL_SLICE
  for (Req& q : reqs) {
    q.kc.resize(L); q.vc.resize(L);
    for (int l = 0; l < L; ++l) { q.kc[l] = dalloc<bf>(cache_elems); q.vc[l] = dalloc<bf>(cache_elems); }
    q.token_in = ualloc<int32_t>(steps + 2); q.token_out = ualloc<int32_t>(steps + 1); q.token_time = dalloc<uint64_t>(2 * (steps + 1));
  }
// ETX_VL_BATCH: the last six argument words are nb + the per-column strides; a tile that does not batch takes nb = 1
#define ETX_VL_NOBATCH .i32(1).i32(0).i32(0).i32(0).i32(0).i32(0)
#define ETX_VL_BIND(Q)                                                                                                    \
  bf *resid = (Q).resid, *resid2 = (Q).resid2, *hn = (Q).hn, *qkv = (Q).qkv, *qn = (Q).qn, *kn = (Q).kn, *dx = (Q).dx, *ox = (Q).ox, *actx = (Q).actx;  \
  bf *attn_out = (Q).attn_out, *o = (Q).o, *gu = (Q).gu, *act = (Q).act, *dn = (Q).dn, *logits = (Q).logits, *tmp_out = (Q).tmp_out; \
  bf *rlog = (Q).rlog, *mc1 = (Q).mc1, *mc2 = (Q).mc2, *mc3 = (Q).mc3;                                                    \
  float *exp_sums = (Q).exp_sums, *max_logits = (Q).max_logits, *topk_w = (Q).topk_w;                                    \
  int64_t *positions = (Q).positions, *slot_mapping = (Q).slot_mapping; int32_t *seq_lens = (Q).seq_lens;                \
  int32_t *token_in = (Q).token_in, *token_out = (Q).token_out, *topk_ids = (Q).topk_ids, *tei = (Q).tei;                \
  uint64_t* token_time = (Q).token_time; const std::vector<bf*>& kc = (Q).kc; const std::vector<bf*>& vc = (Q).vc;        \
  (void)resid; (void)resid2; (void)hn; (void)qkv; (void)qn; (void)kn; (void)dx; (void)ox; (void)actx; (void)attn_out; (void)o; (void)gu;  \
  (void)act; (void)dn; (void)logits; (void)tmp_out; (void)rlog; (void)mc1; (void)mc2; (void)mc3; (void)exp_sums;         \
  (void)max_logits; (void)topk_w; (void)positions; (void)slot_mapping; (void)seq_lens; (void)token_in; (void)token_out;  \
  (void)topk_ids; (void)tei; (void)token_time; (void)kc; (void)vc;

  // ---- argument blocks, instance by instance (see model.py for the vLLM call each role mirrors)
  std::vector<uint8_t> ab(total, 0);
  for (const Inst& x : insts) for (int rr = 0; rr < x.rep; ++rr) {
    if (x.req >= nreq) die("plan instance of a request the plan header does not declare");
    ETX_VL_BIND(reqs[x.req])
    Args a(x.exp.c_str(), ab.data() + x.off + rr * x.stride);
    const std::string r = x.role; const std::string pl = "L" + std::to_string(x.layer) + ".";
    const bool per_xcd = NR > 1;                                          // replicas read / write their XCD's copies
    bf* res_r = resid + (size_t)rr * H;
    bf* hn_r = hn + (size_t)rr * H; bf* dx_r = dx + (size_t)rr * H; bf* ox_r = ox + (size_t)rr * H; bf* act_r = actx + (size_t)rr * I;
    bf* d_norm = per_xcd ? dx_r : dn;                                     // down output after the in / final norm
    bf* o_norm = per_xcd ? ox_r : o;                                      // o_proj output after the post norm
    bf* act_out = per_xcd ? act_r : act;
    bf* qkv_in = x.layer == 0 ? hn_r : d_norm;                            // the normed hidden state entering qkv
    float* tw_r = topk_w + (size_t)rr * TOPK; int32_t* ti_r = topk_ids + (size_t)rr * TOPK;   // this XCD's top-k copy
    // q / k after the norm: separate buffers (rms_norm_3d writes out of place), or qkv itself (no q/k norm, or the fused
    // q/k norm + rope, in place)
    const bool qn_sep = qk_norm && !qknr;
    bf* qbuf = qn_sep ? qn : qkv; bf* kbuf = qn_sep ? kn : qkv + NH * HD;
    const long long qstr = qn_sep ? NH * HD : NQKV, kstr = qn_sep ? NKV * HD : NQKV;
    auto gemv = [&](const void* w, const void* in, void* out, const void* b) {   // ops.wvSplitK(weight, x, cu_count, bias)
      a.i32(x.K).i32(x.K).i32(x.K).i32(x.M).i32(b ? x.M : 1).i32(1).ptr(w).ptr(in).ptr(b).ptr(out).i32(x.wv).i32(304);
    };
    auto copy = [&](const void* src, void* dst, size_t bytes) {         // the replica's copy before an in-place kernel
      if (x.copyoff < 0) return;
      const unsigned long long c3[4] = {(unsigned long long)src, (unsigned long long)dst, bytes, 0};
      memcpy(ab.data() + x.copyoff + rr * 32, c3, 32);
    };
    auto fused_add = [&](bf* in, const void* w) { a.ptr(in).i64(H).ptr(res_r).ptr(w).f32(eps).i32(1).i32(H).i64(H); };
    // the DSL GEMVs (etx.ops.gemv): W, x, out, ids, wts, expert stride, M, K, rows per task, tasks per expert
    if (x.exp == "dsl_gemv") {
      const int nblk = x.gx * x.gy * x.gz, rpt = (x.M + nblk - 1) / nblk;
      if (x.K > 4096 || x.K % 256) die("dsl_gemv: K must be a multiple of 256 and <= 4096 (etx/ops/gemv.py)");
      const void* w; const void* in; void* out; size_t sx = 0, sout = 0;   // sx / sout: the request stride of in / out
      if (r == "qkv") { if (bias) die("dsl_gemv: qkv bias not supported"); w = W(pl + "wqkv"); in = qkv_in; out = qkv; sx = x.layer == 0 ? n_hn : n_dx; sout = n_qkv; }
      else if (r == "o_proj") { w = W(pl + "wo"); in = attn_out; out = o; sx = n_attn_out; sout = n_o; }
      else if (r == "router") { w = W(pl + "wr"); in = o_norm; out = rlog; sx = n_ox; sout = n_rlog; }
      else if (r == "gate_up") { w = W(pl + "wgu"); in = o_norm; out = gu; sx = n_ox; sout = n_gu; }
      else if (r == "down") { w = W(pl + "wd"); in = act_out; out = dn; sx = n_actx; sout = n_dn; }
      else if (r == "lm_head") { w = tie ? W("embed") : W("lm_head"); in = d_norm; out = logits; sx = n_dx; sout = n_logits; }
      else die("dsl_gemv: role " + r);
      a.ptr(w).ptr(in).ptr(out).ptr(nullptr).ptr(nullptr).i64(0).i32(x.M).i32(x.K).i32(rpt).i32(0)
       .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(0).f32(0.f).i32(0).i32(0).ptr(nullptr)
       .i32(x.nb).i32((int)sx).i32((int)sout).i32(0).i32(0).i32(0);
    }
    // dsl_attnprep (etx/ops/attnprep.py): q/k norm + rope of one kv head in place on qkv, its k / v rows into the cache
    else if (x.exp == "dsl_attnprep" && (HD != 128 || NH % NKV || NH / NKV + 2 > ETX_THREADS / 64 || !qk_norm))
      die("dsl_attnprep: needs HD 128, NH % NKV == 0, G + 2 <= waves per workgroup (G + 1 head waves + the v wave) and q/k norm");
    else if (x.exp == "dsl_attnprep") a.ptr(qkv).ptr(W(pl + "qn")).ptr(W(pl + "kn")).ptr(W("cos_sin")).ptr(positions).ptr(kc[x.layer]).ptr(vc[x.layer])
                                       .ptr(slot_mapping).i32(NH).i32(NKV).i32(HD).f32(eps).i32(BS).i32(X);
    else if (x.exp == "dsl_topk" && E > 128) die("dsl_topk: E > 128 (etx/ops/topk.py: one wave, 2 logits per lane)");
    // batched (x.nb > 1): wave w of the one replica set does request w's column; the last word packs nb | (sv << 8),
    // sv = the request stride of wts/ids/tei (n_tk elements; the slabs are one allocation, ETX_VL_SLICE). The logits
    // request stride is E itself (n_rlog), carried by no field. nb = 1 keeps the single-wave behaviour exactly.
    else if (x.exp == "dsl_topk" && x.nb > 1 && (size_t)n_tk >= (1u << 24)) die("dsl_topk: n_tk does not fit the packed stride");
    else if (x.exp == "dsl_topk") a.ptr(rlog).ptr(tw_r).ptr(ti_r).ptr(tei + (size_t)rr * TOPK).i32(E).i32(TOPK).i32(norm_topk ? 1 : 0)
                                   .i32(x.nb > 1 ? (x.nb | ((int)n_tk << 8)) : 1);
    else if (x.exp == "dsl_gemv_fold") {
      // norm fold (mode 1): qkv (L > 0) = norm(residual + the 8 expert rows) with ln1; router = norm(residual + o) with ln2;
      // lm_head = norm(residual + the last layer's expert rows) with final_norm
      const int nblk = x.gx * x.gy * x.gz, rpt = (x.M + nblk - 1) / nblk;
      if (x.K > 4096 || x.K % 256) die("dsl_gemv_fold: K must be a multiple of 256 and <= 4096");
      if (E == 0) die("dsl_gemv_fold: the fold is written for the MoE model (expert rows)");
      const void* w; void* out; const bf* rows; const bf* rin; bf* rout; const void* ln; int R; bf* xout = nullptr;
      // the residual alternates: X = resid (copy 0; the prologue's embedding row, what layer 0's in_norm reads) and Y = resid2.
      // router(L) reads X, writes Y = X + o; qkv(L >= 1) reads Y, writes X = Y + the expert rows; lm_head reads Y + the last
      // expert rows. Readers and the one writer of a grid never share a buffer; layer 0 has no qkv fold (imported in_norm)
      size_t sx = 0, sout = 0, srows = 0, srin = 0, srout = 0;              // the request strides of xout / out / rows / rin / rout
      if (r == "qkv") { w = W(pl + "wqkv"); out = qkv; rows = mc3; rin = resid2; rout = resid; ln = W(pl + "ln1"); R = TOPK;
                        sout = n_qkv; srows = n_mc3; srin = n_resid2; srout = n_resid; }
      else if (r == "router") { w = W(pl + "wr"); out = rlog; rows = o; rin = resid; rout = resid2; ln = W(pl + "ln2"); R = 1; xout = ox;   // ox: what moe_up reads
                        sx = n_ox; sout = n_rlog; srows = n_o; srin = n_resid; srout = n_resid2; }
      else if (r == "lm_head") { w = tie ? W("embed") : W("lm_head"); out = logits; rows = mc3; rin = resid2; rout = nullptr; ln = W("final_norm"); R = TOPK;
                        sout = n_logits; srows = n_mc3; srin = n_resid2; }
      else die("dsl_gemv_fold: role " + r);
      a.ptr(w).ptr(xout).ptr(out).ptr(nullptr).ptr(nullptr).i64(0).i32(x.M).i32(x.K).i32(rpt).i32(0)
       .ptr(rows).ptr(rin).ptr(rout).ptr(ln).i32(R).f32(eps).i32(1).i32(0).ptr(nullptr)
       .i32(x.nb).i32((int)sx).i32((int)sout).i32((int)srows).i32((int)srin).i32((int)srout);
    }
    // top-k fold (mode 2 / 3 = renormalised): the expert tasks recompute the top-k from the router logits
    else if ((x.exp == "dsl_moe_up_fold" || x.exp == "dsl_moe_down_fold") && (MI % 32 || H > 4096 || H % 256 || H % 64 || MI > 4096 || MI % 256 || E > 128))
      die("dsl_moe_*_fold: needs MI % 32 == 0, H % 256 == 0, MI % 256 == 0, H and MI <= 4096, E <= 128");
// ETX_EMIT_FOLD_TOPK: did the plan fold the top-k into the expert tasks, or give it its own phase? Derived from the
// model's shapes (model.py fold_topk) and published through plan_data.h, which the kernel and this file both include,
// so the two cannot disagree. It was a -D until round 30, where build.sh defaulted it to "fold" while the plan had
// derived "keep": the expert tiles recomputed a top-k that already had its own phase, 0.63 ms/token for nothing.
#if !ETX_EMIT_FOLD_TOPK
    // batched MoE (x.nb > 1): one node, requests x slots tasks (S = s * nb + rq, etx/ops/gemv.py); the request strides
    // ride the dense fields -- sx: x, sout: out, srows: ids, srin: wts. Each task serves one column: the joins go, the
    // duplicated expert weight stream stays (the union dedup is a further step).
    else if (x.exp == "dsl_moe_up_fold") a.ptr(W(pl + "w13")).ptr(ox).ptr(mc2).ptr(ti_r).ptr(nullptr).i64((long long)2 * MI * H).i32(MI).i32(H).i32(32).i32(MI / 32)
                                           .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(TOPK).f32(0.f).i32(0).i32(E).ptr(rlog)
                                           .i32(x.nb).i32(x.nb > 1 ? (int)n_ox : 0).i32(x.nb > 1 ? (int)n_mc2 : 0)
                                           .i32(x.nb > 1 ? (int)n_tk : 0).i32(x.nb > 1 ? (int)n_tk : 0).i32(0);
    else if (x.exp == "dsl_moe_down_fold") a.ptr(W(pl + "w2")).ptr(mc2).ptr(mc3).ptr(ti_r).ptr(tw_r).i64((long long)H * MI).i32(H).i32(MI).i32(64).i32(H / 64)
                                             .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(TOPK).f32(0.f).i32(0).i32(E).ptr(rlog)
                                             .i32(x.nb).i32(x.nb > 1 ? (int)n_mc2 : 0).i32(x.nb > 1 ? (int)n_mc3 : 0)
                                             .i32(x.nb > 1 ? (int)n_tk : 0).i32(x.nb > 1 ? (int)n_tk : 0).i32(0);
#else
    else if ((x.exp == "dsl_moe_up_fold" || x.exp == "dsl_moe_down_fold") && x.nb > 1)
      die("ETX_VL_BATCH: a batched MoE needs the standalone top-k (the fold would recompute it per task per column)");
    else if (x.exp == "dsl_moe_up_fold") a.ptr(W(pl + "w13")).ptr(ox).ptr(mc2).ptr(nullptr).ptr(nullptr).i64((long long)2 * MI * H).i32(MI).i32(H).i32(32).i32(MI / 32)
                                           .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(TOPK).f32(0.f).i32(norm_topk ? 3 : 2).i32(E).ptr(rlog) ETX_VL_NOBATCH;
    else if (x.exp == "dsl_moe_down_fold") a.ptr(W(pl + "w2")).ptr(mc2).ptr(mc3).ptr(nullptr).ptr(nullptr).i64((long long)H * MI).i32(H).i32(MI).i32(64).i32(H / 64)
                                             .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(TOPK).f32(0.f).i32(norm_topk ? 3 : 2).i32(E).ptr(rlog) ETX_VL_NOBATCH;
#endif
    else if (x.exp == "dsl_moe_up" && (MI % 32 || H > 4096 || H % 256)) die("dsl_moe_up: MI % 32 or H > 4096 (etx/ops/gemv.py K_MAX, RCH)");
    else if (x.exp == "dsl_moe_down" && (H % 64 || MI > 4096 || MI % 256)) die("dsl_moe_down: H % 64 or MI > 4096");
    else if (x.exp == "dsl_moe_up") a.ptr(W(pl + "w13")).ptr(o_norm).ptr(mc2).ptr(ti_r).ptr(nullptr).i64((long long)2 * MI * H).i32(MI).i32(H).i32(32).i32(MI / 32)
                                      .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(0).f32(0.f).i32(0).i32(0).ptr(nullptr)
                                      .i32(x.nb).i32(x.nb > 1 ? (int)n_ox : 0).i32(x.nb > 1 ? (int)n_mc2 : 0)
                                      .i32(x.nb > 1 ? (int)n_tk : 0).i32(x.nb > 1 ? (int)n_tk : 0).i32(0);
    else if (x.exp == "dsl_moe_down") a.ptr(W(pl + "w2")).ptr(mc2).ptr(mc3).ptr(ti_r).ptr(tw_r).i64((long long)H * MI).i32(H).i32(MI).i32(64).i32(H / 64)
                                        .ptr(nullptr).ptr(nullptr).ptr(nullptr).ptr(nullptr).i32(0).f32(0.f).i32(0).i32(0).ptr(nullptr)
                                        .i32(x.nb).i32(x.nb > 1 ? (int)n_mc2 : 0).i32(x.nb > 1 ? (int)n_mc3 : 0)
                                        .i32(x.nb > 1 ? (int)n_tk : 0).i32(x.nb > 1 ? (int)n_tk : 0).i32(0);
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
    else if (r == "attn") a.ptr(qbuf).ptr(kc[x.layer]).ptr(vc[x.layer]).i32(NKV).f32(scale).ptr(block_tables).ptr(seq_lens).ptr(nullptr).i32(MAXBLK).ptr(nullptr)
                            .i32((int)qstr).i32(NKV * HD * BS).i32(HD * BS).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(attn_out).i32(MAXBLK).ptr(kv_scale).ptr(kv_scale);
    else if (r == "reduce") a.ptr(attn_out).ptr(exp_sums).ptr(max_logits).ptr(tmp_out).ptr(seq_lens).ptr(nullptr).i32(parts).ptr(nullptr);
    else if (r == "o_proj") gemv(W(pl + "wo"), attn_out, o, nullptr);
    else if (r == "post_norm") { copy(o, o_norm, H * 2); fused_add(o_norm, W(pl + "ln2")); }
    else if (r == "gate_up") gemv(W(pl + "wgu"), o_norm, gu, nullptr);
    else if (r == "moe_act") a.ptr(mc2).ptr(mc1).i32(MI).f32(0.0f).f32(1.0f).f32(0.0f);    // one row per expert
    else if (r == "act") a.ptr(act_out).ptr(gu).i32(I).f32(0.0f).f32(1.0f).f32(0.0f);
    else if (r == "router") gemv(W(pl + "wr"), o_norm, rlog, nullptr);
    // topk_softmax -> topkGatingLauncherHelper(gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,
    // num_tokens, topk, 0, num_experts, renormalize, bias, 1.0, stream, is_padding)
    else if (r == "topk") a.ptr(rlog).ptr(nullptr).ptr(tw_r).i32(1).ptr(ti_r).ptr(tei + (size_t)rr * TOPK).i32(TOPK).i32(0).i32(E).b8(norm_topk)
                            .ptr(nullptr).f64(1.0).ptr(nullptr);
    // invoke_fused_moe_triton_kernel, naive block assignment: EM = M * top_k * BLOCK_SIZE_M, num_valid_tokens = M * top_k;
    // Triton dropped the None pointers and the strides equal to 1 (stride_ak, stride_bk, stride_cn) from the signature
    // and appended its global / profile scratch pointers (both sizes 0)
    else if (r == "moe_up") a.ptr(o_norm).ptr(W(pl + "w13")).ptr(mc1).ptr(ti_r).ptr(ntpp)
                              .i32(2 * MI).i32(H).i32(TOPK * MOE_BM).i32(TOPK)
                              .i32(H).i32(2 * MI * H).i32(H).i32(2 * MI).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).ptr(nullptr).ptr(nullptr);
    else if (r == "moe_down") a.ptr(mc2).ptr(W(pl + "w2")).ptr(mc3).ptr(tw_r).ptr(ti_r).ptr(ntpp)
                                .i32(H).i32(MI).i32(TOPK * MOE_BM).i32(TOPK)
                                .i32(MI).i32(H * MI).i32(MI).i32(H).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).i32(0).ptr(nullptr).ptr(nullptr);
    // moe_sum, vectorised: (out, input, num_tokens, hidden, stride_token, stride_topk, nullptr, nullptr, 0, 0); the output
    // is the buffer the next fused_add_rms_norm reads (vLLM copies it there)
    else if (r == "moe_sum") a.ptr(d_norm).ptr(mc3).i64(1).i32(H).i64((long long)TOPK * H).i64(H).ptr(nullptr).ptr(nullptr).i64(0).i64(0);
    else if (r == "down") gemv(W(pl + "wd"), act_out, dn, nullptr);
    else if (r == "final_norm") { copy(dn, d_norm, H * 2); fused_add(d_norm, W("final_norm")); }
    else if (r == "lm_head") gemv(tie ? W("embed") : W("lm_head"), d_norm, logits, nullptr);
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

  std::vector<VlParams> vps(nreq);
  for (int r = 0; r < nreq; ++r) {
    VlParams& v = vps[r]; const Req& q = reqs[r];
    v = VlParams{}; v.H = H; v.V = V; v.pad = NR; v.embed = W("embed"); v.resid = q.resid; v.positions = q.positions; v.seq_lens = q.seq_lens;
    v.slot_mapping = q.slot_mapping; v.logits = q.logits; v.token_in = q.token_in; v.token_out = q.token_out; v.token_time = q.token_time;
  }
  VlParams& vp = vps[0];

  if (check_layers > 0 && NR > 1) die("--check needs a plan without per-XCD replicas (build without ETX_VL_XCD)");
  if (check_layers > 0 && nreq > 1) die("--check needs a one-request plan (build without ETX_VL_REQUESTS)");
  if (check_layers > 0) {
    ETX_VL_BIND(reqs[0])
#ifndef ETX_VL_CHECK
    die("--check needs a build with ETX_VL_CHECK=1 (the original kernels linked in)");
#else
    // one decode step at position 0 of the prompt, launch by launch: vLLM's original kernel and the imported one
    // on the same inputs (a snapshot of every buffer they can touch), compared byte for byte
    std::vector<std::pair<void*, size_t>> bufs = {{resid, H * 2}, {hn, H * 2}, {qkv, NQKV * 2}, {qn, NH * HD * 2}, {kn, NKV * HD * 2}, {attn_out, NH * HD * 2},
      {o, H * 2}, {gu, 4 * I}, {act, 2 * I}, {dn, H * 2}, {logits, (size_t)V * 2}, {exp_sums, (size_t)NH * parts * 4}, {max_logits, (size_t)NH * parts * 4},
      {tmp_out, (size_t)NH * parts * HD * 2}, {rlog, (size_t)std::max(E, 1) * 2}, {topk_w, (size_t)std::max(TOPK, 1) * 4},
      {topk_ids, (size_t)std::max(TOPK, 1) * 4}, {tei, (size_t)std::max(TOPK, 1) * 4}, {mc1, (size_t)std::max(TOPK, 1) * 4 * MI},
      {mc2, (size_t)std::max(TOPK, 1) * 2 * MI}, {mc3, (size_t)std::max(TOPK, 1) * 2 * H}};
    const char* names[] = {"resid", "hn", "qkv", "qn", "kn", "attn_out", "o", "gu", "act", "dn", "logits", "exp_sums", "max_logits", "tmp_out",
                           "router_logits", "topk_weights", "topk_ids", "token_expert_indices", "moe_cache1", "moe_cache2", "moe_cache3"};
    const size_t NB = sizeof(names) / sizeof(names[0]);
    for (int l = 0; l < std::min(L, check_layers); ++l) { bufs.push_back({kc[l], cache_elems * 2}); bufs.push_back({vc[l], cache_elems * 2}); }
    std::vector<void*> save(bufs.size()), ref(bufs.size());
    for (size_t b = 0; b < bufs.size(); ++b) { save[b] = dalloc<char>(bufs[b].second); ref[b] = dalloc<char>(bufs[b].second); }
    std::vector<int32_t> tin(steps + 2, 0); for (size_t i = 0; i < prompt.size(); ++i) tin[i] = prompt[i];
    ETX_CHECK(hipMemcpy(token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
    vp.tok = 0; vp.pos = 0; vp.write_next = 0; vl_set_params(&vp, 1, hst.stream);
    auto launch_imported = [&](int li) {
      int32_t first = etx_launch_first[li]; void* kargs[] = {&hst.p, &first};
      ETX_CHECK(hipLaunchKernel(hst.unfused_kernel, dim3(etx_launch_count[li]), dim3(ETX_THREADS), kargs, 0, hst.stream));
    };
    launch_imported(0);                                              // prologue: embedding row, positions, lengths
    if (!wait_stream(hst.stream, 5.0)) die("prologue hung");
    int bad = 0;
    hipEvent_t e0, e1; ETX_CHECK(hipEventCreate(&e0)); ETX_CHECK(hipEventCreate(&e1));
    auto elapsed_us = [&]() { float ms = 0.f; ETX_CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.0 * ms; };
    std::map<std::string, std::pair<double, double>> by_kernel;      // export -> (original us, imported us), summed
    double sum_o = 0, sum_i = 0;
    for (const Inst& x : insts) {
      if (x.layer >= check_layers || (x.layer < 0 && check_layers < L)) break;
      const etx_imp_desc* d = etx_imp_find(x.exp.c_str());
      const void* fn = etx_orig_find(x.exp.c_str());
      const char *tri_path = nullptr, *tri_fn = nullptr; int tri_shared = 0; hipFunction_t tri = nullptr;
      if (!fn) {
        if (!etx_orig_triton(x.exp.c_str(), &tri_path, &tri_fn, &tri_shared)) die("no original for " + x.exp);
        static std::map<std::string, hipFunction_t> mods;
        if (!mods.count(x.exp)) { hipModule_t mod; ETX_CHECK(hipModuleLoad(&mod, tri_path)); ETX_CHECK(hipModuleGetFunction(&mods[x.exp], mod, tri_fn)); }
        tri = mods[x.exp];
      }
      const int xtasks = x.tasks == 1 && x.rep > 1 ? x.rep : x.tasks;         // an x-form instance runs one task per XCD
      if (etx_launch_count[x.n + 1] != xtasks) die("launch table does not follow the plan at instance " + std::to_string(x.n));
      for (size_t b = 0; b < bufs.size(); ++b) ETX_CHECK(hipMemcpyAsync(save[b], bufs[b].first, bufs[b].second, hipMemcpyDeviceToDevice, hst.stream));
      std::vector<void*> av(d->nparams); for (int k = 0; k < d->nparams; ++k) av[k] = ab.data() + x.off + d->off[k];
      ETX_CHECK(hipEventRecord(e0, hst.stream));
      if (fn) ETX_CHECK(hipLaunchKernel(fn, dim3(x.gx, x.gy, x.gz), dim3(d->block[0], d->block[1], d->block[2]), av.data(), 0, hst.stream));
      else ETX_CHECK(hipModuleLaunchKernel(tri, x.gx, x.gy, x.gz, d->block[0], d->block[1], d->block[2], tri_shared, hst.stream, av.data(), nullptr));
      ETX_CHECK(hipEventRecord(e1, hst.stream));
      if (!wait_stream(hst.stream, 5.0)) { printf("HANG original %s L%d %s\n", x.role.c_str(), x.layer, x.exp.c_str()); return 3; }
      const double us_o = elapsed_us();
      for (size_t b = 0; b < bufs.size(); ++b) {
        ETX_CHECK(hipMemcpyAsync(ref[b], bufs[b].first, bufs[b].second, hipMemcpyDeviceToDevice, hst.stream));
        ETX_CHECK(hipMemcpyAsync(bufs[b].first, save[b], bufs[b].second, hipMemcpyDeviceToDevice, hst.stream));
      }
      ETX_CHECK(hipEventRecord(e0, hst.stream));
      launch_imported(x.n + 1);
      ETX_CHECK(hipEventRecord(e1, hst.stream));
      if (!wait_stream(hst.stream, 5.0)) { printf("HANG imported %s L%d %s (%d tasks)\n", x.role.c_str(), x.layer, x.exp.c_str(), x.tasks); return 3; }
      const double us_i = elapsed_us();
      by_kernel[x.exp].first += us_o; by_kernel[x.exp].second += us_i; sum_o += us_o; sum_i += us_i;
      std::string diff;
      for (size_t b = 0; b < bufs.size(); ++b) {
        std::vector<uint8_t> A(bufs[b].second), B(bufs[b].second);
        ETX_CHECK(hipMemcpy(A.data(), bufs[b].first, A.size(), hipMemcpyDeviceToHost)); ETX_CHECK(hipMemcpy(B.data(), ref[b], B.size(), hipMemcpyDeviceToHost));
        size_t n = 0, first = 0; for (size_t k = 0; k < A.size(); ++k) if (A[k] != B[k]) { if (!n) first = k; ++n; }
        if (n) diff += std::string(" ") + (b < NB ? names[b] : (((b - NB) % 2) ? "vc" : "kc")) + ":" + std::to_string(n) + "B@" + std::to_string(first);
      }
      printf("  %-10s L%-3d %-20s %s%s  original %6.1f us  imported %6.1f us\n", x.role.c_str(), x.layer, x.exp.c_str(),
             diff.empty() ? "identical" : "DIFFERS", diff.c_str(), us_o, us_i);
      if (!diff.empty()) ++bad;
    }
    {   // the ETX tiles with no vLLM counterpart (vLLM uses torch's embedding and argmax kernels)
      const int last = (int)insts.size() + 1;
      for (int li : {0, last}) {
        ETX_CHECK(hipEventRecord(e0, hst.stream)); launch_imported(li); ETX_CHECK(hipEventRecord(e1, hst.stream));
        if (!wait_stream(hst.stream, 5.0)) die("native tile hung");
        printf("  %-20s imported %6.1f us  (%d task(s))\n", li == 0 ? "prologue (ETX)" : "argmax (ETX)", elapsed_us(), etx_launch_count[li]);
      }
    }
    printf("kernel time, summed over the checked launches (GPU events, one launch each):\n");
    for (auto& kv : by_kernel) printf("  %-20s original %8.1f us  imported %8.1f us  (%+.0f%%)\n", kv.first.c_str(), kv.second.first, kv.second.second,
                                      100.0 * (kv.second.second / std::max(kv.second.first, 1e-9) - 1.0));
    printf("  %-20s original %8.1f us  imported %8.1f us\n", "total", sum_o, sum_i);
    printf("CHECK %s: %d launch(es) differ\n", bad ? "FAILED" : "OK", bad);
    return bad ? 4 : 0;
#endif
  }
  std::vector<std::vector<int32_t>> seqs(nreq);
  for (int r = 0; r < nreq; ++r) { seqs[r] = prompts[r]; seqs[r].insert(seqs[r].end(), goldens[r].begin(), goldens[r].end()); }
  std::vector<int32_t> out(steps + 1); std::vector<uint64_t> tt(2 * (steps + 1));
  std::vector<double> step_ms, wall_ms;
  const int progress = getenv("ETX_VL_PROGRESS") ? atoi(getenv("ETX_VL_PROGRESS")) : 0;   // print every N-th step's wall time
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<int> matched(nreq, 0);
  for (int run = 0; run < repeat; ++run) {
    for (int r = 0; r < nreq; ++r) {
      std::vector<int32_t> tin(steps + 2, 0);
      for (int i = 0; i < (int)(teacher ? seqs[r].size() : prompts[r].size()); ++i) tin[i] = seqs[r][i];
      ETX_CHECK(hipMemcpy(reqs[r].token_in, tin.data(), tin.size() * 4, hipMemcpyHostToDevice));
    }
    wall_ms.clear();
    const int nsteps = max_steps > 0 ? std::min(steps, max_steps) : steps;
    for (int t = 0; t < nsteps; ++t) {
      for (VlParams& v : vps) { v.tok = t; v.pos = t; v.write_next = (!teacher && t + 1 >= (int)prompt.size()) ? 1 : 0; }
      const auto w0 = std::chrono::steady_clock::now();
      vl_set_params(vps.data(), nreq, hst.stream);
      if (unfused) { if (!hst.run_unfused()) { printf("step %d: unfused step failed\n", t); return 2; } }
      else if (!hst.run(60.0)) { printf("step %d: ETX step failed\n", t); return 2; }
      const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
      if (t + 1 >= (int)prompt.size()) wall_ms.push_back(wms);
      if (progress && (t % progress == 0 || t < 3)) { printf("  step %d: %.3f ms wall\n", t, wms); fflush(stdout); }
    }
    if (nsteps < steps) { printf("run %d: stopped after %d of %d steps (--max-steps): no token check\n", run, nsteps, steps); continue; }
    // a step starts at the first request's prologue and ends at the last request's argmax
    std::vector<uint64_t> t0(steps, ~0ull), t1(steps, 0);
    for (int r = 0; r < nreq; ++r) {
      ETX_CHECK(hipMemcpy(out.data(), reqs[r].token_out, steps * 4, hipMemcpyDeviceToHost));
      ETX_CHECK(hipMemcpy(tt.data(), reqs[r].token_time, 2 * steps * 8, hipMemcpyDeviceToHost));
      int first_div = -1; matched[r] = 0;
      for (int i = 0; i < gen; ++i) {
        const int t = (int)prompt.size() - 1 + i;
        if (out[t] == goldens[r][i]) ++matched[r]; else if (first_div < 0) first_div = i;
      }
      for (int t = 0; t < steps; ++t) { t0[t] = std::min(t0[t], tt[2 * t]); t1[t] = std::max(t1[t], tt[2 * t + 1]); }
      printf("run %d%s: %d/%d generated tokens match HF greedy%s", run, nreq > 1 ? (" request " + std::to_string(r)).c_str() : "",
             matched[r], gen, teacher ? " (teacher-forced argmax)" : "");
      if (first_div >= 0) printf("; first divergence at %d (ours %d, HF %d)", first_div, out[(int)prompt.size() - 1 + first_div], goldens[r][first_div]);
      printf("\n");
    }
    step_ms.clear();
    for (int i = 0; i < gen; ++i) { const int t = (int)prompt.size() - 1 + i; step_ms.push_back((t1[t] - t0[t]) * 1e-5); }
  }
  // --check-dsl: the last layer's DSL GEMVs against a CPU reference from the same device inputs and the packed weights.
  // The tiles accumulate in fp32 in a different order than vLLM's kernels (and than this loop): a bf16 output may differ
  // by an ulp; the gate is "no output beyond 2 bf16 ulps of the reference", plus the token match above.
  if (check_dsl && nreq > 1) die("--check-dsl needs a one-request plan");
  if (check_dsl) {
    ETX_VL_BIND(reqs[0])
    const int Lc = L - 1; const std::string pl = "L" + std::to_string(Lc) + ".";
    auto b2f = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
    auto f2b = [](float f) { uint32_t u; memcpy(&u, &f, 4); const uint32_t r = ((u >> 16) & 1u) + 0x7fffu; return (uint16_t)((u + r) >> 16); };
    auto host_w = [&](const std::string& n, size_t off_elems, size_t elems) {
      auto it = m.t.find(n); if (it == m.t.end()) die("manifest lacks " + n);
      std::vector<uint16_t> v(elems); wf.clear(); wf.seekg((std::streamoff)(it->second.first + off_elems * 2)); wf.read((char*)v.data(), elems * 2);
      if ((size_t)wf.gcount() != elems * 2) die("short read of " + n); return v; };
    auto dev = [&](const void* p, size_t elems) { std::vector<uint16_t> v(elems); ETX_CHECK(hipMemcpy(v.data(), p, elems * 2, hipMemcpyDeviceToHost)); return v; };
    auto dot = [&](const uint16_t* wr, const std::vector<float>& x, int K) { double s = 0; for (int k = 0; k < K; ++k) s += (double)b2f(wr[k]) * x[k]; return (float)s; };
    auto tof = [&](const std::vector<uint16_t>& v) { std::vector<float> f(v.size()); for (size_t i = 0; i < v.size(); ++i) f[i] = b2f(v[i]); return f; };
    int bad_total = 0;
    // run one launch of the step alone (the unfused kernel) so its output is compared before any in-place successor
    // (q/k norm + rope on qkv, final_norm on the normed input, post_norm on o) rewrites it
    auto is_fold = [&](const char* role) { for (const Inst& x : insts) if (x.layer == Lc && x.role == role && x.req == 0) return x.exp.size() > 5 && x.exp.compare(x.exp.size() - 5, 5, "_fold") == 0; return false; };
    // the CPU side of the norm fold: z = bf16(bf16(sum of R rows) + rin), x = bf16(z * rsqrt(mean z^2 + eps) * ln)
    auto norm_fold = [&](const std::vector<uint16_t>& rows_b, int R, const std::vector<uint16_t>& rin, const std::vector<uint16_t>& ln) {
      std::vector<float> xv(H); double ss = 0;
      for (int i = 0; i < H; ++i) { float acc = 0.f; for (int s2 = 0; s2 < R; ++s2) acc += b2f(rows_b[(size_t)s2 * H + i]);
        const float inv = R > 1 ? b2f(f2b(acc)) : acc; const float z = b2f(f2b(inv + b2f(rin[i]))); xv[i] = z; ss += (double)z * z; }
      const float sv = 1.0f / sqrtf((float)(ss / H) + eps);
      for (int i = 0; i < H; ++i) xv[i] = b2f(f2b(xv[i] * sv * b2f(ln[i])));
      return xv; };
    // an instance's launch: usually launch x.n + 1 (launch 0 is the one prologue, then one launch per instance), but
    // located by the launch's type name ("role_layer", or "role_layer_x" for a per-XCD replica form) so the check does
    // not depend on that arithmetic. An x-form instance (tasks == 1, rep > 1: one replica per XCD, e.g. the non-fold
    // top-k) runs rep tasks, not x.tasks.
    const int n_launch = etx_launch_begin[ETX_N_DEVICES];
    auto inst_tasks = [](const Inst& x) { return x.tasks == 1 && x.rep > 1 ? x.rep : x.tasks; };
    auto launch_of = [&](const Inst& x) {
      const std::string n1 = x.layer >= 0 ? x.role + "_" + std::to_string(x.layer) : x.role, n2 = n1 + "_x";
      auto named = [&](int li) { const char* tn = etx_type_names[etx_descs[etx_launch_first[li]].type]; return n1 == tn || n2 == tn; };
      if (x.n + 1 < n_launch && named(x.n + 1)) return x.n + 1;
      for (int li = 1; li < n_launch; ++li) if (named(li)) return li;                 // the plan order drifted: search
      die("--check-dsl: no launch of type " + n1 + " (or " + n2 + ")"); return -1;
    };
    auto relaunch = [&](const char* role, const char* exp) {
      for (const Inst& x : insts) if (x.layer == Lc && x.role == role && x.req == 0) {
        const std::string want = is_fold(role) ? std::string(exp) + "_fold" : std::string(exp);
        if (x.exp != want) die(std::string("--check-dsl: ") + role + " is " + x.exp + ", not " + want);
        const int li = launch_of(x);
        if (etx_launch_count[li] != inst_tasks(x)) die(std::string("--check-dsl: launch ") + std::to_string(li) + " of " + role + " has " +
            std::to_string(etx_launch_count[li]) + " tasks, the plan says " + std::to_string(inst_tasks(x)));
        int32_t first = etx_launch_first[li]; void* kargs[] = {&hst.p, &first};
        ETX_CHECK(hipLaunchKernel(hst.unfused_kernel, dim3(etx_launch_count[li]), dim3(ETX_THREADS), kargs, 0, hst.stream));
        if (!wait_stream(hst.stream, 5.0)) die(std::string("--check-dsl: ") + role + " hung");
        return;
      }
      die(std::string("--check-dsl: no ") + role + " launch in layer " + std::to_string(Lc));
    };
    auto cmp = [&](const char* what, const std::vector<uint16_t>& got, const std::vector<float>& ref) {
      int bad = 0; double worst = 0, worst_abs = 0; float scale = 0.f;
      for (float v : ref) scale = std::max(scale, fabsf(v));
      const float floor_tol = scale / 4096.0f;                     // fp32 accumulation-order noise near zero, relative to the vector
      for (size_t i = 0; i < ref.size(); ++i) {
        const float g = b2f(got[i]), r = b2f(f2b(ref[i]));
        const float tol = std::max(2.0f * fabsf(r) / 128.0f, floor_tol);        // 2 bf16 ulps of the rounded reference
        const double d = fabs((double)g - r); worst = std::max(worst, d / std::max((double)fabsf(r), 1e-6)); worst_abs = std::max(worst_abs, d);
        if (d > tol) ++bad;
      }
      bad_total += bad;
      printf("  %-9s %7zu values: %d beyond 2 ulps (floor %.1e), worst relative %.2e, worst absolute %.2e, scale %.2e\n", what, ref.size(), bad, floor_tol, worst, worst_abs, scale);
    };
    // qkv: W [NQKV][H] . normed input (replica 0's copy)
    { const bf* in = Lc == 0 ? hn : (NR > 1 ? dx : dn); std::vector<float> x = tof(dev(in, H)); auto Wm = host_w(pl + "wqkv", 0, (size_t)NQKV * H);
      if (is_fold("qkv")) x = norm_fold(dev(mc3, (size_t)TOPK * H), TOPK, dev(resid2, H), host_w(pl + "ln1", 0, H));
      relaunch("qkv", "dsl_gemv");
      std::vector<float> ref(NQKV); for (int r = 0; r < NQKV; ++r) ref[r] = dot(Wm.data() + (size_t)r * H, x, H); cmp("qkv", dev(qkv, NQKV), ref); }
    for (const Inst& x2 : insts) if (x2.layer == Lc && x2.role == "prep" && x2.exp == "dsl_attnprep" && x2.req == 0) {
      std::vector<float> in = tof(dev(qkv, NQKV));
      long long pos = 0, slot = 0; ETX_CHECK(hipMemcpy(&pos, positions, 8, hipMemcpyDeviceToHost)); ETX_CHECK(hipMemcpy(&slot, slot_mapping, 8, hipMemcpyDeviceToHost));
      auto qn = host_w(pl + "qn", 0, HD), kn = host_w(pl + "kn", 0, HD), cs = host_w("cos_sin", (size_t)pos * HD, HD);
      std::vector<float> ref((size_t)(NH + NKV) * HD);
      for (int hh = 0; hh < NH + NKV; ++hh) {
        const float* e = in.data() + (size_t)hh * HD; const std::vector<uint16_t>& w = hh < NH ? qn : kn;
        double ss = 0; for (int d = 0; d < HD; ++d) ss += (double)e[d] * e[d];
        const float rr = 1.0f / sqrtf((float)(ss / HD) + eps);
        std::vector<float> n(HD); for (int d = 0; d < HD; ++d) n[d] = e[d] * (rr * b2f(w[d]));
        for (int d = 0; d < HD; ++d) { const int half = HD / 2, dd = d % half; const float c = b2f(cs[dd]), sn = b2f(cs[half + dd]);
          const float partner = d < half ? -n[d + half] : n[d - half]; ref[(size_t)hh * HD + d] = n[d] * c + partner * sn; }
      }
      relaunch("prep", "dsl_attnprep");
      cmp("prep q/k", dev(qkv, (size_t)(NH + NKV) * HD), ref);
      // the cache slot of every kv head: key [block][h][d / X][off][d % X], value [block][h][d][off]
      const long long blk = slot / BS, off = slot % BS; std::vector<float> kref((size_t)NKV * HD), vref((size_t)NKV * HD); std::vector<uint16_t> kgot((size_t)NKV * HD), vgot((size_t)NKV * HD);
      auto kc_h = dev(kc[Lc], cache_elems), vc_h = dev(vc[Lc], cache_elems);
      for (int hh = 0; hh < NKV; ++hh) for (int d = 0; d < HD; ++d) {
        kref[(size_t)hh * HD + d] = ref[(size_t)(NH + hh) * HD + d]; vref[(size_t)hh * HD + d] = in[(size_t)(NH + NKV + hh) * HD + d];
        kgot[(size_t)hh * HD + d] = kc_h[((((size_t)blk * NKV + hh) * (HD / X) + d / X) * BS + off) * X + d % X];
        vgot[(size_t)hh * HD + d] = vc_h[(((size_t)blk * NKV + hh) * HD + d) * BS + off];
      }
      cmp("prep kc", kgot, kref); cmp("prep vc", vgot, vref);
      break;
    }
    { std::vector<float> x = tof(dev(attn_out, (size_t)NH * HD)); auto Wm = host_w(pl + "wo", 0, (size_t)H * NH * HD);
      relaunch("o_proj", "dsl_gemv");
      std::vector<float> ref(H); for (int r = 0; r < H; ++r) ref[r] = dot(Wm.data() + (size_t)r * NH * HD, x, NH * HD); cmp("o_proj", dev(o, H), ref); }
    if (E > 0) {
      const bf* on = NR > 1 ? ox : o; std::vector<float> x = tof(dev(on, H));
      if (is_fold("router")) x = norm_fold(dev(o, H), 1, dev(resid, H), host_w(pl + "ln2", 0, H));
      { auto Wm = host_w(pl + "wr", 0, (size_t)E * H); relaunch("router", "dsl_gemv");
        std::vector<float> ref(E); for (int r = 0; r < E; ++r) ref[r] = dot(Wm.data() + (size_t)r * H, x, H); cmp("router", dev(rlog, E), ref); }
      for (const Inst& x2 : insts) if (x2.layer == Lc && x2.role == "topk" && x2.exp == "dsl_topk") { relaunch("topk", "dsl_topk"); break; }
      std::vector<int32_t> ids(TOPK); std::vector<float> wts(TOPK);
      ETX_CHECK(hipMemcpy(ids.data(), topk_ids, TOPK * 4, hipMemcpyDeviceToHost)); ETX_CHECK(hipMemcpy(wts.data(), topk_w, TOPK * 4, hipMemcpyDeviceToHost));
      if (is_fold("moe_up")) {                                     // no top-k buffers: the CPU top-k of the logits stands in
        std::vector<float> lg = tof(dev(rlog, E)); float mx = -INFINITY; for (float v : lg) mx = std::max(mx, v);
        std::vector<float> pr(E); double sum = 0; for (int e = 0; e < E; ++e) { pr[e] = expf(lg[e] - mx); sum += pr[e]; }
        for (int e = 0; e < E; ++e) pr[e] = (float)(pr[e] / sum);
        std::vector<bool> used(E, false); double sel = 0;
        for (int t = 0; t < TOPK; ++t) { int b = -1; for (int e = 0; e < E; ++e) if (!used[e] && (b < 0 || pr[e] > pr[b])) b = e; used[b] = true; ids[t] = b; wts[t] = pr[b]; sel += pr[b]; }
        if (norm_topk) for (int t = 0; t < TOPK; ++t) wts[t] = (float)(wts[t] / sel);
        x = tof(dev(ox, H));                                       // the router's stored normed vector
      }
      if (!is_fold("moe_up")) { std::vector<float> lg = tof(dev(rlog, E)); float mx = -INFINITY; for (float v : lg) mx = std::max(mx, v);
        std::vector<float> pr(E); double sum = 0; for (int e = 0; e < E; ++e) { pr[e] = expf(lg[e] - mx); sum += pr[e]; }
        for (int e = 0; e < E; ++e) pr[e] = (float)(pr[e] / sum);
        std::vector<int> ref_ids; std::vector<float> ref_w; std::vector<bool> used(E, false); double sel = 0;
        for (int t = 0; t < TOPK; ++t) { int b = -1; for (int e = 0; e < E; ++e) if (!used[e] && (b < 0 || pr[e] > pr[b])) b = e; used[b] = true; ref_ids.push_back(b); ref_w.push_back(pr[b]); sel += pr[b]; }
        int bad = 0; double worst = 0;
        for (int t = 0; t < TOPK; ++t) { const float rw = norm_topk ? (float)(ref_w[t] / sel) : ref_w[t]; if (ids[t] != ref_ids[t]) ++bad;
          worst = std::max(worst, fabs((double)wts[t] - rw) / std::max((double)rw, 1e-9)); if (fabs((double)wts[t] - rw) > 1e-4 * rw) ++bad; }
        bad_total += bad; printf("  %-9s %7d values: %d wrong (ids exact, weights within 1e-4), worst relative difference %.2e\n", "topk", TOPK, bad, worst); }
      std::vector<float> act_ref((size_t)TOPK * MI), down_ref((size_t)TOPK * H);
      relaunch("moe_up", "dsl_moe_up"); relaunch("moe_down", "dsl_moe_down");
      auto act_got = dev(mc2, (size_t)TOPK * MI);
      for (int s2 = 0; s2 < TOPK; ++s2) {
        auto We = host_w(pl + "w13", (size_t)ids[s2] * 2 * MI * H, (size_t)2 * MI * H);
        for (int r = 0; r < MI; ++r) { const float g = b2f(f2b(dot(We.data() + (size_t)r * H, x, H))), u = b2f(f2b(dot(We.data() + (size_t)(MI + r) * H, x, H)));
          act_ref[(size_t)s2 * MI + r] = g / (1.f + expf(-g)) * u; }
        std::vector<float> xs(MI); for (int k = 0; k < MI; ++k) xs[k] = b2f(act_got[(size_t)s2 * MI + k]);    // the device's own activation
        auto Wd = host_w(pl + "w2", (size_t)ids[s2] * H * MI, (size_t)H * MI);
        for (int r = 0; r < H; ++r) down_ref[(size_t)s2 * H + r] = dot(Wd.data() + (size_t)r * MI, xs, MI) * wts[s2];
      }
      cmp("moe_up", act_got, act_ref); cmp("moe_down", dev(mc3, (size_t)TOPK * H), down_ref);
    }
    printf("CHECK-DSL %s: %d values beyond 2 bf16 ulps of the CPU reference (layer %d)\n", bad_total ? "FAILED" : "OK", bad_total, Lc);
  }
  // ETX_TRACE=1: per-type wait / body / span of the last step, and the raw per-task stamps (ETX_TRACE_RAW=<file>)
  if (getenv("ETX_TRACE") && !unfused) hst.report_phases(etx_type_names, ETX_N_TYPES);
  std::vector<double> s(step_ms), w(wall_ms); std::sort(s.begin(), s.end()); std::sort(w.begin(), w.end());
  const double med = s.empty() ? 0 : s[s.size() / 2], wmed = w.empty() ? 0 : w[w.size() / 2];
  printf("per-token latency on device (prologue -> argmax): median %.3f ms  min %.3f  max %.3f; host wall per step median %.3f ms\n",
         med, s.empty() ? 0 : s.front(), s.empty() ? 0 : s.back(), wmed);
  printf("mode: %s\n", unfused ? "unfused (same imported kernels, one launch per grid, HIP graph)" : "ETX megakernel (vLLM kernels imported)");
  int mall = matched[0]; std::string ms = std::to_string(matched[0]);
  for (int r = 1; r < nreq; ++r) { mall = std::min(mall, matched[r]); ms += "," + std::to_string(matched[r]); }
  printf("RESULT model=%s requests=%d matched=%d gen=%d median_ms=%.3f wall_ms=%.3f tokens_per_s=%.0f matched_each=%s\n", dir.c_str(), nreq, mall, gen,
         med, wmed, med > 0 ? nreq * 1e3 / med : 0.0, ms.c_str());
  return 0;
}
