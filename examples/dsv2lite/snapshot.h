// Measurement-only host support, compiled ONLY with -DETX_DSV2_SNAPSHOT.
// The graph's capture prefix stops immediately before native qkv_5/router_5.
// The replay graph contains exactly that phase. Every invocation reloads the
// identical saved bytes; no repeated in-place folding is used as a timing sample.
#pragma once

struct DsSnapshotBuffer { void* data; size_t bytes; };

static std::vector<DsSnapshotBuffer> ds_snapshot_buffers(FleetParams& fp, bool qkv, bool output) {
  auto& a = fp.act; const auto& d = fp.d;
  const size_t hidden = (size_t)d.hidden * sizeof(float);
  if (qkv && !output) return {{a.x[1], hidden}, {a.expert_out, (size_t)kNumXCDs * d.hidden * 2}};
  if (!qkv && !output) return {{a.x[0], hidden}, {a.oproj_partial, (size_t)kNumXCDs * hidden}};
  if (qkv) return {{a.x[0], hidden}, {a.q, (size_t)d.heads * d.q_head_dim * 4},
                   {a.kv_a, (size_t)kNumXCDs * (d.kv_lora + d.qk_rope) * 4}};
  return {{a.x[1], hidden}, {a.x_norm, (size_t)kNumXCDs * hidden},
          {a.router_logits, (size_t)kNumXCDs * d.n_routed * 4}};
}

static void ds_snapshot_transfer(const std::string& path, const std::string& header,
                                 const std::vector<DsSnapshotBuffer>& buffers, bool read) {
  if (read) {
    std::ifstream file(path, std::ios::binary);
    std::string got; std::getline(file, got);
    if (!file || got != header) die("snapshot header mismatch: " + path);
    for (const auto& b : buffers) {
      std::vector<char> bytes(b.bytes); file.read(bytes.data(), bytes.size());
      if (!file) die("snapshot short read: " + path);
      HIP_CHECK(hipMemcpy(b.data, bytes.data(), bytes.size(), hipMemcpyHostToDevice));
    }
    if (file.peek() != EOF) die("snapshot trailing data: " + path);
  } else {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << header << '\n';
    for (const auto& b : buffers) {
      std::vector<char> bytes(b.bytes);
      HIP_CHECK(hipMemcpy(bytes.data(), b.data, bytes.size(), hipMemcpyDeviceToHost));
      file.write(bytes.data(), bytes.size());
    }
    file.close(); if (!file) die("snapshot write failed: " + path);
  }
}

static int ds_snapshot_run(etx_host& host, FleetParams& fp, FleetParams* device_fp, int first_token,
                           int tokens, int repeat, bool unfused) {
  const char* mode = getenv("ETX_DSV2_SNAPSHOT_MODE");
  const char* phase = getenv("ETX_DSV2_SNAPSHOT_PHASE");
  const char* file = getenv("ETX_DSV2_SNAPSHOT_FILE");
  if (!mode || !phase || !file) die("snapshot needs MODE, PHASE and FILE environment variables");
  const bool capture = std::string(mode) == "capture";
  if (!capture && std::string(mode) != "replay") die("snapshot MODE must be capture or replay");
  const bool qkv = std::string(phase) == "qkv_5";
  if (!qkv && std::string(phase) != "router_5") die("snapshot PHASE must be qkv_5 or router_5");
  if (ETX_DS_REQUESTS != 1 || tokens != 1 || repeat != 1 || unfused || fp.d.context != 1024)
    die("snapshot requires one request, --tokens 1 --repeat 1 --context 1024, fused execution");
  const std::string expected = capture ? (qkv ? "down_4" : "oproj_5") : "iso0x0_r0_" + std::string(phase);
  if (std::string(etx_type_names[ETX_N_TYPES - 1]) != expected || (!capture && ETX_N_TYPES != 1))
    die("snapshot mode/phase does not match compiled graph");
  const std::string header = "ETX-DS-SNAPSHOT-1 " + std::string(phase) + " context=1024 hidden=" + std::to_string(fp.d.hidden);
  HIP_CHECK(hipMemcpy(fp.act.token_in, &first_token, 4, hipMemcpyHostToDevice));
  fp.tok = 0; fp.epoch = 1; fp.pos = 1024; fp.dump = 0;
  if (!capture) ds_snapshot_transfer(file, header, ds_snapshot_buffers(fp, qkv, false), true);
  HIP_CHECK(hipMemcpy(device_fp, &fp, sizeof(fp), hipMemcpyHostToDevice));
  fleet_set_params(&fp, host.stream, 0);
  std::printf("SNAPSHOT ready mode=%s phase=%s file=%s\n", mode, phase, file); std::fflush(stdout);
  if (!host.run(10.0)) die("snapshot phase step failed");
  if (capture) ds_snapshot_transfer(file, header, ds_snapshot_buffers(fp, qkv, false), false);
  else {
    const char* output = getenv("ETX_DSV2_SNAPSHOT_OUTPUT");
    if (!output) die("snapshot replay needs OUTPUT path");
    ds_snapshot_transfer(output, header, ds_snapshot_buffers(fp, qkv, true), false);
    if (!getenv("ETX_TRACE") || !getenv("ETX_TRACE_RAW")) die("snapshot replay needs ETX_TRACE and ETX_TRACE_RAW");
    host.report_phases(etx_type_names, ETX_N_TYPES);
  }
  std::printf("SNAPSHOT complete mode=%s phase=%s\n", mode, phase); std::fflush(stdout);
  return 0;
}
