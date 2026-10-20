// Synthetic two-GPU transport qualification; no GLM inference or performance verdict.
// Uses the existing multi-device etx_host API illustrated by examples/hosts/gemm_rs.hip.
#include "plan_data.h"
#include "etx/host.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_megakernel_d1(etx_params);

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const int reps = argc == 2 ? atoi(argv[1]) : 8;
  if (argc > 2 || reps < 2 || reps > 100) { fprintf(stderr, "need 2..100 repetitions\n"); return 2; }
  int count = 0; ETX_CHECK(hipGetDeviceCount(&count));
  if (count != 2) { fprintf(stderr, "BLOCKED: expose exactly two GPUs\n"); return 2; }
  for (int rank = 0; rank < 2; ++rank) {
    hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop, rank));
    printf("TRANSPORT-DEVICE rank=%d name=%s arch=%s bytes=%zu\n", rank, prop.name, prop.gcnArchName, prop.totalGlobalMem);
    if (strncmp(prop.gcnArchName, "gfx942", 6)) { fprintf(stderr, "BLOCKED: probe compiled for gfx942\n"); return 2; }
    int access = 0, atomics = 0;
    ETX_CHECK(hipDeviceCanAccessPeer(&access, rank, 1-rank));
    ETX_CHECK(hipDeviceGetP2PAttribute(&atomics, hipDevP2PAttrNativeAtomicSupported, rank, 1-rank));
    if (!access || !atomics) { fprintf(stderr, "BLOCKED: peer access and native peer integer atomics required\n"); return 2; }
    ETX_CHECK(hipSetDevice(rank));
    const hipError_t peer = hipDeviceEnablePeerAccess(1-rank, 0);
    if (peer == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
    else ETX_CHECK(peer);
  }
  const int batch = etx_shape[ETX_SYM_B], n = batch * 16384;
  if (batch != 1 && batch != 2 && batch != 4 && batch != 8) return 2;
  std::vector<void*> buffers(ETX_N_ARGS, nullptr);
  for (int arg = 0; arg < ETX_N_ARGS; ++arg) {
    int owner = etx_arg_writer_device[arg]; if (owner < 0) owner = 0;
    ETX_CHECK(hipSetDevice(owner));
    // The two peer-visible payloads follow P6's fine-grained placement contract.
    const bool shared = !strcmp(etx_arg_names[arg], "mailbox") || !strcmp(etx_arg_names[arg], "token");
    if (shared) ETX_CHECK(hipExtMallocWithFlags(&buffers[arg], etx_arg_bytes[arg], hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&buffers[arg], etx_arg_bytes[arg]));
    ETX_CHECK(hipMemset(buffers[arg], 0, etx_arg_bytes[arg]));
  }
  int32_t* events = etx_host::alloc_events(0);
  std::vector<etx_host> hosts; hosts.reserve(2);
  for (int rank = 0; rank < 2; ++rank) {
    hosts.emplace_back(rank);
    hosts.back().kernel = rank == 0 ? (const void*)etx_megakernel_d0 : (const void*)etx_megakernel_d1;
    hosts.back().args = buffers; hosts.back().setup(events); hosts.back().owns_events = rank == 0;
    if (!hosts.back().check_residency()) return 2;
  }
  std::vector<uint16_t> input(n), received(n);
  std::vector<int32_t> returned(batch);
  for (int step = 0; step < reps; ++step) {
    for (int i = 0; i < n; ++i) input[i] = (i + 17 * step) % 65535;
    ETX_CHECK(hipSetDevice(0));
    ETX_CHECK(hipMemcpy(buffers[etx_host::arg_index("input")], input.data(), n * 2, hipMemcpyHostToDevice));
    for (auto& host : hosts) host.reset_step();
    // Complete BOTH resets before either consumer can see a previous epoch's counters.
    for (auto& host : hosts) { ETX_CHECK(hipSetDevice(host.device)); ETX_CHECK(hipStreamSynchronize(host.stream)); }
    const auto start = std::chrono::steady_clock::now();
    for (auto& host : hosts) host.launch();
    for (auto& host : hosts) if (!host.wait(5.0, start)) return 3;
    ETX_CHECK(hipSetDevice(1));
    ETX_CHECK(hipMemcpy(received.data(), buffers[etx_host::arg_index("received")], n * 2, hipMemcpyDeviceToHost));
    ETX_CHECK(hipSetDevice(0));
    ETX_CHECK(hipMemcpy(returned.data(), buffers[etx_host::arg_index("returned")], batch * 4, hipMemcpyDeviceToHost));
    if (received != input) { fprintf(stderr, "TRANSPORT FAILED step=%d state differs\n", step); return 4; }
    for (int r = 0; r < batch; ++r) if (returned[r] != (int)input[r * 16384] + (int)input[(r+1) * 16384 - 1]) {
      fprintf(stderr, "TRANSPORT FAILED step=%d request=%d feedback differs\n", step, r); return 4;
    }
    printf("TRANSPORT-STEP OK step=%d batch=%d state_bytes=%d token_bytes=%d\n", step, batch, n * 2, batch * 4);
  }
  printf("TRANSPORT-RESULT {\"schema\":1,\"status\":\"PASS\",\"batch\":%d,\"repetitions\":%d,\"model_inference\":false,\"performance\":\"unmeasured\"}\n", batch, reps);
  return 0;
}
