// Greedy speculative decoding. No vendor types cross the host C ABI.
#ifndef ETX_RUNTIME_SPEC_H
#define ETX_RUNTIME_SPEC_H
#include "step.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ETX_SPEC_HD __host__ __device__
#else
#define ETX_SPEC_HD
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define ETX_SPEC_ABI_VERSION 1u

typedef struct etx_spec_result {
  uint32_t L;                       // emitted tokens, including correction/bonus
  uint32_t accepted;                // emitted matching draft tokens
  uint32_t terminated;              // emitted EOS
} etx_spec_result;

// Single device thread after verification/argmax (WP-31) is complete. target
// has k+1 predictions: row i is conditioned on input followed by draft[0:i].
// Emit the longest matching prefix then target[first mismatch], or target[k]
// on full acceptance. Capacity/EOS may truncate that sequence. L counts output
// tokens, not matching drafts. k=0 is ordinary greedy decoding. All token IDs
// are nonnegative; negative eos disables termination. Buffers must not overlap.
// No allocation, synchronization, copies, or host calls. Caller supplies valid
// pointers, k+1 target entries, and capacity output entries; capacity > 0.
ETX_SPEC_HD static inline etx_spec_result etx_spec_accept_greedy(
    const int32_t* draft, const int32_t* target, uint32_t k,
    uint32_t capacity, int32_t eos, int32_t* output) {
  etx_spec_result r = {0, 0, 0};
  while (r.L < capacity) {
    uint32_t i = r.L;
    int match = i < k && draft[i] == target[i];
    int32_t token = match ? draft[i] : target[i];
    output[r.L++] = token;
    if (match) ++r.accepted;
    if (eos >= 0 && token == eos) { r.terminated = 1; break; }
    if (!match) break;
  }
  return r;
}

typedef struct etx_spec_request {
  int64_t position;
  int32_t token;
  uint32_t k;
  uint32_t capacity;                 // <= k+1 and remaining output budget
  int32_t eos;
  double timeout_s;
} etx_spec_request;

// submit enqueues ONE complete step: draft, target verify, device acceptance,
// rollback/commit (WP-22), and output publication. There must be no host read,
// wait, or decision between those phases. It may use etx_spec_accept_greedy on
// device. wait retires all submitted work and publishes host-visible output
// and result. Output/result/request remain alive until wait returns. Both
// callbacks return zero on success and must honor the deadline. A failed or
// throwing submit must leave no work outstanding; wait must drain on failure.
// On any error backend state requires reset before retry, even when host state
// is unchanged. Temperature/rejection sampling is deliberately not this ABI.
struct etx_spec_backend {
  int (*submit)(void*, const etx_spec_request*, int32_t*, etx_spec_result*);
  int (*wait)(void*, double);
};
typedef struct etx_spec_descriptor {
  uint32_t abi_version;
  uint32_t struct_size;
  const struct etx_spec_backend* backend;
  void* context;
  uint32_t k;                        // draft count; UINT32_MAX is invalid
  uint32_t reserved;                 // zero
  double timeout_s;                  // finite, positive
  int64_t position;
  int32_t token;
  etx_spec_result last;              // most recent successful step's L
} etx_spec_descriptor;

// One submit + one completion wait. Host state advances only on success.
// Output is scratch/undefined on failure; descriptor remains unchanged.
int etx_spec_step(etx_spec_descriptor*, uint32_t capacity, int32_t eos,
                  int32_t* output);
// Reports each successful step's L in lengths, including a final partial step.
// lengths needs capacity entries (worst case L=1); tokens needs capacity entries.
// Successful prefix and descriptor survive a later failure. Null output arrays
// are allowed only for capacity=0. Both counts are initialized even on error.
int etx_spec_decode(etx_spec_descriptor*, uint64_t capacity, int32_t eos,
                    int32_t* tokens, uint32_t* lengths,
                    uint64_t* completed, uint64_t* steps);
#ifdef __cplusplus
}
#endif
#undef ETX_SPEC_HD
#endif
