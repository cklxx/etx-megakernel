// Stable, synchronous C entry point. No vendor types cross this ABI.
#ifndef ETX_RUNTIME_STEP_H
#define ETX_RUNTIME_STEP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ETX_STEP_ABI_VERSION 1u
enum etx_status { ETX_OK=0, ETX_INVALID=1, ETX_BACKEND_ERROR=2 };
struct etx_step_descriptor;
// Callbacks return zero on success. They must not throw across the C ABI.
// State and all pointed-to buffers remain caller-owned and live until return.
typedef int (*etx_step_hook)(void*, struct etx_step_descriptor*);
typedef int (*etx_rank_hook)(void*, uint32_t, double);
struct etx_step_backend {
  etx_rank_hook reset;
  // A failed launch must leave this rank with no outstanding work.
  etx_rank_hook launch;
  // wait must retire the rank even if another rank failed. The timeout is
  // per callback; implementations must diagnose and abort a stuck launch.
  etx_rank_hook wait;
};
typedef struct etx_step_descriptor {
  uint32_t abi_version;
  uint32_t struct_size;
  const struct etx_step_backend* backend;
  void* context;
  uint32_t ranks;
  uint32_t reserved;                 // must be zero
  double timeout_s;                  // finite and positive
  int64_t position;                  // committed only after a successful step
  int32_t token;                     // input; replaced by next_token on success
  int32_t next_token;                // finish hook writes greedy output here
  etx_step_hook prepare;             // optional: stage input/position/KV metadata
  etx_step_hook finish;              // optional: read output after every rank retires
} etx_step_descriptor;
// Order: prepare, reset all ranks, launch all ranks, wait all launched ranks,
// finish, commit token/position. On failure decode state is unchanged; device
// buffers may be modified, so retry requires the backend's full reset.
int etx_step(etx_step_descriptor* descriptor);
// Generates up to capacity tokens, stopping after eos (negative disables EOS).
// completed is always set; successful prefix and state survive a later error.
int etx_decode(etx_step_descriptor*, uint64_t capacity, int32_t eos,
               int32_t* tokens, uint64_t* completed);
const char* etx_status_string(int status);
#ifdef __cplusplus
}
#endif
#endif
