#include "spec.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace {
bool valid(const etx_spec_descriptor* d) {
  return d && d->abi_version == ETX_SPEC_ABI_VERSION &&
    d->struct_size == sizeof(*d) && d->backend && d->backend->submit &&
    d->backend->wait && !d->reserved && d->k != UINT32_MAX &&
    std::isfinite(d->timeout_s) && d->timeout_s > 0 && d->position >= 0;
}
}
extern "C" int etx_spec_step(etx_spec_descriptor* d, uint32_t capacity,
                              int32_t eos, int32_t* output) {
  if (!valid(d) || !capacity || !output) return ETX_INVALID;
  uint32_t limit = std::min(capacity, d->k + 1);
  if (d->position > std::numeric_limits<int64_t>::max() - limit)
    return ETX_INVALID;
  const etx_spec_request request = {d->position, d->token, d->k, limit,
                                    eos, d->timeout_s};
  etx_spec_result result = {};
  try {
    if (d->backend->submit(d->context, &request, output, &result))
      return ETX_BACKEND_ERROR;
    if (d->backend->wait(d->context, d->timeout_s)) return ETX_BACKEND_ERROR;
  } catch (...) { return ETX_BACKEND_ERROR; }
  if (!result.L || result.L > limit || result.accepted > d->k ||
      result.accepted > result.L || result.accepted + 1 < result.L ||
      result.terminated > 1)
    return ETX_BACKEND_ERROR;
  // A nonterminal, untruncated step always ends in a correction/bonus.
  if (result.accepted == result.L && result.L < limit && !result.terminated)
    return ETX_BACKEND_ERROR;
  for (uint32_t i = 0; i < result.L; ++i) {
    if (output[i] < 0 || (eos >= 0 && output[i] == eos && i + 1 != result.L))
      return ETX_BACKEND_ERROR;
  }
  if (result.terminated != uint32_t(eos >= 0 && output[result.L-1] == eos))
    return ETX_BACKEND_ERROR;
  d->position += result.L;
  d->token = output[result.L-1];
  d->last = result;
  return ETX_OK;
}
extern "C" int etx_spec_decode(etx_spec_descriptor* d, uint64_t capacity,
    int32_t eos, int32_t* tokens, uint32_t* lengths,
    uint64_t* completed, uint64_t* steps) {
  if (completed) *completed = 0;
  if (steps) *steps = 0;
  if (!completed || !steps || completed == steps || !valid(d) ||
      (capacity && (!tokens || !lengths)) ||
      capacity > uint64_t(std::numeric_limits<int64_t>::max() - d->position))
    return ETX_INVALID;
  while (*completed < capacity) {
    uint32_t limit = uint32_t(std::min<uint64_t>(capacity - *completed,
                                               uint64_t(d->k) + 1));
    int status = etx_spec_step(d, limit, eos, tokens + *completed);
    if (status) return status;
    lengths[(*steps)++] = d->last.L;
    *completed += d->last.L;
    if (d->last.terminated) break;
  }
  return ETX_OK;
}
