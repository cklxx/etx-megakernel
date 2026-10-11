#pragma once
static __device__ __forceinline__ uint64_t etx_inbody_tick() {
  uint64_t t;
  asm volatile("s_memrealtime %0" : "=s"(t) :: "memory");
  return t;
}
// Lambda executes the ORIGINAL arrival once. No extra fence, counter poll or
// consumer wait. The single final producer owns this event's diagnostic row.
#define ETX_INBODY_ARRIVE(p, word, task, expression) ([&]() { \
  uint64_t b = (p).inbody_event ? etx_inbody_tick() : 0; \
  int32_t r = (expression); \
  uint64_t e = (p).inbody_event ? etx_inbody_tick() : 0; \
  if (r == 0 && (p).inbody_event) { \
    (p).inbody_event[(size_t)(word) * 3] = b; \
    (p).inbody_event[(size_t)(word) * 3 + 1] = e; \
    (p).inbody_event[(size_t)(word) * 3 + 2] = task; \
  } \
  return r; \
}())
