#pragma once
#ifdef ETX_NO_ALLWAVE_DRAIN
#error "inbody diagnostics require the original all-wave drain"
#endif
static __device__ __forceinline__ uint64_t etx_inbody_tick() {
  uint64_t t;
  // Inline asm hides the asynchronous SMEM result from LLVM's wait insertion.
  // Wait inside the asm before C++ can read or reuse either destination SGPR.
  asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory");
  return t;
}

#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
static __device__ __forceinline__ int etx_arrival_slot(const etx_params& p, int task, int word) {
  if (!p.arrival_stamps) return -1;
  for (int i = p.arrival_offsets[task]; i < p.arrival_offsets[task + 1]; ++i)
    if (p.arrival_words[i] == word && p.arrival_stamps[(size_t)i * 2] == 0) return i;
  return -1;
}
#define ETX_LEDGER_BEFORE(p, word, task) \
  int ledger_slot = etx_arrival_slot(p, task, word); \
  if (ledger_slot >= 0) (p).arrival_stamps[(size_t)ledger_slot * 2] = etx_inbody_tick()
#define ETX_LEDGER_AFTER(p) \
  if (ledger_slot >= 0) (p).arrival_stamps[(size_t)ledger_slot * 2 + 1] = etx_inbody_tick()
#else
#define ETX_LEDGER_BEFORE(p, word, task) ((void)0)
#define ETX_LEDGER_AFTER(p) ((void)0)
#endif
// Lambda executes the ORIGINAL arrival once. No extra fence, counter poll or
// consumer wait. The single final producer owns this event's diagnostic row.
#define ETX_INBODY_ARRIVE(p, word, task, expression) ([&]() { \
  uint64_t b = (p).inbody_event ? etx_inbody_tick() : 0; \
  ETX_LEDGER_BEFORE(p, word, task); \
  int32_t r = (expression); \
  ETX_LEDGER_AFTER(p); \
  uint64_t e = (p).inbody_event ? etx_inbody_tick() : 0; \
  if (r == 0 && (p).inbody_event) { \
    (p).inbody_event[(size_t)(word) * 3] = b; \
    (p).inbody_event[(size_t)(word) * 3 + 1] = e; \
    (p).inbody_event[(size_t)(word) * 3 + 2] = task; \
  } \
  return r; \
}())
