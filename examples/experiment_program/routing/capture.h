// G04 capture-only ABI. A request owns a disjoint persistent output allocation.
// Published after native selected values were consumed, before LDS reuse.
#pragma once
#include <stdint.h>
struct G04Record {
  uint32_t epoch, seen;
  int32_t ids[6];
  float mix[6];
  uint32_t reserved[2];
};
static_assert(sizeof(G04Record)==64,"G04 exact native capture ABI");
static constexpr size_t G04_RECORDS=32*26*8;
static constexpr size_t G04_BYTES=G04_RECORDS*sizeof(G04Record);
