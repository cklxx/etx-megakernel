// C6 probe ABI. State is cached, agent-atomic, and private to a launch.
#pragma once
#include "etx/abi.h"
struct alignas(128) C6State {
  unsigned long long epoch, ready, output;
  unsigned claim, mask, arrived, elected;
};
struct C6Args {
  etx_ctx prep, attention, reduce;
  const etx_params* retained; // always null; keeps original dispatcher code in the compiled entry
  C6State* state;                 // four head groups
  unsigned* tickets;             // eight discovered XCDs
  int* abort;
  unsigned short* observed;      // output read on XCD 7, checked by host
  unsigned* outputs;             // one normal arrival per head, initial 4
  unsigned long long epoch;
  unsigned spin_limit, mode;     // mode 0 control, 1 combined candidate
  unsigned fault, skew_partition, skew_cycles; // bounded correctness stress only
  unsigned long long* stamps;    // diagnostic only; null in performance build
};
