// Experimental F/K switches. Compile BOTH translation units with the same
// values, then select --local-fold / --local-kva in taskgraph.py. With neither
// switch enabled this header emits no declarations, and changes no shared ABI.
#pragma once
#ifndef FLEET_LOCAL_FOLD
#define FLEET_LOCAL_FOLD 0
#endif
#ifndef FLEET_LOCAL_KVA
#define FLEET_LOCAL_KVA 0
#endif
#ifndef FLEET_VERIFY_LOCAL_ASSEMBLY
#define FLEET_VERIFY_LOCAL_ASSEMBLY 0
#endif
#if FLEET_VERIFY_LOCAL_ASSEMBLY && !(FLEET_LOCAL_FOLD || FLEET_LOCAL_KVA)
#error "local assembly verification requires an enabled assembly path"
#endif

#if FLEET_LOCAL_FOLD || FLEET_LOCAL_KVA
namespace fleet {
// Optional descriptor extension, kept out of the frozen runtime ABI.
// TASK_REDUCE is the existing retired task kind; head selects its new body.
constexpr int kLocalFoldFlag = 4096;  // taskgraph.Flags.LOCAL_FOLD
constexpr int kLocalKvaFlag = 8192;   // taskgraph.Flags.LOCAL_KVA
constexpr int kFoldOproj = 0;
constexpr int kFoldExpert = 1;
constexpr int kGatherKva = 2;
constexpr unsigned kAbortLocalPayload = 0x40000000u;
}
#endif
