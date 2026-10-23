// Compile-time host runtime selection; HIP remains the default.
// Define ETX_HOST_CUDA for a CUDA translation unit compiled by a host compiler.
#pragma once
#if defined(ETX_HOST_CUDA) || (defined(__CUDACC__) && !defined(__HIPCC__))
#include "backend_cuda.h"
#else
#include "backend_hip.h"
#endif

// Interface contract (etx::host_backend): all operations return `error`, use
// byte counts, and preserve native asynchronous/capture semantics. Handles and
// device_properties are backend-native aliases, so existing callers can pass
// streams to external libraries without casts. This header never initializes
// a device; selection is per translation unit and must be consistent in a TU.
//
// malloc/free: device allocation. malloc_fine_grained/free: system-coherent
// allocation usable by concurrent devices (CUDA requires concurrent managed
// access and returns not-supported otherwise). malloc_host_coherent/free_host:
// mapped, coherent pinned host memory. Peer access must be enabled explicitly.
// Neither coherent allocation is a promise of atomic capabilities on arbitrary
// hardware: the caller still needs the device's system-scope atomic support.
