# Step runtime (ABI v1)

`host/step.h` is a C header with no SDK dependency. Build the shared library:

```sh
c++ -std=c++17 -O2 -fPIC -shared etx/runtime/host/step.cc -o libetx_rt.so
```

`etx_step` performs one synchronous iteration. `etx_decode` feeds each output
back as the next input, advances position after success, stops after EOS, and
preserves a successful prefix on failure. The ctypes `Runtime` exposes both.
Importing the Python module does not load the library or initialize a device.

The descriptor owns no memory. Retain backend callbacks and their state until
the call returns. Calls using the same device buffers must be serialized.
`prepare` stages the token, position and KV metadata; `finish` reads the greedy
output into `next_token`. These hooks must return nonzero on failure. Python
callbacks must catch their own exceptions: ctypes cannot propagate exceptions
through a C callback. Callbacks must not mutate the original descriptor or the
backend table. Hook changes to fields other than `next_token` are not committed.

Reset runs on every rank before any rank launches. Every successfully launched
rank is waited, even if a later launch/wait fails. A launch callback returning
failure must leave no outstanding work on that rank (or abort/drain it itself).
A wait callback must retire or abort its rank before returning failure. Device
buffers are not rolled back on error; reset must restore all step control state
before retry. Token and position are committed together only after finish.
Timeouts are per rank, not a deadline for the complete multi-rank iteration.

## Generated-plan executable

`host/run.cc` is the common executable source, linked with `libetx_rt`, WP-21's
`loader.cc`, and a generated kernel. It accepts:

```text
run PLAN_JSON IO_MANIFEST_JSON WEIGHT_MAP_JSON [STEPS [INITIAL_TOKEN]]
```

The loader supplies pointers in plan argument order. Runtime metadata comes
from `plan_data.h`; the runner uses the existing host reset, residency and
watchdog checks and prints progress after every iteration. It is a short-lived
process: metadata allocations are reclaimed at exit. The shared step library
is independent of that executable and never allocates device memory.

For a single-rank generated plan in `$BUILD`, CUDA linking is:

```sh
$CUDA/bin/nvcc -std=c++17 -DETX_HOST_CUDA -x cu -arch=sm_90 \
  -Ietx/runtime/include -Ietx/runtime/host -I"$BUILD" -I. \
  "$BUILD/megakernel_d0.cu" etx/runtime/host/run.cc \
  etx/runtime/host/loader.cc -L. -letx_rt -o "$BUILD/run"
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH "$BUILD/run" plan.json manifest.json weights.json
```

Use `$ROCM/bin/hipcc -x hip --offload-arch=gfx942`, omit `ETX_HOST_CUDA`, and
substitute `megakernel_d0.hip` for HIP. The generated tiles must use the selected
SDK. The `splitk_sum` example ships HIP reference tiles; the CUDA acceptance
test translates just their SDK includes and bfloat16 spelling in a temporary
copy, without changing the graph or arithmetic.

For a model, compile with `-DETX_RUN_ADAPTER='"adapter.h"'`; the header defines
`int etx_run_prepare(void*, etx_step_descriptor*)` and `etx_run_finish` with the
same signature. Cast context to `etx_run_context*` to access `plan.argument(name)`
and per-rank hosts/streams. The default hooks perform no token I/O, suitable
for non-decoder graphs such as split-K; real models require these hooks. This
interface enables migration without modifying either legacy model host.

Multi-rank builds also supply `ETX_RUN_KERNELS`, a header declaring the other
kernel entry points and defining `ETX_RUN_KERNEL_TABLE` as a brace-enclosed
array of kernel addresses in device order. Link every generated device kernel.

## Acceptance

```sh
/data/etx-venv/bin/python -m pytest -q tests/test_step_api.py
```

The CPU tests build and load a real shared C++ library via ctypes. SDK tests
compile and link `run`, never execute it. Set `CUDA` and `ROCM` to SDK roots or
compiler commands; `ETX_HOST_CUDA_FLAGS` and `ETX_HOST_ROCM_FLAGS` allow split
SDK layouts. Missing SDKs are reported as skips, not verified links.

Local validation (2026-10-23): the ctypes contracts and generated CUDA link
passed with `/opt/cuda`; the ROCm link test skipped because no installed HIP
runtime library/SDK was available. A compile-only probe with the extracted
ROCm 7.2.4 headers/compiler failed inside `__clang_cuda_complex_builtins.h`
(`max` undeclared). This is not evidence of a successful ROCm link. No GPU was
used and no packages were installed.
