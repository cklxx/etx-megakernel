// N workgroups pop from one ring queue until it is empty; reports pops/us vs N.
// Feeds the queue_contention curve and t_pop_ns used by Pass 4.
#include <hip/hip_runtime.h>
#include <cstdio>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void drain(etx_queue q, int32_t* done, long long* out) {
  if (threadIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (;;) {
    int32_t v = etx_pop(&q);
    if (v < 0) { if (atomicAdd(q.head, 0) >= atomicAdd(q.tail, 0)) break; continue; }
    atomicAdd(done, 1);
  }
  long long t1 = ETX_TIMER();
  if (blockIdx.x == 0) out[0] = t1 - t0;
}

int main() {
  const int cap = 1 << 16;
  int32_t *slots, *head, *tail, *done; long long* out;
  hipMalloc(&slots, cap * 4); hipMalloc(&head, 4); hipMalloc(&tail, 4); hipMalloc(&done, 4); hipMallocManaged(&out, 8);
  for (int n = 1; n <= 512; n *= 2) {
    // fill: slots[i] = i, tail = cap
    std::vector<int32_t> h(cap); for (int i = 0; i < cap; ++i) h[i] = i;
    hipMemcpy(slots, h.data(), cap * 4, hipMemcpyHostToDevice);
    hipMemset(head, 0, 4); int32_t t = cap; hipMemcpy(tail, &t, 4, hipMemcpyHostToDevice); hipMemset(done, 0, 4);
    etx_queue q{slots, cap, head, tail};
    drain<<<n, 64>>>(q, done, out); hipDeviceSynchronize();
    int32_t d; hipMemcpy(&d, done, 4, hipMemcpyDeviceToHost);
    printf("N=%4d pops=%d  %.1f ns/pop (per popper)\n", n, d, (double)out[0] * 10.0 * n / d);
  }
  return 0;
}
