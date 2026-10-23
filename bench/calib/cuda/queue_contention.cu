// CUDA syntax port of bench/calib/queue_contention.hip. Status: compiled, never run.
// Source comments below describe the HIP experiment; see README.md for CUDA mappings and exclusions.
// N workgroups drain one ring of 64K task ids. Compares the CAS pop (v0.1) with
// the ticket pop (v0.2, etx_try_pop) and reports aggregate throughput, which is
// what the scheduler's C_queue term needs.
#include "port.cuh"
#include <cstdio>
#include <vector>



static __device__ __forceinline__ int32_t pop_cas(const etx_queue* q) {
  int32_t h = atomicAdd(q->head, 0);
  int32_t t = atomicAdd(q->tail, 0);
  if (h >= t) return -1;
  if (atomicCAS(q->head, h, h + 1) != h) return -2;
  int32_t* slot = q->slots + (h % q->capacity);
  int32_t v;
  while ((v = atomicExch(slot, -1)) < 0) { ETX_BACKOFF(); }
  return v;
}

template <int TICKET>
__global__ void drain(etx_queue q, int32_t* done, long long* out) {
  if (threadIdx.x != 0) return;
  int32_t ticket = -1;
  long long t0 = ETX_TIMER();
  for (;;) {
    int32_t v = TICKET ? etx_try_pop(&q, &ticket) : pop_cas(&q);
    if (v >= 0) { atomicAdd(done, 1); continue; }
    if (v == -2) continue;                                  // CAS lost, retry
    const int32_t h = ETX_POLL_DEVICE(q.head), t = ETX_POLL_DEVICE(q.tail);
    if (ticket < 0 && h >= t) break;                        // drained
    if (ticket >= t) break;                                 // ticket beyond the final tail: nothing will come
  }
  long long t1 = ETX_TIMER();
  if (blockIdx.x == 0) out[0] = t1 - t0;
}

int main(int argc, char** argv) {
  if (port_help(argc, argv, "queue_contention (CAS and ticket queue throughput)")) return 0;
  const int cap = 1 << 16;
  int32_t *slots, *head, *tail, *done; long long* out;
  PORT_CHECK(cudaMalloc(&slots, cap * 4)); PORT_CHECK(cudaMalloc(&head, 4)); PORT_CHECK(cudaMalloc(&tail, 4)); PORT_CHECK(cudaMalloc(&done, 4)); PORT_CHECK(cudaMallocManaged(&out, 8));
  std::vector<int32_t> h(cap); for (int i = 0; i < cap; ++i) h[i] = i;
  printf("%6s %14s %14s\n", "N", "cas Mpops/s", "ticket Mpops/s");
  for (int n = 1; n <= 512; n *= 2) {
    double rate[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
      PORT_CHECK(cudaMemcpy(slots, h.data(), cap * 4, cudaMemcpyHostToDevice));
      PORT_CHECK(cudaMemset(head, 0, 4)); int32_t t = cap; PORT_CHECK(cudaMemcpy(tail, &t, 4, cudaMemcpyHostToDevice)); PORT_CHECK(cudaMemset(done, 0, 4));
      etx_queue q{slots, cap, head, tail};
      if (mode == 0) drain<0><<<n, 64>>>(q, done, out); else drain<1><<<n, 64>>>(q, done, out);
      cudaDeviceSynchronize();
      int32_t d; PORT_CHECK(cudaMemcpy(&d, done, 4, cudaMemcpyDeviceToHost));
      rate[mode] = d / ((double)out[0] * 10.0 / 1000.0);            // pops per microsecond
      if (d != cap) printf("  (mode %d N=%d popped %d of %d)\n", mode, n, d, cap);
    }
    printf("%6d %14.2f %14.2f\n", n, rate[0], rate[1]);
  }
  return 0;
}
