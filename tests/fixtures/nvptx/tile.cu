extern "C" __attribute__((global)) void tile(unsigned *out) {
  unsigned tid = __nvvm_read_ptx_sreg_tid_x();
  unsigned bid = __nvvm_read_ptx_sreg_ctaid_x();
  __shared__ unsigned scratch[32];
  scratch[tid] = bid + tid;
  __syncthreads();
  out[bid * 32 + tid] = scratch[31 - tid];
}
