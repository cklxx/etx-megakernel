#include "layernorm_kernels.hip"
extern "C" const void* etx_orig_rms_norm_2d() { return (const void*)&vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>; }
