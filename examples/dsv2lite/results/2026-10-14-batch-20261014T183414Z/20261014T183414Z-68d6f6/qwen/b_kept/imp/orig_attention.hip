#include "attention.hip"
extern "C" const void* etx_orig_attn_mfma16_g8() { return (const void*)&paged_attention_ll4mi_QKV_mfma16_kernel<__hip_bfloat16, __hip_bfloat16, vllm::Fp8KVCacheDataType::kAuto, __hip_bfloat16, 16, 128, 256, false, 8, MFMAType::F16>; }
extern "C" const void* etx_orig_attn_reduce() { return (const void*)&paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 128, 128, 256, 1>; }
