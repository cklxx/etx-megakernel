#include <hip/hip_runtime.h>
#include <cstring>
extern "C" bool etx_orig_triton(const char* n, const char** hsaco, const char** fn, int* shared) {
  return false;
}
extern "C" const void* etx_orig_attn_mfma16_g8();
extern "C" const void* etx_orig_attn_reduce();
extern "C" const void* etx_orig_rms_norm_2d();
extern "C" const void* etx_orig_find(const char* n) {
  if (!strcmp(n, "attn_mfma16_g8")) return etx_orig_attn_mfma16_g8();
  if (!strcmp(n, "attn_reduce")) return etx_orig_attn_reduce();
  if (!strcmp(n, "rms_norm_2d")) return etx_orig_rms_norm_2d();
  return nullptr;
}
