// Experimental Qwen producer scoring. Shared by HIP and the CPU replay.
#pragma once
#ifndef ETX_PRODUCER_INLINE
#define ETX_PRODUCER_INLINE inline
#endif
ETX_PRODUCER_INLINE float etx_producer_dot8(const float* w, const float* z, const float* gamma) {
  float acc = 0.f;
  for (int j = 0; j < 8; ++j) {
    const float weighted = z[j] * gamma[j];
    acc = fmaf(w[j], weighted, acc);
  }
  return acc;
}
ETX_PRODUCER_INLINE float etx_producer_square8(const float* z) {
  float acc = 0.f;
  for (int j = 0; j < 8; ++j) acc = fmaf(z[j], z[j], acc);
  return acc;
}
