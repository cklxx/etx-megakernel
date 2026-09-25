// etx importer shim: the part of thrust::complex that torch/headeronly/util/complex.h touches (conversions only;
// no vLLM kernel the importer handles uses complex arithmetic)
#pragma once
namespace thrust {
template <class T> struct complex {
  T re_, im_;
  __host__ __device__ constexpr complex(T r = T(), T i = T()) : re_(r), im_(i) {}
  __host__ __device__ constexpr T real() const { return re_; }
  __host__ __device__ constexpr T imag() const { return im_; }
};
template <class T> __host__ __device__ complex<T> polar(const T& r, const T& t = T()) { return complex<T>(r, t); }  // never evaluated
}  // namespace thrust
