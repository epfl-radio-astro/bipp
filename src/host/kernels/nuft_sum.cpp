#include "host/kernels/nuft_sum.hpp"

#include <cmath>
#include <complex>

namespace bipp {
namespace host {

template <typename T>
auto nuft_sum(T alpha, std::size_t nIn, const std::complex<T>* __restrict__ input, const T* u,
              const T* v, const T* w, std::size_t nOut, const T* x, const T* y, const T* z, T* out)
    -> void {
  for (std::size_t idxOut = 0; idxOut < nOut; ++idxOut) {
    const auto xVal = x[idxOut];
    const auto yVal = y[idxOut];
    const auto zVal = z[idxOut];

    T sum = 0;
    for (std::size_t idxIn = 0; idxIn < nIn; ++idxIn) {
      const auto p = alpha * (xVal * u[idxIn] + yVal * v[idxIn] + zVal * w[idxIn]);
      const auto real = std::cos(p);
      const auto imag = std::sin(p);

      const auto inVal = input[idxIn];

      sum += inVal.real() * real - inVal.imag() * imag;
    }

    out[idxOut] += sum;
  }
}

template auto nuft_sum<float>(float alpha, std::size_t nIn,
                              const std::complex<float>* __restrict__ input, const float* u,
                              const float* v, const float* w, std::size_t nOut, const float* x,
                              const float* y, const float* z, float* out) -> void;

template auto nuft_sum<double>(double alpha, std::size_t nIn,
                               const std::complex<double>* __restrict__ input, const double* u,
                               const double* v, const double* w, std::size_t nOut, const double* x,
                               const double* y, const double* z, double* out) -> void;

}  // namespace host
}  // namespace bipp
