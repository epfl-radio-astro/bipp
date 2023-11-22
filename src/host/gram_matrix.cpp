#include "host/gram_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T>
static T calc_pi_sinc(T x) {
  constexpr auto pi = T(3.14159265358979323846);
  return x ? std::sin(pi * x) / (pi * x) : T(1);
}

template <typename T>
auto gram_matrix(ContextInternal& ctx, std::size_t m, std::size_t n, const std::complex<T>* w,
                 std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl, std::complex<T>* g,
                 std::size_t ldg) -> void {
  auto bufferBase = Buffer<std::complex<T>>(ctx.host_alloc(), m * m);
  auto basePtr = bufferBase.get();

  auto x = xyz;
  auto y = xyz + ldxyz;
  auto z = xyz + 2 * ldxyz;
  
  for (std::size_t i = 0; i < m; ++i) {
    basePtr[i * m + i] = T(1);
    for (std::size_t j = i + 1; j < m; ++j) {
      auto diffX = x[i] - x[j];
      auto diffY = y[i] - y[j];
      auto diffZ = z[i] - z[j];
      auto norm = std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
      basePtr[i * m + j] = calc_pi_sinc(T(2) * norm / wl);
    }
  }

  auto bufferC = Buffer<std::complex<T>>(ctx.host_alloc(), m * n);

  blas::symm(CblasColMajor, CblasLeft, CblasLower, m, n, {1, 0}, basePtr, m, w, ldw, {0, 0},
             bufferC.get(), m);
  blas::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, m, {1, 0}, w, ldw, bufferC.get(), m,
             {0, 0}, g, ldg);

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gram", n, n, g, ldg);
}

template auto gram_matrix<float>(ContextInternal& ctx, std::size_t m, std::size_t n,
                                 const std::complex<float>* w, std::size_t ldw, const float* xyz,
                                 std::size_t ldxyz, float wl, std::complex<float>* g,
                                 std::size_t ldg) -> void;

template auto gram_matrix<double>(ContextInternal& ctx, std::size_t m, std::size_t n,
                                  const std::complex<double>* w, std::size_t ldw, const double* xyz,
                                  std::size_t ldxyz, double wl, std::complex<double>* g,
                                  std::size_t ldg) -> void;

}  // namespace host
}  // namespace bipp
