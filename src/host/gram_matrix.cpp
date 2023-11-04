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
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace host {

template <typename T>
static T calc_pi_sinc(T a, T x) {
  return x ? std::sin(a * x) / x : T(3.14159265358979323846);
}

template <typename T>
auto gram_matrix(ContextInternal& ctx, ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz,
                 T wl, HostView<std::complex<T>, 2> g) -> void {
  const auto nAntenna= w.shape()[0];
  const auto nBeam= w.shape()[1];

  auto buffer = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {nAntenna, nAntenna});

  auto x = xyz.slice_view(0);
  auto y = xyz.slice_view(1);
  auto z = xyz.slice_view(2);
  T sincScale = 2 * 3.14159265358979323846 / wl;
  for (std::size_t i = 0; i < nAntenna; ++i) {
    buffer[{i, i}] = 4 * 3.14159265358979323846;
    for (std::size_t j = i + 1; j < nAntenna; ++j) {
      auto diffX = x[{i}] - x[{j}];
      auto diffY = y[{i}] - y[{j}];
      auto diffZ = z[{i}] - z[{j}];
      auto norm = std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
      buffer[{j, i}] = 4 * calc_pi_sinc(sincScale, norm);
    }
  }

  auto bufferC = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {nAntenna, nBeam});

  blas::symm(CblasLeft, CblasLower, {1, 0}, buffer, w, {0, 0}, bufferC);
  blas::gemm(CblasConjTrans, CblasNoTrans, {1, 0}, w, bufferC, {0, 0}, g);

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gram", g);
}

template auto gram_matrix<float>(ContextInternal& ctx, ConstHostView<std::complex<float>, 2> w,
                                 ConstHostView<float, 2> xyz, float wl,
                                 HostView<std::complex<float>, 2> g) -> void;

template auto gram_matrix<double>(ContextInternal& ctx, ConstHostView<std::complex<double>, 2> w,
                                  ConstHostView<double, 2> xyz, double wl,
                                  HostView<std::complex<double>, 2> g) -> void;

}  // namespace host
}  // namespace bipp
