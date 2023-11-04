#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/kernels/gram.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto gram_matrix(ContextInternal& ctx, ConstDeviceView<api::ComplexType<T>, 2> w,
                 ConstDeviceView<T, 2> xyz, T wl, DeviceView<api::ComplexType<T>, 2> g) -> void {
  const auto nAntenna= w.shape()[0];
  const auto nBeam= w.shape()[1];

  auto& queue = ctx.gpu_queue();

  auto buffer = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna, nAntenna});

  gram(queue, nAntenna, xyz.slice_view(0).data(), xyz.slice_view(1).data(),
       xyz.slice_view(2).data(), wl, buffer.data(), buffer.shape()[1]);

  api::ComplexType<T> alpha{1, 0};
  api::ComplexType<T> beta{0, 0};

  auto bufferC = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna, nBeam});

  api::blas::symm<api::ComplexType<T>>(queue.blas_handle(), api::blas::side::left,
                                       api::blas::fill::lower, alpha, buffer, w, beta, bufferC);
  api::blas::gemm<api::ComplexType<T>>(queue.blas_handle(),
                                       api::blas::operation::ConjugateTranspose,
                                       api::blas::operation::None, alpha, w, bufferC, beta, g);

  ctx.logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gram", g);
}

template auto gram_matrix<float>(ContextInternal& ctx,
                                 ConstDeviceView<api::ComplexType<float>, 2> w,
                                 ConstDeviceView<float, 2> xyz, float wl,
                                 DeviceView<api::ComplexType<float>, 2> g) -> void;

template auto gram_matrix<double>(ContextInternal& ctx,
                                  ConstDeviceView<api::ComplexType<double>, 2> w,
                                  ConstDeviceView<double, 2> xyz, double wl,
                                  DeviceView<api::ComplexType<double>, 2> g) -> void;

}  // namespace gpu
}  // namespace bipp
