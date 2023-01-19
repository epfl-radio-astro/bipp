#include "gpu/virtual_vis.hpp"

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/interval_indices.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto virtual_vis(ContextInternal& ctx, std::size_t nFilter, const BippFilter* filterHost,
                 std::size_t nIntervals, const T* intervalsHost, std::size_t ldIntervals,
                 std::size_t nEig, const T* D, std::size_t nAntenna, const api::ComplexType<T>* V,
                 std::size_t ldv, std::size_t nBeam, const api::ComplexType<T>* W, std::size_t ldw,
                 api::ComplexType<T>* virtVis, std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                 std::size_t ldVirtVis3) -> void {
  using ComplexType = api::ComplexType<T>;
  auto& queue = ctx.gpu_queue();

  const auto zero = ComplexType{0, 0};
  const auto one = ComplexType{1, 0};

  Buffer<api::ComplexType<T>> VUnbeamBuffer;
  if (W) {
    VUnbeamBuffer = queue.create_device_buffer<api::ComplexType<T>>(nAntenna * nEig);

    api::blas::gemm(queue.blas_handle(), api::blas::operation::None, api::blas::operation::None,
                    nAntenna, nEig, nBeam, &one, W, ldw, V, ldv, &zero, VUnbeamBuffer.get(),
                    nAntenna);
    V = VUnbeamBuffer.get();
    ldv = nAntenna;
  }
  // V is alwayts of shape (nAntenna, nEig) from here on

  auto VMulDBuffer = queue.create_device_buffer<api::ComplexType<T>>(nEig * nAntenna);

  auto DBufferHost = queue.create_pinned_buffer<T>(nEig);
  auto DFilteredBuffer = queue.create_device_buffer<T>(nEig);

  api::memcpy_async(DBufferHost.get(), D, nEig * sizeof(T), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  // Make sure D is available on host
  queue.sync();

  for (std::size_t i = 0; i < static_cast<std::size_t>(nFilter); ++i) {
    apply_filter(queue, filterHost[i], nEig, D, DFilteredBuffer.get());

    for (std::size_t j = 0; j < static_cast<std::size_t>(nIntervals); ++j) {
      std::size_t start, size;
      std::tie(start, size) = host::find_interval_indices(
          nEig, DBufferHost.get(), intervalsHost[j * static_cast<std::size_t>(ldIntervals)],
          intervalsHost[j * static_cast<std::size_t>(ldIntervals) + 1]);

      auto virtVisCurrent = virtVis + i * static_cast<std::size_t>(ldVirtVis1) +
                            j * static_cast<std::size_t>(ldVirtVis2);
      if (size) {
        // Multiply each col of V with the selected eigenvalue
        scale_matrix<T>(queue, nAntenna, size, V + start * ldv, ldv, DFilteredBuffer.get() + start,
                        VMulDBuffer.get(), nAntenna);

        // Matrix multiplication of the previously scaled V and the original V
        // with the selected eigenvalues
        api::blas::gemm(queue.blas_handle(), api::blas::operation::None,
                        api::blas::operation::ConjugateTranspose, nAntenna, nAntenna, size, &one,
                        VMulDBuffer.get(), nAntenna, V + start * ldv, ldv, &zero, virtVisCurrent,
                        ldVirtVis3);

      } else {
        api::memset_2d_async(virtVisCurrent, ldVirtVis3 * sizeof(ComplexType), 0,
                             nAntenna * sizeof(ComplexType), nAntenna, queue.stream());
      }
    }
  }
}

template auto virtual_vis<float>(
    ContextInternal& ctx, std::size_t nFilter, const BippFilter* filter, std::size_t nIntervals,
    const float* intervals, std::size_t ldIntervals, std::size_t nEig, const float* D,
    std::size_t nAntenna, const api::ComplexType<float>* V, std::size_t ldv, std::size_t nBeam,
    const api::ComplexType<float>* W, std::size_t ldw, api::ComplexType<float>* virtVis,
    std::size_t ldVirtVis1, std::size_t ldVirtVis2, std::size_t ldVirtVis3) -> void;

template auto virtual_vis<double>(
    ContextInternal& ctx, std::size_t nFilter, const BippFilter* filter, std::size_t nIntervals,
    const double* intervals, std::size_t ldIntervals, std::size_t nEig, const double* D,
    std::size_t nAntenna, const api::ComplexType<double>* V, std::size_t ldv, std::size_t nBeam,
    const api::ComplexType<double>* W, std::size_t ldw, api::ComplexType<double>* virtVis,
    std::size_t ldVirtVis1, std::size_t ldVirtVis2, std::size_t ldVirtVis3) -> void;

}  // namespace gpu
}  // namespace bipp
