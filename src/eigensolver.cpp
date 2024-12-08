#include "host/eigensolver.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "bipp/context.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/eigensolver.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_guard.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T, typename>
BIPP_EXPORT auto eigh(Context& ctx, T wl, std::size_t nAntenna, std::size_t nBeam,
                      const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                      std::size_t ldw, const T* xyz, std::size_t ldxyz, T* d, std::complex<T>* v,
                      std::size_t ldv) -> std::pair<std::size_t, std::size_t> {
  auto& ctxInternal = *InternalContextAccessor::get(ctx);
  std::pair<std::size_t, std::size_t> pev {0, 0};
  if (ctxInternal.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctxInternal.device_id());

    auto& queue = ctxInternal.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> sDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s), {nBeam, nBeam}, {1, lds});
    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> wDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(w), {nAntenna, nBeam}, {1, ldw});
    ConstDeviceAccessor<T, 2> xyzDevice(queue, xyz, {nAntenna, 3}, {1, ldxyz});
    DeviceAccessor<T, 1> dDevice(queue, d, nBeam, 1);

    // call eigh on GPU
    pev = gpu::eigh<T>(ctxInternal, wl, sDevice.view(), wDevice.view(), xyzDevice.view(), dDevice.view());

    dDevice.copy_back(queue);
#else
    throw GPUSupportError();
#endif
  } else {
    pev = host::eigh<T>(
        ctxInternal, wl, ConstHostView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
        ConstHostView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
        ConstHostView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}), HostView<T, 1>(d, nBeam, 1),
        HostView<std::complex<T>, 2>(v, {nAntenna, nBeam}, {1, ldv}));
  }

  return pev;
}

template auto eigh<float, void>(Context& ctx, float wl, std::size_t nAntenna, std::size_t nBeam,
                                const std::complex<float>* s, std::size_t lds,
                                const std::complex<float>* w, std::size_t ldw, const float* xyz,
                                std::size_t ldxyz, float* d, std::complex<float>* v,
                                std::size_t ldv) -> std::pair<std::size_t, std::size_t>;

template auto eigh<double, void>(Context& ctx, double wl, std::size_t nAntenna, std::size_t nBeam,
                                 const std::complex<double>* s, std::size_t lds,
                                 const std::complex<double>* w, std::size_t ldw, const double* xyz,
                                 std::size_t ldxyz, double* d, std::complex<double>* v,
                                 std::size_t ldv) -> std::pair<std::size_t, std::size_t>;

}  // namespace bipp
