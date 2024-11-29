#include "host/gram_matrix.hpp"

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/gram_matrix.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_guard.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/view.hpp"
#endif

namespace bipp {

template <typename T, typename>
BIPP_EXPORT auto gram_matrix(Context& ctx, std::size_t nAntenna, std::size_t nBeam, const std::complex<T>* w,
                             std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl,
                             std::complex<T>* g, std::size_t ldg) -> void {
  auto& ctxInternal = *InternalContextAccessor::get(ctx);
  if (ctxInternal.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    gpu::DeviceGuard deviceGuard(ctxInternal.device_id());

    auto& queue = ctxInternal.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    ConstDeviceAccessor<T, 2> xyzDevice(queue, xyz, {nAntenna, 3}, {1, ldxyz});
    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> wDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(w), {nAntenna, nBeam}, {1, ldw});
    DeviceAccessor<gpu::api::ComplexType<T>, 2> gDevice(
        queue, reinterpret_cast<gpu::api::ComplexType<T>*>(g), {nBeam, nBeam}, {1, ldg});

    // call gram on gpu
    gpu::gram_matrix<T>(ctxInternal, wDevice.view(), xyzDevice.view(), wl, gDevice.view());

    gDevice.copy_back(queue);

#else
    throw GPUSupportError();
#endif
  } else {
    host::gram_matrix<T>(ctxInternal, ConstHostView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                         ConstHostView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}), wl,
                         HostView<std::complex<T>, 2>(g, {nBeam, nBeam}, {1, ldg}));
  }
}

template auto gram_matrix(Context& ctx, std::size_t nAntenna, std::size_t nBeam, const std::complex<float>* w,
                          std::size_t ldw, const float* xyz, std::size_t ldxyz, float wl,
                          std::complex<float>* g, std::size_t ldg) -> void;

template auto gram_matrix(Context& ctx, std::size_t nAntenna, std::size_t nBeam, const std::complex<double>* w,
                          std::size_t ldw, const double* xyz, std::size_t ldxyz, double wl,
                          std::complex<double>* g, std::size_t ldg) -> void;

extern "C" {
BIPP_EXPORT BippError bipp_gram_matrix_f(BippContext ctx, size_t nAntenna, size_t nBeam, const void* w,
                                         size_t ldw, const float* xyz, size_t ldxyz, float wl,
                                         void* g, size_t ldg) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<float>(*reinterpret_cast<Context*>(ctx), nAntenna, nBeam,
                       reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz, wl,
                       reinterpret_cast<std::complex<float>*>(g), ldg);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_gram_matrix(BippContext ctx, size_t nAntenna, size_t nBeam, const void* w,
                                       size_t ldw, const double* xyz, size_t ldxyz, double wl,
                                       void* g, size_t ldg) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<double>(*reinterpret_cast<Context*>(ctx), nAntenna, nBeam,
                        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz, wl,
                        reinterpret_cast<std::complex<double>*>(g), ldg);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_GENERIC_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
