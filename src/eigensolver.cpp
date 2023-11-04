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
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"
#endif

namespace bipp {
template <typename T, typename>
BIPP_EXPORT auto eigh(Context& ctx, std::size_t m, std::size_t nEig, const std::complex<T>* a,
                      std::size_t lda, const std::complex<T>* b, std::size_t ldb,
                      std::size_t* nEigOut, T* d, std::complex<T>* v, std::size_t ldv) -> void {
  auto& ctxInternal = *InternalContextAccessor::get(ctx);
  if (ctxInternal.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctxInternal.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    ConstHostAccessor<gpu::api::ComplexType<T>, 2> aHost(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(a), {m, m}, {1, lda});

    queue.sync(); // make sure a is on host

    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> aDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(a), {m, m}, {1, lda});
    ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> bDevice(
        queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(b), {b ? m : 0, b ? m : 0},
        {1, ldb});
    DeviceAccessor<gpu::api::ComplexType<T>, 2> vDevice(
        queue, reinterpret_cast<gpu::api::ComplexType<T>*>(v), {m, nEig}, {1, ldv});
    DeviceAccessor<T, 1> dDevice(queue, d, {nEig}, {1});

    // call eigh on GPU
    gpu::eigh<T>(ctxInternal, nEig, aHost.view(), aDevice.view(), bDevice.view(), dDevice.view(),
                 vDevice.view());

    dDevice.copy_back(queue);
    vDevice.copy_back(queue);
#else
    throw GPUSupportError();
#endif
  } else {
    host::eigh<T>(ctxInternal, nEig, ConstHostView<std::complex<T>, 2>(a, {m, m}, {1, lda}),
                  ConstHostView<std::complex<T>, 2>(b, {m, m}, {1, ldb}),
                  HostView<T, 1>(d, {nEig}, {1}),
                  HostView<std::complex<T>, 2>(v, {m, nEig}, {1, ldv}));
  }
  *nEigOut = nEig;
}

extern "C" {
BIPP_EXPORT BippError bipp_eigh_f(BippContext ctx, size_t m, size_t nEig, const void* a, size_t lda,
                                  const void* b, size_t ldb, size_t* nEigOut, float* d, void* v,
                                  size_t ldv) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    eigh<float>(*reinterpret_cast<Context*>(ctx), m, nEig,
                reinterpret_cast<const std::complex<float>*>(a), lda,
                reinterpret_cast<const std::complex<float>*>(b), ldb, nEigOut, d,
                reinterpret_cast<std::complex<float>*>(v), ldv);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_eigh(BippContext ctx, size_t m, size_t nEig, const void* a, size_t lda,
                                const void* b, size_t ldb, size_t* nEigOut, double* d, void* v,
                                size_t ldv) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    eigh<double>(*reinterpret_cast<Context*>(ctx), m, nEig,
                 reinterpret_cast<const std::complex<double>*>(a), lda,
                 reinterpret_cast<const std::complex<double>*>(b), ldb, nEigOut, d,
                 reinterpret_cast<std::complex<double>*>(v), ldv);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

template auto eigh<float, void>(Context& ctx, std::size_t m, std::size_t nEig,
                                const std::complex<float>* a, std::size_t lda,
                                const std::complex<float>* b, std::size_t ldb, std::size_t* nEigOut,
                                float* d, std::complex<float>* v, std::size_t ldv) -> void;

template auto eigh<double, void>(Context& ctx, std::size_t m, std::size_t nEig,
                                 const std::complex<double>* a, std::size_t lda,
                                 const std::complex<double>* b, std::size_t ldb,
                                 std::size_t* nEigOut, double* d, std::complex<double>* v,
                                 std::size_t ldv) -> void;

}  // namespace bipp
