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

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/eigensolver.hpp"
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

    Buffer<gpu::api::ComplexType<T>> aBuffer, bBuffer, vBuffer;
    Buffer<T> dBuffer;
    auto aDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(a);
    auto bDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(b);
    auto dDevice = d;
    auto vDevice = reinterpret_cast<gpu::api::ComplexType<T>*>(v);
    std::size_t ldaDevice = lda;
    std::size_t ldbDevice = ldb;
    std::size_t ldvDevice = ldv;

    // copy input if required
    if (!gpu::is_device_ptr(a)) {
      aBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(m * m);
      ldaDevice = m;
      aDevice = aBuffer.get();
      gpu::api::memcpy_2d_async(aBuffer.get(), ldaDevice * sizeof(gpu::api::ComplexType<T>), a,
                                lda * sizeof(gpu::api::ComplexType<T>),
                                m * sizeof(gpu::api::ComplexType<T>), m,
                                gpu::api::flag::MemcpyHostToDevice, queue.stream());
    }

    if (b && !gpu::is_device_ptr(b)) {
      bBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(m * m);
      ldbDevice = m;
      bDevice = bBuffer.get();
      gpu::api::memcpy_2d_async(bBuffer.get(), ldbDevice * sizeof(gpu::api::ComplexType<T>), b,
                                ldb * sizeof(gpu::api::ComplexType<T>),
                                m * sizeof(gpu::api::ComplexType<T>), m,
                                gpu::api::flag::MemcpyHostToDevice, queue.stream());
    }

    if (!gpu::is_device_ptr(v)) {
      vBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(nEig * m);
      ldvDevice = m;
      vDevice = vBuffer.get();
    }

    if (!gpu::is_device_ptr(d)) {
      dBuffer = queue.create_device_buffer<T>(nEig);
      dDevice = dBuffer.get();
    }

    // call eigh on GPU
    gpu::eigh<T>(ctxInternal, m, nEig, aDevice, ldaDevice, bDevice, ldbDevice, nEigOut, dDevice,
                 vDevice, ldvDevice);

    // copy back results if required
    if (vBuffer) {
      gpu::api::memcpy_2d_async(v, ldv * sizeof(gpu::api::ComplexType<T>), vBuffer.get(),
                                ldvDevice * sizeof(gpu::api::ComplexType<T>),
                                m * sizeof(gpu::api::ComplexType<T>), nEig,
                                gpu::api::flag::MemcpyDeviceToHost, queue.stream());
    }
    if (dBuffer) {
      gpu::api::memcpy_async(d, dDevice, nEig * sizeof(T), gpu::api::flag::MemcpyDeviceToHost,
                             queue.stream());
    }

#else
    throw GPUSupportError();
#endif
  } else {
    host::eigh<T>(ctxInternal, m, nEig, a, lda, b, ldb, nEigOut, d, v, ldv);
  }
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
