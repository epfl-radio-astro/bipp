#include "host/gram_matrix.hpp"

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/bipp.hpp"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/gram_matrix.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"
#endif

namespace bipp {

template <typename T, typename>
BIPP_EXPORT auto gram_matrix(Context& ctx, std::size_t m, std::size_t n, const std::complex<T>* w,
                             std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl,
                             std::complex<T>* g, std::size_t ldg) -> void {
  auto& ctxInternal = *InternalContextAccessor::get(ctx);
  if (ctxInternal.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctxInternal.gpu_queue();
    // Syncronize with default stream.
    queue.sync_with_stream(nullptr);
    // syncronize with stream to be synchronous with host before exiting
    auto syncGuard = queue.sync_guard();

    Buffer<gpu::api::ComplexType<T>> wBuffer, gBuffer;
    Buffer<T> xyzBuffer;
    auto wDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(w);
    auto gDevice = reinterpret_cast<gpu::api::ComplexType<T>*>(g);
    auto xyzDevice = xyz;
    std::size_t ldwDevice = ldw;
    std::size_t ldgDevice = ldg;
    std::size_t ldxyzDevice = ldxyz;

    // copy input if required
    if (!gpu::is_device_ptr(w)) {
      wBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(m * n);
      ldwDevice = m;
      wDevice = wBuffer.get();
      gpu::api::memcpy_2d_async(wBuffer.get(), m * sizeof(gpu::api::ComplexType<T>), w,
                                ldw * sizeof(gpu::api::ComplexType<T>),
                                m * sizeof(gpu::api::ComplexType<T>), n,
                                gpu::api::flag::MemcpyDefault, queue.stream());
    }

    if (!gpu::is_device_ptr(xyz)) {
      xyzBuffer = queue.create_device_buffer<T>(3 * m);
      ldxyzDevice = m;
      xyzDevice = xyzBuffer.get();
      gpu::api::memcpy_2d_async(xyzBuffer.get(), m * sizeof(T), xyz, ldxyz * sizeof(T),
                                m * sizeof(T), 3, gpu::api::flag::MemcpyDefault, queue.stream());
    }
    if (!gpu::is_device_ptr(g)) {
      gBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(n * n);
      ldgDevice = n;
      gDevice = gBuffer.get();
    }

    // call gram on gpu
    gpu::gram_matrix<T>(ctxInternal, m, n, wDevice, ldwDevice, xyzDevice, ldxyzDevice, wl, gDevice,
                        ldgDevice);

    // copy back results if required
    if (gBuffer) {
      gpu::api::memcpy_2d_async(g, ldg * sizeof(gpu::api::ComplexType<T>), gBuffer.get(),
                                n * sizeof(gpu::api::ComplexType<T>),
                                n * sizeof(gpu::api::ComplexType<T>), n,
                                gpu::api::flag::MemcpyDefault, queue.stream());
    }
#else
    throw GPUSupportError();
#endif
  } else {
    host::gram_matrix<T>(ctxInternal, m, n, w, ldw, xyz, ldxyz, wl, g, ldg);
  }
}

template auto gram_matrix(Context& ctx, std::size_t m, std::size_t n, const std::complex<float>* w,
                          std::size_t ldw, const float* xyz, std::size_t ldxyz, float wl,
                          std::complex<float>* g, std::size_t ldg) -> void;

template auto gram_matrix(Context& ctx, std::size_t m, std::size_t n, const std::complex<double>* w,
                          std::size_t ldw, const double* xyz, std::size_t ldxyz, double wl,
                          std::complex<double>* g, std::size_t ldg) -> void;

extern "C" {
BIPP_EXPORT BippError bipp_gram_matrix_f(BippContext ctx, size_t m, size_t n, const void* w,
                                         size_t ldw, const float* xyz, size_t ldxyz, float wl,
                                         void* g, size_t ldg) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<float>(*reinterpret_cast<Context*>(ctx), m, n,
                       reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz, wl,
                       reinterpret_cast<std::complex<float>*>(g), ldg);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_gram_matrix(BippContext ctx, size_t m, size_t n, const void* w,
                                       size_t ldw, const double* xyz, size_t ldxyz, double wl,
                                       void* g, size_t ldg) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<double>(*reinterpret_cast<Context*>(ctx), m, n,
                        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz, wl,
                        reinterpret_cast<std::complex<double>*>(g), ldg);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
