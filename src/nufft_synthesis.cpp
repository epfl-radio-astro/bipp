#include "bipp/nufft_synthesis.hpp"

#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft_synthesis.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"
#endif

namespace bipp {

template <typename T>
struct NufftSynthesisInternal {
  NufftSynthesisInternal(const std::shared_ptr<ContextInternal>& ctx, T tol, std::size_t nAntenna,
                         std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                         const BippFilter* filter, std::size_t nPixel, const T* lmnX, const T* lmnY,
                         const T* lmnZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals), nPixel_(nPixel) {
    if (ctx_->processing_unit() == BIPP_PU_CPU) {
      planHost_.emplace(ctx_, tol, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, lmnX, lmnY,
                        lmnZ);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      Buffer<T> lmnXBuffer, lmnYBuffer, lmnZBuffer;
      auto lmnXDevice = lmnX;
      auto lmnYDevice = lmnY;
      auto lmnZDevice = lmnZ;

      if (!gpu::is_device_ptr(lmnX)) {
        lmnXBuffer = queue.create_device_buffer<T>(nPixel);
        lmnXDevice = lmnXBuffer.get();
        gpu::api::memcpy_async(lmnXBuffer.get(), lmnX, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(lmnY)) {
        lmnYBuffer = queue.create_device_buffer<T>(nPixel);
        lmnYDevice = lmnYBuffer.get();
        gpu::api::memcpy_async(lmnYBuffer.get(), lmnY, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(lmnZ)) {
        lmnZBuffer = queue.create_device_buffer<T>(nPixel);
        lmnZDevice = lmnZBuffer.get();
        gpu::api::memcpy_async(lmnZBuffer.get(), lmnZ, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }

      planGPU_.emplace(ctx_, tol, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, lmnXDevice,
                       lmnYDevice, lmnZDevice);
#else
      throw GPUSupportError();
#endif
    }
  }

  void collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw) {
    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz, uvw,
                                lduvw);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      queue.sync_with_stream(nullptr);
      queue.sync();  // make sure unused allocated memory is available

      Buffer<gpu::api::ComplexType<T>> wBuffer, sBuffer;
      Buffer<T> xyzBuffer, uvwBuffer;

      auto sDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(s);
      auto ldsDevice = lds;
      auto wDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(w);
      auto ldwDevice = ldw;
      auto xyzDevice = xyz;
      auto ldxyzDevice = ldxyz;
      auto uvwDevice = uvw;
      auto lduvwDevice = lduvw;

      if (s && !gpu::is_device_ptr(s)) {
        sBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(nBeam_ * nBeam_);
        ldsDevice = nBeam_;
        sDevice = sBuffer.get();
        gpu::api::memcpy_2d_async(sBuffer.get(), nBeam_ * sizeof(gpu::api::ComplexType<T>), s,
                                  lds * sizeof(gpu::api::ComplexType<T>),
                                  nBeam_ * sizeof(gpu::api::ComplexType<T>), nBeam_,
                                  gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(w)) {
        wBuffer = queue.create_device_buffer<gpu::api::ComplexType<T>>(nAntenna_ * nBeam_);
        ldwDevice = nAntenna_;
        wDevice = wBuffer.get();
        gpu::api::memcpy_2d_async(wBuffer.get(), nAntenna_ * sizeof(gpu::api::ComplexType<T>), w,
                                  ldw * sizeof(gpu::api::ComplexType<T>),
                                  nAntenna_ * sizeof(gpu::api::ComplexType<T>), nBeam_,
                                  gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(xyz)) {
        xyzBuffer = queue.create_device_buffer<T>(3 * nAntenna_);
        ldxyzDevice = nAntenna_;
        xyzDevice = xyzBuffer.get();
        gpu::api::memcpy_2d_async(xyzBuffer.get(), nAntenna_ * sizeof(T), xyz, ldxyz * sizeof(T),
                                  nAntenna_ * sizeof(T), 3, gpu::api::flag::MemcpyHostToDevice,
                                  queue.stream());
      }
      if (!gpu::is_device_ptr(uvw)) {
        uvwBuffer = queue.create_device_buffer<T>(3 * nAntenna_ * nAntenna_);
        uvwDevice = uvwBuffer.get();
        lduvwDevice = nAntenna_ * nAntenna_;
        gpu::api::memcpy_2d_async(uvwBuffer.get(), nAntenna_ * nAntenna_ * sizeof(T), uvw,
                                  lduvw * sizeof(T), nAntenna_ * nAntenna_ * sizeof(T), 3,
                                  gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }

      // sync before call, such that host memory can be safely discarded by
      // caller, while computation is continued asynchronously
      queue.sync();

      planGPU_->collect(nEig, wl, intervals, ldIntervals, sDevice, ldsDevice, wDevice, ldwDevice,
                        xyzDevice, ldxyzDevice, uvwDevice, lduvwDevice);
#else
      throw GPUSupportError();
#endif
    }
  }

  auto get(BippFilter f, T* out, std::size_t ld) -> void {
    if (planHost_) {
      planHost_.value().get(f, out, ld);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      planGPU_->get(f, out, ld);
      ctx_->gpu_queue().sync();
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  std::size_t nAntenna_, nBeam_, nIntervals_, nPixel_;
  std::optional<host::NufftSynthesis<T>> planHost_;
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::NufftSynthesis<T>> planGPU_;
#endif
};

template <typename T>
NufftSynthesis<T>::NufftSynthesis(Context& ctx, T tol, std::size_t nAntenna, std::size_t nBeam,
                                  std::size_t nIntervals, std::size_t nFilter,
                                  const BippFilter* filter, std::size_t nPixel, const T* lmnX,
                                  const T* lmnY, const T* lmnZ)
    : plan_(new NufftSynthesisInternal<T>(InternalContextAccessor::get(ctx), tol, nAntenna, nBeam,
                                          nIntervals, nFilter, filter, nPixel, lmnX, lmnY, lmnZ),
            [](auto&& ptr) { delete reinterpret_cast<NufftSynthesisInternal<T>*>(ptr); }) {}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
                                const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                                std::size_t ldw, const T* xyz, std::size_t ldxyz, const T* uvw,
                                std::size_t lduvw) -> void {
  reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz, uvw, lduvw);
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())->get(f, out, ld);
}

template class BIPP_EXPORT NufftSynthesis<double>;

template class BIPP_EXPORT NufftSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_nufft_synthesis_create_f(BippContext ctx, float tol, size_t nAntenna,
                                                    size_t nBeam, size_t nIntervals, size_t nFilter,
                                                    const BippFilter* filter, size_t nPixel,
                                                    const float* lmnX, const float* lmnY,
                                                    const float* lmnZ, BippNufftSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesisInternal<float>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), tol, nAntenna, nBeam,
        nIntervals, nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_destroy_f(BippNufftSynthesisF* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<NufftSynthesisInternal<float>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_collect_f(BippNufftSynthesisF plan, size_t nEig,
                                                     float wl, const float* intervals,
                                                     size_t ldIntervals, const void* s, size_t lds,
                                                     const void* w, size_t ldw, const float* xyz,
                                                     size_t ldxyz, const float* uvw, size_t lduvw) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<float>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<float>*>(s), lds,
        reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz, uvw, lduvw);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_get_f(BippNufftSynthesisF plan, BippFilter f, float* img,
                                                 size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<float>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_create(BippContext ctx, double tol, size_t nAntenna,
                                                  size_t nBeam, size_t nIntervals, size_t nFilter,
                                                  const BippFilter* filter, size_t nPixel,
                                                  const double* lmnX, const double* lmnY,
                                                  const double* lmnZ, BippNufftSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesisInternal<double>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), tol, nAntenna, nBeam,
        nIntervals, nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_destroy(BippNufftSynthesis* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<NufftSynthesisInternal<double>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_collect(BippNufftSynthesis plan, size_t nEig, double wl,
                                                   const double* intervals, size_t ldIntervals,
                                                   const void* s, size_t lds, const void* w,
                                                   size_t ldw, const double* xyz, size_t ldxyz,
                                                   const double* uvw, size_t lduvw) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<double>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<double>*>(s), lds,
        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz, uvw, lduvw);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_get(BippNufftSynthesis plan, BippFilter f, double* img,
                                               size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<double>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
