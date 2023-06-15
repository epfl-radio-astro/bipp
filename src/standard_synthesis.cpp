#include "bipp/standard_synthesis.hpp"

#include <chrono>
#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/buffer.hpp"
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {

template <typename T>
struct StandardSynthesisInternal {
  StandardSynthesisInternal(const std::shared_ptr<ContextInternal>& ctx, std::size_t nAntenna,
                            std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                            const BippFilter* filter, std::size_t nPixel, const T* pixelX,
                            const T* pixelY, const T* pixelZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals), nPixel_(nPixel) {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                       "{} StandardSynthesis.create({}, opt, {}, {} ,{} ,{} {}, {}, {}, {}, {})",
                       (const void*)this, (const void*)ctx_.get(), nAntenna, nBeam, nIntervals,
                       nFilter, (const void*)filter, nPixel, (const void*)pixelX,
                       (const void*)pixelY, (const void*)pixelZ);

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnX", nPixel_, 1, pixelX, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnY", nPixel_, 1, pixelY, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnZ", nPixel_, 1, pixelZ, nPixel_);

    if (ctx_->processing_unit() == BIPP_PU_CPU) {
      planHost_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, pixelX, pixelY,
                        pixelZ);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      Buffer<T> pixelXBuffer, pixelYBuffer, pixelZBuffer;
      auto pixelXDevice = pixelX;
      auto pixelYDevice = pixelY;
      auto pixelZDevice = pixelZ;

      if (!gpu::is_device_ptr(pixelX)) {
        pixelXBuffer = queue.create_device_buffer<T>(nPixel);
        pixelXDevice = pixelXBuffer.get();
        gpu::api::memcpy_async(pixelXBuffer.get(), pixelX, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(pixelY)) {
        pixelYBuffer = queue.create_device_buffer<T>(nPixel);
        pixelYDevice = pixelYBuffer.get();
        gpu::api::memcpy_async(pixelYBuffer.get(), pixelY, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }
      if (!gpu::is_device_ptr(pixelZ)) {
        pixelZBuffer = queue.create_device_buffer<T>(nPixel);
        pixelZDevice = pixelZBuffer.get();
        gpu::api::memcpy_async(pixelZBuffer.get(), pixelZ, nPixel * sizeof(T),
                               gpu::api::flag::MemcpyHostToDevice, queue.stream());
      }

      planGPU_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter, nPixel, pixelXDevice,
                       pixelYDevice, pixelZDevice);
#else
      throw GPUSupportError();
#endif
    }
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis created with Context {}",
                       static_cast<const void*>(this), (const void*)ctx_.get());
  }

  ~StandardSynthesisInternal() {
    try{
      ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis destroyed",
                         static_cast<const void*>(this));
    } catch (...) {
    }
  }

  void collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz) {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "------------");
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                       "{} NufftSynthesis.collect({}, {}, {}, {}, {} ,{} ,{} {}, {}, {})",
                       (const void*)this, nEig, wl, (const void*)intervals, ldIntervals,
                       (const void*)s, lds, (const void*)w, ldw, (const void*)xyz, ldxyz);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "intervals", 2, nIntervals_, intervals, ldIntervals);
    if (s) ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "S", nBeam_, nBeam_, s, lds);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "W", nAntenna_, nBeam_, w, ldw);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "XYZ", nAntenna_, 3, xyz, ldxyz);

    const auto start = std::chrono::high_resolution_clock::now();

    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);

      Buffer<gpu::api::ComplexType<T>> wBuffer, sBuffer;
      Buffer<T> xyzBuffer;

      auto sDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(s);
      auto ldsDevice = lds;
      auto wDevice = reinterpret_cast<const gpu::api::ComplexType<T>*>(w);
      auto ldwDevice = ldw;

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

      // Always copy xyz, even when on device, since it will be overwritten
      xyzBuffer = queue.create_device_buffer<T>(3 * nAntenna_);
      auto ldxyzDevice = nAntenna_;
      auto xyzDevice = xyzBuffer.get();
      gpu::api::memcpy_2d_async(xyzBuffer.get(), nAntenna_ * sizeof(T), xyz, ldxyz * sizeof(T),
                                nAntenna_ * sizeof(T), 3, gpu::api::flag::MemcpyDefault,
                                queue.stream());

      planGPU_->collect(nEig, wl, intervals, ldIntervals, sDevice, ldsDevice, wDevice, ldwDevice,
                        xyzDevice, ldxyzDevice);
      queue.sync();
#else
      throw GPUSupportError();
#endif
    }

    const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} StandardSynthesis.collect() time: {}ms",
                       (const void*)this, time.count());

    if (ctx_->processing_unit() == BIPP_PU_CPU)
      ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} Context memory usage: host {}MB",
                         (const void*)ctx_.get(), ctx_->host_alloc()->size() / 1000000);
    else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      const auto queueMem = ctx_->gpu_queue().allocated_memory();
      ctx_->logger().log(
          BIPP_LOG_LEVEL_INFO, "{} Context memory usage: host {}MB, pinned {}MB, device {}MB",
          (const void*)ctx_.get(), (ctx_->host_alloc()->size() + std::get<0>(queueMem)) / 1000000,
          std::get<1>(queueMem) / 1000000, std::get<2>(queueMem) / 1000000);
#endif
    }
  }

  auto get(BippFilter f, T* img, std::size_t ld) -> void {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "{} StandardSynthesis.get({}, {}, {})",
                       (const void*)this, (int)f, (const void*)img, ld);
    if (planHost_) {
      planHost_.value().get(f, img, ld);
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      planGPU_->get(f, img, ld);
      ctx_->gpu_queue().sync();
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  std::size_t nAntenna_, nBeam_, nIntervals_, nPixel_;
  std::optional<host::StandardSynthesis<T>> planHost_;
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::StandardSynthesis<T>> planGPU_;
#endif
};

template <typename T>
StandardSynthesis<T>::StandardSynthesis(Context& ctx, std::size_t nAntenna, std::size_t nBeam,
                                        std::size_t nIntervals, std::size_t nFilter,
                                        const BippFilter* filter, std::size_t nPixel,
                                        const T* pixelX, const T* pixelY, const T* pixelZ) {
  try {
    plan_ = decltype(plan_)(
        new StandardSynthesisInternal<T>(InternalContextAccessor::get(ctx), nAntenna, nBeam,
                                         nIntervals, nFilter, filter, nPixel, pixelX, pixelY,
                                         pixelZ),
        [](auto&& ptr) { delete reinterpret_cast<StandardSynthesisInternal<T>*>(ptr); });
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "StandardSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals,
                                   std::size_t ldIntervals, const std::complex<T>* s,
                                   std::size_t lds, const std::complex<T>* w, std::size_t ldw,
                                   const T* xyz, std::size_t ldxyz) -> void {
  try {
  reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz);
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())
          ->ctx_->logger()
          .log(BIPP_LOG_LEVEL_ERROR, "{} StandardSynthesis.get() error: {}",
               (const void*)plan_.get(), e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  reinterpret_cast<StandardSynthesisInternal<T>*>(plan_.get())->get(f, out, ld);
}

template class BIPP_EXPORT StandardSynthesis<double>;

template class BIPP_EXPORT StandardSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_standard_synthesis_create_f(BippContext ctx, size_t nAntenna,
                                                       size_t nBeam, size_t nIntervals,
                                                       size_t nFilter, const BippFilter* filter,
                                                       size_t nPixel, const float* lmnX,
                                                       const float* lmnY, const float* lmnZ,
                                                       BippStandardSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<float>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nAntenna, nBeam, nIntervals,
        nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy_f(BippStandardSynthesisF* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesisInternal<float>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect_f(BippStandardSynthesisF plan, size_t nEig,
                                                        float wl, const float* intervals,
                                                        size_t ldIntervals, const void* s,
                                                        size_t lds, const void* w, size_t ldw,
                                                        const float* xyz, size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<float>*>(s), lds,
        reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get_f(BippStandardSynthesisF plan, BippFilter f,
                                                    float* img, size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_create(BippContext ctx, size_t nAntenna, size_t nBeam,
                                                     size_t nIntervals, size_t nFilter,
                                                     const BippFilter* filter, size_t nPixel,
                                                     const double* lmnX, const double* lmnY,
                                                     const double* lmnZ,
                                                     BippStandardSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesisInternal<double>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)), nAntenna, nBeam, nIntervals,
        nFilter, filter, nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy(BippStandardSynthesis* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesisInternal<double>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect(BippStandardSynthesis plan, size_t nEig,
                                                      double wl, const double* intervals,
                                                      size_t ldIntervals, const void* s, size_t lds,
                                                      const void* w, size_t ldw, const double* xyz,
                                                      size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->collect(
        nEig, wl, intervals, ldIntervals, reinterpret_cast<const std::complex<double>*>(s), lds,
        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get(BippStandardSynthesis plan, BippFilter f,
                                                  double* img, size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->get(f, img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
