#include "bipp/nufft_synthesis.hpp"

#include <chrono>
#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft_synthesis.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#endif

namespace bipp {

template <typename T>
struct NufftSynthesisInternal {
  NufftSynthesisInternal(const std::shared_ptr<ContextInternal>& ctx, NufftSynthesisOptions opt,
                         std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                         std::size_t nFilter, const BippFilter* filter, std::size_t nPixel,
                         const T* lmnX, const T* lmnY, const T* lmnZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals), nPixel_(nPixel) {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                       "{} NufftSynthesis.create({}, opt, {}, {} ,{} ,{} {}, {}, {}, {}, {})",
                       (const void*)this, (const void*)ctx_.get(), nAntenna, nBeam, nIntervals,
                       nFilter, (const void*)filter, nPixel, (const void*)lmnX, (const void*)lmnY,
                       (const void*)lmnZ);

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnX", nPixel_, 1, lmnX, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnY", nPixel_, 1, lmnY, nPixel_);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "lmnZ", nPixel_, 1, lmnZ, nPixel_);

    if (ctx_->processing_unit() == BIPP_PU_CPU) {
      planHost_.emplace(ctx_, opt, nAntenna, nBeam, nIntervals,
                        ConstHostView<BippFilter, 1>(filter, nFilter, 1),
                        ConstHostView<T, 1>(lmnX, nPixel, 1), ConstHostView<T, 1>(lmnY, nPixel, 1),
                        ConstHostView<T, 1>(lmnZ, nPixel, 1));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      auto filterArray = queue.create_host_array<BippFilter, 1>(nFilter);
      copy(queue, ConstView<BippFilter, 1>(filter, nFilter, 1), filterArray);
      queue.sync();  // make sure filters are available

      auto pixelArray = queue.create_device_array<T, 2>({nPixel_, 3});
      copy(queue, ConstView<T, 1>(lmnX, nPixel_, 1), pixelArray.slice_view(0));
      copy(queue, ConstView<T, 1>(lmnY, nPixel_, 1), pixelArray.slice_view(1));
      copy(queue, ConstView<T, 1>(lmnZ, nPixel_, 1), pixelArray.slice_view(2));

      planGPU_.emplace(ctx_, std::move(opt), nAntenna, nBeam, nIntervals, std::move(filterArray),
                       std::move(pixelArray));
#else
      throw GPUSupportError();
#endif
    }
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} NufftSynthesis created with Context {}",
                       static_cast<const void*>(this), (const void*)ctx_.get());
  }

  ~NufftSynthesisInternal() {
    try {
      ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} NufftSynthesis destroyed",
                         static_cast<const void*>(this));
    } catch (...) {
    }
  }

  void collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw) {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "------------");
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                       "{} NufftSynthesis.collect({}, {}, {}, {}, {} ,{} ,{} {}, {}, {}, {}, {})",
                       (const void*)this, nEig, wl, (const void*)intervals, ldIntervals,
                       (const void*)s, lds, (const void*)w, ldw, (const void*)xyz, ldxyz,
                       (const void*)uvw, lduvw);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "intervals", 2, nIntervals_, intervals,
                              ldIntervals);
    if (s) ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "S", nBeam_, nBeam_, s, lds);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "W", nAntenna_, nBeam_, w, ldw);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "XYZ", nAntenna_, 3, xyz, ldxyz);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "UVW", nAntenna_ * nAntenna_, 3, uvw, lduvw);

    const auto start = std::chrono::high_resolution_clock::now();

    if (planHost_) {
      auto& p = planHost_.value();
      planHost_.value().collect(
          nEig, wl, ConstHostView<T, 2>(intervals, {2, p.num_intervals()}, {1, ldIntervals}),
          s ? ConstHostView<std::complex<T>, 2>(s, {p.num_beam(), p.num_beam()}, {1, lds})
            : ConstHostView<std::complex<T>, 2>{},
          ConstHostView<std::complex<T>, 2>(w, {p.num_antenna(), p.num_beam()}, {1, ldw}),
          ConstHostView<T, 2>(xyz, {p.num_antenna(), 3}, {1, ldxyz}),
          ConstHostView<T, 2>(uvw, {p.num_antenna() * p.num_antenna(), 3}, {1, lduvw}));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      // Syncronize with default stream.
      queue.sync_with_stream(nullptr);
      // syncronize with stream to be synchronous with host before exiting
      auto syncGuard = queue.sync_guard();

      typename View<T, 2>::IndexType sShape = {0, 0};
      if (s) sShape = {nBeam_, nBeam_};

      ConstHostAccessor<T, 2> hostIntervals(queue, intervals, {2, nIntervals_}, {1, ldIntervals});
      ConstHostAccessor<gpu::api::ComplexType<T>, 2> sHost(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s), sShape, {1, lds});
      queue.sync();  // make sure it's accessible after construction

      ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> sDevice(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(s), sShape, {1, lds});
      ConstDeviceAccessor<gpu::api::ComplexType<T>, 2> wDevice(
          queue, reinterpret_cast<const gpu::api::ComplexType<T>*>(w), {nAntenna_, nBeam_},
          {1, ldw});
      ConstDeviceAccessor<T, 2> xyzDevice(queue, xyz, {nAntenna_, 3}, {1, ldxyz});
      ConstDeviceAccessor<T, 2> uvwDevice(queue, uvw, {nAntenna_ * nAntenna_, 3}, {1, lduvw});

      planGPU_->collect(nEig, wl, hostIntervals.view(), sHost.view(), sDevice.view(),
                        wDevice.view(), xyzDevice.view(), uvwDevice.view());
#else
      throw GPUSupportError();
#endif
    }

    const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "{} NufftSynthesis.collect() time: {}ms",
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

  auto get(BippFilter f, T* out, std::size_t ld) -> void {
    ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "{} NufftSynthesis.get({}, {}, {})", (const void*)this,
                       (int)f, (const void*)out, ld);
    if (planHost_) {
      auto& p = planHost_.value();
      p.get(f, HostView<T, 2>(out, {p.num_pixel(), p.num_intervals()}, {1, ld}));
    } else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      auto& queue = ctx_->gpu_queue();
      DeviceAccessor<T, 2> imgDevice(queue, out, {nPixel_, nIntervals_}, {1, ld});
      planGPU_->get(f, imgDevice.view());
      imgDevice.copy_back(queue);
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
NufftSynthesis<T>::NufftSynthesis(Context& ctx, NufftSynthesisOptions opt, std::size_t nAntenna,
                                  std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
                                  const BippFilter* filter, std::size_t nPixel, const T* lmnX,
                                  const T* lmnY, const T* lmnZ) {
  try {
    plan_ = decltype(plan_)(
        new NufftSynthesisInternal<T>(InternalContextAccessor::get(ctx), std::move(opt), nAntenna,
                                      nBeam, nIntervals, nFilter, filter, nPixel, lmnX, lmnY, lmnZ),
        [](auto&& ptr) { delete reinterpret_cast<NufftSynthesisInternal<T>*>(ptr); });
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "NufftSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
                                const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                                std::size_t ldw, const T* xyz, std::size_t ldxyz, const T* uvw,
                                std::size_t lduvw) -> void {
  try {
    reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())
        ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz, uvw, lduvw);
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())
          ->ctx_->logger()
          .log(BIPP_LOG_LEVEL_ERROR, "NufftSynthesis.collect() error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  try {
    reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())->get(f, out, ld);
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<NufftSynthesisInternal<T>*>(plan_.get())
          ->ctx_->logger()
          .log(BIPP_LOG_LEVEL_ERROR, "{} NufftSynthesis.get() error: {}", (const void*)plan_.get(),
               e.what());
    } catch (...) {
    }
    throw;
  }
}

template class BIPP_EXPORT NufftSynthesis<double>;

template class BIPP_EXPORT NufftSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_ns_options_create(BippNufftSynthesisOptions* opt) {
  try {
    *reinterpret_cast<NufftSynthesisOptions**>(opt) = new NufftSynthesisOptions();
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_destroy(BippNufftSynthesisOptions* opt) {
  if (!opt || !(*opt)) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<NufftSynthesisOptions**>(opt);
    *reinterpret_cast<Context**>(opt) = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_tolerance(BippNufftSynthesisOptions opt, float tol) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_tolerance(tol);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_collect_group_size(BippNufftSynthesisOptions opt,
                                                             size_t size) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_collect_group_size(size);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError
bipp_ns_options_set_local_image_partition_auto(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition({Partition::Auto{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError
bipp_ns_options_set_local_image_partition_none(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition({Partition::None{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_image_partition_grid(BippNufftSynthesisOptions opt,
                                                                     size_t dimX, size_t dimY,
                                                                     size_t dimZ) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition(
        {Partition::Grid{{dimX, dimY, dimZ}}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_auto(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition({Partition::Auto{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_none(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition({Partition::None{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_grid(BippNufftSynthesisOptions opt,
                                                                   size_t dimX, size_t dimY,
                                                                   size_t dimZ) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition(
        {Partition::Grid{{dimX, dimY, dimZ}}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_create_f(BippContext ctx, BippNufftSynthesisOptions opt,
                                                    size_t nAntenna, size_t nBeam,
                                                    size_t nIntervals, size_t nFilter,
                                                    const BippFilter* filter, size_t nPixel,
                                                    const float* lmnX, const float* lmnY,
                                                    const float* lmnZ, BippNufftSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesisInternal<float>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)),
        *reinterpret_cast<const NufftSynthesisOptions*>(opt), nAntenna, nBeam, nIntervals, nFilter,
        filter, nPixel, lmnX, lmnY, lmnZ);
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

BIPP_EXPORT BippError bipp_nufft_synthesis_create(BippContext ctx, BippNufftSynthesisOptions opt,
                                                  size_t nAntenna, size_t nBeam, size_t nIntervals,
                                                  size_t nFilter, const BippFilter* filter,
                                                  size_t nPixel, const double* lmnX,
                                                  const double* lmnY, const double* lmnZ,
                                                  BippNufftSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesisInternal<double>(
        InternalContextAccessor::get(*reinterpret_cast<Context*>(ctx)),
        *reinterpret_cast<const NufftSynthesisOptions*>(opt), nAntenna, nBeam, nIntervals, nFilter,
        filter, nPixel, lmnX, lmnY, lmnZ);
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
