#include "gpu/standard_synthesis.hpp"

#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <cassert>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/interval_indices.hpp"

namespace bipp {
namespace gpu {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna,
                                        std::size_t nBeam, std::size_t nIntervals,
                                        std::size_t nFilter, const BippFilter* filterHost,
                                        std::size_t nPixel, const T* pixelX, const T* pixelY,
                                        const T* pixelZ, const bool filter_negative_eigenvalues)
    : ctx_(std::move(ctx)),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      filter_negative_eigenvalues_(filter_negative_eigenvalues) {
  auto& queue = ctx_->gpu_queue();
  filterHost_ = queue.create_host_buffer<BippFilter>(nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BippFilter) * nFilter_);
  pixelX_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(pixelX_.get(), pixelX, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  pixelY_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(pixelY_.get(), pixelY, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());
  pixelZ_ = queue.create_device_buffer<T>(nPixel_);
  api::memcpy_async(pixelZ_.get(), pixelZ, sizeof(T) * nPixel_, api::flag::MemcpyDeviceToDevice,
                    queue.stream());

  img_ = queue.create_device_buffer<T>(nPixel_ * nIntervals_ * nFilter_);
  api::memset_async(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T), queue.stream());
}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervalsHost,
                                   std::size_t ldIntervals, const api::ComplexType<T>* s,
                                   std::size_t lds, const api::ComplexType<T>* w, std::size_t ldw,
                                   T* xyz, std::size_t ldxyz, const std::size_t nz_vis) -> void {
  auto& queue = ctx_->gpu_queue();
  auto v = queue.create_device_buffer<api::ComplexType<T>>(nBeam_ * nEig);
  auto d = queue.create_device_buffer<T>(nEig);
  auto vUnbeam = queue.create_device_buffer<api::ComplexType<T>>(nAntenna_ * nEig);
  auto unlayeredStats = queue.create_device_buffer<T>(nPixel_ * nEig);

  // Center coordinates for much better performance of cos / sin
  for (std::size_t i = 0; i < 3; ++i) {
    center_vector<T>(queue, nAntenna_, xyz + i * ldxyz);
  }

  auto g = queue.create_device_buffer<api::ComplexType<T>>(nBeam_ * nBeam_);

  gram_matrix<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyz, ldxyz, wl, g.get(), nBeam_);

  std::size_t nEigOut = 0;
  
  char range = filter_negative_eigenvalues_ ? 'V' : 'A';

  // Note different order of s and g input
  if (s)
    eigh<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, range, &nEigOut, d.get(), v.get(), nBeam_);
  else
    eigh<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, range, &nEigOut, d.get(), v.get(), nBeam_);

  if (not filter_negative_eigenvalues_)
      assert (nEig == nEigOut);


  auto DBufferHost = queue.create_pinned_buffer<T>(nEig);
  auto DFilteredBufferHost = queue.create_host_buffer<T>(nEig);
  api::memcpy_async(DBufferHost.get(), d.get(), nEig * sizeof(T), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  // Make sure D is available on host
  queue.sync();

  api::ComplexType<T> one{1, 0};
  api::ComplexType<T> zero{0, 0};
  api::blas::gemm(queue.blas_handle(), api::blas::operation::None, api::blas::operation::None,
                  nAntenna_, nEig, nBeam_, &one, w, ldw, v.get(), nBeam_, &zero, vUnbeam.get(),
                  nAntenna_);

  T alpha = 2.0 * M_PI / wl;
  gemmexp<T>(queue, nEig, nPixel_, nAntenna_, alpha, vUnbeam.get(), nAntenna_, xyz, ldxyz,
             pixelX_.get(), pixelY_.get(), pixelZ_.get(), unlayeredStats.get(), nPixel_);
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.get(),
                            nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < static_cast<std::size_t>(nFilter_); ++idxFilter) {
    host::apply_filter(filterHost_.get()[idxFilter], nEig, DBufferHost.get(),
                       DFilteredBufferHost.get());

    for (std::size_t idxInt = 0; idxInt < static_cast<std::size_t>(nIntervals_); ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = host::find_interval_indices(
          nEig, DBufferHost.get(), intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals)],
          intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals) + 1]);

      auto imgCurrent = img_.get() + (idxFilter * nIntervals_ + idxInt) * nPixel_;
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        ctx_->logger().log(
            BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
            *(DBufferHost.get() + idxEig), *(DFilteredBufferHost.get() + idxEig),
            intervalsHost[idxInt * ldIntervals], intervalsHost[idxInt * ldIntervals + 1]);
        const auto scale = nz_vis > 0 ?  DFilteredBufferHost.get()[idxEig] / nz_vis : DFilteredBufferHost.get()[idxEig];
        auto unlayeredStatsCurrent = unlayeredStats.get() + nPixel_ * idxEig;
        api::blas::axpy(queue.blas_handle(), nPixel_, &scale, unlayeredStatsCurrent, 1, imgCurrent,
                        1);
      }
    }
  }
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, T* outHostOrDevice, std::size_t ld) -> void {
  auto& queue = ctx_->gpu_queue();
  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filterHost_.get();
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    if (filterPtr[idxFilter] == f) {
      index = idxFilter;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  api::memcpy_2d_async(outHostOrDevice, ld * sizeof(T), img_.get() + index * nIntervals_ * nPixel_,
                       nPixel_ * sizeof(T), nPixel_ * sizeof(T), nIntervals_,
                       api::flag::MemcpyDefault, queue.stream());
  for (std::size_t i = 0; i < nIntervals_; ++i) {
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", nPixel_, 1,
                              outHostOrDevice + i * ld, nPixel_);
  }
}

template class StandardSynthesis<float>;
template class StandardSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
