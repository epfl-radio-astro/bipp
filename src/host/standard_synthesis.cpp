#include "host/standard_synthesis.hpp"

#include <algorithm>
#include <complex>
#include <limits>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/kernels/apply_filter.hpp"
#include "host/kernels/gemmexp.hpp"
#include "host/kernels/interval_indices.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T>
static auto center_vector(std::size_t n, const T* __restrict__ in, T* __restrict__ out) -> void {
  T mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    mean += in[i];
  }
  mean /= n;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = in[i] - mean;
  }
}

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna,
                                        std::size_t nBeam, std::size_t nIntervals,
                                        std::size_t nFilter, const BippFilter* filter,
                                        std::size_t nPixel, const T* pixelX, const T* pixelY,
                                        const T* pixelZ)
    : ctx_(std::move(ctx)),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      count_(0) {
  filter_ = Buffer<BippFilter>(ctx_->host_alloc(), nFilter_);
  std::memcpy(filter_.get(), filter, sizeof(BippFilter) * nFilter_);
  pixelX_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(pixelX_.get(), pixelX, sizeof(T) * nPixel_);
  pixelY_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(pixelY_.get(), pixelY, sizeof(T) * nPixel_);
  pixelZ_ = Buffer<T>(ctx_->host_alloc(), nPixel_);
  std::memcpy(pixelZ_.get(), pixelZ, sizeof(T) * nPixel_);

  img_ = Buffer<T>(ctx_->host_alloc(), nPixel_ * nIntervals_ * nFilter_);
  std::memset(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T));
}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals,
                                   std::size_t ldIntervals, const std::complex<T>* s,
                                   std::size_t lds, const std::complex<T>* w, std::size_t ldw,
                                   const T* xyz, std::size_t ldxyz, const std::size_t nz_vis) -> void {
  auto v = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nEig);
  auto vUnbeam = Buffer<std::complex<T>>(ctx_->host_alloc(), nAntenna_ * nEig);
  auto unlayeredStats = Buffer<T>(ctx_->host_alloc(), nPixel_ * nEig);
  auto d = Buffer<T>(ctx_->host_alloc(), nEig);
  auto dFiltered = Buffer<T>(ctx_->host_alloc(), nEig);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = Buffer<T>(ctx_->host_alloc(), 3 * nAntenna_);
  center_vector(nAntenna_, xyz, xyzCentered.get());
  center_vector(nAntenna_, xyz + ldxyz, xyzCentered.get() + nAntenna_);
  center_vector(nAntenna_, xyz + 2 * ldxyz, xyzCentered.get() + 2 * nAntenna_);

  {
    auto g = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nBeam_);

    gram_matrix<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyzCentered.get(), nAntenna_, wl, g.get(),
                   nBeam_);

    // Note different order of s and g input
    if (s)
      eigh<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, d.get(), v.get(), nBeam_);
    else {
      eigh<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, d.get(), v.get(), nBeam_);
    }
  }

  blas::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nAntenna_, nEig, nBeam_, {1, 0}, w, ldw,
             v.get(), nBeam_, {0, 0}, vUnbeam.get(), nAntenna_);

  T alpha = 2.0 * M_PI / wl;

  gemmexp(nEig, nPixel_, nAntenna_, alpha, vUnbeam.get(), nAntenna_, xyzCentered.get(), nAntenna_,
          pixelX_.get(), pixelY_.get(), pixelZ_.get(), unlayeredStats.get(), nPixel_);
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.get(),
                            nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    apply_filter(filter_.get()[idxFilter], nEig, d.get(), dFiltered.get());
    for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices<T>(
          nEig, d.get(), intervals[idxInt * ldIntervals], intervals[idxInt * ldIntervals + 1]);

      auto imgCurrent = img_.get() + (idxFilter * nIntervals_ + idxInt) * nPixel_;
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = nz_vis > 0 ? dFiltered.get()[idxEig] / nz_vis : dFiltered.get()[idxEig];
        auto unlayeredStatsCurrent = unlayeredStats.get() + nPixel_ * idxEig;

        constexpr auto maxInt = static_cast<std::size_t>(std::numeric_limits<int>::max());

        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                           "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
                           *(d.get() + idxEig), scale, intervals[idxInt * ldIntervals],
                           intervals[idxInt * ldIntervals + 1]);

        for (std::size_t idxPix = 0; idxPix < nPixel_; idxPix += maxInt) {
          const auto nPixBlock = std::min(nPixel_ - idxPix, maxInt);
          blas::axpy(nPixBlock, scale, unlayeredStatsCurrent + idxPix, 1, imgCurrent + idxPix, 1);
        }
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filter_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    const T* __restrict__ localImg = img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_;
    T* __restrict__ outputImg = out + i * ld;
    const T scale = count_ ?  static_cast<T>(1.0 / static_cast<double>(count_)) : 0;

    for (std::size_t j = 0; j < nPixel_; ++j) {
      outputImg[j] = scale * localImg[j];
    }
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", nPixel_, 1, out + i * ld,
                              nPixel_);
  }
}

template class StandardSynthesis<double>;

template class StandardSynthesis<float>;

}  // namespace host
}  // namespace bipp
