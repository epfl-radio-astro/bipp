#include "host/standard_synthesis.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <cassert>

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
#include "memory/copy.hpp"
#include "memory/array.hpp"

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
                                        ConstHostView<BippFilter, 1> filter,
                                        ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                        ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      nIntervals_(nIntervals),
      nFilter_(filter.size()),
      nPixel_(pixelX.size()),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      count_(0),
      filter_(ctx_->host_alloc(), filter.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nIntervals_, nFilter_}) {

  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  copy(filter, filter_);
  copy(pixelX, pixel_.slice_view(0));
  copy(pixelY, pixel_.slice_view(1));
  copy(pixelZ, pixel_.slice_view(2));
  img_.zero();
}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
                                   ConstHostView<std::complex<T>, 2> s,
                                   ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz)
    -> void {
  assert(xyz.shape()[0] == nAntenna_);
  assert(xyz.shape()[1] == 3);
  assert(intervals.shape()[1] == nIntervals_);
  assert(intervals.shape()[0] == 2);
  assert(w.shape()[0] == nAntenna_);
  assert(w.shape()[1] == nBeam_);
  assert(!s.size() || s.shape()[0] == nBeam_);
  assert(!s.size() || s.shape()[1] == nBeam_);

  auto v = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nBeam_, nEig});
  auto vUnbeam = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nAntenna_, nEig});

  auto unlayeredStats = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEig});

  auto d = HostArray<T, 1>(ctx_->host_alloc(), nEig);
  auto dFiltered = HostArray<T, 1>(ctx_->host_alloc(), nEig);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = HostArray<T, 2>(ctx_->host_alloc(), {nAntenna_, 3});
  center_vector(nAntenna_, xyz.slice_view(0).data(), xyzCentered.data());
  center_vector(nAntenna_, xyz.slice_view(1).data(), xyzCentered.slice_view(1).data());
  center_vector(nAntenna_, xyz.slice_view(2).data(), xyzCentered.slice_view(2).data());

  {
    auto g = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nBeam_, nBeam_});

    gram_matrix<T>(*ctx_, w, xyzCentered, wl, g);

    // Note different order of s and g input
    if (s.size())
      eigh<T>(*ctx_, nEig, s, g, d, v);
    else {
      eigh<T>(*ctx_, nEig, g, s, d, v);
    }
  }

  blas::gemm<std::complex<T>>(CblasNoTrans, CblasNoTrans, {1, 0}, w, v, {0, 0}, vUnbeam);

  T alpha = 2.0 * M_PI / wl;

  gemmexp(nEig, nPixel_, nAntenna_, alpha, vUnbeam.data(), nAntenna_, xyzCentered.data(), nAntenna_,
          &pixel_[{0, 0}], &pixel_[{0, 1}], &pixel_[{0, 2}], unlayeredStats.data(), nPixel_);
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.data(),
                            nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    apply_filter(filter_[{idxFilter}], nEig, d.data(), dFiltered.data());
    for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) =
          find_interval_indices<T>(nEig, d.data(), intervals[{0, idxInt}], intervals[{1, idxInt}]);

      auto imgCurrent = img_.slice_view(idxFilter).slice_view(idxInt);
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = dFiltered[{idxEig}];

        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                           "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
                           d[{idxEig}], scale, intervals[{0, idxInt}], intervals[{1, idxInt}]);

        blas::axpy(nPixel_, scale, &unlayeredStats[{0, idxEig}], 1, imgCurrent.data(), 1);
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, HostView<T, 2> out) -> void {
  assert(out.shape()[0] == nPixel_);
  assert(out.shape()[1] == nIntervals_);

  std::size_t index = nFilter_;
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filter_[{i}] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    const T* __restrict__ localImg = &img_[{0, i, index}];
    T* __restrict__ outputImg = &out[{0, i}];
    const T scale = count_ ? static_cast<T>(1.0 / static_cast<double>(count_)) : 0;

    for (std::size_t j = 0; j < nPixel_; ++j) {
      outputImg[j] = scale * localImg[j];
    }
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", out.slice_view(i));
  }
}

template class StandardSynthesis<double>;

template class StandardSynthesis<float>;

}  // namespace host
}  // namespace bipp
