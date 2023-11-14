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
auto StandardSynthesis<T>::collect(T wl, std::function<void(std::size_t, std::size_t, T*)> eigMaskFunc,
                                   ConstHostView<std::complex<T>, 2> s,
                                   ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz)
    -> void {
  assert(xyz.shape(0) == nAntenna_);
  assert(xyz.shape(1) == 3);
  assert(intervals.shape(1) == nIntervals_);
  assert(intervals.shape(0) == 2);
  assert(w.shape(0) == nAntenna_);
  assert(w.shape(1) == nBeam_);
  assert(!s.size() || s.shape(0) == nBeam_);
  assert(!s.size() || s.shape(1) == nBeam_);

  const auto nEig = nBeam_; // TODO remove

  auto vUnbeamArray = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nAntenna_, nBeam_});

  auto dArray = HostArray<T, 1>(ctx_->host_alloc(), nBeam_);
  auto dFiltered = HostArray<T, 1>(ctx_->host_alloc(), nEig);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = HostArray<T, 2>(ctx_->host_alloc(), {nAntenna_, 3});
  center_vector(nAntenna_, xyz.slice_view(0).data(), xyzCentered.data());
  center_vector(nAntenna_, xyz.slice_view(1).data(), xyzCentered.slice_view(1).data());
  center_vector(nAntenna_, xyz.slice_view(2).data(), xyzCentered.slice_view(2).data());


  eigh<T>(*ctx_, wl, s, w, xyzCentered, dArray, vUnbeamArray);

  auto vUnbeam = vUnbeamArray.sub_view({0, nBeam_ - nEig}, {nAntenna_, nEig});
  auto d = dArray.sub_view(nBeam_ - nEig, nEig);

  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "vUnbeam", vUnbeam);

  auto dMasked = HostArray<T, 2>(ctx_->host_alloc(), {d.size(), nIntervals_});

  auto start = std::chrono::high_resolution_clock::now();
  for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
    copy(d, dMasked.slice_view(idxInt));
    eigMaskFunc(idxInt, nBeam_, dMasked.slice_view(idxInt).data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  // const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
  //     std::chrono::high_resolution_clock::now() - start);
  const std::chrono::duration<double> elapsed_seconds{end - start};
  printf("callback time [s]: %f\n", elapsed_seconds.count());

  // TODO: gather only selected eigenvalues to reduce amount of calc in gemmexp

  auto unlayeredStats = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEig});

  T alpha = 2.0 * M_PI / wl;

  gemmexp(nEig, nPixel_, nAntenna_, alpha, vUnbeam.data(), vUnbeam.strides(1), xyzCentered.data(),
          xyzCentered.strides(1), &pixel_[{0, 0}], &pixel_[{0, 1}], &pixel_[{0, 2}],
          unlayeredStats.data(), unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.data(),
                            nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
      auto dMaskedSlice = dMasked.slice_view(idxInt);

      apply_filter(filter_[{idxFilter}], nEig, dMaskedSlice.data(), dFiltered.data());

      auto imgCurrent = img_.slice_view(idxFilter).slice_view(idxInt);
      for (std::size_t idxEig = 0; idxEig < nBeam_; ++idxEig) {
        if (dMaskedSlice[idxEig]) {
          const auto scale = dFiltered[{idxEig}];

          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                             "Assigning eigenvalue {} (filtered {}) to bin {}",
                             dMaskedSlice[{idxEig}], scale, idxInt);

          blas::axpy(nPixel_, scale, &unlayeredStats[{0, idxEig}], 1, imgCurrent.data(), 1);
        }
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, HostView<T, 2> out) -> void {
  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nIntervals_);

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
