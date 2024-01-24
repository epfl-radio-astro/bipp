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
#include "memory/allocator.hpp"
#include "memory/copy.hpp"
#include "memory/array.hpp"

namespace bipp {
namespace host {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nImages,
                                        ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                        ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      nImages_(nImages),
      nPixel_(pixelX.size()),
      count_(0),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nImages_}) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  copy(pixelX, pixel_.slice_view(0));
  copy(pixelY, pixel_.slice_view(1));
  copy(pixelZ, pixel_.slice_view(2));
  img_.zero();
}

template <typename T>
auto StandardSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  auto data = collector.get_data();

  for (const auto& s : data) {
    this->process_single(s.wl, s.v, s.dMasked, s.xyzUvw);
  }
}

template <typename T>
auto StandardSynthesis<T>::process_single(T wl, ConstView<std::complex<T>, 2> vView,
                                          ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvwView)
    -> void {
  HostArray<std::complex<T>, 2> v(ctx_->host_alloc(),vView.shape());
  copy(ConstHostView<std::complex<T>, 2>(vView), v);
  ConstHostView<T, 2> xyz(xyzUvwView);

  assert(xyz.shape(1) == 3);
  assert(v.shape(0) == xyz.shape(0));
  assert(v.shape(1) == dMasked.shape(0));
  assert(img_.shape(1) == dMasked.shape(1));

  const auto nEig = dMasked.shape(0);
  const auto nAntenna = v.shape(0);

  HostArray<T, 2> dMaskedArray(ctx_->host_alloc(), dMasked.shape());
  copy(dMasked, dMaskedArray);

  auto dCount = HostArray<short, 1>(ctx_->host_alloc(), dMasked.shape(0));
  dCount.zero();
  for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
    auto mask = dMaskedArray.slice_view(idxLevel);
    for(std::size_t i = 0; i < mask.size(); ++i) {
      dCount[i] |= mask[i] != 0;
    }
  }

  // remove any eigenvalue that is zero for all level
  // by copying forward
  std::size_t nEigRemoved = 0;
  for (std::size_t i = 0; i < nEig; ++i) {
    if(dCount[i]) {
      if(nEigRemoved) {
        copy(v.slice_view(i), v.slice_view(i - nEigRemoved));
        for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
          dMaskedArray[{i - nEigRemoved, idxLevel}] = dMaskedArray[{i, idxLevel}];
        }
      }
    } else {
      ++nEigRemoved;
    }
  }

  const auto nEigMasked = nEig - nEigRemoved;


  auto unlayeredStats = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEigMasked});
  auto dMaskedReduced = dMaskedArray.sub_view({0, 0}, {nEigMasked, dMaskedArray.shape(1)});

  T alpha = 2.0 * M_PI / wl;

  gemmexp(nEigMasked, nPixel_, nAntenna, alpha, v.data(), v.strides(1), xyz.data(), xyz.strides(1),
          &pixel_[{0, 0}], &pixel_[{0, 1}], &pixel_[{0, 2}], unlayeredStats.data(),
          unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", unlayeredStats);


  // cluster eigenvalues / vectors based on mask
  for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
    auto dMaskedSlice = dMaskedReduced.slice_view(idxLevel).sub_view(0, nEigMasked);

    auto imgCurrent = img_.slice_view(idxLevel);
    for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
      if (dMaskedSlice[idxEig]) {
        const auto scale = dMaskedSlice[idxEig];

        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} (filtered {}) to bin {}",
                           dMaskedSlice[{idxEig}], scale, idxLevel);

        blas::axpy(nPixel_, scale, &unlayeredStats[{0, idxEig}], 1, imgCurrent.data(), 1);
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(View<T, 2> out) -> void {
  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nImages_);

  HostView<T, 2> outHost(out);

  for (std::size_t i = 0; i < nImages_; ++i) {
    const T* __restrict__ localImg = &img_[{0, i}];
    T* __restrict__ outputImg = &outHost[{0, i}];
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
