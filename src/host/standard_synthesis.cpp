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
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx,
                                        std::size_t nLevel, ConstHostView<BippFilter, 1> filter,
                                        ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                        ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      nLevel_(nLevel),
      nFilter_(filter.size()),
      nPixel_(pixelX.size()),
      count_(0),
      filter_(ctx_->host_alloc(), filter.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nLevel_, nFilter_}) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  copy(filter, filter_);
  copy(pixelX, pixel_.slice_view(0));
  copy(pixelY, pixel_.slice_view(1));
  copy(pixelZ, pixel_.slice_view(2));
  img_.zero();
}

template <typename T>
auto StandardSynthesis<T>::collect(
    T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
    ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
    ConstHostView<T, 2> xyz) -> void {

  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);

  assert(xyz.shape(0) == nAntenna);
  assert(xyz.shape(1) == 3);
  assert(s.shape(0) == nBeam);
  assert(s.shape(1) == nBeam);


  auto vUnbeamArray = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nAntenna, nBeam});

  auto dArray = HostArray<T, 1>(ctx_->host_alloc(), nBeam);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = HostArray<T, 2>(ctx_->host_alloc(), {nAntenna, 3});
  center_vector(nAntenna, xyz.slice_view(0).data(), xyzCentered.data());
  center_vector(nAntenna, xyz.slice_view(1).data(), xyzCentered.slice_view(1).data());
  center_vector(nAntenna, xyz.slice_view(2).data(), xyzCentered.slice_view(2).data());


  const auto nEig = eigh<T>(*ctx_, wl, s, w, xyzCentered, dArray, vUnbeamArray);

  auto d = dArray.sub_view(0, nEig);
  auto vUnbeam = vUnbeamArray.sub_view({0, 0}, {nAntenna, nEig});

  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "vUnbeam", vUnbeam);


  // callback for each level with eigenvalues
  auto dMaskedArray = HostArray<T, 2>(ctx_->host_alloc(), {d.size(), nLevel_});

  for (std::size_t idxLevel = 0; idxLevel < nLevel_; ++idxLevel) {
    copy(d, dMaskedArray.slice_view(idxLevel));
    eigMaskFunc(idxLevel, nBeam, dMaskedArray.slice_view(idxLevel).data());
  }

  auto dCount = HostArray<short, 1>(ctx_->host_alloc(), d.size());
  dCount.zero();
  for (std::size_t idxLevel = 0; idxLevel < nLevel_; ++idxLevel) {
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
        copy(vUnbeam.slice_view(i), vUnbeam.slice_view(i - nEigRemoved));
        for (std::size_t idxLevel = 0; idxLevel < nLevel_; ++idxLevel) {
          dMaskedArray[{i - nEigRemoved, idxLevel}] = dMaskedArray[{i, idxLevel}];
        }
      }
    } else {
      ++nEigRemoved;
    }
  }

  const auto nEigMasked = nEig - nEigRemoved;

  auto unlayeredStats = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEigMasked});
  auto dMasked = dMaskedArray.sub_view({0, 0}, {nEigMasked, dMaskedArray.shape(1)});

  T alpha = 2.0 * M_PI / wl;

  gemmexp(nEigMasked, nPixel_, nAntenna, alpha, vUnbeam.data(), vUnbeam.strides(1), xyzCentered.data(),
          xyzCentered.strides(1), &pixel_[{0, 0}], &pixel_[{0, 1}], &pixel_[{0, 2}],
          unlayeredStats.data(), unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", unlayeredStats);

  auto dFiltered = HostArray<T, 1>(ctx_->host_alloc(), nEigMasked);

  // cluster eigenvalues / vectors based on mask
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    for (std::size_t idxLevel = 0; idxLevel < nLevel_; ++idxLevel) {
      auto dMaskedSlice = dMasked.slice_view(idxLevel);

      apply_filter(filter_[idxFilter], nEigMasked, dMaskedSlice.data(), dFiltered.data());

      auto imgCurrent = img_.slice_view(idxFilter).slice_view(idxLevel);
      for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
        if (dMaskedSlice[idxEig]) {
          const auto scale = dFiltered[idxEig];

          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                             "Assigning eigenvalue {} (filtered {}) to bin {}",
                             dMaskedSlice[{idxEig}], scale, idxLevel);

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
  assert(out.shape(1) == nLevel_);

  std::size_t index = nFilter_;
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filter_[{i}] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  for (std::size_t i = 0; i < nLevel_; ++i) {
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
