#include "gpu/standard_synthesis.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <cstddef>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/kernels/scale_vector.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/kernels/apply_filter.hpp"
#include "memory/copy.hpp"

namespace bipp {
namespace gpu {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna,
                                        std::size_t nBeam, std::size_t nIntervals,
                                        HostArray<BippFilter, 1> filter, DeviceArray<T, 2> pixel)
    : ctx_(std::move(ctx)),
      nIntervals_(nIntervals),
      nFilter_(filter.size()),
      nPixel_(pixel.shape(0)),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      count_(0),
      filter_(std::move(filter)),
      pixel_(std::move(pixel)),
      img_(ctx_->gpu_queue().create_device_array<T, 3>({nPixel_, nIntervals_, nFilter_})) {
  auto& queue = ctx_->gpu_queue();
  api::memset_async(img_.data(), 0, img_.size() * sizeof(T), queue.stream());
}

template <typename T>
auto StandardSynthesis<T>::collect(
    T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
    ConstHostView<api::ComplexType<T>, 2> sHost, ConstDeviceView<api::ComplexType<T>, 2> s,
    ConstDeviceView<api::ComplexType<T>, 2> w, ConstDeviceView<T, 2> xyz) -> void {
  assert(xyz.shape(0) == nAntenna_);
  assert(xyz.shape(1) == 3);
  assert(w.shape(0) == nAntenna_);
  assert(w.shape(1) == nBeam_);
  assert(!s.size() || s.shape(0) == nBeam_);
  assert(!s.size() || s.shape(1) == nBeam_);

  auto& queue = ctx_->gpu_queue();
  auto vUnbeamArray = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna_, nBeam_});
  auto dArray = queue.create_device_array<T, 1>(nBeam_);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = queue.create_device_array<T, 2>(xyz.shape());
  copy(queue, xyz, xyzCentered);

  for (std::size_t i = 0; i < xyzCentered.shape(1); ++i) {
    center_vector<T>(queue, nAntenna_, xyzCentered.slice_view(i).data());
  }

  const auto nEig = eigh<T>(*ctx_, wl, s, w, xyzCentered, dArray, vUnbeamArray);

  auto d = dArray.sub_view(0, nEig);
  auto vUnbeam = vUnbeamArray.sub_view({0, 0}, {nAntenna_, nEig});

  auto dHostArray = queue.create_pinned_array<T, 1>(nEig);

  copy(queue, d, dHostArray);


  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "vUnbeam", vUnbeam);

  // Make sure D is available on host
  queue.sync();


  // callback for each level with eigenvalues
  auto dMaskedArray = queue.create_host_array<T, 2>({d.size(), nIntervals_});

  for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
    copy(dHostArray, dMaskedArray.slice_view(idxInt));
    eigMaskFunc(idxInt, nBeam_, dMaskedArray.slice_view(idxInt).data());
  }

  auto dCount = queue.create_host_array<short, 1>(d.size());
  dCount.zero();
  for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
    auto mask = dMaskedArray.slice_view(idxInt);
    for(std::size_t i = 0; i < mask.size(); ++i) {
      dCount[i] |= mask[i] != 0;
    }
  }

  // remove any eigenvalue that is zero for all levels
  // by copying forward
  std::size_t nEigRemoved = 0;
  for (std::size_t i = 0; i < nEig; ++i) {
    if(dCount[i]) {
      if(nEigRemoved) {
        copy(queue, vUnbeam.slice_view(i), vUnbeam.slice_view(i - nEigRemoved));
        for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
          dMaskedArray[{i - nEigRemoved, idxInt}] = dMaskedArray[{i, idxInt}];
        }
      }
    } else {
      ++nEigRemoved;
    }
  }

  const auto nEigMasked = nEig - nEigRemoved;

  auto unlayeredStats = queue.create_device_array<T, 2>({nPixel_, nEigMasked});
  auto dMasked = dMaskedArray.sub_view({0, 0}, {nEigMasked, dMaskedArray.shape(1)});


  T alpha = 2.0 * M_PI / wl;
  gemmexp<T>(queue, nEig, nPixel_, nAntenna_, alpha, vUnbeam.data(), vUnbeam.strides(1),
             xyzCentered.data(), xyzCentered.strides(1), pixel_.slice_view(0).data(),
             pixel_.slice_view(1).data(), pixel_.slice_view(2).data(), unlayeredStats.data(),
             unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.data(),
                            nPixel_);

  auto dFilteredHost = queue.create_host_array<T, 1>(nEigMasked);





  // cluster eigenvalues / vectors based on mask
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
      auto dMaskedSlice = dMasked.slice_view(idxInt);

      host::apply_filter(filter_[idxFilter], nEigMasked, dMaskedSlice.data(),
                         dFilteredHost.data());

      auto imgCurrent = img_.slice_view(idxFilter).slice_view(idxInt);
      for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
        if (dMaskedSlice[idxEig]) {
          const auto scale = dFilteredHost[idxEig];

          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                             "Assigning eigenvalue {} (filtered {}) to bin {}",
                             dMaskedSlice[{idxEig}], scale, idxInt);

          api::blas::axpy(queue.blas_handle(), nPixel_, &scale,
                          unlayeredStats.slice_view(idxEig).data(), 1, imgCurrent.data(), 1);
        }
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, DeviceView<T, 2> out) -> void {
  auto& queue = ctx_->gpu_queue();

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

  auto filterImg = img_.slice_view(index);

  const T scale = count_ ? static_cast<T>(1.0 / static_cast<double>(count_)) : 0;
  for (std::size_t i = 0; i < nIntervals_; ++i) {
    scale_vector<T>(queue.device_prop(), queue.stream(), nPixel_, filterImg.slice_view(i).data(),
                    scale, out.slice_view(i).data());
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", out.slice_view(i));
  }
}

template class StandardSynthesis<float>;
template class StandardSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
