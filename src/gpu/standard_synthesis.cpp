#include "gpu/standard_synthesis.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>

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
#include "host/kernels/interval_indices.hpp"
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
      nPixel_(pixel.shape()[0]),
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
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
                                   ConstHostView<api::ComplexType<T>, 2> sHost,
                                   ConstDeviceView<api::ComplexType<T>, 2> s,
                                   ConstDeviceView<api::ComplexType<T>, 2> w,
                                   ConstDeviceView<T, 2> xyz) -> void {
  assert(xyz.shape()[0] == nAntenna_);
  assert(xyz.shape()[1] == 3);
  assert(intervals.shape()[1] == nIntervals_);
  assert(intervals.shape()[0] == 2);
  assert(w.shape()[0] == nAntenna_);
  assert(w.shape()[1] == nBeam_);
  assert(!s.size() || s.shape()[0] == nBeam_);
  assert(!s.size() || s.shape()[1] == nBeam_);

  auto& queue = ctx_->gpu_queue();
  auto v = queue.create_device_array<api::ComplexType<T>, 2>({nBeam_, nEig});
  auto vUnbeam = queue.create_device_array<api::ComplexType<T>, 2>({nAntenna_, nEig});

  auto unlayeredStats = queue.create_device_array<T, 2>({nPixel_, nEig});

  auto d = queue.create_device_array<T, 1>(nEig);
  auto dFiltered = queue.create_device_array<T, 1>(nEig);

  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = queue.create_device_array<T, 2>(xyz.shape());
  copy(queue, xyz, xyzCentered);

  for (std::size_t i = 0; i < xyzCentered.shape()[1]; ++i) {
    center_vector<T>(queue, nAntenna_, xyzCentered.slice_view(i).data());
  }

  {
    auto g = queue.create_device_array<api::ComplexType<T>, 2>({nBeam_, nBeam_});

    gram_matrix<T>(*ctx_, w, xyzCentered, wl, g);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s.size())
      eigh<T>(*ctx_, nEig, sHost, s, g, d, v);
    else {
      auto gHost = queue.create_pinned_array<api::ComplexType<T>, 2>(g.shape());
      copy(queue, g, gHost);
      queue.sync();  // finish copy
      eigh<T>(*ctx_, nEig, gHost, g, s, d, v);
    }
  }

  auto dHost = queue.create_pinned_array<T, 1>(nEig);
  auto dFilteredHost = queue.create_host_array<T, 1>(nEig);

  copy(queue, d, dHost);
  // Make sure D is available on host
  queue.sync();

  api::ComplexType<T> one{1, 0};
  api::ComplexType<T> zero{0, 0};
  api::blas::gemm(queue.blas_handle(), api::blas::operation::None, api::blas::operation::None, one,
                  w, v, zero, vUnbeam);

  T alpha = 2.0 * M_PI / wl;
  gemmexp<T>(queue, nEig, nPixel_, nAntenna_, alpha, vUnbeam.data(), vUnbeam.strides()[1],
             xyzCentered.data(), xyzCentered.strides()[1], pixel_.slice_view(0).data(),
             pixel_.slice_view(1).data(), pixel_.slice_view(2).data(), unlayeredStats.data(),
             nPixel_);
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", nPixel_, nEig, unlayeredStats.data(),
                            nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < static_cast<std::size_t>(nFilter_); ++idxFilter) {
    host::apply_filter(filter_[{idxFilter}], nEig, dHost.data(), dFilteredHost.data());

    for (std::size_t idxInt = 0; idxInt < static_cast<std::size_t>(nIntervals_); ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = host::find_interval_indices(
          nEig, dHost.data(), intervals[{0, idxInt}], intervals[{1, idxInt}]);

      auto imgCurrent = img_.slice_view(idxFilter).slice_view(idxInt);
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = dFilteredHost[{idxEig}];

        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                           "Assigning eigenvalue {} (filtered {}) to inverval [{}, {}]",
                           dHost[{idxEig}], scale, intervals[{0, idxInt}], intervals[{1, idxInt}]);

        api::blas::axpy(queue.blas_handle(), nPixel_, &scale,
                        unlayeredStats.slice_view(idxEig).data(), 1, imgCurrent.data(), 1);
      }
    }
  }
  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(BippFilter f, DeviceView<T, 2> out) -> void {
  auto& queue = ctx_->gpu_queue();

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
