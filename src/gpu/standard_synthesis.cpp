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
#include "gpu/kernels/center_vector.hpp"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/kernels/scale_vector.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/copy.hpp"

namespace bipp {
namespace gpu {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nLevel,
                                        DeviceArray<T, 2> pixel)
    : ctx_(std::move(ctx)),
      nImages_(nLevel),
      nPixel_(pixel.shape(0)),
      count_(0),
      pixel_(std::move(pixel)),
      img_(ctx_->gpu_queue().create_device_array<T, 2>({nPixel_, nImages_})) {
  auto& queue = ctx_->gpu_queue();
  api::memset_async(img_.data(), 0, img_.size() * sizeof(T), queue.stream());
}

template <typename T>
auto StandardSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  auto data = collector.get_data();

  for (const auto& s : data) {
    this->process_single(s.wl, s.v, s.dMasked, s.xyzUvw);
    ctx_->gpu_queue().sync();  // make sure memory inside process_single is available again
  }
}

template <typename T>
auto StandardSynthesis<T>::process_single(T wl, ConstView<std::complex<T>, 2> vView,
                                          ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvwView)
    -> void {
  const auto nAntenna = vView.shape(0);
  const auto nEig = dMasked.shape(0);

  assert(dMasked.shape(1) == nImages_);
  assert(vView.shape(1) == nEig);
  assert(xyzUvwView.shape(0) == nAntenna);
  assert(xyzUvwView.shape(1) == 3);

  auto& queue = ctx_->gpu_queue();

  auto v = queue.create_device_array<api::ComplexType<T>, 2>(vView.shape());
  copy(queue,
       ConstView<api::ComplexType<T>, 2>(
           reinterpret_cast<const gpu::api::ComplexType<T>*>(vView.data()), vView.shape(),
           vView.strides()),
       v);

  ConstDeviceAccessor<T, 2> xyzDevice(queue, xyzUvwView);
  auto xyz = xyzDevice.view();

  auto dMaskedArray = queue.create_host_array<T, 2>(dMasked.shape());
  copy(dMasked, dMaskedArray);

  auto dCount = queue.create_host_array<short, 1>(dMasked.shape(0));
  dCount.zero();
  for (std::size_t idxImage = 0; idxImage < nImages_; ++idxImage) {
    auto mask = dMaskedArray.slice_view(idxImage);
    for (std::size_t i = 0; i < mask.size(); ++i) {
      dCount[i] |= mask[i] != 0;
    }
  }

  // remove any eigenvalue that is zero for all level
  // by copying forward
  std::size_t nEigRemoved = 0;
  for (std::size_t i = 0; i < nEig; ++i) {
    if (dCount[i]) {
      if (nEigRemoved) {
        copy(queue, v.slice_view(i), v.slice_view(i - nEigRemoved));
        for (std::size_t idxImage = 0; idxImage < nImages_; ++idxImage) {
          dMaskedArray[{i - nEigRemoved, idxImage}] = dMaskedArray[{i, idxImage}];
        }
      }
    } else {
      ++nEigRemoved;
    }
  }

  const auto nEigMasked = nEig - nEigRemoved;

  auto unlayeredStats = queue.create_device_array<T, 2>({nPixel_, nEigMasked});
  auto dMaskedReduced = dMaskedArray.sub_view({0, 0}, {nEigMasked, dMaskedArray.shape(1)});

  T alpha = 2.0 * M_PI / wl;
  gemmexp<T>(queue, nEigMasked, nPixel_, nAntenna, alpha, v.data(), v.strides(1), xyz.data(),
             xyz.strides(1), pixel_.slice_view(0).data(), pixel_.slice_view(1).data(),
             pixel_.slice_view(2).data(), unlayeredStats.data(), unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", unlayeredStats);

  // cluster eigenvalues / vectors based on mask
  for (std::size_t idxImage = 0; idxImage < nImages_; ++idxImage) {
    auto dMaskedSlice = dMaskedReduced.slice_view(idxImage);

    auto imgCurrent = img_.slice_view(idxImage);
    for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
      if (dMaskedSlice[idxEig]) {
        const auto scale = dMaskedSlice[idxEig];

        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning eigenvalue {} (filtered {}) to bin {}",
                           dMaskedSlice[{idxEig}], scale, idxImage);

        api::blas::axpy(queue.blas_handle(), nPixel_, &scale,
                        unlayeredStats.slice_view(idxEig).data(), 1, imgCurrent.data(), 1);
      }
    }
  }

  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(View<T, 2> out) -> void {
  auto& queue = ctx_->gpu_queue();

  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nImages_);

  DeviceAccessor<T, 2> outDevice(queue, out);

  const T scale = count_ ? static_cast<T>(1.0 / static_cast<double>(count_)) : 0;
  for (std::size_t i = 0; i < nImages_; ++i) {
    scale_vector<T>(queue.device_prop(), queue.stream(), nPixel_, img_.slice_view(i).data(), scale,
                    outDevice.view().slice_view(i).data());
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", outDevice.view().slice_view(i));
  }

  outDevice.copy_back(queue);
}

template class StandardSynthesis<float>;
template class StandardSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
