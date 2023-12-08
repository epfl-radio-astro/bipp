#include "gpu/nufft_synthesis.hpp"

#include <cassert>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/gram_matrix.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/min_max_element.hpp"
#include "gpu/kernels/nuft_sum.hpp"
#include "gpu/kernels/scale_vector.hpp"
#include "gpu/nufft_3d3.hpp"
#include "gpu/util/device_accessor.hpp"
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#include "gpu/virtual_vis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace gpu {

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                  std::size_t nLevel, HostArray<BippFilter, 1> filter,
                                  DeviceArray<T, 2> pixel)

    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nLevel_(nLevel),
      nFilter_(filter.shape()),
      nPixel_(pixel.shape(0)),
      filter_(std::move(filter)),
      pixel_(std::move(pixel)),
      img_(ctx_->gpu_queue().create_device_array<T, 3>({nPixel_, nLevel_, nFilter_})),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      totalCollectCount_(0) {
  auto& queue = ctx_->gpu_queue();
  api::memset_async(img_.data(), 0, img_.size() * sizeof(T), queue.stream());

  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          imgPartition_ = DomainPartition::grid<T>(
              ctx_, arg.dimensions,
              {pixel.slice_view(0), pixel.slice_view(1), pixel.slice_view(2)});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(pixel_.slice_view(0));
  imgPartition_.apply(pixel_.slice_view(1));
  imgPartition_.apply(pixel_.slice_view(2));
}

template <typename T>
auto NufftSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  auto& queue = ctx_->gpu_queue();

  auto data = collector.get_data();
  if (data.empty()) return;

  std::size_t collectPoints = 0;
  for (const auto& s : data) {
    collectPoints += s.xyzUvw.shape(0);
    assert(s.v.shape(0) * s.v.shape(0) == s.xyzUvw.shape(0));
  }

  // compute virtual visibilities
  auto virtualVis =
      queue.create_device_array<api::ComplexType<T>, 3>({collectPoints, nLevel_, nFilter_});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      const auto nAntenna = s.v.shape(0);
      ConstDeviceAccessor<api::ComplexType<T>, 2> vDevice(
          queue, ConstView<api::ComplexType<T>, 2>(
                     reinterpret_cast<const api::ComplexType<T>*>(s.v.data()), s.v.shape(),
                     s.v.strides()));

      auto virtVisCurrent =
          virtualVis.sub_view({currentCount, 0, 0},
                              {nAntenna * nAntenna, virtualVis.shape(1), virtualVis.shape(2)});
      virtual_vis<T>(*ctx_, filter_, s.dMasked, vDevice.view(), virtVisCurrent);
      currentCount += s.xyzUvw.shape(0);
    }
  }

  // copy uvw into contiguous buffer
  auto uvw = queue.create_device_array<T, 2>({collectPoints, 3});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      copy(queue, s.xyzUvw, uvw.sub_view({currentCount, 0}, {s.xyzUvw.shape(0), 3}));
      currentCount += s.xyzUvw.shape(0);
    }
  }

  auto uvwX = uvw.slice_view(0).sub_view({0}, {collectPoints});
  auto uvwY = uvw.slice_view(1).sub_view({0}, {collectPoints});
  auto uvwZ = uvw.slice_view(2).sub_view({0}, {collectPoints});


  auto pixelX = pixel_.slice_view(0);
  auto pixelY = pixel_.slice_view(1);
  auto pixelZ = pixel_.slice_view(2);

  auto inputPartition = std::visit(
      [&](auto&& arg) -> DomainPartition {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          return DomainPartition::grid<T>(ctx_, arg.dimensions, {uvwX, uvwY, uvwZ});

        } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
          return DomainPartition::none(ctx_, collectPoints);

        } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
          auto minMaxDevice = queue.create_device_array<T, 1>(12);
          min_element(queue, collectPoints, uvwX.data(), minMaxDevice.data());
          max_element(queue, collectPoints, uvwX.data(), minMaxDevice.data() + 1);

          min_element(queue, collectPoints, uvwY.data(), minMaxDevice.data() + 2);
          max_element(queue, collectPoints, uvwY.data(), minMaxDevice.data() + 3);

          min_element(queue, collectPoints, uvwZ.data(), minMaxDevice.data() + 4);
          max_element(queue, collectPoints, uvwZ.data(), minMaxDevice.data() + 5);

          min_element(queue, nPixel_, pixelX.data(), minMaxDevice.data() + 6);
          max_element(queue, nPixel_, pixelX.data(), minMaxDevice.data() + 7);

          min_element(queue, nPixel_, pixelY.data(), minMaxDevice.data() + 8);
          max_element(queue, nPixel_, pixelY.data(), minMaxDevice.data() + 9);

          min_element(queue, nPixel_, pixelZ.data(), minMaxDevice.data() + 10);
          max_element(queue, nPixel_, pixelZ.data(), minMaxDevice.data() + 11);

          auto minMaxHost = queue.create_host_array<T, 1>(minMaxDevice.shape());
          copy(queue, minMaxDevice, minMaxHost);

          queue.sync();

          const T* minMaxPtr = minMaxHost.data();

          std::array<double, 3> uvwExtent = {minMaxPtr[1] - minMaxPtr[0],
                                             minMaxPtr[3] - minMaxPtr[2],
                                             minMaxPtr[5] - minMaxPtr[4]};
          std::array<double, 3> imgExtent = {minMaxPtr[7] - minMaxPtr[6],
                                             minMaxPtr[9] - minMaxPtr[8],
                                             minMaxPtr[11] - minMaxPtr[10]};

          // Use at most 12.5% of total memory for fft grid
          const auto gridSize = optimal_nufft_input_partition(
              uvwExtent, imgExtent,
              queue.device_prop().totalGlobalMem / (8 * sizeof(api::ComplexType<T>)));

          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})", gridSize[0],
                             gridSize[1], gridSize[2]);

          // set partition method to grid and create grid partition
          opt_.localUVWPartition.method = Partition::Grid{gridSize};
          return DomainPartition::grid<T>(ctx_, gridSize, {uvwX, uvwY, uvwZ});
        }
      },
      opt_.localUVWPartition.method);

  for (std::size_t i = 0; i < nFilter_; ++i) {
    for (std::size_t j = 0; j < nLevel_; ++j) {
      inputPartition.apply(virtualVis.slice_view(i).slice_view(j));
    }
  }

  inputPartition.apply(uvwX);
  inputPartition.apply(uvwY);
  inputPartition.apply(uvwZ);

  auto output = queue.create_device_array<api::ComplexType<T>, 1>(nPixel_);

  queue.signal_stream(nullptr);  // cufinufft uses default stream

  for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
    if (!inputSize) continue;
    for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
      if (!imgSize) continue;

      if (inputSize <= 1024) {
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
        // Direct evaluation of sum for small input sizes
        for (std::size_t i = 0; i < nFilter_; ++i) {
          for (std::size_t j = 0; j < nLevel_; ++j) {
            auto imgPtr = img_.slice_view(i).slice_view(j).data() + imgBegin;
            nuft_sum<T>(queue.device_prop(), nullptr, 1.0, inputSize,
                        virtualVis.slice_view(i).slice_view(j).data() + inputBegin,
                        uvwX.data() + inputBegin, uvwY.data() + inputBegin,
                        uvwZ.data() + inputBegin, imgSize, pixelX.data() + imgBegin,
                        pixelY.data() + imgBegin, pixelZ.data() + imgBegin, imgPtr);
          }
        }
      } else {
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, calling cuFINUFFT", inputSize);
        // Approximate sum through nufft
        Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwX.data() + inputBegin,
                              uvwY.data() + inputBegin, uvwZ.data() + inputBegin, imgSize,
                              pixelX.data() + imgBegin, pixelY.data() + imgBegin,
                              pixelZ.data() + imgBegin);

        for (std::size_t i = 0; i < nFilter_; ++i) {
          for (std::size_t j = 0; j < nLevel_; ++j) {
            auto imgPtr = img_.slice_view(i).slice_view(j).data() + imgBegin;
            transform.execute(virtualVis.slice_view(i).slice_view(j).data() + inputBegin,
                              output.data());
            ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output", imgSize, 1,
                                      output.data(), imgSize);

            add_vector_real_of_complex<T>(queue.device_prop(), nullptr, imgSize, output.data(),
                                          imgPtr);
          }
        }
      }
    }
  }

  queue.sync_with_stream(nullptr);

  totalCollectCount_ += data.size();
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, View<T, 2> out) -> void {
  auto& queue = ctx_->gpu_queue();


  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nLevel_);

  DeviceAccessor<T,2> outDevice(queue,out);

  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filter_.data();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  const T scale =
      totalCollectCount_ ? static_cast<T>(1.0 / static_cast<double>(totalCollectCount_)) : 0;

  auto filterImg = img_.slice_view(index);

  auto outDeviceView = outDevice.view();

  for (std::size_t i = 0; i < nLevel_; ++i) {
    auto levelImage = filterImg.slice_view(i);
    auto levelOut = outDeviceView.slice_view(i);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", levelImage);
    imgPartition_.reverse<T>(levelImage, levelOut);

    scale_vector<T>(queue.device_prop(), queue.stream(), nPixel_, scale, levelOut.data());
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", levelOut);
  }

  outDevice.copy_back(queue);
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace gpu
}  // namespace bipp
