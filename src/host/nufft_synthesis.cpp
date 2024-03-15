#include "host/nufft_synthesis.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "host/blas_api.hpp"
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/kernels/nuft_sum.hpp"
#include "host/nufft_3d3.hpp"
#include "host/virtual_vis.hpp"
#include "memory/copy.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace host {

static auto system_memory() -> unsigned long long {
  unsigned long long pages = sysconf(_SC_PHYS_PAGES);
  unsigned long long pageSize = sysconf(_SC_PAGE_SIZE);
  unsigned long long memory = pages * pageSize;
  return memory > 0 ? memory : 8ull * 1024ull * 1024ull * 1024ull;
}

template <typename T>
NufftSynthesis<T>::NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                                  std::size_t nImages, ConstHostView<T, 1> pixelX,
                                  ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nImages_(nImages),
      nPixel_(pixelX.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nImages_}),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      totalCollectCount_(0),
      totalVisibilityCount_(0) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  img_.zero();

  // Only partition image if explicitly set. Auto defaults to no partition.
  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: grid ({}, {}, {})",
                             arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
          imgPartition_ =
              DomainPartition::grid<T, 3>(ctx_, arg.dimensions, {pixelX, pixelY, pixelZ});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(pixelX, pixel_.slice_view(0));
  imgPartition_.apply(pixelY, pixel_.slice_view(1));
  imgPartition_.apply(pixelZ, pixel_.slice_view(2));
}

template <typename T>
auto NufftSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "computing nufft for collected data");

  auto data = collector.get_data();
  if (data.empty()) return;

  std::size_t collectPoints = 0;
  std::size_t visibilityCount = 0;
  for (const auto& s : data) {
    collectPoints += s.xyzUvw.shape(0);
    visibilityCount += s.nVis;
    assert(s.v.shape(0) * s.v.shape(0) == s.xyzUvw.shape(0));
  }

  // compute virtual visibilities
  HostArray<std::complex<T>, 2> virtualVis(ctx_->host_alloc(), {collectPoints, nImages_});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      const auto nAntenna = s.v.shape(0);
      auto virtVisCurrent =
          virtualVis.sub_view({currentCount, 0}, {nAntenna * nAntenna, virtualVis.shape(1)});
      virtual_vis<T>(*ctx_, s.dMasked, ConstHostView<std::complex<T>, 2>(s.v), virtVisCurrent);
      currentCount += s.xyzUvw.shape(0);
    }
  }

  // copy uvw into contiguous buffer
  HostArray<T, 2> uvw(ctx_->host_alloc(), {collectPoints, 3});
  {
    std::size_t currentCount = 0;
    for (const auto& s : data) {
      copy(ConstHostView<T, 2>(s.xyzUvw), uvw.sub_view({currentCount, 0}, {s.xyzUvw.shape(0), 3}));
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
          return DomainPartition::grid<T, 3>(ctx_, arg.dimensions, {uvwX, uvwY, uvwZ});
        } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
          return DomainPartition::none(ctx_, collectPoints);

        } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
          std::array<double, 3> uvwExtent{};
          std::array<double, 3> imgExtent{};

          auto minMaxIt = std::minmax_element(uvwX.data(), uvwX.data() + collectPoints);
          uvwExtent[0] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(uvwY.data(), uvwY.data() + collectPoints);
          uvwExtent[1] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(uvwZ.data(), uvwZ.data() + collectPoints);
          uvwExtent[2] = *minMaxIt.second - *minMaxIt.first;

          minMaxIt = std::minmax_element(pixelX.data(), pixelX.data() + nPixel_);
          imgExtent[0] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(pixelY.data(), pixelY.data() + nPixel_);
          imgExtent[1] = *minMaxIt.second - *minMaxIt.first;
          minMaxIt = std::minmax_element(pixelZ.data(), pixelZ.data() + nPixel_);
          imgExtent[2] = *minMaxIt.second - *minMaxIt.first;

          // Use at most 12.5% of total memory for fft grid
          const auto gridSize = optimal_nufft_input_partition(
              uvwExtent, imgExtent, system_memory() / (8 * sizeof(std::complex<T>)));

          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})", gridSize[0],
                             gridSize[1], gridSize[2]);

          // set partition method to grid and create grid partition
          opt_.localUVWPartition.method = Partition::Grid{gridSize};
          return DomainPartition::grid<T>(ctx_, gridSize, {uvwX, uvwY, uvwZ});
        }
      },
      opt_.localUVWPartition.method);

  for (std::size_t j = 0; j < nImages_; ++j) {
    inputPartition.apply(virtualVis.slice_view(j));
  }

  inputPartition.apply(uvwX);
  inputPartition.apply(uvwY);
  inputPartition.apply(uvwZ);
  auto output = HostArray<std::complex<T>, 1>(ctx_->host_alloc(), nPixel_);

  for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
    if (!inputSize) continue;
    auto uvwXSlice = uvwX.sub_view(inputBegin, inputSize);
    auto uvwYSlice = uvwY.sub_view(inputBegin, inputSize);
    auto uvwZSlice = uvwZ.sub_view(inputBegin, inputSize);

    if (inputSize <= 32) {
      // Direct evaluation of sum for small input sizes
      ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
      for (std::size_t j = 0; j < nImages_; ++j) {
        auto virtVisCurrentSlice =
            virtualVis.slice_view(j).sub_view(inputBegin, inputSize);
        auto* imgPtr = &img_[{0, j}];
        nuft_sum<T>(1.0, inputSize, virtVisCurrentSlice.data(), uvwXSlice.data(), uvwYSlice.data(),
                    uvwZSlice.data(), img_.shape(0), pixelX.data(), pixelY.data(), pixelZ.data(),
                    imgPtr);
      }
    } else {
      // Compute Nufft for each input and output partition combination
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        auto pixelXSlice = pixelX.sub_view(imgBegin, imgSize);
        auto pixelYSlice = pixelY.sub_view(imgBegin, imgSize);
        auto pixelZSlice = pixelZ.sub_view(imgBegin, imgSize);

        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate x", uvwXSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate y", uvwYSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input coordinate z", uvwZSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate x", pixelXSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate y", pixelYSlice);
        ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output coordinate z", pixelZSlice);

        // Approximate sum through nufft
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, calling fiNUFFT", inputSize);
        Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwXSlice.data(), uvwYSlice.data(),
                              uvwZSlice.data(), imgSize, pixelXSlice.data(), pixelYSlice.data(),
                              pixelZSlice.data());

          for (std::size_t j = 0; j < nImages_; ++j) {
            auto virtVisCurrentSlice =
                virtualVis.slice_view(j).sub_view(inputBegin, inputSize);
            ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT input", virtVisCurrentSlice);
            transform.execute(virtVisCurrentSlice.data(), output.data());
            ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output",
                                      output.sub_view({0}, {imgSize}));

            auto* __restrict__ outputPtr = output.data();
            auto* __restrict__ imgPtr = &img_[{imgBegin, j}];
            for (std::size_t k = 0; k < imgSize; ++k) {
              imgPtr[k] += outputPtr[k].real();
            }
          }
      }
    }
  }
  totalCollectCount_ += data.size();
  totalVisibilityCount_ += visibilityCount;
}

template <typename T>
auto NufftSynthesis<T>::get(View<T, 2> out) -> void {
  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nImages_);

  HostView<T, 2> outHost(out);

  const T visScale =
      totalVisibilityCount_ ? static_cast<T>(1.0 / static_cast<double>(totalVisibilityCount_)) : 0;

  ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                     "NufftSynthesis<T>::get totalVisibilityCount_ = {}, visScale = {}",
                     totalVisibilityCount_, visScale);

  for (std::size_t i = 0; i < nImages_; ++i) {
    auto currentImg = img_.slice_view(i);
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", currentImg);

    imgPartition_.reverse<T>(currentImg, outHost.slice_view(i));

    if (opt_.normalizeImage) {
      T* __restrict__ outPtr = &outHost[{0, i}];
      for (std::size_t j = 0; j < nPixel_; ++j) {
          outPtr[j] *= visScale;
      }
    }

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", outHost.slice_view(i));
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace host
}  // namespace bipp
