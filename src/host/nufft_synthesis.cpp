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
                                  std::size_t nLevel, ConstHostView<BippFilter, 1> filter,
                                  ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                  ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nLevel_(nLevel),
      nFilter_(filter.shape()),
      nPixel_(pixelX.shape()),
      filter_(ctx_->host_alloc(), filter.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nLevel_, nFilter_}),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      collectPoints_(0),
      totalCollectCount_(0) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());

  copy(filter, filter_);
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
auto NufftSynthesis<T>::collect(
    T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
    ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
    ConstHostView<T, 2> xyz, ConstHostView<T, 2> uvw) -> void {

  const auto nAntenna = w.shape(0);
  const auto nBeam = w.shape(1);

  assert(xyz.shape(0) == nAntenna);
  assert(xyz.shape(1) == 3);
  assert(s.shape(0) == nBeam);
  assert(s.shape(1) == nBeam);
  assert(uvw.shape(0) == nAntenna * nAntenna);
  assert(uvw.shape(1) == 3);


  // allocate initial memory if not yet done
  if (uvw_.shape(0) <= collectPoints_ + nAntenna * nAntenna) {
    this->computeNufft();

    const std::size_t minSizeUvw = 3 * nAntenna * nAntenna;
    const std::size_t minSizeVirtVis = nLevel_ * nFilter_ * nAntenna * nAntenna;
    const double memFracUvw =
        double(minSizeUvw * sizeof(T)) /
        double(minSizeUvw * sizeof(T) + minSizeVirtVis * sizeof(std::complex<T>));

    const auto systemMemory = system_memory();

    const std::size_t requestedMem =
        std::max(0.0f, std::min(opt_.collectMemory, 1.0f)) * systemMemory;

    const std::size_t requestedMemUvw = std::size_t(memFracUvw * requestedMem);

    const std::size_t requestedSizeMultiplier =
        std::max<std::size_t>(1, requestedMemUvw / (3 * nAntenna * nAntenna * sizeof(T)));

    const std::size_t maxPoints = requestedSizeMultiplier * nAntenna * nAntenna;

    // free up existing memory first
    virtualVis_ = decltype(virtualVis_)();
    uvw_ = decltype(uvw_)();

    virtualVis_ =
        HostArray<std::complex<T>, 3>(ctx_->host_alloc(), {maxPoints, nLevel_, nFilter_});
    uvw_ = HostArray<T, 2>(ctx_->host_alloc(), {maxPoints, 3});
  }

  // store coordinates
  copy(uvw, uvw_.sub_view({collectPoints_, 0}, {nAntenna * nAntenna, 3}));

  auto vUnbeamArray = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nAntenna, nBeam});
  auto dArray = HostArray<T, 1>(ctx_->host_alloc(), nBeam);

  const auto nEig = eigh<T>(*ctx_, wl, s, w, xyz, dArray, vUnbeamArray);

  auto d = dArray.sub_view(0, nEig);
  auto vUnbeam = vUnbeamArray.sub_view({0, 0}, {nAntenna, nEig});

  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "vUnbeam", vUnbeam);

  // callback for each level with eigenvalues
  auto dMaskedArray = HostArray<T, 2>(ctx_->host_alloc(), {d.size(), nLevel_});

  for (std::size_t idxLevel = 0; idxLevel < nLevel_; ++idxLevel) {
    copy(d, dMaskedArray.slice_view(idxLevel));
    eigMaskFunc(idxLevel, nBeam, dMaskedArray.slice_view(idxLevel).data());
  }

  // slice virtual visibility for current step
  auto virtVisCurrent =
      virtualVis_.sub_view({collectPoints_, 0, 0},
                           {nAntenna * nAntenna, virtualVis_.shape(1), virtualVis_.shape(2)});

  // compute virtual visibilities
  virtual_vis<T>(*ctx_, filter_, dMaskedArray, vUnbeam, virtVisCurrent);

  collectPoints_ += nAntenna * nAntenna;
  ++totalCollectCount_;
  ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "collect count: {}", collectPoints_);
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  if (collectPoints_) {
    auto output = HostArray<std::complex<T>, 1>(ctx_->host_alloc(), nPixel_);

    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "computing nufft for collected data");

    auto uvwX = uvw_.slice_view(0).sub_view({0}, {collectPoints_});
    auto uvwY = uvw_.slice_view(1).sub_view({0}, {collectPoints_});;
    auto uvwZ = uvw_.slice_view(2).sub_view({0}, {collectPoints_});;

    auto virtVisCurrent = virtualVis_.sub_view(
        {0, 0, 0}, {collectPoints_, virtualVis_.shape(1), virtualVis_.shape(2)});

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
            return DomainPartition::none(ctx_, collectPoints_);

          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
            std::array<double, 3> uvwExtent{};
            std::array<double, 3> imgExtent{};

            auto minMaxIt = std::minmax_element(uvwX.data(), uvwX.data() + collectPoints_);
            uvwExtent[0] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwY.data(), uvwY.data() + collectPoints_);
            uvwExtent[1] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwZ.data(), uvwZ.data() + collectPoints_);
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

            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                               gridSize[0], gridSize[1], gridSize[2]);

            // set partition method to grid and create grid partition
            opt_.localUVWPartition.method = Partition::Grid{gridSize};
            return DomainPartition::grid<T>(ctx_, gridSize, {uvwX, uvwY, uvwZ});
          }
        },
        opt_.localUVWPartition.method);

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nLevel_; ++j) {
        inputPartition.apply(virtVisCurrent.slice_view(i).slice_view(j));
      }
    }

    inputPartition.apply(uvwX);
    inputPartition.apply(uvwY);
    inputPartition.apply(uvwZ);

    for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
      if (!inputSize) continue;
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        if (inputSize <= 32) {
          // Direct evaluation of sum for small input sizes
          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nLevel_; ++j) {
              auto imgPtr = img_.data() + (j + i * nLevel_) * nPixel_ + imgBegin;
              nuft_sum<T>(1.0, inputSize, &virtVisCurrent[{inputBegin, j, i}],
                          uvwX.data() + inputBegin, uvwY.data() + inputBegin,
                          uvwZ.data() + inputBegin, imgSize, pixelX.data() + imgBegin,
                          pixelY.data() + imgBegin, pixelZ.data() + imgBegin, imgPtr);
            }
          }
        } else {
          // Approximate sum through nufft
          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, calling fiNUFFT", inputSize);
          Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwX.data() + inputBegin,
                                uvwY.data() + inputBegin, uvwZ.data() + inputBegin, imgSize,
                                pixelX.data() + imgBegin, pixelY.data() + imgBegin,
                                pixelZ.data() + imgBegin);

          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nLevel_; ++j) {
              transform.execute(&virtVisCurrent[{inputBegin, j, i}], output.data());
              ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output", output.sub_view({0}, {imgSize}));

              auto* __restrict__ outputPtr = output.data();
              auto* __restrict__ imgPtr = &img_[{imgBegin, j, i}];
              for (std::size_t k = 0; k < imgSize; ++k) {
                imgPtr[k] += outputPtr[k].real();
              }
            }
          }
        }
      }
    }
  }

  collectPoints_ = 0;
}

template <typename T>
auto NufftSynthesis<T>::get(BippFilter f, HostView<T, 2> out) -> void {
  computeNufft();  // make sure all input has been processed

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

  auto currentImg = img_.slice_view(index);

  for (std::size_t i = 0; i < nLevel_; ++i) {

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", currentImg);

    imgPartition_.reverse<T>(currentImg.slice_view(i), out.slice_view(i));

    T* __restrict__ outPtr = &out[{0, i}];
    for(std::size_t j = 0; j < nPixel_; ++j) {
      outPtr[j] *= scale;
    }

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", out.slice_view(i));
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace host
}  // namespace bipp
