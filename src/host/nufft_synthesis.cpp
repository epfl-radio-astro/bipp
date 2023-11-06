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
                                  std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                                  ConstHostView<BippFilter, 1> filter, ConstHostView<T, 1> pixelX,
                                  ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nIntervals_(nIntervals),
      nFilter_(filter.shape()[0]),
      nPixel_(pixelX.shape()[0]),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      filter_(ctx_->host_alloc(), filter.shape()),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nIntervals_, nFilter_}),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      collectCount_(0),
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
              DomainPartition::grid<T, 3>(ctx_, arg.dimensions, nPixel_, {pixelX.data(), pixelY.data(), pixelZ.data()});
        } else if constexpr (std::is_same_v<ArgType, Partition::None> ||
                             std::is_same_v<ArgType, Partition::Auto>) {
          ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "image partition: none");
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(pixelX.data(), pixel_.slice_view(0).data());
  imgPartition_.apply(pixelY.data(), pixel_.slice_view(1).data());
  imgPartition_.apply(pixelZ.data(), pixel_.slice_view(2).data());

  if (opt_.collectGroupSize && opt_.collectGroupSize.value() > 0) {
    nMaxInputCount_ = opt_.collectGroupSize.value();
  } else {
    // use at most 20% of memory for accumulation, but not more than 200 iterations in total
    nMaxInputCount_ = (system_memory() / 5) /
                      (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * sizeof(std::complex<T>));
    nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);
  }

  virtualVis_ = HostArray<std::complex<T>, 3>(
      ctx_->host_alloc(), {nAntenna_ * nAntenna_ * nMaxInputCount_, nIntervals_, nFilter_});
  uvw_ = HostArray<T, 2>(ctx_->host_alloc(), {nAntenna_ * nAntenna_ * nMaxInputCount_, 3});
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
                                ConstHostView<std::complex<T>, 2> s,
                                ConstHostView<std::complex<T>, 2> w, ConstHostView<T, 2> xyz,
                                ConstHostView<T, 2> uvw) -> void {
  assert(xyz.shape()[0] == nAntenna_);
  assert(xyz.shape()[1] == 3);
  assert(intervals.shape()[1] == nIntervals_);
  assert(intervals.shape()[0] == 2);
  assert(w.shape()[0] == nAntenna_);
  assert(w.shape()[1] == nBeam_);
  assert(!s.size() || s.shape()[0] == nBeam_);
  assert(!s.size() || s.shape()[1] == nBeam_);
  assert(uvw.shape()[0] == nAntenna_ * nAntenna_);
  assert(uvw.shape()[1] == 3);

  // store coordinates
  copy(uvw, uvw_.sub_view({collectCount_ * nAntenna_ * nAntenna_, 0}, {nAntenna_ * nAntenna_, 3}));

  auto v = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nBeam_, nEig});
  auto d = HostArray<T, 1>(ctx_->host_alloc(), {nEig});

  {
    auto g = HostArray<std::complex<T>, 2>(ctx_->host_alloc(), {nBeam_, nBeam_});

    gram_matrix<T>(*ctx_, w, xyz, wl, g);

    // Note different order of s and g input
    if (s.size())
      eigh<T>(*ctx_, nEig, s, g, d, v);
    else {
      eigh<T>(*ctx_, nEig, g, s, d, v);
    }
  }

  // Reverse beamforming
  HostArray<std::complex<T>, 2> vUnbeam(ctx_->host_alloc(), {nAntenna_, v.shape()[1]});
  blas::gemm(CblasNoTrans, CblasNoTrans, {1, 0}, w, v, {0, 0}, vUnbeam);

  // slice virtual visibility for current step
  auto virtVisCurrent =
      virtualVis_.sub_view({collectCount_ * nAntenna_ * nAntenna_, 0, 0},
                           {nAntenna_ * nAntenna_, virtualVis_.shape()[1], virtualVis_.shape()[2]});

  // compute virtual visibilities
  virtual_vis(*ctx_, filter_, intervals, d, vUnbeam, virtVisCurrent);

  ++collectCount_;
  ++totalCollectCount_;
  ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "collect count: {} / {}", collectCount_, nMaxInputCount_);
  if (collectCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  if (collectCount_) {
    auto output = HostArray<std::complex<T>, 1>(ctx_->host_alloc(), {nPixel_});
    auto outputPtr = output.data();

    const auto nInputPoints = nAntenna_ * nAntenna_ * collectCount_;

    ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "computing nufft for collected data");

    auto uvwX = uvw_.slice_view(0);
    auto uvwY = uvw_.slice_view(1);
    auto uvwZ = uvw_.slice_view(2);

    auto pixelX = pixel_.slice_view(0);
    auto pixelY = pixel_.slice_view(1);
    auto pixelZ = pixel_.slice_view(2);

    auto inputPartition = std::visit(
        [&](auto&& arg) -> DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                               arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
            return DomainPartition::grid<T, 3>(ctx_, arg.dimensions, nInputPoints,
                                               {uvwX.data(), uvwY.data(), uvwZ.data()});
          } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
            ctx_->logger().log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
            return DomainPartition::none(ctx_, nInputPoints);

          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
            std::array<double, 3> uvwExtent{};
            std::array<double, 3> imgExtent{};

            auto minMaxIt = std::minmax_element(uvwX.data(), uvwX.data() + nInputPoints);
            uvwExtent[0] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwY.data(), uvwY.data() + nInputPoints);
            uvwExtent[1] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwZ.data(), uvwZ.data() + nInputPoints);
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
            return DomainPartition::grid<T>(ctx_, gridSize, nInputPoints,
                                            {uvwX.data(), uvwY.data(), uvwZ.data()});
          }
        },
        opt_.localUVWPartition.method);

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        assert(i * ldVirtVis1 + j * ldVirtVis2 + inputPartition.num_elements() <=
               virtualVis_.size());
        inputPartition.apply(virtualVis_.data() + i * ldVirtVis1 + j * ldVirtVis2);
      }
    }

    inputPartition.apply(uvwX.data());
    inputPartition.apply(uvwY.data());
    inputPartition.apply(uvwZ.data());

    for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
      if (!inputSize) continue;
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        if (inputSize <= 32) {
          // Direct evaluation of sum for small input sizes
          ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "nufft size {}, direct evaluation", inputSize);
          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.data() + (j + i * nIntervals_) * nPixel_ + imgBegin;
              nuft_sum<T>(1.0, inputSize,
                          virtualVis_.data() + i * ldVirtVis1 + j * ldVirtVis2 + inputBegin,
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
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.data() + (j + i * nIntervals_) * nPixel_ + imgBegin;

              transform.execute(virtualVis_.data() + i * ldVirtVis1 + j * ldVirtVis2 + inputBegin,
                                outputPtr);
              ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "NUFFT output", imgSize, 1, outputPtr,
                                      imgSize);

              for (std::size_t k = 0; k < imgSize; ++k) {
                imgPtr[k] += outputPtr[k].real();
              }
            }
          }
        }
      }
    }
  }

  collectCount_ = 0;
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

  for (std::size_t i = 0; i < nIntervals_; ++i) {

    const T* __restrict__ imgPtr = &img_[{0, i, index}];
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image permuted", nPixel_, 1, imgPtr, nPixel_);

    T* __restrict__ outPtr = &out[{0, i}];

    imgPartition_.reverse(imgPtr, outPtr);

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
