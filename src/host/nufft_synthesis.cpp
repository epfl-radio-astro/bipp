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
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/kernels/nuft_sum.hpp"
#include "host/nufft_3d3.hpp"
#include "host/virtual_vis.hpp"
#include "memory/buffer.hpp"
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
                                  std::size_t nFilter, const BippFilter* filter, std::size_t nPixel,
                                  const T* lmnX, const T* lmnY, const T* lmnZ)
    : ctx_(std::move(ctx)),
      opt_(std::move(opt)),
      nIntervals_(nIntervals),
      nFilter_(nFilter),
      nPixel_(nPixel),
      nAntenna_(nAntenna),
      nBeam_(nBeam),
      filter_(ctx_->host_alloc(), nFilter_),
      lmnX_(ctx_->host_alloc(), nPixel_),
      lmnY_(ctx_->host_alloc(), nPixel_),
      lmnZ_(ctx_->host_alloc(), nPixel_),
      imgPartition_(DomainPartition::none(ctx_, nPixel_)),
      collectCount_(0) {
  std::memcpy(filter_.get(), filter, sizeof(BippFilter) * nFilter_);

  // Only partition image if explicitly set. Auto defaults to no partition.
  std::visit(
      [&](auto&& arg) -> void {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
          imgPartition_ =
              DomainPartition::grid<T, 3>(ctx_, arg.dimensions, nPixel_, {lmnX, lmnY, lmnZ});
        }
      },
      opt_.localImagePartition.method);

  imgPartition_.apply(lmnX, lmnX_.get());
  imgPartition_.apply(lmnY, lmnY_.get());
  imgPartition_.apply(lmnZ, lmnZ_.get());

  if (opt_.collectGroupSize && opt_.collectGroupSize.value() > 0) {
    nMaxInputCount_ = opt_.collectGroupSize.value();
  } else {
    // use at most 20% of memory for accumulation, but not more than 200 iterations in total
    nMaxInputCount_ = (system_memory() / 5) /
                      (nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * sizeof(std::complex<T>));
    nMaxInputCount_ = std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);
  }

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  virtualVis_ = Buffer<std::complex<T>>(ctx_->host_alloc(), virtualVisBufferSize);
  uvwX_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwY_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwZ_ = Buffer<T>(ctx_->host_alloc(), nAntenna_ * nAntenna_ * nMaxInputCount_);

  img_ = Buffer<T>(ctx_->host_alloc(), nPixel_ * nIntervals_ * nFilter_);
  std::memset(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T));
}

template <typename T>
auto NufftSynthesis<T>::collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
                                const std::complex<T>* s, std::size_t lds, const std::complex<T>* w,
                                std::size_t ldw, const T* xyz, std::size_t ldxyz, const T* uvw,
                                std::size_t lduvw) -> void {
  // store coordinates
  std::memcpy(uvwX_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwY_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw + lduvw,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwZ_.get() + collectCount_ * nAntenna_ * nAntenna_, uvw + 2 * lduvw,
              sizeof(T) * nAntenna_ * nAntenna_);

  auto v = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nEig);
  auto d = Buffer<T>(ctx_->host_alloc(), nEig);

  {
    auto g = Buffer<std::complex<T>>(ctx_->host_alloc(), nBeam_ * nBeam_);

    gram_matrix<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyz, ldxyz, wl, g.get(), nBeam_);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s)
      eigh<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, &nEigOut, d.get(), v.get(), nBeam_);
    else {
      eigh<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, &nEigOut, d.get(), v.get(), nBeam_);
    }
  }

  auto virtVisPtr = virtualVis_.get() + collectCount_ * nAntenna_ * nAntenna_;

  virtual_vis(*ctx_, nFilter_, filter_.get(), nIntervals_, intervals, ldIntervals, nEig, d.get(),
              nAntenna_, v.get(), nBeam_, nBeam_, w, ldw, virtVisPtr,
              nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
              nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  ++collectCount_;
  if (collectCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T>
auto NufftSynthesis<T>::computeNufft() -> void {
  if (collectCount_) {
    auto output = Buffer<std::complex<T>>(ctx_->host_alloc(), nPixel_);
    auto outputPtr = output.get();

    const auto nInputPoints = nAntenna_ * nAntenna_ * collectCount_;

    auto inputPartition = std::visit(
        [&](auto&& arg) -> DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            return DomainPartition::grid<T, 3>(ctx_, arg.dimensions, nInputPoints,
                                               {uvwX_.get(), uvwY_.get(), uvwZ_.get()});

          } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
            return DomainPartition::none(ctx_, nInputPoints);

          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {
            std::array<double, 3> uvwExtent{};
            std::array<double, 3> imgExtent{};

            auto minMaxIt = std::minmax_element(uvwX_.get(), uvwX_.get() + nInputPoints);
            uvwExtent[0] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwY_.get(), uvwY_.get() + nInputPoints);
            uvwExtent[1] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(uvwZ_.get(), uvwZ_.get() + nInputPoints);
            uvwExtent[2] = *minMaxIt.second - *minMaxIt.first;

            minMaxIt = std::minmax_element(lmnX_.get(), lmnX_.get() + nPixel_);
            imgExtent[0] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(lmnY_.get(), lmnY_.get() + nPixel_);
            imgExtent[1] = *minMaxIt.second - *minMaxIt.first;
            minMaxIt = std::minmax_element(lmnZ_.get(), lmnZ_.get() + nPixel_);
            imgExtent[2] = *minMaxIt.second - *minMaxIt.first;

            // Use at most 12.5% of total memory for fft grid
            const auto gridSize = optimal_nufft_input_partition(
                uvwExtent, imgExtent, system_memory() / (8 * sizeof(std::complex<T>)));

            // set partition method to grid and create grid partition
            opt_.localUVWPartition.method = Partition::Grid{gridSize};
            return DomainPartition::grid<T>(ctx_, gridSize, nInputPoints,
                                            {uvwX_.get(), uvwY_.get(), uvwZ_.get()});
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
        inputPartition.apply(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2);
      }
    }

    inputPartition.apply(uvwX_.get());
    inputPartition.apply(uvwY_.get());
    inputPartition.apply(uvwZ_.get());

    for (const auto& [inputBegin, inputSize] : inputPartition.groups()) {
      if (!inputSize) continue;
      for (const auto& [imgBegin, imgSize] : imgPartition_.groups()) {
        if (!imgSize) continue;

        if (inputSize <= 32) {
          // Direct evaluation of sum for small input sizes
          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_ + imgBegin;
              nuft_sum<T>(1.0, inputSize,
                          virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2 + inputBegin,
                          uvwX_.get() + inputBegin, uvwY_.get() + inputBegin,
                          uvwZ_.get() + inputBegin, imgSize, lmnX_.get() + imgBegin,
                          lmnY_.get() + imgBegin, lmnZ_.get() + imgBegin, imgPtr);
            }
          }
        } else {
          // Approximate sum through nufft
          Nufft3d3<T> transform(1, opt_.tolerance, 1, inputSize, uvwX_.get() + inputBegin,
                                uvwY_.get() + inputBegin, uvwZ_.get() + inputBegin, imgSize,
                                lmnX_.get() + imgBegin, lmnY_.get() + imgBegin,
                                lmnZ_.get() + imgBegin);

          for (std::size_t i = 0; i < nFilter_; ++i) {
            for (std::size_t j = 0; j < nIntervals_; ++j) {
              auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_ + imgBegin;

              transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2 + inputBegin,
                                outputPtr);

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
auto NufftSynthesis<T>::get(BippFilter f, T* out, std::size_t ld) -> void {
  computeNufft();  // make sure all input has been processed

  std::size_t index = nFilter_;
  const BippFilter* filterPtr = filter_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_) throw InvalidParameterError();

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    imgPartition_.reverse(img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_, out + i * ld);
  }
}

template class NufftSynthesis<float>;
template class NufftSynthesis<double>;

}  // namespace host
}  // namespace bipp
