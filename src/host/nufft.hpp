#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <neonufft/plan.hpp>
#include <unistd.h>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "bipp/image_synthesis.hpp"
#include "context_internal.hpp"
#include "host/domain_partition.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "nufft_interface.hpp"
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
class NUFFT : public NUFFTInterface<T> {
public:
  NUFFT(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
        ConstHostView<T, 2> pixelXYZ, std::size_t nImages, std::size_t nBaselines,
        std::size_t collectGroupSize)
      : nImages_(nImages),
        nBaselines_(nBaselines),
        collectGroupSize_(collectGroupSize),
        ctx_(std::move(ctx)),
        opt_(std::move(opt)),
        images_(ctx_->host_alloc(), {pixelXYZ.shape(0), nImages}),
        pixelXYZ_(ctx_->host_alloc(), pixelXYZ.shape()),
        valueCollection_(ctx_->host_alloc(), {collectGroupSize * nBaselines, nImages}),
        uvwCollection_(ctx_->host_alloc(), {collectGroupSize * nBaselines, 3}) {
    images_.zero();

    copy(pixelXYZ, pixelXYZ_);

    auto pixelX = pixelXYZ_.slice_view(0);
    auto pixelY = pixelXYZ_.slice_view(1);
    auto pixelZ = pixelXYZ_.slice_view(2);

    auto pixelXMinMax = std::minmax_element(pixelX.data(), pixelX.data() + pixelX.size());
    auto pixelYMinMax = std::minmax_element(pixelY.data(), pixelY.data() + pixelY.size());
    auto pixelZMinMax = std::minmax_element(pixelZ.data(), pixelZ.data() + pixelZ.size());

    pixelMin_ = {*pixelXMinMax.first, *pixelYMinMax.first, *pixelZMinMax.first};
    pixelMax_ = {*pixelXMinMax.second, *pixelYMinMax.second, *pixelZMinMax.second};
  }

  auto add(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 2> values) -> void override {
    assert(values.shape(0) == nBaselines_);
    assert(uvw.shape(0) == nBaselines_);

    copy(uvw, uvwCollection_.sub_view({count_ * nBaselines_, 0}, {nBaselines_, 3}));
    copy(values, valueCollection_.sub_view({count_ * nBaselines_, 0}, {nBaselines_, nImages_}));

    ++count_;
    if (count_ >= collectGroupSize_) {
      this->transform();
    }
  }

  auto get_image(std::size_t imgIdx, HostView<float, 1> image) -> void override {
    this->transform();
    copy(images_.slice_view(imgIdx), image);
  }

private:
  auto transform() -> void {
    if (!count_) return;

    neonufft::Options neoOpt;
    neoOpt.tol = opt_.tolerance;
    neoOpt.sort_input = false;
    neoOpt.sort_output = false;

    auto uvw = uvwCollection_.sub_view({0, 0}, {count_ * nBaselines_, 3});
    auto values = valueCollection_.sub_view({0, 0}, {count_ * nBaselines_, nImages_});

    host::DomainPartition uvwPartition = std::visit(
        [&](auto&& arg) -> host::DomainPartition {
          using ArgType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<ArgType, Partition::Grid>) {
            globLogger.log(BIPP_LOG_LEVEL_INFO, "uvw partition: grid ({}, {}, {})",
                               arg.dimensions[0], arg.dimensions[1], arg.dimensions[2]);
            return host::DomainPartition::grid<T, 3>(
                ctx_->host_alloc(), arg.dimensions,
                {uvw.slice_view(0), uvw.slice_view(1), uvw.slice_view(2)});
          } else if constexpr (std::is_same_v<ArgType, Partition::Auto>) {

            // Use at most66% of memory for fft grid
            const auto maxMem = system_memory() / 3 * 2;

            auto grid =
                optimal_parition_size<T>(uvw, pixelMin_, pixelMax_, maxMem,
                                         [&](std::array<T, 3> uvwMin, std::array<T, 3> uvwMax,
                                             std::array<T, 3> xyzMin,
                                             std::array<T, 3> xyzMax) -> unsigned long long {
                                           return neonufft::PlanT3<T, 3>::grid_memory_size(
                                               neoOpt, uvwMin, uvwMax, xyzMin, xyzMax);
                                         });

            globLogger.log(BIPP_LOG_LEVEL_INFO, "auto uvw partition: grid ({}, {}, {})", grid[0],
                           grid[1], grid[2]);
            return host::DomainPartition::grid<T, 3>(
                ctx_->host_alloc(), grid,
                {uvw.slice_view(0), uvw.slice_view(1), uvw.slice_view(2)});

          } else if constexpr (std::is_same_v<ArgType, Partition::None>) {
            globLogger.log(BIPP_LOG_LEVEL_INFO, "uvw partition: none");
            return host::DomainPartition::none(ctx_->host_alloc(), uvw.shape(0));
          }
        },
        opt_.localUVWPartition.method);

    uvwPartition.apply(uvw.slice_view(0));
    uvwPartition.apply(uvw.slice_view(1));
    uvwPartition.apply(uvw.slice_view(2));
    for (std::size_t imageIdx = 0; imageIdx < nImages_; ++imageIdx) {
      uvwPartition.apply(values.slice_view(imageIdx));
    }

    const auto maxGroupSize =
        std::max_element(
            uvwPartition.groups().begin(), uvwPartition.groups().end(),
            [](const PartitionGroup& g1, const PartitionGroup& g2) { return g1.size < g2.size; })
            ->size;

    HostArray<std::complex<T>, 2> imageCpx(ctx_->host_alloc(), {pixelXYZ_.shape(0), nImages_});


    for (const auto& [uvwBegin, uvwSize] : uvwPartition.groups()) {
      if (!uvwSize) continue;

      globLogger.log(BIPP_LOG_LEVEL_DEBUG, "uvw partition begin = {}, size = {}", uvwBegin,
                         uvwSize);

      auto uvwPart = uvw.sub_view({uvwBegin, 0}, {uvwSize, 3});
      auto valuesPart = values.sub_view({uvwBegin, 0}, {uvwSize, nImages_});

      auto uView = uvwPart.slice_view(0);
      auto vView = uvwPart.slice_view(1);
      auto wView = uvwPart.slice_view(2);

      auto uMinMax = std::minmax_element(uView.data(), uView.data() + uView.size());
      auto vMinMax = std::minmax_element(vView.data(), vView.data() + vView.size());
      auto wMinMax = std::minmax_element(wView.data(), wView.data() + wView.size());

      std::array<T, 3> uvwMin = {*uMinMax.first, *vMinMax.first, *wMinMax.first};
      std::array<T, 3> uvwMax = {*uMinMax.second, *vMinMax.second, *wMinMax.second};

      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points l", pixelXYZ_.slice_view(0));
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points m", pixelXYZ_.slice_view(1));
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points n", pixelXYZ_.slice_view(2));

      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points u", uView);
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points v", vView);
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points w", wView);


      // Batched version
      /*
      neonufft::PlanT3<T, 3> plan(neoOpt, 1, uvwMin, uvwMax, pixelMin_, pixelMax_, nImages_);
      plan.set_input_points(uvwSize, {uView.data(), vView.data(), wView.data()});
      plan.set_output_points(pixelXYZ_.shape(0), {pixelXYZ_.data(), pixelXYZ_.slice_view(1).data(),
                                                  pixelXYZ_.slice_view(2).data()});

      plan.add_input(valuesPart.data(), valuesPart.strides(1));
      plan.transform(imageCpx.data(), imageCpx.strides(1));

      for (std::size_t imageIdx = 0; imageIdx < nImages_; ++imageIdx) {
        const T* __restrict__ sourcePtr =
            reinterpret_cast<const T*>(imageCpx.slice_view(imageIdx).data());
        float* targetPtr = images_.slice_view(imageIdx).data();
        const auto nPixel = images_.shape(0);
        for (std::size_t pixelIdx = 0; pixelIdx < nPixel; ++pixelIdx) {
          targetPtr[pixelIdx] += sourcePtr[2 * pixelIdx];  // add real part
        }
      }
      */

      neonufft::PlanT3<T, 3> plan(neoOpt, 1, uvwMin, uvwMax, pixelMin_, pixelMax_);
      plan.set_input_points(uvwSize, {uView.data(), vView.data(), wView.data()});
      plan.set_output_points(pixelXYZ_.shape(0), {pixelXYZ_.data(), pixelXYZ_.slice_view(1).data(),
                                                  pixelXYZ_.slice_view(2).data()});

      for (std::size_t imageIdx = 0; imageIdx < nImages_; ++imageIdx) {
        auto valuesSlice = values.slice_view(imageIdx).sub_view(uvwBegin, uvwSize);
        plan.add_input(valuesSlice.data());
        globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input", valuesSlice);
        plan.transform(imageCpx.data());
        globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft output", imageCpx);

        const T* __restrict__ sourcePtr = reinterpret_cast<const T*>(imageCpx.data());
        float* targetPtr = images_.slice_view(imageIdx).data();
        const auto nPixel = images_.shape(0);
        for (std::size_t pixelIdx = 0; pixelIdx < nPixel; ++pixelIdx) {
          targetPtr[pixelIdx] += sourcePtr[2 * pixelIdx];  // add real part
        }

        if (imageIdx < nImages_ - 1) plan.reset();
      }
    }

    count_ = 0;
  }

  std::size_t nImages_, nBaselines_, collectGroupSize_;
  std::size_t count_ = 0;
  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;

  HostArray<float, 2> images_;
  HostArray<T, 2> pixelXYZ_;
  HostArray<std::complex<T>, 2> valueCollection_;
  HostArray<T, 2> uvwCollection_;

  std::array<T, 3> pixelMin_;
  std::array<T, 3> pixelMax_;
};
}  // namespace host
}  // namespace bipp
