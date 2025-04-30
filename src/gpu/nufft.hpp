#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <neonufft/allocator.hpp>
#include <neonufft/gpu/plan.hpp>
#include <neonufft/gpu/types.hpp>
#include <neonufft/plan.hpp>
#include <numeric>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "bipp/image_synthesis.hpp"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/domain_partition.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "memory/view.hpp"
#include "nufft_interface.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "nufft_util.hpp"

namespace bipp {
namespace gpu {
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
        images_(ctx_->gpu_queue().create_device_array<float, 2>({pixelXYZ.shape(0), nImages})),
        pixelXYZ_(ctx_->gpu_queue().create_device_array<T, 2>(pixelXYZ.shape())),
        valueCollection_(ctx_->gpu_queue().create_pinned_array<std::complex<T>, 2>(
            {collectGroupSize * nBaselines, nImages})),
        uvwCollection_(
            ctx_->gpu_queue().create_pinned_array<T, 2>({collectGroupSize * nBaselines, 3})) {
    auto& queue = ctx_->gpu_queue();

    images_.zero(queue.stream());
    copy(queue, pixelXYZ, pixelXYZ_);

    auto pixelX = pixelXYZ.slice_view(0);
    auto pixelY = pixelXYZ.slice_view(1);
    auto pixelZ = pixelXYZ.slice_view(2);

    auto pixelXMinMax = std::minmax_element(pixelX.data(), pixelX.data() + pixelX.size());
    auto pixelYMinMax = std::minmax_element(pixelY.data(), pixelY.data() + pixelY.size());
    auto pixelZMinMax = std::minmax_element(pixelZ.data(), pixelZ.data() + pixelZ.size());

    pixelMin_ = {*pixelXMinMax.first, *pixelYMinMax.first, *pixelZMinMax.first};
    pixelMax_ = {*pixelXMinMax.second, *pixelYMinMax.second, *pixelZMinMax.second};

    queue.sync();  // make sure pixel data has been copied
  }

  auto add(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 2> values) -> void override {
    assert(values.shape(0) == nBaselines_);
    assert(uvw.shape(0) == nBaselines_);

    auto& queue = ctx_->gpu_queue();

    copy(uvw, uvwCollection_.sub_view({count_ * nBaselines_, 0}, {nBaselines_, 3}));
    copy(values, valueCollection_.sub_view({count_ * nBaselines_, 0}, {nBaselines_, nImages_}));

    ++count_;
    if (count_ >= collectGroupSize_) {
      this->transform();
    }
  }

  auto get_image(std::size_t imgIdx, HostView<float, 1> image) -> void override {
    this->transform();
    auto& queue = ctx_->gpu_queue();
    copy(queue, images_.slice_view(imgIdx), image);
    queue.sync();
  }

private:
  auto transform() -> void {
    if (!count_) return;

    neonufft::Options neoOpt;
    neoOpt.tol = opt_.tolerance;

    auto& queue = ctx_->gpu_queue();
    queue.sync();

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
            const auto maxMem = queue.device_prop().totalGlobalMem / 3 * 2;

            auto grid = optimal_parition_size<T>(
                uvw, pixelMin_, pixelMax_, maxMem,
                [&](std::array<T, 3> uvwMin, std::array<T, 3> uvwMax, std::array<T, 3> xyzMin,
                    std::array<T, 3> xyzMax) -> unsigned long long {
                  return neonufft::gpu::PlanT3<T, 3>::grid_memory_size(neoOpt, uvwMin, uvwMax,
                                                                       xyzMin, xyzMax);
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

    auto uvwBuffer = queue.create_device_array<T, 1>(3 * maxGroupSize);

    auto valuesBuffer = queue.create_device_array<std::complex<T>, 2>({maxGroupSize, nImages_});

    auto imageCpx = queue.create_device_array<api::ComplexType<T>, 1>(pixelXYZ_.shape(0));


    struct NeoAlloc : neonufft::Allocator {
      explicit NeoAlloc(std::shared_ptr<bipp::Allocator> alloc) : alloc_(std::move(alloc)) {}

      void* allocate(std::size_t size) override {
        return alloc_->allocate(size);
      }

      void deallocate(void* ptr) noexcept override { alloc_->deallocate(ptr); }

      std::shared_ptr<bipp::Allocator> alloc_;
    };
    std::shared_ptr<neonufft::Allocator> neoAlloc(new NeoAlloc(queue.device_alloc()));

    for (const auto& [uvwBegin, uvwSize] : uvwPartition.groups()) {
      if (!uvwSize) continue;

      globLogger.log(BIPP_LOG_LEVEL_DEBUG, "uvw partition begin = {}, size = {}", uvwBegin,
                         uvwSize);

      assert(uvwSize <= 3 * uvwBuffer.shape(0));
      DeviceView<T, 2> uvwDevice(uvwBuffer.data(), {uvwSize, 3}, {1, uvwSize});
      assert(nImages_ * uvwSize <= valuesBuffer.size());
      DeviceView<std::complex<T>, 2> valuesDevice(valuesBuffer.data(), {uvwSize, nImages_},
                                                  {1, uvwSize});

      auto uvwPart = uvw.sub_view({uvwBegin, 0}, {uvwSize, 3});
      copy(queue, uvwPart, uvwDevice);
      copy(queue, values.sub_view({uvwBegin, 0}, {uvwSize, nImages_}), valuesDevice);

      auto uView = uvwPart.slice_view(0);
      auto vView = uvwPart.slice_view(1);
      auto wView = uvwPart.slice_view(2);

      auto uMinMax = std::minmax_element(uView.data(), uView.data() + uView.size());
      auto vMinMax = std::minmax_element(vView.data(), vView.data() + vView.size());
      auto wMinMax = std::minmax_element(wView.data(), wView.data() + wView.size());

      std::array<T, 3> uvwMin = {*uMinMax.first, *vMinMax.first, *wMinMax.first};
      std::array<T, 3> uvwMax = {*uMinMax.second, *vMinMax.second, *wMinMax.second};

      // TODO: add allocator
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points u", uvwDevice.slice_view(0));
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points v", uvwDevice.slice_view(1));
      globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft points w", uvwDevice.slice_view(2));
      neonufft::gpu::PlanT3<T, 3> plan(neoOpt, 1, uvwMin, uvwMax, pixelMin_, pixelMax_,
                                       queue.stream(), 1, neoAlloc);
      plan.set_input_points(uvwSize, {uvwDevice.data(), uvwDevice.slice_view(1).data(),
                                      uvwDevice.slice_view(2).data()});
      plan.set_output_points(pixelXYZ_.shape(0), {pixelXYZ_.data(), pixelXYZ_.slice_view(1).data(),
                                                  pixelXYZ_.slice_view(2).data()});

      for (std::size_t imageIdx = 0; imageIdx < nImages_; ++imageIdx) {
        plan.add_input(reinterpret_cast<neonufft::gpu::ComplexType<T>*>(
            valuesDevice.slice_view(imageIdx).data()));
        globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft input",
                                  valuesDevice.slice_view(imageIdx));
        plan.transform(reinterpret_cast<neonufft::gpu::ComplexType<T>*>(imageCpx.data()));
        globLogger.log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft output", imageCpx);
        add_vector_real_of_complex<T>(queue.device_prop(), queue.stream(), imageCpx.shape(0),
                                      imageCpx.data(), images_.slice_view(imageIdx).data());
      }
    }

    queue.sync();
    count_ = 0;
  }

  std::size_t nImages_, nBaselines_, collectGroupSize_;
  std::size_t count_ = 0;
  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;

  DeviceArray<float, 2> images_;
  DeviceArray<T, 2> pixelXYZ_;
  HostArray<std::complex<T>, 2> valueCollection_;
  HostArray<T, 2> uvwCollection_;

  std::array<T, 3> pixelMin_;
  std::array<T, 3> pixelMax_;
};
}  // namespace gpu
}  // namespace bipp
