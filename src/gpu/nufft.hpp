#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <neonufft/gpu/plan.hpp>
#include <neonufft/gpu/types.hpp>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"
#include "context_internal.hpp"
#include "nufft_interface.hpp"

namespace bipp {
namespace gpu {
template <typename T>
class NUFFT : public NUFFTInterface<T> {
public:
  using ComplexType = neonufft::gpu::ComplexType<T>;

  NUFFT(std::shared_ptr<ContextInternal> ctx, std::size_t collectSize, std::size_t maxInputSize,
        neonufft::Options opt, int sign, std::array<T, 3> input_min, std::array<T, 3> input_max,
        std::array<T, 3> output_min, std::array<T, 3> output_max)
      : maxInputSize_(maxInputSize),
        count_(0),
        collectSize_(collectSize),
        ctx_(std::move(ctx)),
        collectUVW_(ctx_->gpu_queue().create_pinned_array<T, 2>({maxInputSize * collectSize, 3})),
        collectUVWDevice_(ctx_->gpu_queue().create_device_array<T, 2>(collectUVW_.shape())),
        collectValues_(
            ctx_->gpu_queue().create_pinned_array<std::complex<T>, 1>(maxInputSize * collectSize)),
        collectValuesDevice_(
            ctx_->gpu_queue().create_device_array<ComplexType, 1>(collectValues_.shape())),
        plan_(std::move(opt), sign, input_min, input_max, output_min, output_max,
              ctx_->gpu_queue().stream()) {
    // TODO: set plan allocator
  }

  auto set_output_points(ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                         ConstHostView<T, 1> pixelZ) -> void override {
    auto& queue = ctx_->gpu_queue();
    auto pixelDevice = queue.create_device_array<T, 2>({pixelX.shape(0), 3});
    copy(queue, pixelX, pixelDevice.slice_view(0));
    copy(queue, pixelY, pixelDevice.slice_view(1));
    copy(queue, pixelZ, pixelDevice.slice_view(2));

    plan_.set_output_points(pixelX.size(),
                            {pixelDevice.slice_view(0).data(), pixelDevice.slice_view(1).data(),
                             pixelDevice.slice_view(2).data()});
  }

  auto add_input(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 1> values)
      -> void override {
    copy(uvw, collectUVW_.sub_view({count_, 0}, uvw.shape()));
    copy(values, collectValues_.sub_view(count_, values.shape()));

    count_ += values.size();

    if (count_ + maxInputSize_ >= collectValues_.size()) {
      this->add_to_plan();
    }
  }

  auto transform_and_add(HostView<float, 1> out) -> void override {
    auto& queue = ctx_->gpu_queue();

    this->add_to_plan();

    auto outCpx = queue.create_pinned_array<ComplexType, 1>(out.shape());
    auto outCpxDevice = queue.create_device_array<ComplexType, 1>(out.shape());

    plan_.transform(outCpxDevice.data());
    copy(queue, outCpxDevice, outCpx);

    queue.sync();

    const ComplexType* __restrict__ cpxPtr = outCpx.data();
    float* __restrict__ outPtr = out.data();
    for (std::size_t i = 0; i < out.size(); ++i) {
      outPtr[i] += cpxPtr[i].x;
    }
  }

private:
  auto add_to_plan() {
    if (count_) {
      auto& queue = ctx_->gpu_queue();

      copy(queue, collectUVW_, collectUVWDevice_);
      copy(queue,
           ConstHostView<ComplexType, 1>(reinterpret_cast<ComplexType*>(collectValues_.data()),
                                         collectValues_.shape(), 1),
           collectValuesDevice_);

      plan_.set_input_points(
          count_, {collectUVWDevice_.slice_view(0).data(), collectUVWDevice_.slice_view(1).data(),
                   collectUVWDevice_.slice_view(2).data()});

      plan_.add_input(collectValuesDevice_.data());

      count_ = 0;
    }
  }

  std::size_t maxInputSize_ = 0;
  std::size_t count_ = 0;
  std::size_t collectSize_ = 1;

  std::shared_ptr<ContextInternal> ctx_;
  HostArray<T, 2> collectUVW_;
  DeviceArray<T, 2> collectUVWDevice_;
  HostArray<std::complex<T>, 1> collectValues_;
  DeviceArray<neonufft::gpu::ComplexType<T>, 1> collectValuesDevice_;
  neonufft::gpu::PlanT3<T, 3> plan_;
};
}  // namespace host
}  // namespace bipp
