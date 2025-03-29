#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <neonufft/plan.hpp>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "memory/copy.hpp"
#include "context_internal.hpp"
#include "nufft_interface.hpp"

namespace bipp {
namespace host {
template <typename T>
class NUFFT : public NUFFTInterface<T> {
public:
  NUFFT(std::shared_ptr<ContextInternal> ctx, std::size_t collectSize, std::size_t maxInputSize,
        neonufft::Options opt, int sign, std::array<T, 3> input_min, std::array<T, 3> input_max,
        std::array<T, 3> output_min, std::array<T, 3> output_max)
      : maxInputSize_(maxInputSize),
        count_(0),
        collectSize_(collectSize),
        ctx_(std::move(ctx)),
        collectUVW_(ctx_->host_alloc(), {maxInputSize * collectSize, 3}),
        collectValues_(ctx_->host_alloc(), maxInputSize * collectSize),
        plan_(std::move(opt), sign, input_min, input_max, output_min, output_max) {}

  auto set_output_points(ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                         ConstHostView<T, 1> pixelZ) -> void override {
    plan_.set_output_points(pixelX.size(), {pixelX.data(), pixelY.data(), pixelZ.data()});
  }

  auto add_input(ConstHostView<T, 2> uvw, ConstHostView<std::complex<T>, 1> values) -> void override {
    copy(uvw, collectUVW_.sub_view({count_, 0}, uvw.shape()));
    copy(values, collectValues_.sub_view(count_, values.shape()));

    count_ += values.size();

    if (count_ + maxInputSize_ >= collectValues_.size()) {
      this->add_to_plan();
    }
  }

  auto transform_and_add(HostView<float, 1> out) -> void override {
    this->add_to_plan();

    HostArray<std::complex<T>, 1> outCpx(ctx_->host_alloc(), out.shape());

    plan_.transform(outCpx.data());

    const std::complex<T>* __restrict__ cpxPtr = outCpx.data();
    float* __restrict__ outPtr = out.data();
    for(std::size_t i = 0; i < out.size(); ++i) {
      outPtr[i] += cpxPtr[i].real();
    }
  }

private:
  auto add_to_plan() {
    if (count_) {
      plan_.set_input_points(count_,
                             {collectUVW_.slice_view(0).data(), collectUVW_.slice_view(1).data(),
                              collectUVW_.slice_view(2).data()});
      plan_.add_input(collectValues_.data());

      count_ = 0;
    }
  }

  std::size_t maxInputSize_ = 0;
  std::size_t count_ = 0;
  std::size_t collectSize_ = 1;

  std::shared_ptr<ContextInternal> ctx_;
  HostArray<T, 2> collectUVW_;
  HostArray<std::complex<T>, 1> collectValues_;
  neonufft::PlanT3<T, 3> plan_;
};
}  // namespace host
}  // namespace bipp
