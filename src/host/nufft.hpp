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
  NUFFT(std::shared_ptr<ContextInternal> ctx, neonufft::Options opt, int sign,
        std::array<T, 3> uvwMin, std::array<T, 3> uvwMax, ConstHostView<T, 2> uvw,
        std::array<T, 3> pixelMin, std::array<T, 3> pixelMax, ConstHostView<T, 1> pixelX,
        ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
      : ctx_(std::move(ctx)), plan_(std::move(opt), sign, uvwMin, uvwMax, pixelMin, pixelMax) {
    plan_.set_input_points(uvw.shape(0),
                           {uvw.data(), uvw.slice_view(1).data(), uvw.slice_view(2).data()});
    plan_.set_output_points(pixelX.shape(0), {pixelX.data(), pixelY.data(), pixelZ.data()});
  }

  auto transform_and_add(ConstHostView<std::complex<T>, 1> values, HostView<float, 1> out)
      -> void override {
    plan_.add_input(values.data());

    HostArray<std::complex<T>, 1> outCpx(ctx_->host_alloc(), out.shape());

    plan_.transform(outCpx.data());
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft output", outCpx);

    const std::complex<T>* __restrict__ cpxPtr = outCpx.data();
    float* __restrict__ outPtr = out.data();
    for (std::size_t i = 0; i < out.size(); ++i) {
      outPtr[i] += cpxPtr[i].real();
    }
  }

private:
  std::shared_ptr<ContextInternal> ctx_;
  neonufft::PlanT3<T, 3> plan_;
};
}  // namespace host
}  // namespace bipp
