#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <numeric>
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

  NUFFT(std::shared_ptr<ContextInternal> ctx, neonufft::Options opt, int sign,
        std::array<T, 3> uvwMin, std::array<T, 3> uvwMax, ConstHostView<T, 2> uvw,
        std::array<T, 3> pixelMin, std::array<T, 3> pixelMax, ConstHostView<T, 1> pixelX,
        ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ)
      : ctx_(std::move(ctx)),
        plan_(std::move(opt), sign, uvwMin, uvwMax, pixelMin, pixelMax,
              ctx_->gpu_queue().stream()) {
    auto& queue = ctx_->gpu_queue();
    {
      auto uvwDevice = queue.create_device_array<T, 2>(uvw.shape());
      copy(queue, uvw, uvwDevice);
      plan_.set_input_points(uvwDevice.shape(0), {uvwDevice.data(), uvwDevice.slice_view(1).data(),
                                                  uvwDevice.slice_view(2).data()});
    }
    queue.sync(); // free up memory again
    {
      auto pixelXYZDevice = queue.create_device_array<T, 2>({pixelX.shape(0), 3});
      copy(queue, pixelX, pixelXYZDevice.slice_view(0));
      copy(queue, pixelY, pixelXYZDevice.slice_view(1));
      copy(queue, pixelZ, pixelXYZDevice.slice_view(2));
      plan_.set_output_points(pixelXYZDevice.shape(0),
                              {pixelXYZDevice.data(), pixelXYZDevice.slice_view(1).data(),
                               pixelXYZDevice.slice_view(2).data()});
    }
    queue.sync(); // free up memory again

    // TODO: set plan allocator
  }
  auto transform_and_add(ConstHostView<std::complex<T>, 1> values, HostView<float, 1> out)
      -> void override {
    auto& queue = ctx_->gpu_queue();

    auto valuesDevice = queue.create_device_array<std::complex<T>, 1>(values.shape());
    auto outCpx = queue.create_pinned_array<std::complex<T>, 1>(out.shape());
    auto outCpxDevice = queue.create_device_array<std::complex<T>, 1>(out.shape());

    copy(queue, values, valuesDevice);
    plan_.add_input(reinterpret_cast<const neonufft::gpu::ComplexType<T>*>(valuesDevice.data()));
    plan_.transform(reinterpret_cast<neonufft::gpu::ComplexType<T>*>(outCpxDevice.data()));

    copy(queue, outCpxDevice, outCpx);

    queue.sync();

    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "nufft output", outCpx);

    const std::complex<T>* __restrict__ cpxPtr = outCpx.data();
    float* __restrict__ outPtr = out.data();
    for (std::size_t i = 0; i < out.size(); ++i) {
      outPtr[i] += cpxPtr[i].real();
    }
  }

private:
  std::shared_ptr<ContextInternal> ctx_;
  neonufft::gpu::PlanT3<T, 3> plan_;
};
}  // namespace host
}  // namespace bipp
