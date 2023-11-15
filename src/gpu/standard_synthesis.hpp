#pragma once

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class StandardSynthesis {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna, std::size_t nBeam,
                    std::size_t nIntervals, HostArray<BippFilter, 1> filter,
                    DeviceArray<T, 2> pixel);

  auto collect(T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               ConstHostView<api::ComplexType<T>, 2> sHost,
               ConstDeviceView<api::ComplexType<T>, 2> s, ConstDeviceView<api::ComplexType<T>, 2> w,
               ConstDeviceView<T, 2> xyz) -> void;

  auto get(BippFilter f, DeviceView<T, 2> out) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

  inline auto num_filter() const -> std::size_t { return nFilter_; }
  inline auto num_pixel() const -> std::size_t { return nPixel_; }
  inline auto num_intervals() const -> std::size_t { return nIntervals_; }
  inline auto num_antenna() const -> std::size_t { return nAntenna_; }
  inline auto num_beam() const -> std::size_t { return nBeam_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nFilter_, nPixel_, nIntervals_, nAntenna_, nBeam_;
  std::size_t count_;
  HostArray<BippFilter, 1> filter_;
  DeviceArray<T, 2> pixel_;
  DeviceArray<T, 3> img_;
};

}  // namespace gpu
}  // namespace bipp
