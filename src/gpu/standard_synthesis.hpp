#pragma once

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class StandardSynthesis : public SynthesisInterface<T> {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nLevel,
                    HostArray<BippFilter, 1> filter, DeviceArray<T, 2> pixel);

  auto collect(T wl, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
               ConstView<T, 2> xyzUvwView) -> void override;

  auto get(BippFilter f, View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::Standard; }

  auto filter(std::size_t idx) const -> BippFilter override { return filter_[idx]; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto gpu_enabled() const -> bool override { return false; }

  auto image() -> View<T, 3> override { return img_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nFilter_, nPixel_, nLevel_;
  std::size_t count_;
  HostArray<BippFilter, 1> filter_;
  DeviceArray<T, 2> pixel_;
  DeviceArray<T, 3> img_;
};

}  // namespace gpu
}  // namespace bipp
