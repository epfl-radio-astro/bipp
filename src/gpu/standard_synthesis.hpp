#pragma once

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "bipp/standard_synthesis.hpp"
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
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, StandardSynthesisOptions opt,
                    std::size_t nLevel, DeviceArray<T, 2> pixel);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(View<T, 2> out) -> void override;

  auto get_psf(View<T, 1> out) -> void override { throw NotImplementedError(); }

  auto type() const -> SynthesisType override { return SynthesisType::Standard; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 2> override { return img_; }

  auto normalize_by_nvis() const -> bool override { return opt_.normalizeImageNvis; }

private:
  auto process_single(T wl, const std::size_t nVis, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
                      ConstView<T, 2> xyzUvwView) -> void;

  std::shared_ptr<ContextInternal> ctx_;
  StandardSynthesisOptions opt_;
  const std::size_t nPixel_, nImages_;
  std::size_t count_;
  std::size_t totalVisibilityCount_;
  DeviceArray<T, 2> pixel_;
  DeviceArray<T, 2> img_;
};

}  // namespace gpu
}  // namespace bipp
