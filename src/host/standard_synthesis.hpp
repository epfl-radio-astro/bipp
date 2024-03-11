#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "bipp/standard_synthesis.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

namespace bipp {
namespace host {

template <typename T>
class StandardSynthesis : public SynthesisInterface<T> {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, StandardSynthesisOptions opt,
                    std::size_t nImages, ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                    ConstHostView<T, 1> pixelZ);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::Standard; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 2> override { return img_; }

private:
  auto process_single(T wl, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
                      ConstView<T, 2> xyzUvwView) -> void;

  std::shared_ptr<ContextInternal> ctx_;
  StandardSynthesisOptions opt_;
  const std::size_t nPixel_, nImages_;
  std::size_t count_;
  HostArray<T, 2> pixel_;
  HostArray<T, 2> img_;
};

}  // namespace host
}  // namespace bipp
