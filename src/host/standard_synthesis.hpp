#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "memory/array.hpp"

namespace bipp {
namespace host {

template <typename T>
class StandardSynthesis {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna, std::size_t nBeam,
                    std::size_t nIntervals, ConstHostView<BippFilter, 1> filter,
                    ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                    ConstHostView<T, 1> pixelZ);

  auto collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
               ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
               ConstHostView<T, 2> xyz) -> void;

  auto get(BippFilter f, HostView<T, 2> out) -> void;

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
  HostArray<T, 2> pixel_;
  HostArray<T, 3> img_;
};

}  // namespace host
}  // namespace bipp
