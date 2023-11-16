#pragma once

#include <complex>
#include <memory>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "host/domain_partition.hpp"

namespace bipp {
namespace host {

template <typename T>
class NufftSynthesis {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                 std::size_t nLevel, ConstHostView<BippFilter, 1> filter,
                 ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                 ConstHostView<T, 1> pixelZ);

  auto collect(T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               ConstHostView<std::complex<T>, 2> s, ConstHostView<std::complex<T>, 2> w,
               ConstHostView<T, 2> xyz, ConstHostView<T, 2> uvw) -> void;

  auto get(BippFilter f, HostView<T, 2> out) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

  inline auto num_filter() const -> std::size_t { return nFilter_; }
  inline auto num_pixel() const -> std::size_t { return nPixel_; }
  inline auto num_level() const -> std::size_t { return nLevel_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nLevel_, nFilter_, nPixel_;
  HostArray<BippFilter, 1> filter_;
  HostArray<T, 2> pixel_;
  DomainPartition imgPartition_;

  std::size_t collectPoints_, totalCollectCount_;
  HostArray<std::complex<T>, 3> virtualVis_;
  HostArray<T, 2> uvw_;
  HostArray<T, 3> img_;
};

}  // namespace host
}  // namespace bipp
