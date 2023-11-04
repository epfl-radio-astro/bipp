#pragma once

#include "bipp/config.h"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class NufftSynthesis {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                 std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                 HostArray<BippFilter, 1> filter, DeviceArray<T, 2> pixel);

  auto collect(std::size_t nEig, T wl, ConstHostView<T, 2> intervals,
               ConstHostView<api::ComplexType<T>, 2> sHost,
               ConstDeviceView<api::ComplexType<T>, 2> s, ConstDeviceView<api::ComplexType<T>, 2> w,
               ConstDeviceView<T, 2> xyz, ConstDeviceView<T, 2> uvw) -> void;

  auto get(BippFilter f, DeviceView<T, 2> out) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

  inline auto num_filter() const -> std::size_t { return nFilter_; }
  inline auto num_pixel() const -> std::size_t { return nPixel_; }
  inline auto num_intervals() const -> std::size_t { return nIntervals_; }
  inline auto num_antenna() const -> std::size_t { return nAntenna_; }
  inline auto num_beam() const -> std::size_t { return nBeam_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  HostArray<BippFilter, 1> filter_;
  DeviceArray<T, 2> pixel_;
  DomainPartition imgPartition_;

  std::size_t nMaxInputCount_, collectCount_, totalCollectCount_;
  DeviceArray<api::ComplexType<T>, 3> virtualVis_;
  DeviceArray<T, 2> uvw_;
  DeviceArray<T, 3> img_;
};

}  // namespace gpu
}  // namespace bipp
