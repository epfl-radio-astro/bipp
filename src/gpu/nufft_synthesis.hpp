#pragma once

#include "bipp/config.h"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/array.hpp"
#include "synthesis_interface.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class NufftSynthesis : public SynthesisInterface<T> {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                 std::size_t nLevel,
                 HostArray<BippFilter, 1> filter, DeviceArray<T, 2> pixel);

  auto collect(T wl, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
               ConstView<T, 2> xyzUvwView) -> void override;

  auto get(BippFilter f, View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::NUFFT; }

  auto filter(std::size_t idx) const -> BippFilter override { return filter_[idx]; }

  auto context() -> ContextInternal& override { return *ctx_; }

  auto gpu_enabled() const -> bool override { return false; }

  auto image() -> View<T, 3> override { return img_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nLevel_, nFilter_, nPixel_;
  HostArray<BippFilter, 1> filter_;
  DeviceArray<T, 2> pixel_;
  DomainPartition imgPartition_;

  std::size_t collectPoints_, totalCollectCount_;
  DeviceArray<api::ComplexType<T>, 3> virtualVis_;
  DeviceArray<T, 2> uvw_;
  DeviceArray<T, 3> img_;
};

}  // namespace gpu
}  // namespace bipp
