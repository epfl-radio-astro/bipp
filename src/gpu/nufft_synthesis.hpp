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
                 std::size_t nLevel, DeviceArray<T, 2> pixel);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::NUFFT; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 2> override { return img_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nImages_, nPixel_;
  DeviceArray<T, 2> pixel_;
  DomainPartition imgPartition_;

  std::size_t totalCollectCount_;
  DeviceArray<T, 2> img_;
};

}  // namespace gpu
}  // namespace bipp
