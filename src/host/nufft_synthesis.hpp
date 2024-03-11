#pragma once

#include <complex>
#include <memory>
#include <optional>
#include <vector>
#include <tuple>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "host/domain_partition.hpp"
#include "synthesis_interface.hpp"

namespace bipp {
namespace host {

template <typename T>
class NufftSynthesis : public SynthesisInterface<T> {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                 std::size_t nImages, ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                 ConstHostView<T, 1> pixelZ);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::NUFFT; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 2> override { return img_; }

private:

  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nImages_, nPixel_;
  HostArray<T, 2> pixel_;
  DomainPartition imgPartition_;

  std::size_t totalCollectCount_;
  HostArray<T, 2> img_;
};

}  // namespace host
}  // namespace bipp
