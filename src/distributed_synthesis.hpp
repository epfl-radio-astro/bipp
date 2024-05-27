#pragma once

#include <complex>
#include <cstddef>
#include <variant>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "bipp/nufft_synthesis.hpp"
#include "bipp/standard_synthesis.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#include "host/domain_partition.hpp"
#include "synthesis_interface.hpp"
#include "context_internal.hpp"
#include "memory/array.hpp"
#include "memory/view.hpp"

namespace bipp {

template <typename T>
class DistributedSynthesis : public SynthesisInterface<T> {
public:
  DistributedSynthesis(std::shared_ptr<CommunicatorInternal> comm,
                       std::shared_ptr<ContextInternal> ctx,
                       std::variant<NufftSynthesisOptions, StandardSynthesisOptions> opt,
                       std::size_t nLevel, ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                       ConstHostView<T, 1> pixelZ);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return type_; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 2> override { return img_; }

  auto normalize_by_nvis() const -> bool override { return normalize_by_nvis_; }

private:
  std::shared_ptr<CommunicatorInternal> comm_;
  std::shared_ptr<ContextInternal> ctx_;
  std::size_t id_;
  std::size_t totalCollectCount_;
  std::size_t totalVisibilityCount_;
  HostArray<T, 2> img_;
  host::DomainPartition imgPartition_;
  SynthesisType type_;
  bool normalize_by_nvis_;
};

}  // namespace bipp
#endif
