#pragma once

#include <complex>
#include <cstddef>
#include <optional>

#include "bipp/bipp.h"
#include "bipp/config.h"

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
                       std::optional<NufftSynthesisOptions> nufftOpt, std::size_t nLevel,
                       ConstHostView<BippFilter, 1> filter, ConstHostView<T, 1> pixelX,
                       ConstHostView<T, 1> pixelY, ConstHostView<T, 1> pixelZ);

  auto process(CollectorInterface<T>& collector) -> void override;

  auto get(BippFilter f, View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return type_; }

  auto filter(std::size_t idx) const -> BippFilter override { return filter_[idx]; }

  auto context() -> const std::shared_ptr<ContextInternal>& override { return ctx_; }

  auto image() -> View<T, 3> override { return img_; }

private:
  std::shared_ptr<CommunicatorInternal> comm_;
  std::shared_ptr<ContextInternal> ctx_;
  std::size_t id_;
  std::size_t totalCollectCount_;
  HostArray<T, 3> img_;
  HostArray<BippFilter, 1> filter_;
  host::DomainPartition imgPartition_;
  SynthesisType type_;

};

}  // namespace bipp
#endif
