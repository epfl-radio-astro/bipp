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

  auto collect(T wl, ConstView<std::complex<T>, 2> vView, ConstHostView<T, 2> dMasked,
               ConstView<T, 2> xyzUvwView) -> void override;

  auto get(BippFilter f, View<T, 2> out) -> void override;

  auto type() const -> SynthesisType override { return SynthesisType::Standard; }

  auto context() -> ContextInternal& override { return *ctx_; }

  auto gpu_enabled() const -> bool override { return ctx_->processing_unit() == BIPP_PU_GPU; }

  auto image() -> View<T, 3> override { return img_; }

private:
  std::shared_ptr<CommunicatorInternal> comm_;
  std::shared_ptr<ContextInternal> ctx_;
  std::size_t id_;
  std::size_t count_;
  HostArray<T, 3> img_;
  host::DomainPartition imgPartition_;

};

}  // namespace bipp
#endif
