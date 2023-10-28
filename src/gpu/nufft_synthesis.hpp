#pragma once

#include "bipp/config.h"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "gpu/domain_partition.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class NufftSynthesis {
public:
  NufftSynthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                 std::size_t nAntenna, std::size_t nBeam, std::size_t nIntervals,
                 std::size_t nFilter, const BippFilter* filterHost, std::size_t nPixel,
                 const T* lmnX, const T* lmnY, const T* lmnZ,
                 const bool filter_negative_eigenvalues);

  auto collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const api::ComplexType<T>* s, std::size_t lds, const api::ComplexType<T>* w,
               std::size_t ldw, const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw,
               const std::size_t nz_vis)
      -> void;

  auto get(BippFilter f, T* outHostOrDevice, std::size_t ld) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  NufftSynthesisOptions opt_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  Buffer<BippFilter> filterHost_;
  Buffer<T> lmnX_, lmnY_, lmnZ_;
  DomainPartition imgPartition_;

  std::size_t nMaxInputCount_, collectCount_, totalCollectCount_;
  Buffer<api::ComplexType<T>> virtualVis_;
  Buffer<T> uvwX_, uvwY_, uvwZ_;
  Buffer<T> img_;
};

}  // namespace gpu
}  // namespace bipp
