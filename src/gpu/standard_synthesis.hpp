#pragma once

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class StandardSynthesis {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna, std::size_t nBeam,
                    std::size_t nIntervals, std::size_t nFilter, const BippFilter* filterHost,
                    std::size_t nPixel, const T* pixelX, const T* pixelY, const T* pixelZ,
                    const bool filter_negative_eigenvalues);

  auto collect(std::size_t nEig, T wl, const T* intervalsHost, std::size_t ldIntervals,
               const api::ComplexType<T>* s, std::size_t lds, const api::ComplexType<T>* w,
               std::size_t ldw, T* xyz, std::size_t ldxyz, const std::size_t nz_vis) -> void;

  auto get(BippFilter f, T* outHostOrDevice, std::size_t ld) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  Buffer<BippFilter> filterHost_;
  Buffer<T> pixelX_, pixelY_, pixelZ_;
  Buffer<T> img_;
  const bool filter_negative_eigenvalues_;
};

}  // namespace gpu
}  // namespace bipp
