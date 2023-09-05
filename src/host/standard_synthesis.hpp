#pragma once

#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace host {

template <typename T>
class StandardSynthesis {
public:
  StandardSynthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna, std::size_t nBeam,
                    std::size_t nIntervals, std::size_t nFilter, const BippFilter* filter,
                    std::size_t nPixel, const T* pixelX, const T* pixelY, const T* pixelZ);

  auto collect(std::size_t nEig, T wl, const T* intervals, std::size_t ldIntervals,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const std::size_t nz_vis) -> void;

  auto get(BippFilter f, T* out, std::size_t ld) -> void;

  auto context() -> ContextInternal& { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  std::size_t count_;
  Buffer<BippFilter> filter_;
  Buffer<T> pixelX_, pixelY_, pixelZ_;
  Buffer<T> img_;
};

}  // namespace host
}  // namespace bipp
