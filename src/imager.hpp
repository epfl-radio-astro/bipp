#pragma once

#include <complex>
#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "bipp/nufft_synthesis.hpp"
#include "bipp/standard_synthesis.hpp"
#include "collector_interface.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#include "distributed_synthesis.hpp"
#endif

namespace bipp {

template <typename T>
class Imager {
public:
  static auto standard_synthesis(std::shared_ptr<ContextInternal> ctx, StandardSynthesisOptions opt,
                                 std::size_t nImages, ConstView<T, 1> pixelX,
                                 ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> Imager<T>;

  static auto nufft_synthesis(std::shared_ptr<ContextInternal> ctx, NufftSynthesisOptions opt,
                              std::size_t nImages, ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                              ConstView<T, 1> pixelZ) -> Imager<T>;

#ifdef BIPP_MPI
  static auto distributed_standard_synthesis(std::shared_ptr<CommunicatorInternal> comm,
                                             std::shared_ptr<ContextInternal> ctx,
                                             StandardSynthesisOptions opt, std::size_t nImages,
                                             ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                             ConstView<T, 1> pixelZ) -> Imager<T>;

  static auto distributed_nufft_synthesis(std::shared_ptr<CommunicatorInternal> comm,
                                          std::shared_ptr<ContextInternal> ctx,
                                          NufftSynthesisOptions opt, std::size_t nImages,
                                          ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                          ConstView<T, 1> pixelZ) -> Imager<T>;
#endif


  auto collect(T wl, const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               ConstView<std::complex<T>, 2> s, ConstView<std::complex<T>, 2> w,
               ConstView<T, 2> xyz, ConstView<T, 2> uvw) -> void;

  auto get(T* out, std::size_t ld) -> void;

  auto get_psf(T* out) -> void;

  auto context() -> ContextInternal& { return *synthesis_->context(); };

private:
  Imager(std::shared_ptr<ContextInternal> ctx, std::unique_ptr<SynthesisInterface<T>> syn,
         std::size_t collectGroupSize);

  std::unique_ptr<SynthesisInterface<T>> synthesis_;
  std::unique_ptr<CollectorInterface<T>> collector_;

  std::size_t collectGroupSize_ = 0;
};

}  // namespace bipp
