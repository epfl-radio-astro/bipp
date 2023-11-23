#pragma once

#include <cstddef>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "bipp/nufft_synthesis.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "synthesis_interface.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {

template <typename T>
struct SynthesisFactory {
  static auto create_standard_synthesis(std::shared_ptr<ContextInternal> ctx, std::size_t nLevel,
                                        ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                                        ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ)
      -> std::unique_ptr<SynthesisInterface<T>>;

  static auto create_nufft_synthesis(std::shared_ptr<ContextInternal> ctx,
                                     NufftSynthesisOptions opt, std::size_t nLevel,
                                     ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                                     ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ)
      -> std::unique_ptr<SynthesisInterface<T>>;

#ifdef BIPP_MPI
  static auto create_distributed_standard_synthesis(
      std::shared_ptr<CommunicatorInternal> comm, std::shared_ptr<ContextInternal> ctx,
      std::size_t nLevel, ConstView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
      ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ) -> std::unique_ptr<SynthesisInterface<T>>;

  static auto create_distributed_nufft_synthesis(std::shared_ptr<CommunicatorInternal> comm,
                                                 std::shared_ptr<ContextInternal> ctx,
                                                 NufftSynthesisOptions opt, std::size_t nLevel,
                                                 ConstView<BippFilter, 1> filter,
                                                 ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                                 ConstView<T, 1> pixelZ)
      -> std::unique_ptr<SynthesisInterface<T>>;
#endif
};

}  // namespace bipp
