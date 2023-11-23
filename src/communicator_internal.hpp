#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>
#include <optional>

#include "bipp/config.h"
#include "bipp/nufft_synthesis.hpp"
#include "bipp/communicator.hpp"

#ifdef BIPP_MPI
#include "context_internal.hpp"
#include "host/domain_partition.hpp"
#include "memory/view.hpp"
#include "mpi_util/mpi_comm_handle.hpp"
#include "synthesis_interface.hpp"

namespace bipp {

class CommunicatorInternal {
  public:
    explicit CommunicatorInternal(MPI_Comm comm);

    CommunicatorInternal(const CommunicatorInternal&) = delete;

    CommunicatorInternal(CommunicatorInternal&&) = delete;

    inline auto is_root() const -> bool { return comm_.rank() == 0; }

    inline auto comm() const -> const MPICommHandle& { return comm_; }


    // Must be called by all non-root
    auto attach_non_root(std::shared_ptr<ContextInternal> ctx) -> void;

    // Must only be called by root
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    auto send_synthesis_init(std::optional<NufftSynthesisOptions> nufftOpt, std::size_t nLevel,
                             ConstHostView<BippFilter, 1> filter, ConstView<T, 1> pixelX,
                             ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ,
                             ConstHostView<PartitionGroup, 1> groups) -> std::size_t;

    // Must only be called by root
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    auto send_synthesis_collect(std::size_t id, ConstHostView<std::complex<T>, 2> vView,
                                ConstHostView<T, 2> dMasked, ConstHostView<T, 2> xyzUvwView)
        -> void;

    ~CommunicatorInternal();

  private:
    inline auto generate_local_id() -> std::size_t { return ++idCount_; }

    MPICommHandle comm_;
    std::size_t idCount_ = 0;
    bool rootDestroyed = false;
};

struct InternalCommunicatorAccessor {
  static auto get(const Communicator& comm) -> const std::shared_ptr<CommunicatorInternal>& {
    return comm.comm_;
  }
};

}  // namespace bipp
#endif
