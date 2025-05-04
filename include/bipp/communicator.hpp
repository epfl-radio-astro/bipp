#pragma once

#include <bipp/config.h>

#ifdef BIPP_MPI
#include <mpi.h>
#endif

#include <cstddef>
#include <optional>

/*! \cond PRIVATE */
namespace bipp {

#ifdef BIPP_MPI
class CommunicatorInternal;
#endif

struct InternalCommunicatorAccessor;
/*! \endcond */

class BIPP_EXPORT Communicator {
public:
#ifdef BIPP_MPI
  /**
   * Create a Communicator using a custom MPI_Comm
   *
   * @param[in] comm Communicator to use. Will be dublicated internally, so MPI_Comm_free can be
   * safely called during this Communicator object lifetime.
   *
   * @return Communicator
   */
  static auto custom(const MPI_Comm& comm) -> Communicator;
#endif

  /**
   * Create a world communicator with all ranks launched through MPI.
   * Will return a local Communicator if not compiled with MPI.
   *
   * @return Communicator
   */
  static auto world() -> Communicator;

  /**
   * Create a local only Communicator of size 1.
   * Will not utilize any MPI call internally and not call MPI_Initialize().
   *
   * @return Communicator
   */
  static auto local() -> Communicator;

  /**
   * The process rank
   *
   * @return Rank
   */
  inline auto rank() const -> std::size_t { return rank_; }

  /**
   * The number size / number of processes.
   *
   * @return Size
   */
  inline auto size() const -> std::size_t { return size_; }

#ifdef BIPP_MPI
  /**
   * The mpi communicator used.
   *
   * @return MPI_Comm
   */
  auto mpi_handle() const -> const MPI_Comm&;
#endif


private:
  /*! \cond PRIVATE */

  Communicator() = default;

  std::size_t rank_ = 0;
  std::size_t size_ = 1;
#ifdef BIPP_MPI
  Communicator(const MPI_Comm& comm);

  std::optional<MPI_Comm> comm_;
#endif
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
