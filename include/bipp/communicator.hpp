#pragma once

#include <bipp/config.h>

#ifdef BIPP_MPI
#include <mpi.h>
#endif

#include <cstddef>
#include <memory>
#include <type_traits>

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
  static auto custom(const MPI_Comm& comm);

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
   * Check if calling process is root.
   *
   * @return True or false
   */
  auto is_root() const -> bool;

  /**
   * The process rank
   *
   * @return Rank
   */
  auto rank() const -> std::size_t;

  /**
   * The number size / number of processes.
   *
   * @return Size
   */
  auto size() const -> std::size_t;

private:
  /*! \cond PRIVATE */
  friend InternalCommunicatorAccessor;

  Communicator() = default;

#ifdef BIPP_MPI
  Communicator(const MPI_Comm& comm);

  std::shared_ptr<CommunicatorInternal> comm_;
#endif
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
