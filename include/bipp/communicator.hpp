#pragma once

#include <bipp/config.h>


#ifdef BIPP_MPI
#include <bipp/context.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <mpi.h>

/*! \cond PRIVATE */
namespace bipp {

class CommunicatorInternal;
struct InternalCommunicatorAccessor;
/*! \endcond */

class BIPP_EXPORT Communicator {
public:
  /**
   * Create a Communicator.
   *
   * @param[in] comm Communicator to use. Will be dublicated internally, so MPI_Comm_free can be
   * safely called during this Communicator object lifetime.
   */
  explicit Communicator(const MPI_Comm& comm);

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

  /**
   * Attach to root process to enable usage support image synthesis.
   * Must be called by all non-root processes. Must NOT be called by the root process.
   * Will only return once the Communicator of the root process is destroyed.
   *
   * @param[in] ctx Context to use for processing.
   */
  auto attach_non_root(Context& ctx) -> void;

private:
  /*! \cond PRIVATE */
  friend InternalCommunicatorAccessor;

  std::shared_ptr<CommunicatorInternal> comm_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
#endif
